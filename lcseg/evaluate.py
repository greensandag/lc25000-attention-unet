"""Evaluation (Section III.H) and the comparison with the mask generator (Section IV.C).

All numbers are computed on the test partition only. Per-image pixel counts are stored so
that every table can be recomputed without re-running the network.
"""
import os
import time

import cv2
import numpy as np

from .config import BENIGN, CANCER, SIZE
from .masks import generate_mask
from .utils import jdump, read_meta, read_split, unpack


def counts(pred, ref):
    """Per-image (TP, FP, FN, TN) for binary arrays of shape (N, H, W)."""
    p = pred.reshape(len(pred), -1).astype(bool)
    r = ref.reshape(len(ref), -1).astype(bool)
    tp = (p & r).sum(1)
    fp = (p & ~r).sum(1)
    fn = (~p & r).sum(1)
    tn = p.shape[1] - tp - fp - fn
    return np.stack([tp, fp, fn, tn], 1).astype(np.int64)


def boundary_errors(pred, ref, px=2):
    """Per-image number of error pixels and of error pixels within `px` pixels of a reference boundary."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    tot, band = [], []
    for p, r in zip(pred, ref):
        b = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_GRADIENT, k).astype(bool)
        e = p.astype(bool) != r.astype(bool)
        tot.append(int(e.sum()))
        band.append(int((e & b).sum()))
    return np.array(tot), np.array(band)


def test_arrays(out):
    split, _ = read_split(out)
    te = np.where(split == "test")[0]
    X = np.load(os.path.join(out, "cache", "X.npy"), mmap_mode="r")
    Y = np.load(os.path.join(out, "cache", "Y.npy"))[te]
    meta = read_meta(out)
    cls = np.array([meta[i]["cls"] for i in te])
    return te, X, Y, cls


def predict_probs(model, X, rows, batch=32, chunk=256):
    out = []
    for s in range(0, len(rows), chunk):
        x = np.asarray(X[rows[s:s + chunk]]).astype(np.float32) / 255.0
        p = model.predict(x, batch_size=batch, verbose=0)
        if isinstance(p, dict):
            p = p["main"]
        out.append(np.asarray(p, np.float32)[..., 0])
    return np.concatenate(out)


def per_image_dice(c):
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)


def evaluate_run(cfg, out, variant, seed, model=None):
    from .train import setup_tf
    rdir = os.path.join(out, "runs", f"{variant}_s{seed}")
    if model is None:
        tf, keras = setup_tf()
        model = keras.models.load_model(os.path.join(rdir, "model.keras"), compile=False)
    e = cfg["EVAL"]
    te, X, Yp, cls = test_arrays(out)
    grid = np.array(e["thr_grid"], np.float32)
    n = len(te)
    cnt = np.zeros((n, 4), np.int64)
    cnt_grid = np.zeros((len(grid), n, 4), np.int64)
    area = np.zeros(n, np.float64)
    etot = np.zeros(n, np.int64)
    eband = np.zeros(n, np.int64)
    pred_s = 0.0
    for s in range(0, n, 256):
        sl = slice(s, min(s + 256, n))
        t0 = time.perf_counter()
        prob = predict_probs(model, X, te[sl])
        pred_s += time.perf_counter() - t0
        ref = unpack(Yp[sl])
        pred = (prob >= e["thr"]).astype(np.uint8)
        cnt[sl] = counts(pred, ref)
        for gi, th in enumerate(grid):
            cnt_grid[gi, sl] = counts((prob >= th).astype(np.uint8), ref)
        area[sl] = pred.reshape(len(pred), -1).mean(1)
        etot[sl], eband[sl] = boundary_errors(pred, ref, e["boundary_px"])
    dice = per_image_dice(cnt)

    # images for the figures: one typical image per class (median Dice / median FP area),
    # the three colon_aca images and the lung_aca and lung_scc images with the lowest Dice,
    # and the benign image with the largest false positive area
    pick = {}
    for c in CANCER:
        ii = np.where(cls == c)[0]
        pick[f"typical_{c}"] = int(ii[np.argsort(dice[ii])[len(ii) // 2]])
    for c in BENIGN:
        ii = np.where(cls == c)[0]
        pick[f"typical_{c}"] = int(ii[np.argsort(cnt[ii, 1])[len(ii) // 2]])
    ca = np.where(cls == "colon_aca")[0]
    for r, j in enumerate(ca[np.argsort(dice[ca])[:3]]):
        pick[f"worst_colon_aca_{r + 1}"] = int(j)
    for c in ["lung_aca", "lung_scc"]:
        ii = np.where(cls == c)[0]
        pick[f"worst_{c}"] = int(ii[np.argmin(dice[ii])])
    bb = np.where(np.isin(cls, BENIGN))[0]
    pick["worst_benign_fp"] = int(bb[np.argmax(cnt[bb, 1])])
    sel = np.array(sorted(set(pick.values())))
    probs = predict_probs(model, X, te[sel])
    np.savez_compressed(os.path.join(rdir, "eval.npz"), test_idx=te, cls=cls, counts=cnt,
                        counts_grid=cnt_grid, grid=grid, area=area, err_tot=etot, err_band=eband,
                        fig_rows=sel, fig_probs=probs.astype(np.float16),
                        fig_keys=np.array(list(pick.keys())), fig_vals=np.array(list(pick.values())))
    jdump(dict(predict_seconds_all_test=pred_s, n_test=int(n)), os.path.join(rdir, "eval_time.json"))
    print(f"[eval] {variant} s{seed}: cancer per-image Dice mean "
          f"{100 * dice[np.isin(cls, CANCER)].mean():.2f}", flush=True)


def stage_generator(cfg, out):
    """Apply the mask generator to all test images WITHOUT class labels (Table 7) and time it."""
    gdir = os.path.join(out, "generator")
    if os.path.exists(os.path.join(gdir, "generator.json")):
        print("[generator] exists, skipping")
        return
    os.makedirs(gdir, exist_ok=True)
    te, X, Yp, cls = test_arrays(out)
    cv2.setNumThreads(1)                       # one CPU core
    area = np.zeros(len(te))
    times = []
    for k, i in enumerate(te):
        rgb = np.asarray(X[i])
        t0 = time.perf_counter()
        m, _ = generate_mask(rgb, cfg["MASK"], qc=True)
        times.append(time.perf_counter() - t0)
        area[k] = m.mean()
    np.savez_compressed(os.path.join(gdir, "generator.npz"), area=area, cls=cls)
    jdump(dict(ms_per_image_1core=1000 * float(np.median(times)),
               ms_per_image_1core_mean=1000 * float(np.mean(times))),
          os.path.join(gdir, "generator.json"))
    print("[generator] done", flush=True)


def stage_bench(cfg, out, variants=("full",), seed=0):
    """Parameters, GFLOPs, file size, GPU/CPU latency, GPU throughput and peak memory (Table 9)."""
    from .model import conv_flops
    from .train import setup_tf
    tf, keras = setup_tf()
    bdir = os.path.join(out, "bench")
    os.makedirs(bdir, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    res = {}
    try:
        res["cpu_name"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                               if l.startswith("model name"))
    except Exception:
        res["cpu_name"] = None
    if gpus:
        try:
            res["gpu_name"] = tf.config.experimental.get_device_details(gpus[0]).get("device_name")
        except Exception:
            res["gpu_name"] = "GPU"

    def timed(fn, x, n, warm):
        for _ in range(warm):
            np.asarray(fn(x))
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            np.asarray(fn(x))
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    for v in variants:
        path = os.path.join(out, "runs", f"{v}_s{seed}", "model.keras")
        if not os.path.exists(path):
            continue
        m = keras.models.load_model(path, compile=False)
        r = dict(params=int(m.count_params()), gflops=conv_flops(m) / 1e9, size_mb=os.path.getsize(path) / 1e6)
        x1 = np.random.rand(1, SIZE, SIZE, 3).astype(np.float32)
        x32 = np.random.rand(32, SIZE, SIZE, 3).astype(np.float32)
        if gpus:
            with tf.device("/GPU:0"):
                f = tf.function(lambda x: m(x, training=False))
                r["gpu_ms_b1"] = 1000 * timed(f, x1, 200, 20)
                try:
                    tf.config.experimental.reset_memory_stats("GPU:0")
                except Exception:
                    pass
                r["gpu_img_per_s_b32"] = 32 / timed(f, x32, 50, 5)
                try:
                    r["gpu_peak_mb_b32"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
                except Exception:
                    r["gpu_peak_mb_b32"] = None
        with tf.device("/CPU:0"):
            fc = tf.function(lambda x: m(x, training=False))
            r["cpu_ms_b1"] = 1000 * timed(fc, x1, 30, 5)
        res[v] = r
        print("[bench]", v, r, flush=True)
    jdump(res, os.path.join(bdir, "bench.json"))
