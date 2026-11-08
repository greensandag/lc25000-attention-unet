"""Stage `report`: compute every number of Tables 5-11 and draw the result figures.

Output: <out>/report/results.json (all values), results.md (tables in manuscript order),
and the figures. Numbers in the manuscript should be copied from these files only.
"""
import os

import numpy as np

from .config import BENIGN, CANCER, CLASSES
from .evaluate import per_image_dice
from .utils import jdump, jload

VARIANT_NAME = {"unet": "U-Net", "att": "Attention U-Net", "full": "Complete model",
                "full_nofocal": "Complete model, BCE + Dice loss"}


def pooled(c):
    tp, fp, fn, tn = [float(v) for v in np.asarray(c).reshape(-1, 4).sum(0)]
    nan = float("nan")
    return dict(dice=100 * 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan,
                iou=100 * tp / (tp + fp + fn) if tp + fp + fn else nan,
                pixel_acc=100 * (tp + tn) / (tp + fp + fn + tn),
                precision=100 * tp / (tp + fp) if tp + fp else nan,
                recall=100 * tp / (tp + fn) if tp + fn else nan,
                fp_share=100 * fp / (tp + fp + fn + tn))


def load_run(out, variant, seed):
    rdir = os.path.join(out, "runs", f"{variant}_s{seed}")
    if not os.path.exists(os.path.join(rdir, "eval.npz")):
        return None
    z = dict(np.load(os.path.join(rdir, "eval.npz"), allow_pickle=False))
    z["run"] = jload(os.path.join(rdir, "run.json"))
    tpath = os.path.join(rdir, "eval_time.json")
    z["time"] = jload(tpath) if os.path.exists(tpath) else {}
    return z


def run_summary(z):
    cls, c = z["cls"], z["counts"]
    can, ben = np.isin(cls, CANCER), np.isin(cls, BENIGN)
    d = per_image_dice(c)
    r = dict(cancer=pooled(c[can]), all=pooled(c), per_class={})
    for k in CLASSES:
        r["per_class"][k] = pooled(c[cls == k])
        r["per_class"][k]["n"] = int(np.sum(cls == k))
    r["n"] = dict(cancer=int(can.sum()), benign=int(ben.sum()), all=int(len(cls)))
    r["benign_fp_share"] = pooled(c[ben])["fp_share"]
    r["per_image_dice_mean"] = 100 * float(d[can].mean())
    r["epochs_run"] = z["run"]["epochs_run"]
    r["best_epoch"] = z["run"]["best_epoch"]
    r["train_hours"] = z["run"]["train_hours"]
    return r


def per_image_stats(z, low):
    cls, c = z["cls"], z["counts"]
    can = np.isin(cls, CANCER)
    d = per_image_dice(c)[can]
    cc = cls[can]
    below = d < low
    et, eb = z["err_tot"][can], z["err_band"][can]
    return dict(mean=100 * float(d.mean()), sd=100 * float(d.std(ddof=1)), median=100 * float(np.median(d)),
                min=100 * float(d.min()), max=100 * float(d.max()),
                pct_below=100 * float(below.mean()), n_below=int(below.sum()),
                colon_aca_share_below=100 * float(np.mean(cc[below] == "colon_aca")) if below.any() else float("nan"),
                per_class_mean={k: 100 * float(d[cc == k].mean()) for k in CANCER},
                boundary_error_pct=100 * float(eb.sum() / max(et.sum(), 1)))


def image_level(area, cls, thr):
    from sklearn.metrics import roc_auc_score
    can, ben = np.isin(cls, CANCER), np.isin(cls, BENIGN)
    pos = area >= thr
    return dict(benign_pixels_pct=100 * float(area[ben].mean()),
                benign_images_region_pct=100 * float(pos[ben].mean()),
                sensitivity=100 * float(pos[can].mean()),
                specificity=100 * float((~pos[ben]).mean()),
                auroc=float(roc_auc_score(can[can | ben].astype(int), area[can | ben])))


def paired(za, zb, n_boot, seed):
    from scipy.stats import wilcoxon
    can = np.isin(za["cls"], CANCER)
    assert np.array_equal(za["test_idx"], zb["test_idx"])
    a, b = per_image_dice(za["counts"])[can], per_image_dice(zb["counts"])[can]
    diff = 100 * (b - a)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return dict(mean_diff=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                wilcoxon_p=float(wilcoxon(b, a).pvalue), n=int(len(diff)))


def mean_sd(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)), values=v.tolist())


def stage_report(cfg, out):
    e = cfg["EVAL"]
    rdir = os.path.join(out, "report")
    os.makedirs(rdir, exist_ok=True)
    R = dict(groups=jload(os.path.join(out, "groups", "group_stats.json")),
             prepare=jload(os.path.join(out, "cache", "prepare.json")))
    runs = {v: {s: load_run(out, v, s) for s in seeds} for v, seeds in cfg["RUNS"].items()}
    done = {v: {s: z for s, z in d.items() if z is not None} for v, d in runs.items()}
    pending = [f"{v}:{s}" for v, d in runs.items() for s, z in d.items() if z is None]
    R["pending_runs"] = pending
    if pending:
        print("[report] WARNING: runs not finished:", pending)

    main = done.get("full", {}).get(0)
    if main is not None:
        R["table5"] = run_summary(main)
        R["table6"] = per_image_stats(main, e["low_dice"])
        R["table7_network"] = image_level(main["area"], main["cls"], e["img_area_thr"])
        grid_dice = [pooled(main["counts_grid"][i])["dice"] for i in range(len(main["grid"]))]
        best = int(np.nanargmax(grid_dice))
        R["table8"] = dict(fixed_thr=e["thr"], dice_all_fixed=R["table5"]["all"]["dice"],
                           best_thr=float(main["grid"][best]), dice_all_best=float(grid_dice[best]),
                           grid=[float(t) for t in main["grid"]], grid_dice=grid_dice)
    gpath = os.path.join(out, "generator", "generator.npz")
    if os.path.exists(gpath):
        g = np.load(gpath)
        R["table7_generator"] = image_level(g["area"], g["cls"], e["img_area_thr"])
        R["generator_timing"] = jload(os.path.join(out, "generator", "generator.json"))
    bpath = os.path.join(out, "bench", "bench.json")
    if os.path.exists(bpath):
        R["table9"] = jload(bpath)
        if main is not None:
            R["table9"]["predict_seconds_all_test"] = main["time"].get("predict_seconds_all_test")
    R["table10"] = {v: run_summary(d[0]) for v, d in done.items() if 0 in d}
    R["table11"] = {}
    for a, b, name in [("unet", "att", "attention gates"), ("att", "full", "deep supervision"),
                       ("full_nofocal", "full", "focal term"), ("unet", "full", "all components")]:
        if 0 in done.get(a, {}) and 0 in done.get(b, {}):
            R["table11"][f"{a}->{b}"] = dict(change=name, **paired(done[a][0], done[b][0], e["n_boot"], e["boot_seed"]))
    R["seeds"] = {}
    for v, d in done.items():
        if len(d) > 1:
            S = [run_summary(z) for z in d.values()]
            R["seeds"][v] = dict(seeds=sorted(d.keys()),
                                 cancer_dice=mean_sd([s["cancer"]["dice"] for s in S]),
                                 cancer_iou=mean_sd([s["cancer"]["iou"] for s in S]),
                                 per_image_dice=mean_sd([s["per_image_dice_mean"] for s in S]),
                                 all_dice=mean_sd([s["all"]["dice"] for s in S]),
                                 benign_fp=mean_sd([s["benign_fp_share"] for s in S]),
                                 epochs=[s["epochs_run"] for s in S], best_epoch=[s["best_epoch"] for s in S])
    jdump(R, os.path.join(rdir, "results.json"))
    write_markdown(R, os.path.join(rdir, "results.md"))
    try:
        from .figures import make_figures
        make_figures(cfg, out, done, R)
    except Exception as ex:          # figures must never block the numbers
        print("[report] figure error:", repr(ex))
    print("[report] written to", rdir, flush=True)
    return R


def _f(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def write_markdown(R, path):
    L = ["# Results (generated by `lcseg report`; copy numbers from here only)", ""]
    g = R["groups"]
    L += ["## Source grouping (Section III.C)", "",
          f"- groups per class: {g['groups_per_class']}",
          f"- singletons: {g.get('singletons')}",
          f"- split: {g['n_split']}; test per class: {g.get('n_test_per_class')}",
          f"- test images with a verified copy in train after the group split: {g['leak_after']}",
          f"- random image-level split: {100 * g['leak_random_frac']:.1f}% of test images share a source group with train",
          f"- mask QC actions: {R['prepare']['qc']}; mean coverage cancer {100 * R['prepare']['coverage_cancer_mean']:.1f}%, "
          f"all {100 * R['prepare']['coverage_all_mean']:.1f}%", ""]
    if "table5" in R:
        t = R["table5"]
        L += ["## Table 5 (complete model, seed 0, threshold 0.5, %)", "",
              "| Class / subset | Type | n | Dice | IoU | Pixel Acc. | Precision | Recall |", "|---|---|---|---|---|---|---|---|"]
        rows = [("Cancer only", "Cancer", t["n"]["cancer"], t["cancer"])]
        rows += [(k, "Cancer", t["per_class"][k]["n"], t["per_class"][k]) for k in CANCER]
        rows += [(k, "Benign", t["per_class"][k]["n"], t["per_class"][k]) for k in BENIGN]
        rows += [("All images", "All", t["n"]["all"], t["all"])]
        for name, typ, n, m in rows:
            if typ == "Benign":
                L.append(f"| {name} | {typ} | {n:,} | Undefined | Undefined | {_f(m['pixel_acc'])} | Undefined | Undefined |")
            else:
                L.append(f"| {name} | {typ} | {n:,} | {_f(m['dice'])} | {_f(m['iou'])} | {_f(m['pixel_acc'])} | "
                         f"{_f(m['precision'])} | {_f(m['recall'])} |")
        L += ["", f"Benign pixels marked as target: lung_n {_f(t['per_class']['lung_n']['fp_share'])}%, "
                  f"colon_n {_f(t['per_class']['colon_n']['fp_share'])}%", ""]
        s = R["table6"]
        L += ["## Table 6 (per-image Dice, cancer test images)", "", "| Statistic | Value |", "|---|---|",
              f"| Mean ± SD (%) | {_f(s['mean'])} ± {_f(s['sd'])} |", f"| Median (%) | {_f(s['median'])} |",
              f"| Range (%) | {_f(s['min'])}–{_f(s['max'])} |",
              f"| Images with Dice below 85% (%) | {_f(s['pct_below'])} ({s['n_below']} images) |",
              f"| Share of colon_aca among them (%) | {_f(s['colon_aca_share_below'])} |",
              "| Mean per-image Dice lung_aca / lung_scc / colon_aca (%) | "
              + " / ".join(_f(s["per_class_mean"][k]) for k in CANCER) + " |",
              f"| Error pixels within 2 px of a reference boundary (%) | {_f(s['boundary_error_pct'])} |", ""]
    if "table7_network" in R:
        n_, g_ = R["table7_network"], R.get("table7_generator", {})
        L += ["## Table 7 (image-level recognition)", "", "| Measure | Network | Mask generator (no class label) |",
              "|---|---|---|"]
        for k, lab, d in [("benign_pixels_pct", "Benign test images: pixels marked as target (%)", 2),
                          ("benign_images_region_pct", "Benign images with a region ≥ 1% (%)", 2),
                          ("sensitivity", "Image-level sensitivity, cancer images (%)", 2),
                          ("specificity", "Image-level specificity, benign images (%)", 2),
                          ("auroc", "AUROC of the predicted area fraction", 3)]:
            L.append(f"| {lab} | {_f(n_[k], d)} | {_f(g_.get(k), d)} |")
        L.append("")
    if "table8" in R:
        t = R["table8"]
        L += ["## Table 8 (threshold, reference only)", "", "| Threshold | Selected on | Dice, all test images (%) |",
              "|---|---|---|", f"| {t['fixed_thr']:.2f} (fixed) | Not tuned | {_f(t['dice_all_fixed'])} |",
              f"| {t['best_thr']:.2f} (grid 0.20–0.85) | Test images (reference only) | {_f(t['dice_all_best'])} |", ""]
    if "table9" in R:
        b = R["table9"].get("full", {})
        gt = R.get("generator_timing", {})
        L += ["## Table 9 (computational cost, complete model)", "", "| Measure | Value |", "|---|---|",
              f"| Parameters (inference graph) | {b.get('params', 0):,} |",
              f"| Convolution GFLOPs per 256×256 image | {_f(b.get('gflops'))} |",
              f"| Model file size (MB) | {_f(b.get('size_mb'))} |",
              f"| GPU latency, batch 1 (ms) | {_f(b.get('gpu_ms_b1'), 1)} |",
              f"| GPU throughput, batch 32 (images/s) | {_f(b.get('gpu_img_per_s_b32'), 1)} |",
              f"| GPU peak memory, batch 32 (MB) | {_f(b.get('gpu_peak_mb_b32'), 0)} |",
              f"| CPU latency, batch 1 (ms) | {_f(b.get('cpu_ms_b1'), 1)} |",
              f"| Mask generator, one CPU core (ms/image) | {_f(gt.get('ms_per_image_1core'), 1)} |",
              f"| Prediction time for all test images (s) | {_f(R['table9'].get('predict_seconds_all_test'), 1)} |",
              f"| GPU / CPU | {R['table9'].get('gpu_name')} / {R['table9'].get('cpu_name')} |", ""]
    if R.get("table10"):
        L += ["## Table 10 (ablation, seed 0, %)", "",
              "| Configuration | Cancer Dice | Cancer IoU | Per-image Dice | All-images Dice | Benign FP (%) | Epochs (best) | Training (h) |",
              "|---|---|---|---|---|---|---|---|"]
        for v in ["unet", "att", "full", "full_nofocal"]:
            if v in R["table10"]:
                s = R["table10"][v]
                L.append(f"| {VARIANT_NAME[v]} | {_f(s['cancer']['dice'])} | {_f(s['cancer']['iou'])} | "
                         f"{_f(s['per_image_dice_mean'])} | {_f(s['all']['dice'])} | {_f(s['benign_fp_share'])} | "
                         f"{s['epochs_run']} ({s['best_epoch']}) | {_f(s['train_hours'], 1)} |")
        L.append("")
    if R.get("table11"):
        L += ["## Table 11 (paired per-image comparison, cancer test images, seed 0)", "",
              "| Change | Mean Δ (points) | 95% bootstrap CI | Wilcoxon p | n |", "|---|---|---|---|---|"]
        for k, p in R["table11"].items():
            a, b = k.split("->")
            L.append(f"| {VARIANT_NAME[a]} → {VARIANT_NAME[b]} ({p['change']}) | {p['mean_diff']:+.2f} | "
                     f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | {p['wilcoxon_p']:.2e} | {p['n']:,} |")
        L.append("")
    if R.get("seeds"):
        L += ["## Repeated runs (mean ± SD across seeds)", "",
              "| Configuration | Seeds | Cancer Dice | Cancer IoU | Per-image Dice | All-images Dice | Benign FP (%) |",
              "|---|---|---|---|---|---|---|"]
        for v, s in R["seeds"].items():
            ms = lambda m: f"{m['mean']:.2f} ± {m['sd']:.2f}"
            L.append(f"| {VARIANT_NAME[v]} | {s['seeds']} | {ms(s['cancer_dice'])} | {ms(s['cancer_iou'])} | "
                     f"{ms(s['per_image_dice'])} | {ms(s['all_dice'])} | {ms(s['benign_fp'])} |")
        L.append("")
    if R["pending_runs"]:
        L += [f"**Unfinished runs:** {R['pending_runs']} — numbers above are not final.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
