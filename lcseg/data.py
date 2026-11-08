"""Stage `prepare`: find LC25000, resize to 256x256, generate the reference masks."""
import csv
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from .config import CLASSES, CANCER, SIZE, NPIX
from .masks import generate_mask
from .utils import jdump, load_rgb, packbits, read_split


def find_lc25000(root):
    """Return {class: sorted list of image paths}; searches recursively for the class folders."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        name = os.path.basename(dirpath)
        if name in CLASSES:
            files = sorted(os.path.join(dirpath, f) for f in filenames
                           if f.lower().endswith((".jpeg", ".jpg", ".png")))
            if files and name not in out:
                out[name] = files
    missing = [c for c in CLASSES if c not in out]
    if missing:
        raise FileNotFoundError(f"LC25000 class folders not found under {root}: {missing}")
    return out


def _prep_one(args):
    path, cls, mp = args
    rgb = load_rgb(path)                          # bilinear resize (Table 3)
    if cls in CANCER:
        ref, action = generate_mask(rgb, mp, qc=True)
    else:                                         # benign: all-zero mask, generator not applied
        ref, action = np.zeros((SIZE, SIZE), np.uint8), "benign"
    return rgb, packbits(ref), action, float(ref.mean())


def stage_prepare(cfg, lc_root, out):
    cdir = os.path.join(out, "cache")
    if os.path.exists(os.path.join(cdir, "meta.csv")):
        print("[prepare] cache exists, skipping")
        return
    os.makedirs(cdir, exist_ok=True)
    files = find_lc25000(lc_root)
    items = []
    for c in CLASSES:
        fl = files[c][: cfg["LIMIT_PER_CLASS"]] if cfg["LIMIT_PER_CLASS"] else files[c]
        items += [(f, c) for f in fl]
    n = len(items)
    X = np.lib.format.open_memmap(os.path.join(cdir, "X.npy"), mode="w+", dtype=np.uint8,
                                  shape=(n, SIZE, SIZE, 3))
    Y = np.zeros((n, NPIX // 8), np.uint8)
    meta = []
    t0 = time.time()
    with Pool(cfg["WORKERS"]) as pool:
        for i, (rgb, yb, action, cov) in enumerate(
                pool.imap(_prep_one, [(f, c, cfg["MASK"]) for f, c in items], chunksize=16)):
            X[i] = rgb
            Y[i] = yb
            meta.append(dict(idx=i, cls=items[i][1], file=os.path.basename(items[i][0]),
                             path=items[i][0], qc=action, coverage=round(cov, 6)))
            if i % 2500 == 0:
                print(f"[prepare] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    X.flush()
    np.save(os.path.join(cdir, "Y.npy"), Y)
    with open(os.path.join(cdir, "meta.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        wr.writeheader()
        wr.writerows(meta)
    qc = {a: sum(m["qc"] == a for m in meta) for a in ["none", "fallback", "eroded", "benign"]}
    cov_c = [m["coverage"] for m in meta if m["cls"] in CANCER]
    cov_all = [m["coverage"] for m in meta]
    jdump(dict(n=n, qc=qc, coverage_cancer_mean=float(np.mean(cov_c)),
               coverage_all_mean=float(np.mean(cov_all)), opencv=cv2.__version__,
               seconds=time.time() - t0), os.path.join(cdir, "prepare.json"))
    print("[prepare] done", qc, flush=True)


def load_split_arrays(out, which):
    """Return (indices, X memmap rows as uint8 array, packed masks) for a partition."""
    split, _ = read_split(out)
    idx = np.where(split == which)[0]
    X = np.load(os.path.join(out, "cache", "X.npy"), mmap_mode="r")
    Y = np.load(os.path.join(out, "cache", "Y.npy"))
    return idx, X, Y[idx]
