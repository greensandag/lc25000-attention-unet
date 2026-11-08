"""Stage `group`: recover the LC25000 source tiles and build a source-disjoint split (Section III.C).

Step 1  candidates: CIELAB + uniform-LBP histogram descriptor, 30 nearest neighbours (cosine)
Step 2  verification: ORB keypoints + ratio test + RANSAC partial affine, >= 25 inliers
Step 3  union-find over accepted pairs -> groups
Step 4  per class, shuffled groups fill test (>= 20%), then validation (>= 10%), rest training
Step 5  leakage audit of the final split and of a random image-level split for comparison
"""
import csv
import os
import time
from multiprocessing import get_context

Pool = get_context("fork").Pool   # workers must inherit the global ORB cache

import cv2
import numpy as np

from .config import CLASSES
from .utils import jdump, load_rgb, read_meta

_ORB = {}
_G = {}


def descriptor(rgb, p):
    from skimage.feature import local_binary_pattern
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    feats = []
    for ch in range(3):
        h = np.histogram(lab[..., ch], bins=p["desc_bins"], range=(0, 256))[0].astype(np.float64)
        feats.append(h / h.sum())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for P_, R_ in p["lbp"]:
        lbp = local_binary_pattern(gray, P_, R_, method="uniform")
        h = np.histogram(lbp, bins=P_ + 2, range=(0, P_ + 2))[0].astype(np.float64)
        feats.append(h / h.sum())
    v = np.sqrt(np.concatenate(feats))                      # square root (Hellinger)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)   # L2 normalisation


def _desc_one(args):
    path, p = args
    return descriptor(load_rgb(path, interp=cv2.INTER_AREA), p)


def _orb_one(args):
    path, size, nfeat = args
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeat)
    out = []
    for img in (g, cv2.flip(g, 1)):          # original and mirrored (ORB itself handles rotation)
        kp, des = orb.detectAndCompute(img, None)
        pts = np.array([k.pt for k in kp], np.float32) if kp else np.zeros((0, 2), np.float32)
        out.append((pts, des if des is not None else np.zeros((0, 32), np.uint8)))
    return out


def verify(feat_a, feat_b, p):
    """Largest number of RANSAC inliers between image a and image b (original or mirrored)."""
    best = 0
    pa, da = feat_a[0]
    if len(da) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    for pb, db in feat_b:
        if len(db) < 2:
            continue
        good = [mm[0] for mm in bf.knnMatch(da, db, k=2)
                if len(mm) == 2 and mm[0].distance < p["ratio"] * mm[1].distance]
        if len(good) < p["min_inliers"]:
            continue
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=p["ransac_thr"])
        if inl is not None:
            best = max(best, int(inl.sum()))
    return best


def _verify_pairs(pairs):
    p = _G["p"]
    return [(i, j, verify(_ORB[i], _ORB[j], p)) for i, j in pairs]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _load_orb(paths_idx, meta, p, workers):
    _ORB.clear()
    with Pool(workers) as pool:
        feats = pool.map(_orb_one, [(meta[i]["path"], p["orb_size"], p["orb_n"]) for i in paths_idx],
                         chunksize=16)
    for i, ft in zip(paths_idx, feats):
        _ORB[int(i)] = ft


def _verify_all(pairs, workers):
    chunks = [pairs[s:s + 500] for s in range(0, len(pairs), 500)]
    with Pool(workers) as pool:      # fork start method: workers inherit _ORB and _G
        res = pool.map(_verify_pairs, chunks)
    return [r for ch in res for r in ch]


def stage_group(cfg, out):
    gdir = os.path.join(out, "groups")
    if os.path.exists(os.path.join(gdir, "groups.csv")):
        print("[group] exists, skipping")
        return
    os.makedirs(gdir, exist_ok=True)
    p, W = cfg["GROUP"], cfg["WORKERS"]
    _G["p"] = p
    meta = read_meta(out)
    N = len(meta)
    t0 = time.time()
    with Pool(W) as pool:
        D = np.stack(pool.map(_desc_one, [(m["path"], p) for m in meta], chunksize=32))
    print(f"[group] descriptors {time.time() - t0:.0f}s", flush=True)

    uf = UnionFind(N)
    pair_log = []
    rng = np.random.default_rng(p["split_seed"])
    split = np.empty(N, dtype=object)
    tr_frac, va_frac, te_frac = p["split"]
    for c in CLASSES:
        idx = np.array([m["idx"] for m in meta if m["cls"] == c])
        t1 = time.time()
        _load_orb(idx, meta, p, W)
        S = D[idx] @ D[idx].T
        np.fill_diagonal(S, -1)
        k = min(p["k_cand"], len(idx) - 1)
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        pairs = sorted({(int(min(idx[a], idx[b])), int(max(idx[a], idx[b])))
                        for a in range(len(idx)) for b in nn[a]})
        acc = 0
        for i, j, inl in _verify_all(pairs, W):
            pair_log.append((i, j, inl))
            if inl >= p["min_inliers"]:
                uf.union(i, j)
                acc += 1
        roots = {}
        for i in idx:
            roots.setdefault(uf.find(int(i)), []).append(int(i))
        groups = list(roots.values())
        order = rng.permutation(len(groups))
        n_cls, n_test, n_val = len(idx), 0, 0
        for gi in order:
            g = groups[gi]
            if n_test < te_frac * n_cls:
                lab = "test"
                n_test += len(g)
            elif n_val < va_frac * n_cls:
                lab = "val"
                n_val += len(g)
            else:
                lab = "train"
            for i in g:
                split[i] = lab
        print(f"[group] {c}: {len(pairs)} candidate pairs, {acc} accepted, {len(groups)} groups, "
              f"{time.time() - t1:.0f}s", flush=True)
    gid = np.array([uf.find(i) for i in range(N)])

    # leakage audit: every test image vs its 10 most similar training images, same criterion
    audit_pairs = []
    for c in CLASSES:
        idx = np.array([m["idx"] for m in meta if m["cls"] == c])
        te_i, tr_i = idx[split[idx] == "test"], idx[split[idx] == "train"]
        S = D[te_i] @ D[tr_i].T
        k = min(p["k_audit"], len(tr_i))
        nn = np.argsort(-S, axis=1)[:, :k]
        audit_pairs += [(int(te_i[a]), int(tr_i[b])) for a in range(len(te_i)) for b in nn[a]]
    need = sorted({i for pr in audit_pairs for i in pr})
    _load_orb(need, meta, p, W)
    leaked = sorted({i for i, j, inl in _verify_all(audit_pairs, W) if inl >= p["min_inliers"]})

    # random image-level split with 20% test images per class, for comparison:
    # share of its test images whose source group also has images in its training set
    rr = np.random.default_rng(p["split_seed"] + 1)
    rand_test = np.zeros(N, bool)
    for c in CLASSES:
        idx = np.array([m["idx"] for m in meta if m["cls"] == c])
        rand_test[rr.choice(idx, int(round(te_frac * len(idx))), replace=False)] = True
    train_groups = set(gid[~rand_test])
    leak_rand = float(np.mean([gid[i] in train_groups for i in np.where(rand_test)[0]]))

    with open(os.path.join(gdir, "groups.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["idx", "cls", "file", "group", "split"])
        for m in meta:
            wr.writerow([m["idx"], m["cls"], m["file"], int(gid[m["idx"]]), split[m["idx"]]])
    np.save(os.path.join(gdir, "pairs.npy"), np.array(pair_log, dtype=np.int64))
    sizes = {c: [] for c in CLASSES}
    for g in set(gid.tolist()):
        members = np.where(gid == g)[0]
        sizes[meta[members[0]]["cls"]].append(int(len(members)))
    stats = dict(groups_per_class={c: len(sizes[c]) for c in CLASSES},
                 singletons={c: int(sum(s == 1 for s in sizes[c])) for c in CLASSES},
                 median_size={c: float(np.median(sizes[c])) for c in CLASSES},
                 max_size={c: int(max(sizes[c])) for c in CLASSES},
                 sizes=sizes, leak_after=len(leaked), leak_after_ids=leaked,
                 leak_random_frac=leak_rand,
                 n_split={s: int(np.sum(split == s)) for s in ["train", "val", "test"]},
                 n_test_per_class={c: int(sum(1 for m in meta if m["cls"] == c and split[m["idx"]] == "test"))
                                   for c in CLASSES},
                 seconds=time.time() - t0)
    jdump(stats, os.path.join(gdir, "group_stats.json"))
    print("groups per class :", stats["groups_per_class"], "(dataset paper: 250 per class)")
    print("split            :", stats["n_split"])
    print("leak after split :", stats["leak_after"], "test images with a verified copy in train")
    print("random split leak:", round(100 * leak_rand, 1), "% of test images")
    print("[group] done", flush=True)
