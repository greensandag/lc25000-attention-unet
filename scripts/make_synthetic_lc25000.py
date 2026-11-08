"""Create a tiny synthetic dataset with the LC25000 folder layout (for smoke tests only).

Each class gets a few random "source tiles" (H&E-like blobs), and every source tile is
copied several times with rotations of up to 25 degrees and flips, as in LC25000.

    python scripts/make_synthetic_lc25000.py --out /tmp/lc_synth --sources 4 --copies 5
"""
import argparse
import os

import cv2
import numpy as np

LAYOUT = {"lung_aca": "lung_image_sets", "lung_scc": "lung_image_sets", "lung_n": "lung_image_sets",
          "colon_aca": "colon_image_sets", "colon_n": "colon_image_sets"}


def source_tile(rng, size=768):
    img = np.full((size, size, 3), (235, 190, 225), np.uint8)          # pink stroma (RGB)
    for _ in range(rng.integers(150, 300)):                            # purple nuclei
        c = tuple(int(v) for v in rng.integers(0, size, 2))
        ax = tuple(int(v) for v in rng.integers(6, 22, 2))
        col = tuple(int(v) for v in (rng.integers(60, 120), rng.integers(30, 70), rng.integers(120, 180)))
        cv2.ellipse(img, c, ax, float(rng.uniform(0, 180)), 0, 360, col, -1)
    for _ in range(rng.integers(3, 8)):                                # white lumens
        c = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.circle(img, c, int(rng.integers(20, 60)), (250, 250, 250), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def augment(img, rng):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-25, 25)), 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", type=int, default=4)
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls, parent in LAYOUT.items():
        d = os.path.join(a.out, "lung_colon_image_set", parent, cls)
        os.makedirs(d, exist_ok=True)
        k = 0
        for s in range(a.sources):
            src = source_tile(rng)
            for c in range(a.copies):
                img = src if c == 0 else augment(src, rng)
                cv2.imwrite(os.path.join(d, f"{cls}{k + 1}.jpeg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                k += 1
    print("synthetic LC25000 written to", a.out)


if __name__ == "__main__":
    main()
