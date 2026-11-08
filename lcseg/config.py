"""Default configuration.

Every value below is taken from the revised manuscript. The section, table, or
equation that defines each value is given in the comment next to it. Change a
value only together with the manuscript.
"""
import copy
import json
import os

CLASSES = ["lung_aca", "lung_scc", "lung_n", "colon_aca", "colon_n"]
CANCER = ["lung_aca", "lung_scc", "colon_aca"]
BENIGN = ["lung_n", "colon_n"]
SIZE = 256                      # input size, Section III.D
NPIX = SIZE * SIZE

DEFAULT_CFG = {
    # ------------------------------------------------------------------
    # Automatic mask generator (Section III.E, Table 3)
    # ------------------------------------------------------------------
    "MASK": {
        "bg_gray": 215,          # gray level > 215 -> white background, excluded
        "a_max": 138,            # keep pixels with 8-bit CIELAB A < 138 (128 = neutral)
        "block": 31,             # adaptive threshold: Gaussian-weighted mean, 31x31 window
        "C": 8,                  # ... minus a constant of 8, inverted (darker pixels selected)
        "close_k": 5,            # closing with 5x5 elliptical element
        "open_k": 3,             # opening with 3x3 elliptical element
        "min_area": 4,           # 8-connected components smaller than 4 px discarded
        "max_area": 2621,        # components larger than (0.2*256)^2 = 2,621 px discarded
        "dilate_k": 3,           # kept components dilated once with 3x3 ellipse
        "qc_low": 0.08,          # coverage < 8%  -> fallback mask
        "fb_block": 51,          # fallback: adaptive threshold 51x51 ...
        "fb_C": 5,               # ... C = 5, background removal, closing 5x5
        "fb_close_k": 5,
        "qc_high": 0.75,         # coverage > 75% -> erosion
        "erode_k": 7,            # erosion with 7x7 ellipse ...
        "erode_iter": 2,         # ... 2 iterations
    },
    # ------------------------------------------------------------------
    # Source grouping and split (Section III.C)
    # ------------------------------------------------------------------
    "GROUP": {
        "desc_bins": 16,         # 16-bin histogram per CIELAB channel
        "lbp": [[8, 1], [16, 2]],  # uniform LBP (neighbors, radius)
        "k_cand": 30,            # 30 most similar images (cosine) as candidates
        "orb_size": 512,         # ORB on gray images resized to 512x512
        "orb_n": 1000,           # up to 1,000 ORB keypoints
        "ratio": 0.75,           # Lowe ratio test
        "ransac_thr": 5.0,       # RANSAC reprojection threshold (pixels)
        "min_inliers": 25,       # accept pair with >= 25 inliers
        "k_audit": 10,           # leakage audit: 10 most similar training images
        "split_seed": 2026,      # seed for shuffling groups
        "split": [0.70, 0.10, 0.20],  # train / val / test (test and val filled first)
    },
    # ------------------------------------------------------------------
    # Model (Section III.F, Table 4)
    # ------------------------------------------------------------------
    "MODEL": {
        "enc_filters": [64, 128, 256, 512],
        "enc_dropout": [0.05, 0.05, 0.10, 0.10],
        "bottleneck_filters": 512,
        "bottleneck_dropout": 0.20,
        "dec_filters": [256, 256, 128, 64],
        "dec_dropout": [0.10, 0.10, 0.05, 0.05],
        "ag_inter": [128, 128, 64, 32],  # attention-gate intermediate channels
    },
    # ------------------------------------------------------------------
    # Training (Section III.G, Equation 3, Table 4)
    # ------------------------------------------------------------------
    "TRAIN": {
        "loss_w": {"bce": 0.3, "dice": 0.4, "focal": 0.3},   # Equation 3
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "ds_weights": {"main": 1.0, "aux4": 0.4, "aux8": 0.2},
        "optimizer": "adam",
        "lr": 1e-4,
        "batch": 8,
        "epochs": 50,
        "es_patience": 5,        # early stopping on validation Dice of the main output
        "lr_factor": 0.3,        # learning rate x 0.3 ...
        "lr_patience": 5,        # ... after 5 epochs without improvement of the validation loss
        "augment": True,
        "mixed_precision": True,   # float16 compute on GPU for speed (outputs and losses stay float32)
        "workers": 4,
    },
    # Runs: (variant, seed). Variants: unet | att | full | full_nofocal (Section III.G, Table 10)
    "RUNS": {
        "full": [0, 1, 2],       # complete model, three seeds (mean +- SD across runs)
        "unet": [0],
        "att": [0],
        "full_nofocal": [0],
    },
    # ------------------------------------------------------------------
    # Evaluation (Section III.H)
    # ------------------------------------------------------------------
    "EVAL": {
        "thr": 0.5,              # fixed decision threshold
        "thr_grid": [round(0.20 + 0.05 * i, 2) for i in range(14)],  # 0.20..0.85, reference only
        "img_area_thr": 0.01,    # image called positive when predicted region >= 1% of the image
        "boundary_px": 2,        # boundary errors: within 2 px of a reference boundary
        "low_dice": 0.85,        # failure cases: per-image Dice below 85%
        "n_boot": 10000,         # bootstrap resamples for paired comparisons
        "boot_seed": 0,
    },
    "WORKERS": max(1, (os.cpu_count() or 2)),
    "LIMIT_PER_CLASS": None,     # for smoke tests only
}


def load_cfg(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULT_CFG)
    user = {}
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
    if overrides:
        user = _merge(user, overrides)
    return _merge(cfg, user)


def _merge(a, b):
    a = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _merge(a[k], v)
        else:
            a[k] = v
    return a
