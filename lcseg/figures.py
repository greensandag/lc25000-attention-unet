"""Result figures drawn directly from the stored outputs (no manual editing)."""
import os

import numpy as np

from .config import BENIGN, CANCER, CLASSES
from .evaluate import per_image_dice
from .utils import read_meta, read_split, unpack

ROW_NAME = {"lung_aca": "Lung adenocarcinoma", "lung_scc": "Lung squamous\ncell carcinoma",
            "lung_n": "Lung benign", "colon_aca": "Colon adenocarcinoma", "colon_n": "Colon benign"}
TP, FN, FP = np.array([0, 230, 0]), np.array([30, 60, 255]), np.array([255, 160, 30])


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _error_map(rgb, pred, ref):
    img = rgb.astype(np.float32).copy()
    for m, col in [((pred == 1) & (ref == 1), TP), ((pred == 0) & (ref == 1), FN), ((pred == 1) & (ref == 0), FP)]:
        img[m] = 0.35 * img[m] + 0.65 * col
    return img.astype(np.uint8)


def _overlay(rgb, pred):
    img = rgb.astype(np.float32).copy()
    img[pred == 1] = 0.4 * img[pred == 1] + 0.6 * np.array([230, 0, 0])
    return img.astype(np.uint8)


def _panel(out, z, keys, labels, path, thr):
    plt = _plt()
    X = np.load(os.path.join(out, "cache", "X.npy"), mmap_mode="r")
    Y = np.load(os.path.join(out, "cache", "Y.npy"))
    kv = dict(zip(z["fig_keys"].tolist(), z["fig_vals"].tolist()))
    rows = [k for k in keys if k in kv]
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 2.1 * len(rows)))
    axes = np.atleast_2d(axes)
    titles = ["Original image", "Generated mask", "Probability map", "Overlay", "Error map"]
    d = per_image_dice(z["counts"])
    for r, k in enumerate(rows):
        j = kv[k]
        gi = int(z["test_idx"][j])
        rgb = np.asarray(X[gi])
        ref = unpack(Y[gi])
        prob = z["fig_probs"][list(z["fig_rows"]).index(j)].astype(np.float32)
        pred = (prob >= thr).astype(np.uint8)
        ims = [rgb, ref * 255, prob, _overlay(rgb, pred), _error_map(rgb, pred, ref)]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            if c == 1:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            elif c == 2:
                ax.imshow(im, cmap="hot", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        cls = str(z["cls"][j])
        extra = f"\nDice {100 * d[j]:.1f}%" if cls in CANCER else f"\nFP {100 * z['area'][j]:.2f}%"
        axes[r, 0].set_ylabel(labels(k, cls) + extra, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_figures(cfg, out, done, R):
    plt = _plt()
    rdir = os.path.join(out, "report")
    thr = cfg["EVAL"]["thr"]
    main = done.get("full", {}).get(0)

    # Figure 2: one generated mask per class (first training image of each class)
    X = np.load(os.path.join(out, "cache", "X.npy"), mmap_mode="r")
    Y = np.load(os.path.join(out, "cache", "Y.npy"))
    meta = read_meta(out)
    split, _ = read_split(out)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    for c, k in enumerate(CLASSES):
        i = next(m["idx"] for m in meta if m["cls"] == k and split[m["idx"]] == "train")
        axes[0, c].imshow(np.asarray(X[i])); axes[0, c].set_title(ROW_NAME[k].replace("\n", " "), fontsize=8)
        axes[1, c].imshow(unpack(Y[i]) * 255, cmap="gray", vmin=0, vmax=255)
        for a in axes[:, c]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(rdir, "fig2_masks.png"), dpi=300); plt.close(fig)

    if main is not None:
        # Figure 4: training curves of the complete model (seed 0)
        h = main["run"]["history"]
        ep = [r["epoch"] for r in h]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
        a1.plot(ep, [r["main_loss"] for r in h], color="#d62728", label="Train loss")
        a1.plot(ep, [r["val_main_loss"] for r in h], color="#2ca02c", label="Validation loss")
        bl = int(np.argmin([r["val_main_loss"] for r in h]))
        a1.scatter([ep[bl]], [h[bl]["val_main_loss"]], color="#1f3fd6", zorder=3, label=f"Lowest val loss = epoch {ep[bl]}")
        a2.plot(ep, [r["dice"] for r in h], color="#d62728", label="Train Dice")
        a2.plot(ep, [r["val_dice"] for r in h], color="#2ca02c", label="Validation Dice")
        be = main["run"]["best_epoch"]
        a2.scatter([be], [h[be - 1]["val_dice"]], color="#1f3fd6", zorder=3, label=f"Best epoch = {be}")
        for a, t in [(a1, "Loss"), (a2, "Dice coefficient")]:
            a.set_title(t + (" (main output)" if t == "Loss" else "")); a.set_xlabel("Epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(rdir, "fig4_curves.png"), dpi=300); plt.close(fig)

        # Figure 5: typical image per class; Figure 7: failure cases
        _panel(out, main, [f"typical_{k}" for k in CLASSES], lambda k, c: ROW_NAME[c],
               os.path.join(rdir, "fig5_examples.png"), thr)
        worst = ["worst_colon_aca_1", "worst_colon_aca_2", "worst_colon_aca_3", "worst_lung_aca",
                 "worst_lung_scc", "worst_benign_fp"]
        _panel(out, main, worst,
               lambda k, c: ROW_NAME[c] + ("\n(largest FP)" if c in BENIGN else "\n(lowest Dice)"),
               os.path.join(rdir, "fig7_failure_cases.png"), thr)

        # per-image Dice distribution by class
        d = 100 * per_image_dice(main["counts"])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.boxplot([d[main["cls"] == k] for k in CANCER], showfliers=True)
        ax.set_xticks([1, 2, 3], CANCER); ax.set_ylabel("Per-image Dice (%)"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(rdir, "fig_per_image_dice.png"), dpi=300); plt.close(fig)

    # ablation bar chart (seed 0; error bars where several seeds exist)
    from .report import VARIANT_NAME
    vs = [v for v in ["unet", "att", "full", "full_nofocal"] if v in R.get("table10", {})]
    if vs:
        vals = [R["seeds"][v]["cancer_dice"]["mean"] if v in R.get("seeds", {}) else R["table10"][v]["cancer"]["dice"]
                for v in vs]
        err = [R["seeds"][v]["cancer_dice"]["sd"] if v in R.get("seeds", {}) else 0 for v in vs]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(range(len(vs)), vals, yerr=err, color=["#62778c", "#305a7d", "#00715f", "#62778c"][:len(vs)], capsize=4)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=9)
        ax.set_xticks(range(len(vs)), [VARIANT_NAME[v].replace(", ", ",\n") for v in vs], fontsize=8)
        ax.set_ylabel("Cancer-only Dice (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(rdir, "fig_ablation.png"), dpi=300); plt.close(fig)
