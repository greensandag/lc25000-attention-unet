# Attention U-Net segmentation of LC25000 trained on automatically generated masks

The study segments **automatically selected stained regions** in the LC25000 lung and colon histopathology
images. Reference masks are produced by a deterministic image-processing pipeline (not by pathologists), the
25,000 images are grouped back into their source tiles so that the training, validation, and test partitions are
source-disjoint, and a U-Net with attention gates and deep supervision is trained and compared with three
ablated configurations and with the mask generator itself. The scores measure agreement with the mask pipeline,
not with a pathologist.

All numbers and result figures of the manuscript are produced by this code:
running [`notebooks/run_experiments_kaggle.ipynb`](notebooks/run_experiments_kaggle.ipynb) writes every value of
Tables 5–11 to `outputs/report/results.md` (and `results.json`).

## Running on Kaggle (recommended)

1. Create a new notebook, upload `notebooks/run_experiments_kaggle.ipynb` (*File → Import notebook*).
2. Settings: **Accelerator: GPU T4 ×2**, **Internet: on**.
3. *Add Input* → dataset `andrewmvd/lung-and-colon-cancer-histopathological-images` (LC25000).
4. In the first code cell set `REPO_URL` to this repository.
5. *Run all*. Training runs are distributed over both GPUs. If a 12-hour session ends before all runs finish,
   *Save Version*, add the saved output as an input of a new session, set `PREV` to its `outputs` folder, and run
   all cells again; finished stages and runs are skipped.

The default runs (in `lcseg/config.py`, `RUNS`) are the complete model with seeds 0, 1, 2 and the three ablated
configurations with seed 0, so that the complete model can be reported as mean ± SD over runs.

## Running locally

```bash
pip install -r requirements.txt
python -m lcseg prepare   --lc-root /path/to/lung_colon_image_set --out outputs
python -m lcseg group     --out outputs
python -m lcseg generator --out outputs
python -m lcseg launch    --out outputs          # or: python -m lcseg train --variant full --seed 0 --out outputs
python -m lcseg bench     --out outputs
python -m lcseg report    --out outputs          # -> outputs/report/results.md
```

Checks: `pytest -q tests` (seconds) and `bash scripts/smoke_test.sh` (complete pipeline on a tiny synthetic
dataset with a tiny model, a few minutes on CPU; its numbers are meaningless).

## Outputs

```
outputs/
  cache/        X.npy (256x256 images), Y.npy (bit-packed masks), meta.csv, prepare.json (coverage-rule counts)
  groups/       groups.csv (image -> source group -> partition), group_stats.json, pairs.npy
  generator/    mask generator applied to the test images without class labels, timing
  runs/<variant>_s<seed>/  model.keras (main output), run.json (history, epochs, time), eval.npz (per-image counts)
  bench/        bench.json (Table 9)
  report/       results.md, results.json, fig2_masks.png, fig4_curves.png, fig5_examples.png,
                fig7_failure_cases.png, fig_per_image_dice.png, fig_ablation.png
```

## Implementation details not stated in the manuscript

These choices are fixed in the code and recorded in every `run.json`; they should be added to the manuscript
(Table 4) if a reviewer asks for them.

- Augmentation probabilities and magnitudes: `AUG_PARAMS` in [`lcseg/augment.py`](lcseg/augment.py).
- `full_nofocal` ablation: the focal term is removed and the other weights are kept (0.3 BCE + 0.4 Dice).
- The learning-rate schedule monitors the total validation loss (main + weighted auxiliary losses); Figure 4 plots
  the Combo loss of the main output.
- Mixed precision (float16 compute, float32 outputs and losses) is used on GPU for speed (`TRAIN.mixed_precision`).
- Figure 5 shows, for each class, the test image with the median per-image Dice (cancer) or median false-positive
  area (benign); Figure 7 shows the three colon adenocarcinoma images and the lung adenocarcinoma and squamous
  cell carcinoma images with the lowest per-image Dice, and the benign image with the largest false-positive area.
- The source grouping reads the original JPEG files with area interpolation; the network inputs and masks use
  bilinear interpolation (Table 3).

## Data

LC25000: A. A. Borkowski et al., "LC25000 lung and colon histopathological image dataset," 2019,
<https://github.com/tampapath/lung_colon_image_set>. The dataset is not redistributed here. The generated masks,
the source groups and partition lists, and the trained checkpoints are produced by the notebook
(`release_bundle.zip`, last cell) and are deposited separately (see the manuscript's Data Availability statement).

## License

MIT (see [LICENSE](LICENSE)).
