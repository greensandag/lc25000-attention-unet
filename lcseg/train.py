"""Training of one (variant, seed) run followed by evaluation on the untouched test partition.

Section III.G: Adam, learning rate 1e-4, batch 8, at most 50 epochs; early stopping on the
validation Dice of the main output (patience 5, best weights kept); learning rate x 0.3 after
5 epochs without improvement of the validation loss; decision threshold fixed at 0.5.
"""
import math
import os
import platform
import time

import numpy as np

from .augment import AUG_PARAMS, build_augmenter
from .losses import make_losses
from .model import build_model, has_deep_supervision, inference_model
from .utils import jdump, unpack


def setup_tf(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    import keras
    return tf, keras


def make_dataset_class(keras):
    base = getattr(keras.utils, "PyDataset", None) or keras.utils.Sequence

    class SegData(base):
        """Batches of (image in [0,1], {output: mask}) with optional online augmentation."""

        def __init__(self, X, rows, Ypacked, keys, batch, shuffle, seed, augment, workers=1):
            try:
                super().__init__(workers=workers, use_multiprocessing=False)
            except TypeError:
                super().__init__()
            self.X, self.rows, self.Y, self.keys = X, np.asarray(rows), Ypacked, keys
            self.batch, self.shuffle = batch, shuffle
            self.rng = np.random.default_rng(seed)
            self.aug = build_augmenter() if augment else None
            self.order = np.arange(len(self.rows))
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return int(math.ceil(len(self.rows) / self.batch))

        def __getitem__(self, i):
            b = self.order[i * self.batch:(i + 1) * self.batch]
            imgs = np.asarray(self.X[self.rows[b]])      # Y rows are aligned with self.rows
            masks = unpack(self.Y[b])
            if self.aug is not None:
                out_i, out_m = [], []
                for im, m in zip(imgs, masks):
                    r = self.aug(image=im, mask=m)
                    out_i.append(r["image"])
                    out_m.append(r["mask"])
                imgs, masks = np.stack(out_i), np.stack(out_m)
            x = imgs.astype(np.float32) / 255.0
            y = masks.astype(np.float32)[..., None]
            return x, {k: y for k in self.keys}

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)

    return SegData


def _find_key(logs, val, word):
    """Find the Keras log key of the main-output metric (names differ between Keras versions)."""
    cands = [k for k in logs if k.startswith("val_") == val and word in k]
    for k in cands:
        if "main" in k:
            return k
    return cands[0] if cands else None


def run_one(cfg, out, variant, seed, evaluate=True):
    from . import evaluate as ev
    from .data import load_split_arrays

    rdir = os.path.join(out, "runs", f"{variant}_s{seed}")
    if os.path.exists(os.path.join(rdir, "eval.npz")):
        print(f"[train] {variant} s{seed} already done, skipping", flush=True)
        return
    os.makedirs(rdir, exist_ok=True)
    tf, keras = setup_tf()
    keras.utils.set_random_seed(seed)
    t = cfg["TRAIN"]
    if t.get("mixed_precision") and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    tr_idx, X, Ytr = load_split_arrays(out, "train")
    va_idx, _, Yva = load_split_arrays(out, "val")
    model = build_model(variant, cfg["MODEL"])
    loss, dice = make_losses(t, variant)
    keys = list(model.output.keys()) if isinstance(model.output, dict) else ["main"]
    weights = {k: t["ds_weights"][k] for k in keys}
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=t["lr"]),
                  loss={k: loss for k in keys}, loss_weights=weights,
                  metrics={"main": [dice]})
    Data = make_dataset_class(keras)
    # training rows are read in random order; validation rows in order
    tr = Data(X, tr_idx, Ytr, keys, t["batch"], True, seed, t["augment"], t["workers"])
    va = Data(X, va_idx, Yva, keys, t["batch"], False, seed, False, t["workers"])

    class EarlyStopLR(keras.callbacks.Callback):
        """Early stopping on val Dice (main output) + ReduceLROnPlateau on val loss, with history."""

        def __init__(s):
            super().__init__()
            s.best_dice, s.best_epoch, s.wait_es = -np.inf, -1, 0
            s.best_loss, s.wait_lr = np.inf, 0
            s.weights, s.hist, s.times = None, [], []

        def on_epoch_begin(s, epoch, logs=None):
            s.t0 = time.time()

        def on_epoch_end(s, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items()}
            s.times.append(time.time() - s.t0)
            k_vd, k_td = _find_key(logs, True, "dice"), _find_key(logs, False, "dice")
            lr = float(keras.ops.convert_to_numpy(s.model.optimizer.learning_rate))
            # loss = total loss (main + weighted auxiliary losses); main_loss = Combo loss of the main output
            row = dict(epoch=epoch + 1, loss=logs.get("loss"), val_loss=logs.get("val_loss"),
                       main_loss=logs.get("main_loss", logs.get("loss")),
                       val_main_loss=logs.get("val_main_loss", logs.get("val_loss")),
                       dice=logs.get(k_td), val_dice=logs.get(k_vd), lr=lr, seconds=s.times[-1])
            s.hist.append(row)
            print(f"[{variant} s{seed}] epoch {epoch + 1} loss {row['loss']:.4f} val_loss {row['val_loss']:.4f} "
                  f"dice {row['dice']:.4f} val_dice {row['val_dice']:.4f} lr {lr:.2e} ({s.times[-1]:.0f}s)",
                  flush=True)
            # learning-rate schedule on validation loss
            if row["val_loss"] < s.best_loss:
                s.best_loss, s.wait_lr = row["val_loss"], 0
            else:
                s.wait_lr += 1
                if s.wait_lr >= t["lr_patience"]:
                    s.model.optimizer.learning_rate.assign(lr * t["lr_factor"])
                    s.wait_lr = 0
            # early stopping on validation Dice of the main output
            if row["val_dice"] > s.best_dice:
                s.best_dice, s.best_epoch, s.wait_es = row["val_dice"], epoch + 1, 0
                s.weights = s.model.get_weights()
            else:
                s.wait_es += 1
                if s.wait_es >= t["es_patience"]:
                    s.model.stop_training = True

        def on_train_end(s, logs=None):
            if s.weights is not None:
                s.model.set_weights(s.weights)

    cb = EarlyStopLR()
    t0 = time.time()
    model.fit(tr, validation_data=va, epochs=t["epochs"], callbacks=[cb], verbose=0)
    train_s = time.time() - t0

    infer = inference_model(model)
    infer.save(os.path.join(rdir, "model.keras"))
    gpus = tf.config.list_physical_devices("GPU")
    gpu_name = None
    if gpus:
        try:
            gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name")
        except Exception:
            gpu_name = "GPU"
    info = dict(variant=variant, seed=seed, epochs_run=len(cb.hist), best_epoch=cb.best_epoch,
                best_val_dice=cb.best_dice, train_seconds=train_s, train_hours=train_s / 3600,
                params_inference=int(infer.count_params()), params_training=int(model.count_params()),
                deep_supervision=has_deep_supervision(variant), history=cb.hist,
                config=dict(TRAIN=t, MODEL=cfg["MODEL"], AUG=AUG_PARAMS),
                precision_policy=keras.mixed_precision.global_policy().name,
                tf=tf.__version__, keras=keras.__version__, python=platform.python_version(), gpu=gpu_name)
    jdump(info, os.path.join(rdir, "run.json"))
    print(f"[train] {variant} s{seed}: {len(cb.hist)} epochs, best {cb.best_epoch}, "
          f"{train_s / 3600:.2f} h", flush=True)
    del tr, va
    if evaluate:
        ev.evaluate_run(cfg, out, variant, seed, model=infer)
    keras.backend.clear_session()
