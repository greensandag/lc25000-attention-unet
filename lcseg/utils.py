import csv
import json
import os
import time

import cv2
import numpy as np

from .config import SIZE, NPIX


def jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_np_default)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_rgb(path, size=SIZE, interp=cv2.INTER_LINEAR):
    """Read an image as uint8 RGB and resize it (bilinear = OpenCV default, Table 3)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=interp)


def packbits(m):
    return np.packbits(np.asarray(m, np.uint8).reshape(-1))


def unpack(b):
    """(N, NPIX/8) uint8 -> (N, SIZE, SIZE) uint8 in {0,1}."""
    b = np.asarray(b)
    if b.ndim == 1:
        return np.unpackbits(b)[:NPIX].reshape(SIZE, SIZE)
    return np.unpackbits(b, axis=1)[:, :NPIX].reshape(-1, SIZE, SIZE)


def read_meta(out):
    with open(os.path.join(out, "cache", "meta.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["idx"] = int(r["idx"])
    return rows


def read_split(out):
    with open(os.path.join(out, "groups", "groups.csv")) as f:
        rows = list(csv.DictReader(f))
    return np.array([r["split"] for r in rows]), np.array([int(r["group"]) for r in rows])


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
