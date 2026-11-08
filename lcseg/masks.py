"""Automatic mask generator (Section III.E, Table 3).

The target is the *automatically selected stained region*: the set of pixels the
generator selects in an image of a cancer class. Benign images receive an all-zero
mask and are not processed (Section III.E, last paragraph).
"""
import cv2
import numpy as np


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))


def background_mask(gray, p):
    """White background: 8-bit gray level above p['bg_gray'] (215)."""
    return gray > p["bg_gray"]


def generate_mask(rgb, p, qc=True):
    """Return (mask uint8 in {0,1}, action) for one 256x256 uint8 RGB image.

    action is one of 'none', 'fallback' (coverage < 8%), 'eroded' (coverage > 75%).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    bg = background_mask(gray, p)

    # adaptive threshold: Gaussian-weighted mean of a 31x31 neighborhood minus C=8;
    # THRESH_BINARY_INV selects pixels darker than the local threshold
    th = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                               int(p["block"]), float(p["C"])).astype(bool)
    color = lab[..., 1] < p["a_max"]          # CIELAB A channel filter, A < 138
    m = (th & color & ~bg).astype(np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(p["close_k"]))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(p["open_k"]))

    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (area >= p["min_area"]) & (area <= p["max_area"])
    keep[0] = False
    m = keep[lab_cc].astype(np.uint8)
    m = cv2.dilate(m, _ellipse(p["dilate_k"]), iterations=1)

    action = "none"
    if qc:
        cov = float(m.mean())
        if cov < p["qc_low"]:
            # fallback mask: adaptive threshold 51x51, C=5, background removal, closing 5x5,
            # without the A filter and without the area filter
            fb = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                       int(p["fb_block"]), float(p["fb_C"])).astype(bool)
            m = (fb & ~bg).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(p["fb_close_k"]))
            action = "fallback"
        elif cov > p["qc_high"]:
            m = cv2.erode(m, _ellipse(p["erode_k"]), iterations=int(p["erode_iter"]))
            action = "eroded"

    # consistency check (Section III.E): same size as the image, only values 0 and 1
    assert m.shape == rgb.shape[:2] and m.dtype == np.uint8 and m.max() <= 1
    return m, action
