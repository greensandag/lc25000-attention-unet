"""Fast checks of the components that define the manuscript's numbers. Run: pytest -q"""
import numpy as np
import pytest

from lcseg.config import load_cfg
from lcseg.masks import generate_mask
from lcseg.evaluate import counts, per_image_dice
from lcseg.report import pooled

CFG = load_cfg()


def _tile(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), (235, 190, 225), np.uint8)
    yy, xx = np.mgrid[:256, :256]
    for _ in range(60):
        cy, cx, r = rng.integers(0, 256, 2).tolist() + [int(rng.integers(3, 9))]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = (90, 50, 150)
    return img


def test_mask_is_binary_and_same_size():
    m, action = generate_mask(_tile(), CFG["MASK"])
    assert m.shape == (256, 256) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}
    assert action in {"none", "fallback", "eroded"}


def test_white_background_is_never_target():
    m, _ = generate_mask(np.full((256, 256, 3), 250, np.uint8), CFG["MASK"])
    assert m.sum() == 0


def test_pooled_metrics():
    ref = np.zeros((2, 4, 4), np.uint8); ref[0, :2] = 1
    pred = np.zeros_like(ref); pred[0, :1] = 1; pred[1, 0, 0] = 1
    c = counts(pred, ref)
    assert c.tolist() == [[4, 0, 4, 8], [0, 1, 0, 15]]
    p = pooled(c)
    assert p["dice"] == pytest.approx(100 * 8 / 13)
    assert per_image_dice(c)[0] == pytest.approx(8 / 12)


def test_parameter_count_of_complete_model():
    pytest.importorskip("tensorflow")
    from lcseg.model import build_model, inference_model
    m = build_model("full", CFG["MODEL"])
    assert set(m.output.keys()) == {"main", "aux4", "aux8"}
    assert inference_model(m).count_params() == 15_241_605
