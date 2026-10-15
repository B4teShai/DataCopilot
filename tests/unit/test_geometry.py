import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from datacopilot.core import geometry as g

coord = st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)
size = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)


def test_xywh_known_values() -> None:
    assert g.xywh_to_xyxy((10, 20, 30, 40)) == (10, 20, 40, 60)
    assert g.xyxy_to_xywh((10, 20, 40, 60)) == (10, 20, 30, 40)


@given(coord, coord, size, size)
def test_xyxy_xywh_xyxy_is_exact_for_boxes_built_from_xywh(
    x: float, y: float, w: float, h: float
) -> None:
    box = g.xywh_to_xyxy((x, y, w, h))
    assert g.xywh_to_xyxy(g.xyxy_to_xywh(box)) == box


@given(coord, coord, size, size)
def test_xywh_xyxy_xywh_preserves_origin_and_is_close(
    x: float, y: float, w: float, h: float
) -> None:
    out = g.xyxy_to_xywh(g.xywh_to_xyxy((x, y, w, h)))
    assert out[0] == x and out[1] == y
    assert math.isclose(out[2], w, rel_tol=1e-9, abs_tol=1e-9 * max(1.0, x))
    assert math.isclose(out[3], h, rel_tol=1e-9, abs_tol=1e-9 * max(1.0, y))


@given(
    st.integers(1, 5000),
    st.integers(1, 5000),
    st.integers(0, 4999),
    st.integers(0, 4999),
    st.integers(1, 5000),
    st.integers(1, 5000),
)
def test_cxcywh_norm_round_trip_is_close(
    width: int, height: int, x1: int, y1: int, bw: int, bh: int
) -> None:
    x1, y1 = min(x1, width - 1), min(y1, height - 1)
    box = (float(x1), float(y1), float(min(x1 + bw, width)), float(min(y1 + bh, height)))
    back = g.cxcywh_norm_to_xyxy(g.xyxy_to_cxcywh_norm(box, width, height), width, height)
    assert back == pytest.approx(box, abs=1e-6)


@given(st.integers(1, 5000), st.integers(1, 5000))
def test_xyxy_norm_round_trip_is_close(width: int, height: int) -> None:
    box = (0.0, 0.0, float(width), float(height))
    norm = g.xyxy_to_norm(box, width, height)
    assert norm == (0.0, 0.0, 1.0, 1.0)
    assert g.norm_to_xyxy(norm, width, height) == box


def test_cxcywh_norm_known_values() -> None:
    assert g.xyxy_to_cxcywh_norm((0, 0, 50, 100), 100, 200) == (0.25, 0.25, 0.5, 0.5)
    assert g.cxcywh_norm_to_xyxy((0.25, 0.25, 0.5, 0.5), 100, 200) == (0, 0, 50, 100)


def test_clip_snaps_tiny_overshoot_and_rejects_large_one() -> None:
    assert g.clip_to_image((0, 0, 100.0000001, 50), 100, 50, tol=1e-3) == (0, 0, 100, 50)
    assert g.clip_to_image((-1e-9, 0, 10, 10), 100, 50, tol=1e-3) == (0, 0, 10, 10)
    with pytest.raises(ValueError, match="outside"):
        g.clip_to_image((0, 0, 101, 50), 100, 50, tol=1e-3)


@pytest.mark.parametrize("bad", [(math.nan, 0, 1, 1), (0, 0, math.inf, 1)])
def test_non_finite_boxes_are_rejected(bad: g.Box) -> None:
    with pytest.raises(ValueError, match="finite"):
        g.xyxy_to_xywh(bad)


def test_non_positive_image_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        g.xyxy_to_cxcywh_norm((0, 0, 1, 1), 0, 10)


# ------------------------------------------------------------------ iou_xyxy (NumPy)
def _iou_reference(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_iou_xyxy_known_values_and_shape() -> None:
    import numpy as np

    from datacopilot.core.geometry import iou_xyxy

    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float64)
    out = iou_xyxy(a, b)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], [1.0, 50 / 150, 0.0])


def test_iou_xyxy_degenerate_and_nonfinite_boxes_give_zero() -> None:
    import numpy as np

    from datacopilot.core.geometry import iou_xyxy

    zero = np.array([[5, 5, 5, 5]], dtype=np.float64)
    nan = np.array([[np.nan, 0, 1, 1]], dtype=np.float64)
    real = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert iou_xyxy(zero, zero)[0, 0] == 0.0
    assert iou_xyxy(nan, real)[0, 0] == 0.0
    assert iou_xyxy(np.empty((0, 4)), real).shape == (0, 1)


def test_iou_xyxy_matches_a_scalar_reference_on_random_boxes() -> None:
    import numpy as np

    from datacopilot.core.geometry import iou_xyxy

    rng = np.random.default_rng(0)

    def boxes(n: int) -> np.ndarray:
        xy = rng.uniform(0, 80, size=(n, 2))
        wh = rng.uniform(0.5, 40, size=(n, 2))
        return np.hstack([xy, xy + wh])

    a, b = boxes(40), boxes(30)
    got = iou_xyxy(a, b)
    for i in range(40):
        for j in range(30):
            assert got[i, j] == pytest.approx(_iou_reference(tuple(a[i]), tuple(b[j])), abs=1e-12)
    np.testing.assert_allclose(iou_xyxy(a, a), iou_xyxy(a, a).T)
