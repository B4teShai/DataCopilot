"""Box conversions between absolute xyxy, xywh and normalised representations.

All functions are pure and operate on 4-tuples of floats (an oriented box: an 8-tuple of corners,
see :data:`Quad`). Absolute coordinates are pixels. Vectorised oriented-box geometry (areas,
rotated IoU) is in :mod:`datacopilot.core.obb`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray

Box = tuple[float, float, float, float]
Quad = tuple[float, float, float, float, float, float, float, float]
"""An oriented box: four corners ``x1, y1, x2, y2, x3, y3, x4, y4`` in drawing order (ADR-0011)."""


def _check_finite(box: Box) -> None:
    if not all(math.isfinite(v) for v in box):
        raise ValueError("box coordinates must be finite")


def _check_size(width: float, height: float) -> None:
    if not (width > 0 and height > 0):
        raise ValueError("image width and height must be positive")


def xywh_to_xyxy(box: Box) -> Box:
    """(x, y, w, h) -> (x_min, y_min, x_max, y_max)."""
    _check_finite(box)
    x, y, w, h = box
    return (x, y, x + w, y + h)


def _exact_extent(lo: float, hi: float) -> float:
    """Return w such that ``lo + w == hi`` exactly when such a float exists, else ``hi - lo``."""
    w = hi - lo
    up = down = w
    for _ in range(4):
        if lo + up == hi:
            return up
        if lo + down == hi:
            return down
        up, down = math.nextafter(up, math.inf), math.nextafter(down, -math.inf)
    return w


def xyxy_to_xywh(box: Box) -> Box:
    """(x_min, y_min, x_max, y_max) -> (x, y, w, h).

    For a box produced by :func:`xywh_to_xyxy`, ``xywh_to_xyxy`` of the result reproduces it
    exactly: plain subtraction can be an ulp off (``x + (x2 - x) != x2``), so the width is nudged
    to the neighbouring float that restores the corner. Arbitrary xyxy floats may have no exact
    xywh form; the nearest width is returned then.
    """
    _check_finite(box)
    x1, y1, x2, y2 = box
    return (x1, y1, _exact_extent(x1, x2), _exact_extent(y1, y2))


def xyxy_to_cxcywh_norm(box: Box, width: float, height: float) -> Box:
    """Absolute xyxy -> normalised (cx, cy, w, h), the YOLO convention."""
    _check_finite(box)
    _check_size(width, height)
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def cxcywh_norm_to_xyxy(box: Box, width: float, height: float) -> Box:
    """Normalised (cx, cy, w, h) -> absolute xyxy."""
    _check_finite(box)
    _check_size(width, height)
    cx, cy, w, h = box
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def xyxy_to_norm(box: Box, width: float, height: float) -> Box:
    """Absolute xyxy -> normalised xyxy in [0, 1]."""
    _check_finite(box)
    _check_size(width, height)
    x1, y1, x2, y2 = box
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def norm_to_xyxy(box: Box, width: float, height: float) -> Box:
    """Normalised xyxy -> absolute xyxy."""
    _check_finite(box)
    _check_size(width, height)
    x1, y1, x2, y2 = box
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def clip_to_image(box: Box, width: float, height: float, *, tol: float) -> Box:
    """Snap a box that overshoots the image by at most ``tol`` pixels back inside.

    Raises:
        ValueError: if any coordinate is more than ``tol`` pixels outside the image.
    """
    _check_finite(box)
    _check_size(width, height)
    x1, y1, x2, y2 = box
    if x1 < -tol or y1 < -tol or x2 > width + tol or y2 > height + tol:
        raise ValueError("box lies outside the image")
    return (max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height)))


def iou_xyxy(a: NDArray[float64], b: NDArray[float64]) -> NDArray[float64]:
    """Pairwise intersection over union of two box arrays, ``(n, 4) x (m, 4) -> (n, m)``.

    Boxes are absolute ``x_min, y_min, x_max, y_max``. A pair with an empty or inverted box, a
    non-finite coordinate, or a zero union has IoU 0.0 (never NaN, never a division warning).
    NumPy is imported here, not at module level, so the tuple helpers stay import-light.
    """
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        left = np.maximum(a[:, None, 0], b[None, :, 0])
        top = np.maximum(a[:, None, 1], b[None, :, 1])
        right = np.minimum(a[:, None, 2], b[None, :, 2])
        bottom = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        valid = (area_a[:, None] > 0) & (area_b[None, :] > 0) & (union > 0)
        out = np.where(valid, inter / union, 0.0)
    return np.where(np.isfinite(out), out, 0.0)


def quad_enclosing_box(quad: Quad) -> Box:
    """The axis-aligned box ``(x_min, y_min, x_max, y_max)`` that encloses the four corners.

    Min and max are exact on floats, so the result is reproducible bit for bit.

    Raises:
        ValueError: not eight finite coordinates.
    """
    if len(quad) != 8:
        raise ValueError("an oriented box has exactly eight coordinates")
    if not all(math.isfinite(v) for v in quad):
        raise ValueError("box coordinates must be finite")
    xs, ys = quad[0::2], quad[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def quad_signed_area(quad: Quad) -> float:
    """Shoelace area of the corners in their order: positive for one turning direction, negative
    for the other (which one is "clockwise" depends on whether y points up or down)."""
    xs, ys = quad[0::2], quad[1::2]
    return 0.5 * sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))


def check_quad(quad: Quad) -> None:
    """Accept only a strictly convex quadrilateral: every corner turns the same way.

    Equal turning signs at all four corners also rule out a self-intersecting ("bow-tie") order,
    repeated corners and three collinear corners, and imply a positive area. Either drawing
    direction is accepted; the order is kept as given.

    Raises:
        ValueError: a non-finite coordinate, or corners that do not form such a quadrilateral.
    """
    quad_enclosing_box(quad)  # length and finiteness
    xs, ys = quad[0::2], quad[1::2]
    turns = []
    for i in range(4):
        ax, ay = xs[(i + 1) % 4] - xs[i], ys[(i + 1) % 4] - ys[i]
        bx, by = xs[(i + 2) % 4] - xs[(i + 1) % 4], ys[(i + 2) % 4] - ys[(i + 1) % 4]
        turns.append(ax * by - ay * bx)
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError(
            "the corners do not form a convex quadrilateral (crossed, repeated or collinear "
            "corners)"
        )


def xyxy_to_quad(box: Box) -> Quad:
    """The four corners of an axis-aligned box, clockwise on screen from the top-left corner."""
    _check_finite(box)
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1, x2, y2, x1, y2)
