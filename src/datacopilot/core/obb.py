"""Vectorised geometry of oriented boxes (ADR-0011): areas, minimum-area rectangles, rotated IoU.

An oriented box is an array row of eight absolute coordinates, the corners ``x1, y1, ..., x4, y4``
of a convex quadrilateral (``core.schema`` validates that on every row). Everything here works on
``(n, 8)`` float arrays with NumPy only (no geometry library: the core dependency allow-list).

The rotated IoU clips one convex quadrilateral by the four edge half-planes of the other
(Sutherland-Hodgman), for many pairs at once: the polygons live in one ``(pairs, k, 2)`` array
with a per-pair vertex count, and each clip step emits, per vertex, the vertex itself (when it is
inside) and the edge crossing (when the edge leaves or enters), then packs the emitted points to
the front with a stable sort. Pairs whose enclosing boxes do not overlap, or whose IoU cannot
exceed a caller's threshold, are never clipped.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

PAIR_CHUNK = 65_536
"""Pairs clipped at once; bounds the working memory to a few tens of megabytes."""


def as_quads(values: object) -> F64:
    """``values`` as an ``(n, 8)`` float64 array (an empty input gives shape ``(0, 8)``)."""
    arr = np.asarray(values, dtype=np.float64)
    return arr.reshape(-1, 8)


def _points(quads: F64) -> F64:
    return quads.reshape(-1, 4, 2)


def _shoelace(pts: F64, count: NDArray[np.int64] | None = None) -> F64:
    """Signed area of each polygon in ``pts`` ``(p, k, 2)``; ``count`` vertices per row (all k
    when None)."""
    k = pts.shape[1]
    idx = np.arange(k)[None, :]
    if count is None:
        nxt = np.broadcast_to((idx + 1) % k, (pts.shape[0], k))
        valid = np.ones((pts.shape[0], k), dtype=bool)
    else:
        nxt = np.where(idx + 1 < count[:, None], idx + 1, 0)
        valid = idx < count[:, None]
    nx = np.take_along_axis(pts[..., 0], nxt, axis=1)
    ny = np.take_along_axis(pts[..., 1], nxt, axis=1)
    terms = np.where(valid, pts[..., 0] * ny - nx * pts[..., 1], 0.0)
    return 0.5 * terms.sum(axis=1)


def areas(quads: F64) -> F64:
    """Area of each oriented box, ``(n, 8) -> (n,)`` (whatever the corner order's direction)."""
    return np.abs(_shoelace(_points(as_quads(quads))))


def enclosing_boxes(quads: F64) -> F64:
    """Axis-aligned ``x_min, y_min, x_max, y_max`` of each oriented box, ``(n, 8) -> (n, 4)``."""
    q = as_quads(quads)
    xs, ys = q[:, 0::2], q[:, 1::2]
    return np.stack([xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1)


def min_area_rects(quads: F64) -> tuple[F64, F64, F64]:
    """Sides of the smallest rectangle (any rotation) that encloses each quadrilateral.

    For a convex polygon that rectangle has one side on a polygon edge, so the four edge
    directions are tried and the smallest area wins. Returns ``(long_side, short_side, area)``,
    each ``(n,)``. For a true rectangle these are its own sides and area; the ratio of a box's
    area to this area says how far it is from a rectangle (YOLO-OBB training code
    replaces every quadrilateral by this rectangle).
    """
    pts = _points(as_quads(quads))
    n = pts.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    with np.errstate(invalid="ignore", divide="ignore"):
        edges = np.roll(pts, -1, axis=1) - pts  # (n, 4 edges, 2)
        length = np.linalg.norm(edges, axis=2, keepdims=True)
        u = np.where(length > 0, edges / length, 0.0)
        v = np.stack([-u[..., 1], u[..., 0]], axis=2)
        pu = np.einsum("npd,ned->nep", pts, u)  # (n, 4 edges, 4 points)
        pv = np.einsum("npd,ned->nep", pts, v)
        ext_u = pu.max(axis=2) - pu.min(axis=2)
        ext_v = pv.max(axis=2) - pv.min(axis=2)
        area = np.where(length[..., 0] > 0, ext_u * ext_v, np.inf)
    best = np.argmin(area, axis=1)
    rows = np.arange(n)
    a, b = ext_u[rows, best], ext_v[rows, best]
    best_area = area[rows, best]
    return np.maximum(a, b), np.minimum(a, b), np.where(np.isfinite(best_area), best_area, 0.0)


def _counter_clockwise(pts: F64) -> F64:
    """The same polygons with their corner order reversed where the signed area is negative."""
    flip = _shoelace(pts) < 0
    return np.where(flip[:, None, None], pts[:, ::-1, :], pts)


def intersection_areas(a: F64, b: F64) -> F64:
    """Area of the overlap of oriented boxes ``a[i]`` and ``b[i]``, ``(p, 8) x (p, 8) -> (p,)``.

    Both must be convex quadrilaterals (any corner direction). Sutherland-Hodgman: ``a`` is clipped
    by each edge half-plane of ``b`` in turn; all pairs advance together.
    """
    subject = _counter_clockwise(_points(as_quads(a)))
    clip = _counter_clockwise(_points(as_quads(b)))
    p = subject.shape[0]
    if p == 0:
        return np.zeros(0, dtype=np.float64)
    poly = subject.copy()
    count = np.full(p, 4, dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for e in range(4):
            k = poly.shape[1]
            start = clip[:, e]
            edge = clip[:, (e + 1) % 4] - start
            rel = poly - start[:, None, :]
            side = edge[:, None, 0] * rel[..., 1] - edge[:, None, 1] * rel[..., 0]  # (p, k)
            idx = np.arange(k)[None, :]
            valid = idx < count[:, None]
            nxt = np.where(idx + 1 < count[:, None], idx + 1, 0)
            side_next = np.take_along_axis(side, nxt, axis=1)
            vert_next = np.take_along_axis(poly, nxt[..., None], axis=1)
            inside = side >= 0
            crossing = valid & (inside != (side_next >= 0))
            t = np.where(crossing, side / np.where(crossing, side - side_next, 1.0), 0.0)
            point = poly + t[..., None] * (vert_next - poly)
            emitted = np.stack([valid & inside, crossing], axis=2).reshape(p, 2 * k)
            candidates = np.stack([poly, point], axis=2).reshape(p, 2 * k, 2)
            order = np.argsort(~emitted, axis=1, kind="stable")
            count = emitted.sum(axis=1).astype(np.int64)
            width = max(int(np.max(count)), 1)
            poly = np.take_along_axis(candidates, order[:, :width, None], axis=1)
    area = np.abs(_shoelace(poly, count))
    return np.where((count >= 3) & np.isfinite(area), area, 0.0)


def iou_obb(a: F64, b: F64, *, min_iou: float = 0.0) -> F64:
    """Pairwise rotated intersection over union, ``(n, 8) x (m, 8) -> (n, m)``.

    A pair with a non-finite coordinate, a zero area or a zero union has IoU 0.0 (never NaN).
    Only pairs whose enclosing boxes overlap and whose IoU could exceed ``min_iou`` are clipped:
    the bound ``min(I_enc, A, B) / (A + B - min(I_enc, A, B))`` (``I_enc`` the overlap of the
    enclosing boxes) caps the true IoU, so a pair at or below the bound's threshold is reported as
    0.0. With the default ``min_iou = 0`` every overlapping pair is exact.
    """
    qa, qb = as_quads(a), as_quads(b)
    n, m = qa.shape[0], qb.shape[0]
    out = np.zeros((n, m), dtype=np.float64)
    if n == 0 or m == 0:
        return out
    area_a, area_b = areas(qa), areas(qb)
    ea, eb = enclosing_boxes(qa), enclosing_boxes(qb)
    ok_a = np.isfinite(qa).all(axis=1) & (area_a > 0)
    ok_b = np.isfinite(qb).all(axis=1) & (area_b > 0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        w = np.minimum(ea[:, None, 2], eb[None, :, 2]) - np.maximum(ea[:, None, 0], eb[None, :, 0])
        h = np.minimum(ea[:, None, 3], eb[None, :, 3]) - np.maximum(ea[:, None, 1], eb[None, :, 1])
        enc = np.clip(w, 0.0, None) * np.clip(h, 0.0, None)
        bound = np.minimum(enc, np.minimum(area_a[:, None], area_b[None, :]))
        upper = bound / (area_a[:, None] + area_b[None, :] - bound)
        candidate = ok_a[:, None] & ok_b[None, :] & (enc > 0) & (upper > min_iou)
    rows, cols = np.nonzero(candidate)
    for s in range(0, rows.size, PAIR_CHUNK):
        r, c = rows[s : s + PAIR_CHUNK], cols[s : s + PAIR_CHUNK]
        inter = intersection_areas(qa[r], qb[c])
        with np.errstate(invalid="ignore", divide="ignore"):
            union = area_a[r] + area_b[c] - inter
            iou = np.where(union > 0, inter / union, 0.0)
        out[r, c] = np.clip(np.where(np.isfinite(iou), iou, 0.0), 0.0, 1.0)
    return out
