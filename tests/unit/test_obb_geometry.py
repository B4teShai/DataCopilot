"""Oriented-box geometry (ADR-0011): the tuple helpers and the vectorised rotated IoU.

The rotated IoU is checked against an independent oracle in exact rational arithmetic: the overlap
of two convex polygons is the convex hull of the corners of each that lie in the other plus the
edge crossings (a different algorithm from the Sutherland-Hodgman clipping under test).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from datacopilot.core import obb
from datacopilot.core.geometry import (
    Quad,
    check_quad,
    iou_xyxy,
    quad_enclosing_box,
    quad_signed_area,
    xyxy_to_quad,
)

Point = tuple[Fraction, Fraction]


def rotated_rect(cx: float, cy: float, w: float, h: float, angle: float) -> Quad:
    c, s = math.cos(angle), math.sin(angle)
    out: list[float] = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out += [cx + dx * c - dy * s, cy + dx * s + dy * c]
    return tuple(out)  # type: ignore[return-value]


# ------------------------------------------------------------------ exact oracle
def _pts(q: Quad) -> list[Point]:
    return [(Fraction(q[2 * i]), Fraction(q[2 * i + 1])) for i in range(4)]


def _cross(o: Point, a: Point, b: Point) -> Fraction:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _ccw(poly: list[Point]) -> list[Point]:
    area = sum(poly[i][0] * poly[i - 3][1] - poly[i - 3][0] * poly[i][1] for i in range(4))
    return poly if area > 0 else poly[::-1]


def _inside(p: Point, poly: list[Point]) -> bool:
    return all(_cross(poly[i], poly[(i + 1) % 4], p) >= 0 for i in range(4))


def _segment_hits(a: list[Point], b: list[Point]) -> list[Point]:
    out = []
    for i in range(4):
        p, r = a[i], (a[(i + 1) % 4][0] - a[i][0], a[(i + 1) % 4][1] - a[i][1])
        for j in range(4):
            q, s = b[j], (b[(j + 1) % 4][0] - b[j][0], b[(j + 1) % 4][1] - b[j][1])
            denom = r[0] * s[1] - r[1] * s[0]
            if denom == 0:
                continue
            qp = (q[0] - p[0], q[1] - p[1])
            t = (qp[0] * s[1] - qp[1] * s[0]) / denom
            u = (qp[0] * r[1] - qp[1] * r[0]) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                out.append((p[0] + t * r[0], p[1] + t * r[1]))
    return out


def _hull_area(points: list[Point]) -> Fraction:
    pts = sorted(set(points))
    if len(pts) < 3:
        return Fraction(0)
    lower: list[Point] = []
    upper: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    twice = sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )
    return abs(twice) / 2


def oracle_iou(qa: Quad, qb: Quad) -> float:
    a, b = _ccw(_pts(qa)), _ccw(_pts(qb))
    candidates = [p for p in a if _inside(p, b)] + [p for p in b if _inside(p, a)]
    inter = _hull_area(candidates + _segment_hits(a, b))
    area_a, area_b = _hull_area(a), _hull_area(b)
    return float(inter / (area_a + area_b - inter))


# ------------------------------------------------------------------ strategies
coords = st.floats(min_value=0.0, max_value=200.0, allow_nan=False)
sides = st.floats(min_value=0.5, max_value=80.0, allow_nan=False)
angles = st.floats(min_value=0.0, max_value=math.pi, allow_nan=False)


@st.composite
def rects(draw: st.DrawFn) -> Quad:
    return rotated_rect(draw(coords), draw(coords), draw(sides), draw(sides), draw(angles))


@st.composite
def convex_quads(draw: st.DrawFn) -> Quad:
    """Four points on a circle at sorted angles: a convex, generally non-rectangular quad."""
    cx, cy, r = draw(coords), draw(coords), draw(st.floats(min_value=1.0, max_value=60.0))
    gaps = [draw(st.floats(min_value=0.3, max_value=2.0)) for _ in range(4)]
    total = sum(gaps)
    theta, out = draw(angles), []
    for g in gaps:
        out += [cx + r * math.cos(theta), cy + r * math.sin(theta)]
        theta += 2 * math.pi * g / total
    quad: Quad = tuple(out)  # type: ignore[assignment]
    if draw(st.booleans()):  # either drawing direction
        quad = tuple(sum(([quad[2 * i], quad[2 * i + 1]] for i in (3, 2, 1, 0)), []))  # type: ignore[assignment]
    return quad


shapes = st.one_of(rects(), convex_quads())


# ------------------------------------------------------------------ tuple helpers
def test_enclosing_box_is_min_and_max_of_the_corners() -> None:
    assert quad_enclosing_box((3, 1, 9, 2, 8, 7, 2, 6)) == (2, 1, 9, 7)
    with pytest.raises(ValueError, match="eight"):
        quad_enclosing_box((1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        quad_enclosing_box((0, 0, 1, 0, 1, math.inf, 0, 1))


@pytest.mark.parametrize(
    "quad",
    [
        (0, 0, 4, 0, 4, 2, 0, 2),  # one direction
        (0, 2, 4, 2, 4, 0, 0, 0),  # the other
        (1, 0, 3, 1, 2, 3, 0, 2),  # rotated
        (0, 0, 5, 0, 4, 2, 1, 2),  # trapezoid: convex, not a rectangle
    ],
)
def test_check_quad_accepts_convex_quadrilaterals(quad: Quad) -> None:
    check_quad(quad)


@pytest.mark.parametrize(
    "quad",
    [
        (0, 0, 4, 2, 4, 0, 0, 2),  # bow-tie (crossed order)
        (0, 0, 2, 0, 4, 0, 0, 2),  # three collinear corners
        (0, 0, 0, 0, 4, 2, 0, 2),  # repeated corner
        (0, 0, 4, 0, 1, 1, 0, 4),  # concave (a dart)
        (1, 1, 1, 1, 1, 1, 1, 1),  # a point
    ],
)
def test_check_quad_refuses_degenerate_shapes(quad: Quad) -> None:
    with pytest.raises(ValueError, match="convex quadrilateral"):
        check_quad(quad)


def test_signed_area_and_axis_aligned_corners() -> None:
    q = xyxy_to_quad((1.0, 2.0, 5.0, 4.0))
    assert q == (1.0, 2.0, 5.0, 2.0, 5.0, 4.0, 1.0, 4.0)
    assert quad_signed_area(q) == 8.0
    assert quad_signed_area(q[6:] + q[4:6] + q[2:4] + q[:2]) == -8.0  # type: ignore[arg-type]


# ------------------------------------------------------------------ vectorised
def test_areas_enclosing_and_min_area_rect_of_a_rotated_rectangle() -> None:
    q = np.array([rotated_rect(50, 40, 30, 10, 0.4)])
    assert obb.areas(q)[0] == pytest.approx(300.0)
    long_side, short_side, area = obb.min_area_rects(q)
    assert (long_side[0], short_side[0], area[0]) == pytest.approx((30.0, 10.0, 300.0))
    assert obb.enclosing_boxes(q)[0] == pytest.approx(quad_enclosing_box(tuple(q[0])))


def test_min_area_rect_of_a_rhombus_is_larger_than_the_rhombus() -> None:
    rhombus = np.array([[0, 0, 10, 0, 12, 5, 2, 5]], dtype=np.float64)
    _, _, rect = obb.min_area_rects(rhombus)
    assert obb.areas(rhombus)[0] == pytest.approx(50.0)
    assert rect[0] > 50.0
    assert obb.min_area_rects(np.zeros((0, 8)))[0].shape == (0,)


def test_a_square_and_the_same_square_turned_by_45_degrees() -> None:
    square = np.array([rotated_rect(1, 1, 2, 2, 0.0)])
    diamond = np.array([rotated_rect(1, 1, 2, 2, math.pi / 4)])
    overlap = 8 * (math.sqrt(2) - 1)  # the regular octagon inside both
    assert obb.intersection_areas(square, diamond)[0] == pytest.approx(overlap)
    assert obb.iou_obb(square, diamond)[0, 0] == pytest.approx(overlap / (8 - overlap))


def test_identical_disjoint_and_degenerate_pairs() -> None:
    a = np.array([rotated_rect(10, 10, 8, 4, 0.3), rotated_rect(90, 90, 8, 4, 1.0)])
    iou = obb.iou_obb(a, a)
    assert iou[0, 0] == pytest.approx(1.0) and iou[1, 1] == pytest.approx(1.0)
    assert iou[0, 1] == 0.0 and iou[1, 0] == 0.0
    bad = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [np.nan] * 8], dtype=np.float64)
    assert not obb.iou_obb(bad, a).any()
    assert obb.iou_obb(np.zeros((0, 8)), a).shape == (0, 2)
    assert obb.intersection_areas(np.zeros((0, 8)), np.zeros((0, 8))).shape == (0,)


def test_min_iou_skips_pairs_that_cannot_exceed_it() -> None:
    a = np.array([rotated_rect(10, 10, 10, 10, 0.0)])
    b = np.array([rotated_rect(15, 10, 10, 10, 0.0)])  # IoU 1/3
    assert obb.iou_obb(a, b)[0, 0] == pytest.approx(1 / 3)
    assert obb.iou_obb(a, b, min_iou=0.5)[0, 0] == 0.0  # bound 1/3 <= 0.5: never clipped
    assert obb.iou_obb(a, b, min_iou=0.2)[0, 0] == pytest.approx(1 / 3)


def test_many_pairs_are_processed_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(obb, "PAIR_CHUNK", 7)
    rng = np.random.default_rng(0)
    quads = np.array(
        [
            rotated_rect(*rng.uniform(20, 40, 2), *rng.uniform(5, 15, 2), rng.uniform(0, 3))
            for _ in range(12)
        ]
    )
    full = obb.iou_obb(quads, quads)
    assert np.allclose(np.diag(full), 1.0)
    assert np.allclose(full, full.T, atol=1e-9)


@settings(max_examples=300, deadline=None)
@given(shapes, shapes)
def test_rotated_iou_matches_the_exact_oracle(qa: Quad, qb: Quad) -> None:
    for q in (qa, qb):
        try:
            check_quad(q)
        except ValueError:
            assume(False)
    got = obb.iou_obb(np.array([qa]), np.array([qb]))[0, 0]
    assert got == pytest.approx(oracle_iou(qa, qb), abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(shapes, shapes)
def test_rotated_iou_is_symmetric_and_ignores_the_corner_order(qa: Quad, qb: Quad) -> None:
    a, b = np.array([qa]), np.array([qb])
    shifted = np.roll(a, 2, axis=1)  # the same box, starting at the next corner
    reversed_ = a.reshape(1, 4, 2)[:, ::-1].reshape(1, 8)
    base = obb.iou_obb(a, b)[0, 0]
    assert obb.iou_obb(b, a)[0, 0] == pytest.approx(base, abs=1e-9)
    assert obb.iou_obb(shifted, b)[0, 0] == pytest.approx(base, abs=1e-9)
    assert obb.iou_obb(reversed_, b)[0, 0] == pytest.approx(base, abs=1e-9)
    assert 0.0 <= base <= 1.0


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(coords, coords, sides, sides).map(
            lambda t: (t[0], t[1], t[0] + t[2], t[1] + t[3])
        ),
        min_size=1,
        max_size=6,
    )
)
def test_axis_aligned_boxes_give_the_axis_aligned_iou(boxes: list[tuple[float, ...]]) -> None:
    xyxy = np.array(boxes, dtype=np.float64)
    quads = np.array([xyxy_to_quad(b) for b in boxes])  # type: ignore[arg-type]
    assert np.allclose(obb.iou_obb(quads, quads), iou_xyxy(xyxy, xyxy), atol=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    rects(), st.floats(min_value=-math.pi, max_value=math.pi), st.floats(0, 50), st.floats(0, 50)
)
def test_rotated_iou_is_invariant_under_rotation_and_translation(
    q: Quad, turn: float, dx: float, dy: float
) -> None:
    other = rotated_rect(q[0] + 3, q[1] + 2, 20, 9, 0.7)

    def move(quad: Quad) -> Quad:
        c, s = math.cos(turn), math.sin(turn)
        out: list[float] = []
        for i in range(4):
            x, y = quad[2 * i], quad[2 * i + 1]
            out += [x * c - y * s + dx, x * s + y * c + dy]
        return tuple(out)  # type: ignore[return-value]

    before = obb.iou_obb(np.array([q]), np.array([other]))[0, 0]
    after = obb.iou_obb(np.array([move(q)]), np.array([move(other)]))[0, 0]
    assert after == pytest.approx(before, abs=1e-7)
