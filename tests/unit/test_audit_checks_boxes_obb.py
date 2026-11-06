"""The ``boxes`` check on oriented boxes, with planted errors (ADR-0011); and the whole audit,
embedding checks included, on a YOLO-OBB dataset."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fixtures import synth
from fixtures.auditdata import dataset
from test_audit_checks_boxes import CATS, rules, run

from datacopilot import api, convert
from datacopilot.core.geometry import quad_enclosing_box
from datacopilot.core.schema import Annotation, Dataset, Finding, Image


def turned(cx: float, cy: float, w: float, h: float, degrees: float) -> tuple[float, ...]:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    out: list[float] = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out += [cx + dx * c - dy * s, cy + dx * s + dy * c]
    return tuple(out)


def oriented(n: int, image: int, cat: int, corners: tuple[float, ...]) -> Annotation:
    """Built without validation, so hostile corners can be represented."""
    finite = all(math.isfinite(v) for v in corners)
    xs, ys = corners[0::2], corners[1::2]
    box = quad_enclosing_box(corners) if finite else (min(xs), min(ys), max(xs), max(ys))  # type: ignore[arg-type]
    return Annotation.model_construct(
        ann_id=f"a{n:06d}",
        image_id=f"{image:016x}",
        category_id=cat,
        kind="obb",
        bbox_xyxy_abs=box,
        obb_xyxyxyxy_abs=corners,
        score=None,
        is_crowd=False,
        occluded=None,
        truncated=None,
        attributes={},
        provenance="human",
        source={},
    )


def make(boxes: list[tuple[int, tuple[float, ...]]], size: tuple[int, int] = (200, 200)) -> Dataset:
    images = [Image(image_id=f"{0:016x}", file_path="/x/0.png", width=size[0], height=size[1])]
    anns = [oriented(n, 0, cat, corners) for n, (cat, corners) in enumerate(boxes)]
    base = dataset(images, [], CATS)
    return Dataset(base.manifest, base.images, tuple(anns))


def by_rule(findings: list[Finding], rule: str) -> list[Finding]:
    return [f for f in findings if f.payload["rule"] == rule]


def test_clean_rotated_boxes_have_no_findings() -> None:
    ds = make([(1, turned(50, 50, 40, 20, 30)), (2, turned(140, 140, 30, 30, 60))])
    assert run(ds) == []


def test_a_thin_turned_stick_is_extreme_although_its_enclosing_box_is_square() -> None:
    ds = make([(1, turned(100, 100, 120, 4, 45))])
    (finding,) = run(ds)
    assert finding.payload["rule"] == "extreme_aspect"
    assert finding.payload["aspect_ratio"] == pytest.approx(30.0, rel=1e-3)


def test_tiny_uses_the_turned_sides_and_the_own_area() -> None:
    ds = make([(1, turned(100, 100, 30, 1.5, 45)), (1, turned(50, 50, 10, 10, 20))])
    found = by_rule(run(ds, max_box_aspect_ratio=100.0), "tiny_box")
    assert [f.ann_id for f in found] == ["a000000"]
    assert found[0].payload["height"] == pytest.approx(1.5, abs=1e-4)
    assert found[0].payload["area_fraction"] == pytest.approx(45 / 40_000, abs=1e-6)


def test_a_corner_outside_the_image_is_critical_and_names_the_corners() -> None:
    ds = make([(1, (150.0, 10.0, 205.0, 40.0, 180.0, 90.0, 140.0, 60.0))])
    (finding,) = by_rule(run(ds), "out_of_bounds")
    assert finding.severity == "CRITICAL"
    assert finding.payload["obb"] == [150.0, 10.0, 205.0, 40.0, 180.0, 90.0, 140.0, 60.0]
    assert finding.payload["bbox"] == [140.0, 10.0, 205.0, 90.0]


@pytest.mark.parametrize(
    "corners",
    [
        (10.0, 10.0, 50.0, 50.0, 50.0, 10.0, 10.0, 50.0),  # bow-tie
        (10.0, 10.0, 30.0, 10.0, 50.0, 10.0, 10.0, 50.0),  # three collinear corners
    ],
)
def test_degenerate_quadrilaterals_are_zero_area(corners: tuple[float, ...]) -> None:
    findings = run(make([(1, corners)]))
    assert [(f.severity, f.payload["rule"]) for f in findings] == [("CRITICAL", "zero_area")]
    assert findings[0].payload["obb"] == list(corners)


def test_non_finite_corners_are_invalid() -> None:
    corners = (10.0, 10.0, 50.0, 10.0, 50.0, float("nan"), 10.0, 50.0)
    assert rules(run(make([(1, corners), (1, turned(100, 100, 20, 20, 10))]))) == {"invalid_box": 1}


def test_a_box_far_from_a_rectangle_is_reported() -> None:
    skewed = (20.0, 20.0, 100.0, 20.0, 120.0, 60.0, 40.0, 60.0)  # a parallelogram: 3200 of 4000
    ds = make([(1, skewed), (2, turned(150, 150, 40, 20, 33))])
    (finding,) = run(ds)
    assert (finding.severity, finding.payload["rule"], finding.ann_id) == (
        "WARN",
        "obb_not_rectangular",
        "a000000",
    )
    assert finding.payload["rectangularity"] == pytest.approx(0.8, abs=1e-4)
    assert run(ds, min_obb_rectangularity=0.75) == []


def test_duplicates_and_conflicts_use_the_rotated_iou() -> None:
    a = turned(100, 100, 80, 20, 30)
    b = turned(100.5, 100, 80, 20, 30)  # the same box, half a pixel to the side
    ds = make([(1, a), (1, b), (2, a)])
    found = {(f.ann_id, f.payload["rule"]) for f in run(ds)}
    assert found == {("a000001", "duplicate_box"), ("a000002", "conflicting_class")}
    dup = by_rule(run(ds), "duplicate_box")[0]
    assert 0.9 < float(dup.payload["iou"]) < 1.0  # type: ignore[arg-type]


def test_crossing_sticks_are_not_duplicates_although_their_enclosing_boxes_are() -> None:
    ds = make([(1, turned(100, 100, 120, 10, 45)), (1, turned(100, 100, 120, 10, -45))])
    assert by_rule(run(ds, max_box_aspect_ratio=100.0), "duplicate_box") == []


def test_the_whole_audit_runs_on_a_yolo_obb_dataset(tmp_path: Path) -> None:
    src = synth.build_yolo_obb(tmp_path / "src")
    convert.ingest(src, "yolo_obb", tmp_path / "o.dcp")
    run_ = api.audit(
        tmp_path / "o.dcp",
        "all",
        embedder="hash",
        allow_test_embedder=True,
        fail_on=(),
        include_model_annotations=True,
    )
    assert {"boxes", "integrity", "near_duplicates", "leakage", "outliers", "label_noise"} <= set(
        run_.checks
    )
    # two fixture boxes are far from rectangles (a skewed one, 0.88, and a trapezoid, 0.75);
    # nothing else is wrong with the boxes
    box_findings = [f for f in run_.findings if f.check == "boxes"]
    assert {(f.ann_id, f.payload["rule"]) for f in box_findings} == {
        ("train/s1.png:0", "obb_not_rectangular"),
        ("train/s1.png:2", "obb_not_rectangular"),
    }
    assert not [f for f in run_.findings if f.check == "integrity"]
