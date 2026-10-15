"""Box sanity: bounds, area, size, aspect, duplicate boxes and conflicting classes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.auditdata import CLASSES, dataset
from test_audit_schema import RUN

from datacopilot.audit import registry
from datacopilot.audit.checks import boxes
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.core.schema import Annotation, Category, Dataset, Finding, Image

Box = tuple[float, float, float, float]
CATS = [Category(id=i + 1, name=n) for i, n in enumerate(CLASSES)]


def ann(n: int, image: int, cat: int, box: Box, **kw: Any) -> Annotation:
    """An annotation built without validation, so hostile geometry can be represented."""
    return Annotation.model_construct(
        ann_id=f"a{n:06d}",
        image_id=f"{image:016x}",
        category_id=cat,
        kind="bbox",
        bbox_xyxy_abs=box,
        score=None,
        is_crowd=False,
        occluded=None,
        truncated=None,
        attributes={},
        provenance="human",
        source={},
        **kw,
    )


def make(
    boxes_by_image: dict[int, list[tuple[int, Box]]], size: tuple[int, int] = (100, 100)
) -> Dataset:
    images = [
        Image(image_id=f"{i:016x}", file_path=f"/x/{i}.png", width=size[0], height=size[1])
        for i in boxes_by_image
    ]
    anns: list[Annotation] = []
    for i, items in boxes_by_image.items():
        for cat, box in items:
            anns.append(ann(len(anns), i, cat, box))
    base = dataset(images, [], CATS)
    return Dataset(base.manifest, base.images, tuple(anns))


def run(ds: Dataset, **cfg: Any) -> list[Finding]:
    ctx = CheckContext(
        run_id=RUN,
        root=Path("/nonexistent"),
        cache_dir=Path("/nonexistent"),
        config=CheckConfig(**cfg),
    )
    return boxes.plugin.run(ds, ctx)


def rules(findings: list[Finding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        out[str(f.payload["rule"])] = out.get(str(f.payload["rule"]), 0) + 1
    return out


def test_plugin_is_registered() -> None:
    assert "boxes" in registry.check_names()


def test_clean_boxes_have_no_findings() -> None:
    ds = make({0: [(1, (10, 10, 50, 50)), (2, (60, 60, 90, 90))], 1: [(1, (0, 0, 100, 100))]})
    assert run(ds) == []


@pytest.mark.parametrize(
    "box", [(-1, 0, 10, 10), (0, -0.5, 10, 10), (0, 0, 100.5, 10), (0, 0, 10, 101)]
)
def test_out_of_bounds_is_critical(box: Box) -> None:
    (f,) = run(make({0: [(1, box)]}))
    assert f.severity == "CRITICAL" and f.payload["rule"] == "out_of_bounds"
    assert f.ann_id == "a000000" and f.image_id == f"{0:016x}"


@pytest.mark.parametrize(
    "box", [(10, 10, 10, 20), (10, 10, 20, 10), (30, 10, 20, 20), (5, 5, 5, 5)]
)
def test_zero_or_inverted_area_is_critical(box: Box) -> None:
    (f,) = run(make({0: [(1, box)]}))
    assert f.severity == "CRITICAL" and f.payload["rule"] == "zero_area"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinates_are_critical_and_do_not_poison_other_boxes(value: float) -> None:
    ds = make({0: [(1, (value, 0, 10, 10)), (1, (20, 20, 40, 40)), (1, (20, 20, 40, 40))]})
    found = rules(run(ds))
    assert found == {"invalid_box": 1, "duplicate_box": 1}


def test_tiny_boxes_by_side_and_by_area_fraction() -> None:
    ds = make({0: [(1, (10, 10, 11, 15)), (1, (50, 50, 52.5, 52.5)), (1, (60, 10, 90, 60))]})
    # width 1 < 2px; 2.5x2.5 is 6.25 px of 10,000 = 0.06 percent, above the 0.01 percent default
    assert [(f.ann_id, f.severity, f.payload["rule"]) for f in run(ds)] == [
        ("a000000", "WARN", "tiny_box")
    ]
    assert [f.ann_id for f in run(ds, tiny_box_side_px=3.0)] == ["a000000", "a000001"]
    assert run(ds, tiny_box_side_px=0.5, tiny_box_area_fraction=1e-5) == []


def test_extreme_aspect_ratio_is_warn_in_both_directions() -> None:
    ds = make({0: [(1, (0, 10, 90, 14)), (1, (10, 0, 14, 90)), (1, (20, 20, 80, 60))]})
    found = [(f.ann_id, f.payload["rule"]) for f in run(ds)]
    assert found == [("a000000", "extreme_aspect"), ("a000001", "extreme_aspect")]
    assert run(ds, max_box_aspect_ratio=50.0) == []


def test_duplicate_boxes_of_the_same_class_above_the_iou_threshold() -> None:
    ds = make({0: [(1, (10, 10, 50, 50)), (1, (10.5, 10, 50, 50.5)), (2, (60, 60, 90, 90))]})
    (f,) = run(ds)
    assert f.severity == "WARN" and f.payload["rule"] == "duplicate_box"
    assert f.ann_id == "a000001" and f.payload["other_ann_id"] == "a000000"
    assert 0.9 < float(f.payload["iou"]) <= 1.0  # type: ignore[arg-type]


def test_iou_exactly_at_the_threshold_is_not_a_duplicate() -> None:
    ds = make({0: [(1, (0, 0, 10, 10)), (1, (0, 0, 10, 9))]})  # IoU = 0.9
    assert run(ds) == []
    assert rules(run(ds, duplicate_box_iou=0.89)) == {"duplicate_box": 1}


def test_same_box_with_different_classes_is_a_conflict() -> None:
    ds = make({0: [(1, (10, 10, 50, 50)), (2, (10, 10, 50, 50))]})
    (f,) = run(ds)
    assert f.payload["rule"] == "conflicting_class"
    assert f.payload["category_id"] == 2 and f.payload["other_category_id"] == 1


def test_identical_boxes_on_different_images_are_not_duplicates() -> None:
    assert run(make({0: [(1, (10, 10, 50, 50))], 1: [(1, (10, 10, 50, 50))]})) == []


def test_each_later_box_is_reported_once_against_its_earliest_match() -> None:
    ds = make({0: [(1, (10, 10, 50, 50))] * 5})
    found = run(ds)
    assert [(f.ann_id, f.payload["other_ann_id"]) for f in found] == [
        (f"a00000{k}", "a000000") for k in range(1, 5)
    ]


def _iou(a: Box, b: Box) -> float:
    """Scalar IoU written independently of ``iou_xyxy``."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_pairs_match_the_documented_definition_across_block_boundaries() -> None:
    """Per later box: its earliest same-class match is a duplicate_box, its earliest
    different-class match a conflicting_class; both can apply to one box."""
    rng = np.random.default_rng(3)
    n = 700  # more than two column blocks of 256
    xy = rng.uniform(0, 60, size=(n, 2))
    wh = rng.uniform(5, 30, size=(n, 2))
    raw = np.hstack([xy, xy + wh])
    cats = [int(c) for c in rng.integers(1, 3, size=n)]
    for later, earlier, same in [
        (600, 5, True),
        (300, 5, False),
        (256, 255, True),
        (699, 0, False),
    ]:
        raw[later] = raw[earlier]
        cats[later] = cats[earlier] if same else 3 - cats[earlier]
    boxes_py: list[Box] = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in raw]
    got = {
        (str(f.ann_id), str(f.payload["other_ann_id"]), str(f.payload["rule"]))
        for f in run(make({0: list(zip(cats, boxes_py, strict=True))}, size=(200, 200)))
        if f.payload["rule"] in {"duplicate_box", "conflicting_class"}
    }
    expected = set()
    for j in range(1, n):
        same = next(
            (i for i in range(j) if cats[i] == cats[j] and _iou(boxes_py[i], boxes_py[j]) > 0.9),
            None,
        )
        diff = next(
            (i for i in range(j) if cats[i] != cats[j] and _iou(boxes_py[i], boxes_py[j]) > 0.9),
            None,
        )
        if same is not None:
            expected.add((f"a{j:06d}", f"a{same:06d}", "duplicate_box"))
        if diff is not None:
            expected.add((f"a{j:06d}", f"a{diff:06d}", "conflicting_class"))
    assert got == expected and len(expected) >= 4


def test_a_box_can_be_both_a_duplicate_and_a_conflict() -> None:
    same_box: Box = (10, 10, 50, 50)
    ds = make({0: [(2, same_box), (1, same_box), (1, same_box)]})
    got = {(str(f.ann_id), str(f.payload["other_ann_id"]), str(f.payload["rule"])) for f in run(ds)}
    assert got == {
        ("a000001", "a000000", "conflicting_class"),
        ("a000002", "a000001", "duplicate_box"),
        ("a000002", "a000000", "conflicting_class"),
    }


def test_classification_and_non_bbox_annotations_are_ignored() -> None:
    base = make({0: [(1, (0, 0, 10, 10))]})
    cls = base.annotations[0].model_copy(update={"kind": "classification", "bbox_xyxy_abs": None})
    ds = Dataset(base.manifest.model_copy(update={"task": "classification"}), base.images, (cls,))
    assert run(ds) == []


def test_per_rule_cap_keeps_a_truncation_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boxes, "MAX_FINDINGS_PER_RULE", 3)
    ds = make({i: [(1, (0, 0, 1, 10))] for i in range(10)})  # ten tiny boxes
    found = run(ds)
    assert rules(found) == {"tiny_box": 3, "boxes_truncated": 1}
    marker = next(f for f in found if f.payload["rule"] == "boxes_truncated")
    assert marker.payload["truncated_rule"] == "tiny_box" and marker.payload["dropped"] == 7
    assert marker.severity == "WARN"


def test_images_with_too_many_boxes_skip_pair_search_and_say_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boxes, "MAX_BOXES_PER_IMAGE_FOR_PAIRS", 4)
    ds = make({0: [(1, (10, 10, 50, 50))] * 6, 1: [(1, (10, 10, 50, 50))] * 2})
    found = run(ds)
    assert rules(found) == {"duplicate_box": 1, "pairwise_skipped": 1}
    skipped = next(f for f in found if f.payload["rule"] == "pairwise_skipped")
    assert skipped.severity == "WARN" and skipped.payload["images"] == 1  # a gate must notice


def test_output_is_deterministic_and_fast_on_fifty_thousand_boxes() -> None:
    rng = np.random.default_rng(1)
    per_image: dict[int, list[tuple[int, Box]]] = {}
    for i in range(5000):
        items = []
        for _ in range(10):
            x, y = rng.uniform(0, 70, 2)
            w, h = rng.uniform(5, 30, 2)
            items.append(
                (int(rng.integers(1, 4)), (float(x), float(y), float(x + w), float(y + h)))
            )
        per_image[i] = items
    ds = make(per_image)
    start = time.perf_counter()
    first = run(ds)
    assert time.perf_counter() - start < 10.0
    assert first == run(ds)
