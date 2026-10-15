"""Imbalance: per-split long-tail ratio, effective number of classes, missing and empty classes."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pytest
from fixtures.auditdata import CLASSES, dataset
from test_audit_schema import RUN

from datacopilot.audit import registry
from datacopilot.audit.checks import imbalance
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.core.schema import Annotation, Category, Dataset, Finding, Image


def make(
    counts: dict[str, dict[int, int]],
    *,
    categories: list[Category] | None = None,
    task: str = "detection",
) -> Dataset:
    """``counts[split][category_id] = n`` boxes (or labels), one image per split holds them all."""
    cats = categories or [Category(id=i + 1, name=n) for i, n in enumerate(CLASSES)]
    images: list[Image] = []
    anns: list[Annotation] = []
    for s, (split, per_class) in enumerate(counts.items()):
        img = Image(
            image_id=f"{s:016x}",
            file_path=f"/x/{s}.png",
            width=100,
            height=100,
            split=split,  # type: ignore[arg-type]
        )
        images.append(img)
        for cid, n in per_class.items():
            for k in range(n):
                anns.append(
                    Annotation(
                        ann_id=f"{s}-{cid}-{k}",
                        image_id=img.image_id,
                        category_id=cid,
                        kind="classification" if task == "classification" else "bbox",
                        bbox_xyxy_abs=None if task == "classification" else (1, 1, 5, 5),
                    )
                )
    return dataset(images, anns, cats) if task == "detection" else _cls(images, anns, cats)


def _cls(images: list[Image], anns: list[Annotation], cats: list[Category]) -> Dataset:
    ds = dataset(images, anns, cats, validate=False)
    manifest = ds.manifest.model_copy(update={"task": "classification"})
    return Dataset(manifest, ds.images, ds.annotations)


def run(ds: Dataset, **cfg: Any) -> list[Finding]:
    ctx = CheckContext(
        run_id=RUN,
        root=Path("/nonexistent"),
        cache_dir=Path("/nonexistent"),
        config=CheckConfig(**cfg),
    )
    return imbalance.plugin.run(ds, ctx)


def by_rule(findings: list[Finding], rule: str) -> list[Finding]:
    return [f for f in findings if f.payload["rule"] == rule]


def test_plugin_is_registered() -> None:
    assert "imbalance" in registry.check_names()


def test_balanced_split_has_no_findings() -> None:
    assert run(make({"train": {1: 10, 2: 10, 3: 10}})) == []


@pytest.mark.parametrize(
    ("counts", "severity"),
    [
        ({1: 30, 2: 10}, "INFO"),  # ratio 3.0: at the info threshold, inclusive
        ({1: 29, 2: 10}, None),  # 2.9
        ({1: 100, 2: 10}, "WARN"),  # 10.0: at the warn threshold, inclusive
        ({1: 99, 2: 10}, "INFO"),
        ({1: 500, 2: 1, 3: 20}, "WARN"),
    ],
)
def test_long_tail_ratio_thresholds(counts: dict[int, int], severity: str | None) -> None:
    found = by_rule(run(make({"train": counts})), "long_tail")
    assert [f.severity for f in found] == ([severity] if severity else [])


def test_thresholds_are_configurable() -> None:
    ds = make({"train": {1: 20, 2: 10}})
    assert by_rule(run(ds), "long_tail") == []
    (info,) = by_rule(run(ds, imbalance_info_ratio=1.5), "long_tail")
    assert info.severity == "INFO"
    (warn,) = by_rule(run(ds, imbalance_info_ratio=1.5, imbalance_warn_ratio=2.0), "long_tail")
    assert warn.severity == "WARN"


def test_payload_carries_ratio_effective_classes_and_head_and_tail() -> None:
    ds = make({"train": {1: 100, 2: 10, 3: 5}})
    (f,) = by_rule(run(ds), "long_tail")
    p = f.payload
    assert p["split"] == "train" and p["ratio"] == 20.0 and p["total"] == 115
    assert p["classes_present"] == 3
    probs = [100 / 115, 10 / 115, 5 / 115]
    expected = math.exp(-sum(q * math.log(q) for q in probs))
    assert p["effective_classes"] == pytest.approx(expected, abs=1e-4)
    assert p["head"][0] == {"id": 1, "name": "cat", "count": 100}  # type: ignore[index]
    assert p["tail"][0] == {"id": 3, "name": "bird", "count": 5}  # type: ignore[index]
    assert f.image_id is None and f.ann_id is None


def test_each_split_is_judged_on_its_own_counts() -> None:
    ds = make({"train": {1: 200, 2: 10, 3: 10}, "val": {1: 5, 2: 5, 3: 5}})
    found = by_rule(run(ds), "long_tail")
    assert [(f.payload["split"], f.severity) for f in found] == [("train", "WARN")]


def test_single_class_split_has_ratio_one_and_no_finding() -> None:
    assert by_rule(run(make({"train": {1: 50}})), "long_tail") == []


def test_class_missing_from_a_split_is_a_warning_naming_the_ids() -> None:
    ds = make({"train": {1: 5, 2: 5, 3: 5}, "val": {1: 4, 2: 4}, "test": {1: 1}})
    found = by_rule(run(ds), "class_missing_in_split")
    assert [(f.payload["split"], f.payload["class_ids"], f.severity) for f in found] == [
        ("val", [3], "WARN"),
        ("test", [2, 3], "WARN"),
    ]


def test_unassigned_images_do_not_count_as_a_split_for_missing_classes() -> None:
    ds = make({"train": {1: 5, 2: 5, 3: 5}, "unassigned": {1: 1}})
    assert by_rule(run(ds), "class_missing_in_split") == []


def test_class_with_no_annotations_anywhere_is_info() -> None:
    ds = make({"train": {1: 5, 2: 5}})
    (f,) = by_rule(run(ds), "class_unused")
    assert f.severity == "INFO" and f.payload["class_ids"] == [3] and f.payload["count"] == 1


def test_no_annotations_does_not_crash() -> None:
    ds = make({"train": {}})
    assert [f.payload["rule"] for f in run(ds)] == ["class_unused"]
    assert run(ds)[0].payload["count"] == 3


def test_classification_datasets_are_supported() -> None:
    ds = make({"train": {1: 100, 2: 2}}, task="classification")
    assert [f.severity for f in by_rule(run(ds), "long_tail")] == ["WARN"]


def test_markup_in_category_names_stays_verbatim_in_data() -> None:
    name = '<script>alert("x")</script>'
    cats = [Category(id=1, name=name), Category(id=2, name="b")]
    (f,) = by_rule(run(make({"train": {1: 30, 2: 1}}, categories=cats)), "long_tail")
    assert f.payload["head"][0]["name"] == name  # type: ignore[index]


def test_many_classes_stay_within_the_payload_limit_and_run_fast() -> None:
    n = 3000
    cats = [Category(id=i + 1, name=f"c{i}") for i in range(n)]
    ds = make({"train": {i + 1: 1 + (i % 7) * 3 for i in range(n)}}, categories=cats)
    start = time.perf_counter()
    findings = run(ds)
    assert time.perf_counter() - start < 3.0
    assert all(len(f.model_dump_json()) < 65_536 for f in findings)
    (cls,) = by_rule(findings, "long_tail")
    assert len(cls.payload["head"]) == len(cls.payload["tail"]) == imbalance.TOP_K  # type: ignore[arg-type]


def test_output_is_deterministic() -> None:
    ds = make({"train": {1: 100, 2: 10, 3: 5}, "val": {1: 1}})
    assert run(ds) == run(ds)
