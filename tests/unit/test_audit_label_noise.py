"""label_noise: a box whose class no neighbouring image shares, from the existing embeddings.

Planted structure in, deterministic findings out. The check also has to fail closed: a missing
embedder, an embedder error, a dataset that is too small and annotations that could not be scored
are explicit findings, never a silent pass.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.embedfakes import near, random_unit
from fixtures.embedscene import Row, scene
from sarif_shape import validate_sarif

from datacopilot.audit import engine, registry
from datacopilot.audit.checks import label_noise
from datacopilot.audit.context import CheckConfig
from datacopilot.core.schema import Dataset, Finding
from datacopilot.embed import knn
from datacopilot.report import render
from datacopilot.report.rules import describe_rule

CENTRE_A = random_unit(b"noise-a")
CENTRE_B = random_unit(b"noise-b")


def of(findings: list[Finding], rule: str) -> list[Finding]:
    return [f for f in findings if f.payload["rule"] == rule]


def clusters(n: int = 16) -> list[Row]:
    """Two well separated groups of images: class index 0 around A, class index 1 around B."""
    rows = [Row(f"a{k}.png", vec=near(CENTRE_A, 0.08, seed=k), cat=0) for k in range(n)]
    rows += [Row(f"b{k}.png", vec=near(CENTRE_B, 0.08, seed=100 + k), cat=1) for k in range(n)]
    return rows


def names(ds: Dataset) -> dict[str, str]:
    return {i.image_id: i.file_path.rsplit("/", 1)[-1] for i in ds.images}


def cfg(ctx: Any, **kw: Any) -> Any:
    return ctx.model_copy(update={"config": CheckConfig(**kw)})


# ------------------------------------------------------------ registration and config
def test_the_check_is_registered_embedding_based_and_ordered() -> None:
    assert "label_noise" in registry.check_names()
    plugin = registry.get_check("label_noise")
    assert plugin.spec.name == "label_noise" and plugin.spec.license == "Apache-2.0"
    assert plugin.spec.extra is None  # no new dependency
    assert "label_noise" in registry.EMBEDDING_CHECKS
    order = registry.check_names()
    assert order.index("boxes") < order.index("label_noise") < order.index("outliers")


def test_all_with_an_embedder_includes_it_and_without_one_leaves_it_out() -> None:
    assert "label_noise" not in engine.select_checks("all")
    assert "label_noise" in engine.select_checks("all", with_embedder=True)
    with pytest.raises(ValueError, match="needs an embedder"):
        engine.select_checks(["label_noise"])
    assert engine.parse_fail_on(["label_noise"]) == ("label_noise",)


@pytest.mark.parametrize(
    "bad",
    [
        {"suspect_k": 0},
        {"suspect_k": 101},
        {"suspect_min_similarity": 0.0},
        {"suspect_min_similarity": 1.5},
        {"suspect_min_similarity": float("nan")},
        {"suspect_min_neighbours": 0},
        {"suspect_max_support": 1.0},
        {"suspect_max_support": -0.1},
        {"suspect_min_images": 4},
        {"max_suspects_per_image": 0},
        {"max_suspects_per_class": 0},
        {"suspect_k": 3, "suspect_min_neighbours": 4},
    ],
)
def test_thresholds_are_bounded(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CheckConfig(**bad)


# ------------------------------------------------------------ the rule
def test_a_flipped_class_is_one_info_finding_with_evidence(tmp_path: Path) -> None:
    rows = clusters()
    rows[3] = Row("a3.png", vec=rows[3].vec, cat=1)  # a photo of group A labelled with class 2
    ds, _, ctx = scene(tmp_path, rows)
    found = label_noise.plugin.run(ds, ctx)
    (f,) = of(found, "suspect_class")
    assert f.severity == "INFO" and f.check == "label_noise"
    assert names(ds)[f.image_id or ""] == "a3.png" and f.ann_id
    p = f.payload
    assert p["category_id"] == 2 and p["category"] == "dog" and p["support"] == 0.0
    assert p["neighbours"] == 10 and 0.5 < float(p["mean_similarity"]) <= 1.0  # type: ignore[arg-type]
    assert p["file_name"] == "a3.png"
    assert p["neighbour_classes"] == [{"category_id": 1, "category": "cat", "images": 10}]
    assert of(found, "skipped") == []


def test_clean_clusters_report_nothing(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    assert label_noise.plugin.run(ds, ctx) == []


def test_a_class_present_on_any_box_of_a_neighbour_counts(tmp_path: Path) -> None:
    rows = [Row(f"a{k}.png", vec=near(CENTRE_A, 0.08, seed=k), cats=(0, 1)) for k in range(16)]
    rows += [Row(f"b{k}.png", vec=near(CENTRE_B, 0.08, seed=100 + k), cat=2) for k in range(16)]
    ds, _, ctx = scene(tmp_path, rows)
    assert label_noise.plugin.run(ds, ctx) == []
    rows[0] = Row("a0.png", vec=rows[0].vec, cats=(0, 1, 2))  # one box of a class nobody shares
    (tmp_path / "x").mkdir()
    ds, _, ctx = scene(tmp_path / "x", rows)
    (f,) = of(label_noise.plugin.run(ds, ctx), "suspect_class")
    assert f.payload["category_id"] == 3


def test_the_support_threshold_is_configurable(tmp_path: Path) -> None:
    rows = clusters()
    rows[0] = Row("a0.png", vec=rows[0].vec, cat=1)  # flipped
    vectors = np.stack([r.vec for r in rows])  # type: ignore[misc]
    idx, _ = knn.knn(vectors, vectors, 4, same_set=True)
    helper = int(idx[0][0])  # one of the four nearest neighbours of a0 shares class 2: support 0.25
    rows[helper] = Row(rows[helper].name, vec=rows[helper].vec, cats=(0, 1))
    ds, _, ctx = scene(tmp_path, rows)
    base = cfg(ctx, suspect_k=4)
    assert of(label_noise.plugin.run(ds, base), "suspect_class") == []
    lax = cfg(ctx, suspect_k=4, suspect_max_support=0.25)
    (f,) = [
        f
        for f in of(label_noise.plugin.run(ds, lax), "suspect_class")
        if names(ds)[f.image_id or ""] == "a0.png"
    ]
    assert f.payload["support"] == 0.25


def test_neighbours_below_the_similarity_floor_are_not_used_and_are_counted(
    tmp_path: Path,
) -> None:
    rows = clusters()
    rows.append(Row("lone.png", vec=random_unit(b"lone"), cat=2))
    ds, _, ctx = scene(tmp_path, rows)
    found = label_noise.plugin.run(ds, cfg(ctx, suspect_min_similarity=0.9))
    assert of(found, "suspect_class") == []  # the loner has no neighbour close enough
    (n,) = of(found, "annotations_not_scored")
    assert n.severity == "INFO" and n.payload["count"] == 1 and n.payload["min_similarity"] == 0.9
    assert n.payload["min_neighbours"] == 3


def test_unannotated_images_are_not_neighbours(tmp_path: Path) -> None:
    rows = clusters()
    rows += [Row(f"u{k}.png", vec=near(CENTRE_A, 0.05, seed=300 + k), cat=None) for k in range(20)]
    rows[2] = Row("a2.png", vec=rows[2].vec, cat=1)
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = of(label_noise.plugin.run(ds, ctx), "suspect_class")
    assert names(ds)[f.image_id or ""] == "a2.png" and f.payload["neighbours"] == 10


# ------------------------------------------------------------ caps and truncation
def test_per_image_and_per_class_caps_are_reported_not_silent(tmp_path: Path) -> None:
    rows = clusters(24)
    for k in range(6):  # six images of group A carry stray class-2 and class-3 boxes
        rows[k] = Row(f"a{k}.png", vec=rows[k].vec, cats=(0, 1, 2))
    ds, _, ctx = scene(tmp_path, rows)
    open_ctx = cfg(ctx, suspect_k=5, suspect_max_support=0.5)
    full = of(label_noise.plugin.run(ds, open_ctx), "suspect_class")
    assert len(full) >= 6  # the scene really produces many findings
    capped_ctx = cfg(
        ctx,
        suspect_k=5,
        suspect_max_support=0.5,
        max_suspects_per_class=2,
        max_suspects_per_image=1,
    )
    found = label_noise.plugin.run(ds, capped_ctx)
    kept = of(found, "suspect_class")
    per_image: dict[str, int] = {}
    per_class: dict[int, int] = {}
    for f in kept:
        per_image[f.image_id or ""] = per_image.get(f.image_id or "", 0) + 1
        cid = int(f.payload["category_id"])  # type: ignore[call-overload]
        per_class[cid] = per_class.get(cid, 0) + 1
    assert max(per_image.values()) == 1 and max(per_class.values()) <= 2
    assert {f.ann_id for f in kept} <= {f.ann_id for f in full}
    (t,) = of(found, "suspects_truncated")
    assert t.severity == "INFO" and t.payload["count"] == len(full) - len(kept) > 0
    assert of(label_noise.plugin.run(ds, open_ctx), "suspects_truncated") == []


# ------------------------------------------------------------ determinism and bounded memory
def test_findings_do_not_depend_on_image_order_and_repeat_exactly(tmp_path: Path) -> None:
    rows = clusters()
    rows[1] = Row("a1.png", vec=rows[1].vec, cat=1)  # one flip in each group
    rows[21] = Row("b5.png", vec=rows[21].vec, cat=0)
    ds, _, ctx = scene(tmp_path, rows)
    shuffled = list(ds.images)
    random.Random(7).shuffle(shuffled)
    other = Dataset(ds.manifest, tuple(shuffled), ds.annotations)
    first = label_noise.plugin.run(ds, ctx)
    assert len(of(first, "suspect_class")) == 2
    assert first == label_noise.plugin.run(other, ctx) == label_noise.plugin.run(ds, ctx)


def test_the_search_is_chunked_and_the_block_size_does_not_change_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = clusters()
    for k in (2, 6):
        rows[k] = Row(f"a{k}.png", vec=rows[k].vec, cat=1)
    ds, _, ctx = scene(tmp_path, rows)
    whole = label_noise.plugin.run(ds, ctx)
    real = knn.iter_knn
    seen: list[int] = []

    def tiny_blocks(*a: Any, **kw: Any) -> Any:
        for block in real(*a, **{**kw, "max_block_bytes": 8 * 32 * 4}):
            seen.append(len(block.indices))
            yield block

    monkeypatch.setattr(knn, "iter_knn", tiny_blocks)
    assert label_noise.plugin.run(ds, ctx) == whole
    assert len(seen) > 1 and max(seen) < 32  # several blocks, none the whole matrix


# ------------------------------------------------------------ fail closed
def test_without_an_embedding_provider_the_check_is_critical_and_says_it_skipped(
    tmp_path: Path,
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    found = label_noise.plugin.run(
        ds, ctx.model_copy(update={"embeddings": None, "embedder": None})
    )
    (missing,) = of(found, "embedder_missing")
    assert missing.severity == "CRITICAL"
    (skip,) = of(found, "skipped")
    assert skip.severity == "WARN" and skip.payload["reason"] == "embedder_missing"


def test_an_embedder_error_is_critical_and_the_check_says_it_skipped(tmp_path: Path) -> None:
    ds, emb, ctx = scene(tmp_path, clusters())
    emb.fail_after = 0
    emb.error = RuntimeError("token=hf_SECRET")
    found = label_noise.plugin.run(ds, ctx)
    assert of(found, "embedder_error")[0].severity == "CRITICAL"
    (skip,) = of(found, "skipped")
    assert skip.payload["reason"] == "no_embeddings" and "hf_SECRET" not in repr(found)


def test_a_dataset_that_is_too_small_is_a_skipped_finding_not_a_pass(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, clusters(8))  # 16 annotated images, floor is 30
    found = label_noise.plugin.run(ds, ctx)
    (skip,) = found
    assert skip.severity == "WARN" and skip.payload["rule"] == "skipped"
    assert skip.payload == {
        "rule": "skipped",
        "reason": "too_few_images",
        "images": 16,
        "min_images": 30,
    }
    lowered = label_noise.plugin.run(ds, cfg(ctx, suspect_min_images=10))
    assert of(lowered, "skipped") == []


def test_a_dataset_without_annotations_is_a_skipped_finding(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, [Row(f"n{k}.png", cat=None) for k in range(40)])
    (skip,) = label_noise.plugin.run(ds, ctx)
    assert skip.payload["reason"] == "no_annotations"


def test_a_single_class_dataset_is_a_skipped_finding(tmp_path: Path) -> None:
    rows = [Row(f"a{k}.png", vec=near(CENTRE_A, 0.08, seed=k), cat=0) for k in range(40)]
    ds, _, ctx = scene(tmp_path, rows)
    (skip,) = label_noise.plugin.run(ds, ctx)
    assert skip.payload["reason"] == "single_class"


def test_a_partly_embedded_dataset_reports_the_missing_images_and_still_checks(
    tmp_path: Path,
) -> None:
    rows = clusters()
    rows[4] = Row("a4.png", vec=rows[4].vec, cat=1)
    ds, _, ctx = scene(tmp_path, rows)
    (tmp_path / "images" / "b0.png").unlink()
    found = label_noise.plugin.run(ds, ctx)
    assert of(found, "images_not_embedded")[0].payload["count"] == 1
    assert len(of(found, "suspect_class")) == 1


def test_a_time_budget_is_an_audit_incomplete_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    real = knn.iter_knn

    def one_block(*a: Any, **kw: Any) -> Any:
        for i, block in enumerate(real(*a, **{**kw, "max_block_bytes": 8 * 32 * 4})):
            if i == 1:
                return
            yield block

    monkeypatch.setattr(knn, "iter_knn", one_block)
    found = label_noise.plugin.run(ds, ctx)
    (inc,) = of(found, "audit_incomplete")
    assert inc.severity == "CRITICAL" and inc.payload["reason"] == "time_budget"
    assert 0 < inc.payload["checked"] < inc.payload["total"]  # type: ignore[operator]
    soft = label_noise.plugin.run(ds, ctx.model_copy(update={"incomplete_is_critical": False}))
    assert of(soft, "audit_incomplete")[0].severity == "WARN"
    assert time.monotonic() > 0


def test_more_images_than_the_search_supports_is_audit_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    monkeypatch.setattr(knn, "MAX_KNN_ROWS", 5)
    found = label_noise.plugin.run(ds, ctx)
    assert of(found, "audit_incomplete")[0].payload["reason"] == "too_many_images"
    assert of(found, "suspect_class") == []


def test_model_annotations_are_not_ground_truth_and_are_left_out_by_the_engine(
    tmp_path: Path,
) -> None:
    rows = clusters()
    ds, _, ctx = scene(tmp_path, rows)
    anns = list(ds.annotations)
    anns[0] = anns[0].model_copy(update={"category_id": 2, "provenance": "model"})
    tainted = Dataset(ds.manifest, ds.images, tuple(anns))
    run = engine.run_audit(tainted, ["label_noise"], ctx)
    assert [f for f in run.findings if f.payload["rule"] == "suspect_class"] == []
    including = engine.run_audit(
        tainted, ["label_noise"], ctx.model_copy(update={"include_model_annotations": True})
    )
    assert len([f for f in including.findings if f.payload["rule"] == "suspect_class"]) == 1


# ------------------------------------------------------------ reports
def test_every_rule_has_a_human_description() -> None:
    for rule in (
        "suspect_class",
        "skipped",
        "annotations_not_scored",
        "suspects_truncated",
        "embedder_missing",
        "images_not_embedded",
    ):
        assert not describe_rule("label_noise", rule).startswith("label_noise:"), rule


def test_the_finding_renders_escaped_in_all_three_formats_and_gates_by_name(
    tmp_path: Path,
) -> None:
    rows = clusters()
    rows[3] = Row("<b>a3</b>.png", vec=rows[3].vec, cat=1)
    (tmp_path / "data").mkdir()
    ds, _, ctx = scene(tmp_path / "data", rows)
    renamed = Dataset(
        ds.manifest,
        ds.images,
        ds.annotations,
    )
    categories = tuple(
        c.model_copy(update={"name": "<script>alert(1)</script>"}) if c.id == 1 else c
        for c in renamed.categories
    )
    ds = Dataset(
        ds.manifest.model_copy(update={"categories": categories}), ds.images, ds.annotations
    )
    run = engine.run_audit(ds, ["label_noise"], ctx, ["label_noise"])
    assert run.passed is False  # --fail-on label_noise gates on any of its findings
    out = tmp_path / "out"
    out.mkdir()
    render(run, html=out / "r.html", json=out / "r.json", sarif=out / "r.sarif")
    html = (out / "r.html").read_text()
    assert "Annotation class disagrees with its neighbouring images" in html
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    validate_sarif(json.loads((out / "r.sarif").read_text()))
    doc = json.loads((out / "r.json").read_text())
    assert [f["payload"]["rule"] for f in doc["findings"]] == ["suspect_class"]
    for name in ("r.html", "r.json", "r.sarif"):
        assert str(tmp_path) not in (out / name).read_text()
