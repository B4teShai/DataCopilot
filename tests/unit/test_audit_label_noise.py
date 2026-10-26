"""label_noise: a box whose class no neighbouring image shares, from the existing embeddings.

Planted structure in, deterministic findings out. The check also has to fail closed: a missing
embedder, an embedder error, a dataset that is too small and annotations that could not be scored
are explicit findings, never a silent pass.
"""

from __future__ import annotations

import json
import random
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.embedfakes import near, random_unit
from fixtures.embedscene import Row
from fixtures.embedscene import scene as _scene
from sarif_shape import validate_sarif

from datacopilot.audit import engine, registry
from datacopilot.audit.checks import embedtools, label_noise
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
    """The image rule with these settings (the crop rule has its own tests)."""
    return ctx.model_copy(update={"config": CheckConfig(**{"suspect_basis": "image", **kw})})


def scene(*args: Any, **kw: Any) -> Any:
    """The planted scene, checked with the whole-image rule (its vectors are per image)."""
    ds, emb, ctx = _scene(*args, **kw)
    return ds, emb, cfg(ctx)


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
    assert n.severity == "WARN" and n.payload["count"] == 1 and n.payload["min_similarity"] == 0.9
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
    assert skip.payload["reason"] == "no_annotations" and skip.severity == "INFO"


def test_a_single_class_dataset_is_a_skipped_finding(tmp_path: Path) -> None:
    rows = [Row(f"a{k}.png", vec=near(CENTRE_A, 0.08, seed=k), cat=0) for k in range(40)]
    ds, _, ctx = scene(tmp_path, rows)
    (skip,) = label_noise.plugin.run(ds, ctx)
    assert skip.payload["reason"] == "single_class" and skip.severity == "INFO"


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


def test_the_real_time_budget_ends_the_search_and_says_so(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    label_noise.plugin.run(ds, ctx)  # embeds once; the provider keeps the result in memory
    tight = cfg(ctx, max_check_seconds=1e-9)
    found = label_noise.plugin.run(ds, tight)
    (inc,) = of(found, "audit_incomplete")
    assert inc.severity == "CRITICAL" and inc.payload["reason"] == "time_budget"
    assert inc.payload["checked"] == 0 and inc.payload["total"] == 32
    soft = tight.model_copy(update={"incomplete_is_critical": False})
    assert of(label_noise.plugin.run(ds, soft), "audit_incomplete")[0].severity == "WARN"


def test_the_budget_is_checked_for_every_row_not_only_between_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    label_noise.plugin.run(ds, ctx)
    ticks, start = iter(range(10_000)), time.monotonic()
    fake = types.SimpleNamespace(monotonic=lambda: start + next(ticks))
    monkeypatch.setattr(label_noise, "time", fake)
    found = label_noise.plugin.run(
        ds, cfg(ctx, max_check_seconds=5.5)
    )  # the whole matrix is one block
    (inc,) = of(found, "audit_incomplete")
    assert inc.payload["checked"] == 5 and inc.payload["total"] == 32


def test_a_dataset_the_embedding_run_refuses_is_skipped_for_that_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    monkeypatch.setattr(knn, "MAX_KNN_ROWS", 5)
    found = label_noise.plugin.run(ds, ctx)
    assert of(found, "audit_incomplete")[0].payload["reason"] == "too_many_images"
    (skip,) = of(found, "skipped")
    assert skip.payload["reason"] == "too_many_images"
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
        "classes_truncated",
        "embedder_missing",
        "images_not_embedded",
        "crop_rule_unavailable",
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
    assert "Annotation class disagrees with its nearest look-alikes" in html
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    validate_sarif(json.loads((out / "r.sarif").read_text()))
    doc = json.loads((out / "r.json").read_text())
    assert [f["payload"]["rule"] for f in doc["findings"]] == ["suspect_class"]
    for name in ("r.html", "r.json", "r.sarif"):
        assert str(tmp_path) not in (out / name).read_text()


# ------------------------------------------------------------ similarity floor, neighbour count
def basis(i: int) -> np.ndarray:
    v = np.zeros(16, dtype=np.float32)
    v[i] = 1.0
    return v


def group(prefix: str, centre: int, noise_axes: list[int], cosine: float, cat: int) -> list[Row]:
    """Images whose pairwise cosine is exactly ``cosine`` (orthogonal noise around one centre)."""
    return [
        Row(
            f"{prefix}{k}.png",
            vec=(np.sqrt(cosine) * basis(centre) + np.sqrt(1 - cosine) * basis(axis)).astype(
                np.float32
            ),
            cat=cat,
        )
        for k, axis in enumerate(noise_axes)
    ]


SMALL = dict(suspect_min_images=5, suspect_k=4)


@pytest.mark.parametrize(("cosine", "scored"), [(0.2, False), (0.4, True)])
def test_the_default_similarity_floor_is_point_three(
    tmp_path: Path, cosine: float, scored: bool
) -> None:
    rows = group("a", 0, [1, 2, 3, 4, 5], cosine, 0) + group("b", 6, [7, 8, 9, 10, 11], cosine, 1)
    rows[0] = Row("a0.png", vec=rows[0].vec, cat=1)  # one flip in group A
    ds, _, ctx = scene(tmp_path, rows)
    found = label_noise.plugin.run(ds, cfg(ctx, **SMALL))
    assert bool(of(found, "suspect_class")) is scored
    assert bool(of(found, "annotations_not_scored")) is (not scored)
    loose = label_noise.plugin.run(ds, cfg(ctx, suspect_min_similarity=0.1, **SMALL))
    assert of(loose, "suspect_class")


def test_exactly_the_minimum_number_of_neighbours_is_enough_and_one_less_is_not(
    tmp_path: Path,
) -> None:
    rows = group("a", 0, [1, 2, 3, 4], 0.5, 0) + group("b", 6, [7, 8, 9], 0.5, 1)
    rows[0] = Row("a0.png", vec=rows[0].vec, cat=1)  # group A: 3 neighbours
    rows[4] = Row("b0.png", vec=rows[4].vec, cat=0)  # group B: 2 neighbours
    ds, _, ctx = scene(tmp_path, rows)
    base = cfg(ctx, suspect_min_images=5, suspect_k=10)
    found = label_noise.plugin.run(ds, base)
    flagged = {names(ds)[f.image_id or ""] for f in of(found, "suspect_class")}
    assert flagged == {"a0.png"}
    (n,) = of(found, "annotations_not_scored")
    assert n.payload["count"] == 3  # the three boxes of group B
    two = label_noise.plugin.run(ds, cfg(ctx, suspect_min_neighbours=2, suspect_min_images=5))
    assert {names(ds)[f.image_id or ""] for f in of(two, "suspect_class")} == {"a0.png", "b0.png"}
    assert of(two, "annotations_not_scored") == []


def test_a_floor_of_one_keeps_images_identical_to_float_precision_as_neighbours(
    tmp_path: Path,
) -> None:
    same = random_unit(b"twins")
    rows = [Row(f"s{k}.png", vec=near(same, 0.0007, seed=k), cat=0) for k in range(6)]
    rows += [Row(f"t{k}.png", vec=random_unit(b"other"), cat=1) for k in range(6)]
    rows[0] = Row("s0.png", vec=rows[0].vec, cat=1)
    ds, _, ctx = scene(tmp_path, rows)
    found = label_noise.plugin.run(
        ds, cfg(ctx, suspect_min_similarity=1.0, suspect_min_images=5, suspect_k=5)
    )
    assert {names(ds)[f.image_id or ""] for f in of(found, "suspect_class")} == {"s0.png"}


# ------------------------------------------------------------ ranking and caps keep the best
def graded_scene() -> list[Row]:
    rows = [
        Row(f"a{k}.png", vec=near(CENTRE_A, 0.04 + 0.005 * k, seed=k), cat=0) for k in range(42)
    ]
    rows += [Row(f"b{k}.png", vec=near(CENTRE_B, 0.08, seed=100 + k), cat=1) for k in range(10)]
    for k in range(0, 42, 3):  # a stray class-3 box on every third image of group A
        rows[k] = Row(rows[k].name, vec=rows[k].vec, cats=(0, 2))
    return rows


def rank(f: Finding) -> tuple[float, float, str]:
    return (float(f.payload["support"]), -float(f.payload["mean_similarity"]), f.ann_id or "")  # type: ignore[arg-type]


def test_the_caps_keep_the_best_suspects_and_the_streaming_prune_is_exact(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, graded_scene())
    wide = cfg(ctx, suspect_k=6, suspect_max_support=0.7, max_suspects_per_class=1000)
    everything = sorted(of(label_noise.plugin.run(ds, wide), "suspect_class"), key=rank)
    stray = [f for f in everything if f.payload["category_id"] == 3]
    assert len(stray) > 5 and len({round(rank(f)[0], 3) for f in stray}) > 1
    capped = cfg(ctx, suspect_k=6, suspect_max_support=0.7, max_suspects_per_class=2)
    found = label_noise.plugin.run(ds, capped)  # more than 2 * cap candidates: the prune runs
    kept = sorted(
        (f for f in of(found, "suspect_class") if f.payload["category_id"] == 3), key=rank
    )
    assert [f.ann_id for f in kept] == [f.ann_id for f in stray[:2]]
    dropped = of(found, "suspects_truncated")[0].payload["count"]
    all_kept = len(of(found, "suspect_class"))
    assert dropped == len(everything) - all_kept


def test_the_global_cap_keeps_the_best_and_counts_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, graded_scene())
    wide = cfg(ctx, suspect_k=6, suspect_max_support=0.7, max_suspects_per_class=1000)
    everything = sorted(of(label_noise.plugin.run(ds, wide), "suspect_class"), key=rank)
    monkeypatch.setattr(label_noise, "MAX_SUSPECTS", 3)
    found = label_noise.plugin.run(ds, wide)
    kept = sorted(of(found, "suspect_class"), key=rank)
    assert [f.ann_id for f in kept] == [f.ann_id for f in everything[:3]]
    assert of(found, "suspects_truncated")[0].payload["count"] == len(everything) - 3


def test_payloads_are_built_only_for_the_suspects_that_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, graded_scene())
    calls = []
    real = embedtools.safe_label
    monkeypatch.setattr(embedtools, "safe_label", lambda *a, **k: calls.append(1) or real(*a, **k))
    tight = cfg(
        ctx,
        suspect_k=6,
        suspect_max_support=0.7,
        max_suspects_per_class=1,
        max_suspects_per_image=1,
    )
    found = label_noise.plugin.run(ds, tight)
    survivors = len(of(found, "suspect_class"))
    assert survivors >= 1 and len(calls) <= survivors * (1 + label_noise.TOP_NEIGHBOUR_CLASSES)
    assert of(found, "suspects_truncated")


# ------------------------------------------------------------ hostile and awkward data
def test_one_image_with_thousands_of_classes_cannot_make_every_row_expensive(
    tmp_path: Path,
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    hub = ds.images[0]
    template = ds.annotations[0]
    extra = [
        template.model_copy(
            update={"ann_id": f"hub-{k}", "image_id": hub.image_id, "category_id": 1000 + k}
        )
        for k in range(3000)
    ]
    other = next(i for i in ds.images if i.image_id != hub.image_id)
    extra.append(  # a box of a class only the hub carries, and only beyond the cap
        template.model_copy(
            update={"ann_id": "far", "image_id": other.image_id, "category_id": 3500}
        )
    )
    hostile = Dataset(ds.manifest, ds.images, (*ds.annotations, *extra))
    started = time.perf_counter()
    found = label_noise.plugin.run(
        hostile, cfg(ctx, suspect_k=31)
    )  # the hub is everyone's neighbour
    assert time.perf_counter() - started < 10
    (cut,) = of(found, "classes_truncated")
    assert cut.severity == "INFO" and cut.payload["count"] == 1
    assert cut.payload["max_classes"] == label_noise.MAX_CARRIED_CLASSES
    # classes beyond the cap do not count as carried: the box is a lead although the hub has it
    assert "far" in {f.ann_id for f in of(found, "suspect_class")}


def test_unscored_boxes_gate_like_unexamined_images(tmp_path: Path) -> None:
    rows = group("a", 0, [1, 2, 3, 4, 5], 0.2, 0) + group("b", 6, [7, 8, 9, 10, 11], 0.2, 1)
    ds, _, ctx = scene(tmp_path, rows)
    found = label_noise.plugin.run(ds, cfg(ctx, **SMALL))
    (n,) = of(found, "annotations_not_scored")
    assert n.payload["count"] == n.payload["total"] == 10 and n.severity == "CRITICAL"
    mild = [*clusters(), Row("lone.png", vec=random_unit(b"mild"), cat=2)]
    (tmp_path / "m").mkdir()
    ds2, _, ctx2 = scene(tmp_path / "m", mild)
    assert of(label_noise.plugin.run(ds2, ctx2), "annotations_not_scored")[0].severity == "WARN"
    strict = ctx2.model_copy(update={"require_local_images": True})
    assert (
        of(label_noise.plugin.run(ds2, strict), "annotations_not_scored")[0].severity == "CRITICAL"
    )
    gated = cfg(ctx, **SMALL)
    assert (
        engine.run_audit(ds, ["label_noise"], gated, ["critical"]).passed is False
    )  # the default gate no longer passes a check that scored nothing


def test_findings_do_not_depend_on_the_order_of_images_or_annotations_with_tied_vectors(
    tmp_path: Path,
) -> None:
    twin = near(CENTRE_A, 0.05, seed=1)
    rows = [Row(f"a{k}.png", vec=twin, cat=0) for k in range(10)]
    rows += [Row(f"b{k}.png", vec=near(CENTRE_B, 0.08, seed=k), cat=1) for k in range(14)]
    rows += [Row(f"c{k}.png", vec=twin, cat=0) for k in range(10)]
    rows[3] = Row("a3.png", vec=twin, cat=1)
    ds, _, ctx = scene(tmp_path, rows)
    ctx = cfg(ctx, suspect_k=3)
    first = label_noise.plugin.run(ds, ctx)
    assert [names(ds)[f.image_id or ""] for f in of(first, "suspect_class")] == ["a3.png"]
    for seed in (1, 2, 3):
        images, anns = list(ds.images), list(ds.annotations)
        random.Random(seed).shuffle(images)
        random.Random(seed + 10).shuffle(anns)
        other = Dataset(ds.manifest, tuple(images), tuple(anns))
        assert label_noise.plugin.run(other, ctx) == first


def test_a_model_box_on_a_neighbour_does_not_rescue_a_suspect(tmp_path: Path) -> None:
    rows = clusters()
    rows[3] = Row("a3.png", vec=rows[3].vec, cat=1)
    ds, _, ctx = scene(tmp_path, rows)
    anns = list(ds.annotations)
    neighbour = next(a for a in anns if names(ds)[a.image_id] == "a4.png")
    anns.append(
        neighbour.model_copy(update={"ann_id": "m1", "category_id": 2, "provenance": "model"})
    )
    tainted = Dataset(ds.manifest, ds.images, tuple(anns))
    run = engine.run_audit(tainted, ["label_noise"], ctx)
    assert [
        f.payload["category_id"] for f in run.findings if f.payload["rule"] == "suspect_class"
    ] == [2]
    inc = engine.run_audit(
        tainted, ["label_noise"], ctx.model_copy(update={"include_model_annotations": True})
    )
    assert len([f for f in inc.findings if f.payload["rule"] == "suspect_class"]) <= 1


def test_the_search_does_not_depend_on_whether_faiss_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _, ctx = scene(tmp_path, clusters())
    seen: list[str] = []
    real = knn.iter_knn

    def spy(*a: Any, **kw: Any) -> Any:
        seen.append(kw.get("backend", "auto"))
        return real(*a, **kw)

    monkeypatch.setattr(knn, "iter_knn", spy)
    label_noise.plugin.run(ds, ctx)
    assert seen == ["numpy"]


def test_the_per_image_cap_keeps_the_lowest_support(tmp_path: Path) -> None:
    rows = group("a", 0, [1, 2, 3, 4, 5], 0.5, 0)
    rows[1] = Row("a1.png", vec=rows[1].vec, cats=(0, 1))  # two neighbours carry class 2
    rows[2] = Row("a2.png", vec=rows[2].vec, cats=(0, 1))
    rows[0] = Row(
        "a0.png", vec=rows[0].vec, cats=(0, 1, 2)
    )  # class 2: shared by 2 of 4; class 3: by none
    rows += group("b", 6, [7, 8, 9, 10, 11], 0.5, 1)
    ds, _, ctx = scene(tmp_path, rows)
    one = cfg(
        ctx, suspect_min_images=5, suspect_k=4, suspect_max_support=0.6, max_suspects_per_image=1
    )
    found = label_noise.plugin.run(ds, one)
    mine = [f for f in of(found, "suspect_class") if names(ds)[f.image_id or ""] == "a0.png"]
    assert [(f.payload["category_id"], f.payload["support"]) for f in mine] == [(3, 0.0)]
    assert of(found, "suspects_truncated")[0].payload["count"] >= 1
