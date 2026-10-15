"""near_duplicates, leakage and outliers: planted structure in, deterministic findings out.

Each check also has to fail closed: images without embeddings, an embedder error, an incomplete
run, a missing embedder and a distrusted cache are explicit findings, never silence.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from fixtures.embedfakes import near, random_unit
from fixtures.embedscene import Row, scene

from datacopilot.audit import embedding, registry
from datacopilot.audit.checks import leakage, near_duplicates, outliers
from datacopilot.audit.context import CheckConfig
from datacopilot.core.schema import Dataset, Finding
from datacopilot.embed import knn

CHECKS = {"near_duplicates": near_duplicates, "leakage": leakage, "outliers": outliers}
BASE = random_unit(b"base-scene")
TWIN = near(BASE, 0.1)  # cosine about 0.995 to BASE


def rules(findings: list[Finding]) -> list[str]:
    return [str(f.payload["rule"]) for f in findings]


def of(findings: list[Finding], rule: str) -> list[Finding]:
    return [f for f in findings if f.payload["rule"] == rule]


def clean_rows(n: int = 6, split: str = "train") -> list[Row]:
    return [Row(f"{split}{k}.png", split) for k in range(n)]


# ------------------------------------------------------------ registration
@pytest.mark.parametrize("name", sorted(CHECKS))
def test_each_check_is_registered_as_a_check_plugin(name: str) -> None:
    assert name in registry.check_names()
    plugin = registry.get_check(name)
    assert plugin.spec.kind == "check" and plugin.spec.name == name
    assert plugin.spec.license == "Apache-2.0"


def test_the_checks_run_in_the_documented_order() -> None:
    names = registry.check_names()
    assert names.index("duplicates") < names.index("near_duplicates") < names.index("leakage")
    assert names.index("boxes") < names.index("outliers")


# ------------------------------------------------------------ near_duplicates
def test_a_planted_near_duplicate_pair_is_one_warn_cluster_with_names_and_short_ids(
    tmp_path: Path,
) -> None:
    rows = [Row("a.png", vec=BASE), Row("b.png", vec=TWIN), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = near_duplicates.plugin.run(ds, ctx)
    assert f.severity == "WARN" and f.check == "near_duplicates"
    p = f.payload
    assert p["rule"] == "near_duplicate" and p["count"] == 2 and p["cross_split"] is False
    assert (
        len(str(p["group_id"])) == 12
        and 0.95 <= float(p["min_cosine"]) <= float(p["max_cosine"]) <= 1.0
    )  # type: ignore[arg-type]
    names = sorted(m["file_name"] for m in p["members"])  # type: ignore[union-attr,index]
    assert names == ["a.png", "b.png"]
    assert str(tmp_path) not in repr(p) and "/" not in repr(p["members"])  # base names only
    assert f.image_id == min(m["image_id"] for m in p["members"])  # type: ignore[union-attr,index]


def test_a_cross_split_pair_is_flagged_here_as_warn_and_as_critical_by_leakage(
    tmp_path: Path,
) -> None:
    rows = [Row("a.png", "train", BASE), Row("b.png", "test", TWIN), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = near_duplicates.plugin.run(ds, ctx)
    assert f.severity == "WARN" and f.payload["cross_split"] is True
    assert f.payload["splits"] == {"test": 1, "train": 1}
    (g,) = leakage.plugin.run(ds, ctx)
    assert g.severity == "CRITICAL"


def test_a_chain_of_pairs_merges_into_one_cluster(tmp_path: Path) -> None:
    a = BASE
    b = near(a, 0.28, seed=1)
    c = near(b, 0.28, seed=2)
    assert float(a @ c) < 0.95 <= min(float(a @ b), float(b @ c))
    rows = [Row("a.png", vec=a), Row("b.png", vec=b), Row("c.png", vec=c), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = near_duplicates.plugin.run(ds, ctx)
    assert f.payload["count"] == 3


def test_pairs_below_the_threshold_are_not_reported_and_the_threshold_is_configurable(
    tmp_path: Path,
) -> None:
    loose = near(BASE, 0.5)
    sim = float(BASE @ loose)
    assert 0.85 < sim < 0.95
    rows = [Row("a.png", vec=BASE), Row("b.png", vec=loose), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    assert near_duplicates.plugin.run(ds, ctx) == []
    relaxed = ctx.model_copy(update={"config": CheckConfig(near_duplicate_cosine=0.85)})
    (f,) = near_duplicates.plugin.run(ds, relaxed)
    assert f.payload["count"] == 2


def test_clean_data_has_no_findings(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, clean_rows(20))
    assert near_duplicates.plugin.run(ds, ctx) == []
    assert leakage.plugin.run(ds, ctx) == []


def test_a_large_cluster_is_one_bounded_finding(tmp_path: Path) -> None:
    rows = [Row(f"d{k}.png", vec=near(BASE, 0.02, seed=k)) for k in range(30)] + clean_rows()
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = near_duplicates.plugin.run(ds, ctx)
    assert f.payload["count"] == 30 and len(f.payload["members"]) == 20  # type: ignore[arg-type]


def test_findings_do_not_depend_on_the_order_of_the_images(tmp_path: Path) -> None:
    rows = [Row("a.png", "train", BASE), Row("b.png", "test", TWIN), *clean_rows(10)]
    ds, _, ctx = scene(tmp_path, rows)
    shuffled = list(ds.images)
    random.Random(4).shuffle(shuffled)
    other = Dataset(ds.manifest, tuple(shuffled), ds.annotations)
    for check in (near_duplicates, leakage, outliers):
        assert check.plugin.run(ds, ctx) == check.plugin.run(other, ctx)


def test_repeating_a_run_gives_identical_findings(tmp_path: Path) -> None:
    rows = [Row("a.png", vec=BASE), Row("b.png", vec=TWIN), *clean_rows(10)]
    ds, _, ctx = scene(tmp_path, rows)
    assert near_duplicates.plugin.run(ds, ctx) == near_duplicates.plugin.run(ds, ctx)


def test_a_single_image_dataset_has_nothing_to_compare(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, [Row("a.png")])
    assert near_duplicates.plugin.run(ds, ctx) == []
    assert leakage.plugin.run(ds, ctx) == []
    assert outliers.plugin.run(ds, ctx) == []


# ------------------------------------------------------------ leakage
def test_a_within_split_near_duplicate_is_not_leakage(tmp_path: Path) -> None:
    rows = [
        Row("a.png", "train", BASE),
        Row("b.png", "train", TWIN),
        *clean_rows(),
        *clean_rows(3, "test"),
    ]
    ds, _, ctx = scene(tmp_path, rows)
    assert leakage.plugin.run(ds, ctx) == []


def test_cross_split_near_duplicates_are_critical_with_splits_and_names(tmp_path: Path) -> None:
    rows = [
        Row("train_a.png", "train", BASE),
        Row("val_a.png", "val", TWIN),
        Row("test_a.png", "test", near(BASE, 0.1, seed=3)),
        *clean_rows(5),
    ]
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = leakage.plugin.run(ds, ctx)
    assert f.severity == "CRITICAL" and f.payload["rule"] == "cross_split_near_duplicate"
    assert f.payload["splits"] == {"test": 1, "train": 1, "val": 1} and f.payload["count"] == 3
    assert sorted(m["file_name"] for m in f.payload["members"]) == [  # type: ignore[union-attr,index]
        "test_a.png",
        "train_a.png",
        "val_a.png",
    ]
    assert f.payload["pairs"] >= 2  # type: ignore[operator]


def test_unassigned_next_to_a_real_split_is_leakage_but_not_among_themselves(
    tmp_path: Path,
) -> None:
    rows = [Row("a.png", "train", BASE), Row("b.png", "unassigned", TWIN), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    assert len(leakage.plugin.run(ds, ctx)) == 1
    rows = [Row("a.png", "unassigned", BASE), Row("b.png", "unassigned", TWIN), *clean_rows()]
    (tmp_path / "x").mkdir()
    ds, _, ctx = scene(tmp_path / "x", rows)
    assert leakage.plugin.run(ds, ctx) == []


def test_leakage_is_found_even_when_more_than_k_same_split_images_are_closer(
    tmp_path: Path,
) -> None:
    crowd = [Row(f"t{k}.png", "train", near(BASE, 0.01, seed=k)) for k in range(15)]
    far_twin = Row("test.png", "test", near(BASE, 0.12, seed=77))
    ds, _, ctx = scene(tmp_path, [*crowd, far_twin, *clean_rows(), *clean_rows(2, "test")])
    ctx = ctx.model_copy(update={"config": CheckConfig(knn_k=3)})
    (f,) = leakage.plugin.run(ds, ctx)
    assert f.payload["splits"]["test"] == 1  # type: ignore[index]


def test_a_group_spanning_two_splits_is_critical_leakage_without_any_similarity(
    tmp_path: Path,
) -> None:
    rows = [
        Row("a.png", "train", group="cam1_day2"),
        Row("b.png", "test", group="cam1_day2"),
        Row("c.png", "train", group="cam2"),
        Row("d.png", "train", group="cam2"),
        *clean_rows(),
    ]
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = of(leakage.plugin.run(ds, ctx), "group_overlap")
    assert (
        f.severity == "CRITICAL" and f.payload["group"] == "cam1_day2" and f.payload["count"] == 2
    )
    assert len(str(f.payload["group_ref"])) == 12 and f.payload["splits"] == {"test": 1, "train": 1}


def test_group_overlap_is_reported_even_when_the_embedder_fails(tmp_path: Path) -> None:
    rows = [Row("a.png", "train", group="g"), Row("b.png", "test", group="g"), *clean_rows()]
    ds, emb, ctx = scene(tmp_path, rows)
    emb.fail_after = 0
    found = leakage.plugin.run(ds, ctx)
    assert set(rules(found)) >= {"group_overlap", "embedder_error"}
    assert any(f.severity == "CRITICAL" for f in of(found, "embedder_error"))


def test_group_ids_are_cut_and_unicode_safe(tmp_path: Path) -> None:
    gid = "g‮" + "x" * 400
    rows = [Row("a.png", "train", group=gid), Row("b.png", "val", group=gid), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    (f,) = of(leakage.plugin.run(ds, ctx), "group_overlap")
    assert len(str(f.payload["group"])) <= 64 and "‮" not in str(f.payload["group"])


def test_a_single_split_has_no_cross_split_pairs(tmp_path: Path) -> None:
    rows = [Row("a.png", "train", BASE), Row("b.png", "train", TWIN), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    assert leakage.plugin.run(ds, ctx) == []


def test_the_leakage_threshold_is_configurable(tmp_path: Path) -> None:
    loose = near(BASE, 0.5)
    rows = [Row("a.png", "train", BASE), Row("b.png", "test", loose), *clean_rows()]
    ds, _, ctx = scene(tmp_path, rows)
    assert leakage.plugin.run(ds, ctx) == []
    strict = ctx.model_copy(update={"config": CheckConfig(leakage_cosine=0.8)})
    assert len(leakage.plugin.run(ds, strict)) == 1


# ------------------------------------------------------------ outliers
def class_rows(n: int = 14) -> list[Row]:
    centre = random_unit(b"class-a")
    return [Row(f"a{k}.png", vec=near(centre, 0.08, seed=k), cat=0) for k in range(n)]


def test_an_image_far_from_its_class_is_an_info_outlier(tmp_path: Path) -> None:
    rows = [*class_rows(), Row("odd.png", vec=random_unit(b"elsewhere"), cat=0)]
    ds, _, ctx = scene(tmp_path, rows)
    found = outliers.plugin.run(ds, ctx)
    (f,) = of(found, "outlier")
    assert f.severity == "INFO" and f.check == "outliers"
    assert f.payload["category_id"] == 1 and f.payload["z"] > 3.5  # type: ignore[operator]
    names = {i.image_id: i.file_path.rsplit("/", 1)[-1] for i in ds.images}
    assert names[f.image_id or ""] == "odd.png"
    assert set(f.payload) >= {"rule", "category_id", "category", "score", "class_median"}


def test_a_tight_class_and_identical_vectors_report_nothing(tmp_path: Path) -> None:
    same = random_unit(b"same")
    ds, _, ctx = scene(tmp_path, [Row(f"s{k}.png", vec=same, cat=0) for k in range(12)])
    assert of(outliers.plugin.run(ds, ctx), "outlier") == []


def test_small_classes_and_unannotated_images_are_counted_not_dropped_silently(
    tmp_path: Path,
) -> None:
    rows = [*class_rows(10), Row("s1.png", cat=1), Row("s2.png", cat=1), Row("n.png", cat=None)]
    ds, _, ctx = scene(tmp_path, rows)
    found = outliers.plugin.run(ds, ctx)
    (skipped,) = of(found, "classes_skipped")
    assert skipped.severity == "INFO" and skipped.payload["count"] == 1
    (none,) = of(found, "images_without_class")
    assert none.payload["count"] == 1


def test_an_image_takes_the_class_of_its_most_frequent_box(tmp_path: Path) -> None:
    centre = random_unit(b"class-a")
    rows = [Row(f"a{k}.png", vec=near(centre, 0.08, seed=k), cats=(0, 0, 1)) for k in range(10)]
    ds, _, ctx = scene(tmp_path, rows)
    found = outliers.plugin.run(ds, ctx)
    assert of(found, "classes_skipped") == []  # one class of ten, not two small classes


def test_outliers_are_capped_per_class_highest_score_first(tmp_path: Path) -> None:
    rows = [*class_rows(30)] + [
        Row(f"o{k}.png", vec=random_unit(f"far{k}".encode()), cat=0) for k in range(6)
    ]
    ds, _, ctx = scene(tmp_path, rows)
    cfg = CheckConfig(max_outliers_per_class=3)
    found = of(outliers.plugin.run(ds, ctx.model_copy(update={"config": cfg})), "outlier")
    assert len(found) == 3
    scores = [float(f.payload["score"]) for f in sorted(found, key=lambda f: f.image_id or "")]  # type: ignore[arg-type]
    assert all(s > 0.5 for s in scores)


# ------------------------------------------------------------ fail closed
@pytest.mark.parametrize("name", sorted(CHECKS))
def test_without_an_embedding_provider_every_embedding_check_is_a_critical_finding(
    tmp_path: Path, name: str
) -> None:
    ds, _, ctx = scene(tmp_path, clean_rows())
    bare = ctx.model_copy(update={"embeddings": None, "embedder": None})
    (f,) = CHECKS[name].plugin.run(ds, bare)
    assert f.severity == "CRITICAL" and f.payload["rule"] == "embedder_missing"


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_an_embedder_error_is_critical_and_unembedded_images_are_counted(
    tmp_path: Path, name: str
) -> None:
    ds, emb, ctx = scene(tmp_path, clean_rows(8))
    emb.fail_after = 0
    emb.error = RuntimeError("token=hf_SECRET")
    found = CHECKS[name].plugin.run(ds, ctx)
    (err,) = of(found, "embedder_error")
    assert err.severity == "CRITICAL" and err.payload["error_type"] == "RuntimeError"
    assert "hf_SECRET" not in repr(found)
    (miss,) = of(found, "images_not_embedded")
    assert miss.payload["count"] == 8 and miss.payload["reasons"] == {"embedder_error": 8}
    assert miss.severity == "CRITICAL"  # every image is unchecked: above the 50 percent line


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_unembedded_images_are_at_least_warn_and_critical_above_half(
    tmp_path: Path, name: str
) -> None:
    rows = [*clean_rows(9), Row("gone.png")]
    ds, _, ctx = scene(tmp_path, rows)
    (tmp_path / "images" / "gone.png").unlink()
    (miss,) = of(CHECKS[name].plugin.run(ds, ctx), "images_not_embedded")
    assert miss.severity == "WARN" and miss.payload["reasons"] == {"file_missing": 1}
    strict = ctx.model_copy(update={"require_local_images": True})
    (miss,) = of(CHECKS[name].plugin.run(ds, strict), "images_not_embedded")
    assert miss.severity == "CRITICAL"


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_an_embedding_time_budget_is_an_audit_incomplete_finding(tmp_path: Path, name: str) -> None:
    import time

    ds, emb, ctx = scene(tmp_path, clean_rows(20), batch_size=2, workers=1)
    real = emb.embed

    def slow(images: Any) -> Any:
        out = real(images)
        time.sleep(0.05)
        return out

    emb.embed = slow  # type: ignore[method-assign]
    ctx = ctx.model_copy(update={"config": CheckConfig(max_check_seconds=0.08)})
    found = CHECKS[name].plugin.run(ds, ctx)
    (inc,) = of(found, "audit_incomplete")
    assert inc.severity == "CRITICAL" and inc.payload["reason"] == "time_budget"
    warn = CHECKS[name].plugin.run(ds, ctx.model_copy(update={"incomplete_is_critical": False}))
    assert of(warn, "audit_incomplete")[0].severity == "WARN"


def test_the_knn_time_budget_and_the_size_cap_are_audit_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [Row("a.png", vec=BASE), Row("b.png", vec=TWIN), *clean_rows(30)]
    ds, _, ctx = scene(tmp_path, rows)
    real = knn.iter_knn

    def one_block(*a: Any, **kw: Any) -> Any:
        for i, block in enumerate(real(*a, **{**kw, "max_block_bytes": 8 * 32 * 4})):
            if i == 1:
                return
            yield block

    monkeypatch.setattr(knn, "iter_knn", one_block)
    found = near_duplicates.plugin.run(ds, ctx)
    (inc,) = of(found, "audit_incomplete")
    assert (
        inc.payload["reason"] == "time_budget" and 0 < inc.payload["checked"] < inc.payload["total"]
    )  # type: ignore[operator]
    monkeypatch.undo()
    monkeypatch.setattr(knn, "MAX_KNN_ROWS", 5)
    found = near_duplicates.plugin.run(ds, ctx)
    (inc,) = of(found, "audit_incomplete")
    assert inc.payload["reason"] == "too_many_images" and of(found, "near_duplicate") == []


def test_a_rejected_cache_is_a_warn_finding_on_every_embedding_check(tmp_path: Path) -> None:
    ds, emb, ctx = scene(tmp_path, clean_rows(12))
    near_duplicates.plugin.run(ds, ctx)  # fills the cache
    shard = next((tmp_path / "ds.dcp" / "embeddings").rglob("shard-*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-10])  # a partial shard
    fresh = ctx.model_copy(update={"embeddings": embedding.PipelineProvider(emb, batch_size=8)})
    for check in CHECKS.values():
        (f,) = of(check.plugin.run(ds, fresh), "embedding_cache_rejected")
        assert f.severity == "WARN" and f.payload["reasons"]


def test_findings_never_contain_directories(tmp_path: Path) -> None:
    rows = [
        Row("a.png", "train", BASE, group="g"),
        Row("b.png", "test", TWIN, group="g"),
        *clean_rows(),
    ]
    ds, _, ctx = scene(tmp_path, rows)
    for check in CHECKS.values():
        text = repr([f.model_dump() for f in check.plugin.run(ds, ctx)])
        assert str(tmp_path) not in text and "images/" not in text
