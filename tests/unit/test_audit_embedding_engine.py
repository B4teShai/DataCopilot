"""The embedding checks through the engine: selection, gating and fail-closed verdicts."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures.embedfakes import near, random_unit
from fixtures.embedscene import Row, scene

from datacopilot.audit import engine, registry
from datacopilot.audit.context import CheckContext

BASE = random_unit(b"engine-base")
TWIN = near(BASE, 0.1)


def clean(n: int = 8, split: str = "train") -> list[Row]:
    return [Row(f"{split}{k}.png", split) for k in range(n)]


def leaking() -> list[Row]:
    return [Row("a.png", "train", BASE), Row("b.png", "test", TWIN), *clean(), *clean(3, "val")]


def bare(ctx: CheckContext) -> CheckContext:
    return ctx.model_copy(update={"embeddings": None, "embedder": None})


def test_all_includes_the_embedding_checks_only_with_an_embedder() -> None:
    without = engine.select_checks("all")
    assert without == ["integrity", "duplicates", "imbalance", "boxes"]
    with_embedder = engine.select_checks("all", with_embedder=True)
    assert with_embedder == [
        "integrity",
        "duplicates",
        "near_duplicates",
        "leakage",
        "imbalance",
        "boxes",
        "label_noise",
        "outliers",
    ]


@pytest.mark.parametrize("name", sorted(registry.EMBEDDING_CHECKS))
def test_naming_an_embedding_check_without_an_embedder_is_a_usage_error(name: str) -> None:
    with pytest.raises(ValueError, match="needs an embedder"):
        engine.select_checks([name])
    assert engine.select_checks([name], with_embedder=True) == [name]


def test_fail_on_leakage_is_valid_and_gates_only_with_an_embedder(tmp_path: Path) -> None:
    assert engine.parse_fail_on(["leakage"]) == ("leakage",)
    ds, _, ctx = scene(tmp_path, clean())
    with pytest.raises(ValueError, match=r"leakage.*not among the checks.*--embedder"):
        engine.run_audit(ds, "all", bare(ctx), ["leakage"])


def test_a_planted_leak_fails_the_leakage_gate_and_clean_data_passes_it(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, leaking())
    run = engine.run_audit(ds, "all", ctx, ["leakage"])
    assert run.passed is False and run.embedder == "fake"
    assert {"near_duplicates", "leakage", "outliers"} <= set(run.checks)
    leaks = [f for f in run.findings if f.check == "leakage"]
    assert [f.payload["rule"] for f in leaks] == ["cross_split_near_duplicate"]
    (tmp_path / "ok").mkdir()
    ds2, _, ctx2 = scene(tmp_path / "ok", [*clean(), *clean(3, "val"), *clean(3, "test")])
    assert engine.run_audit(ds2, "all", ctx2, ["leakage"]).passed is True


def test_the_default_critical_gate_also_catches_the_leak(tmp_path: Path) -> None:
    ds, _, ctx = scene(tmp_path, leaking())
    assert engine.run_audit(ds, "all", ctx, ["critical"]).passed is False


def test_an_embedder_failure_fails_every_gate_that_looks_at_critical_or_the_check(
    tmp_path: Path,
) -> None:
    ds, emb, ctx = scene(tmp_path, leaking())
    emb.fail_after = 0
    for gate in (["critical"], ["leakage"], ["near_duplicates"], ["warn"]):
        run = engine.run_audit(ds, ["leakage", "near_duplicates"], ctx, gate)
        assert run.passed is False, gate
        assert any(f.payload["rule"] == "embedder_error" for f in run.findings)


def test_a_partially_embedded_dataset_cannot_pass_a_check_name_gate(tmp_path: Path) -> None:
    rows = [*clean(9), Row("gone.png")]
    ds, _, ctx = scene(tmp_path, rows)
    (tmp_path / "images" / "gone.png").unlink()
    assert engine.run_audit(ds, ["leakage"], ctx, ["leakage"]).passed is False
    assert engine.run_audit(ds, ["leakage"], ctx, ["warn"]).passed is False
    # an attacker who makes one leaked image unembeddable must not pass the DEFAULT gate
    assert engine.run_audit(ds, ["leakage"], ctx, ["critical"]).passed is False
    run = engine.run_audit(ds, ["leakage"], ctx, [])  # no gate: reported, WARN, exit would be 0
    assert run.passed is True
    assert {f.severity for f in run.findings if f.payload["rule"] == "images_not_embedded"} == {
        "WARN"
    }


def test_without_a_gate_an_embedder_failure_is_still_in_the_report_but_does_not_fail(
    tmp_path: Path,
) -> None:
    ds, emb, ctx = scene(tmp_path, leaking())
    emb.fail_after = 0
    run = engine.run_audit(ds, ["leakage"], ctx, [])
    assert run.passed is True and run.counts.critical >= 1


def test_repeated_runs_give_identical_findings_in_identical_order(
    tmp_path: Path,
) -> None:
    ds, _, ctx = scene(tmp_path, leaking())
    a = engine.run_audit(ds, "all", ctx, ["leakage"])
    b = engine.run_audit(ds, "all", ctx, ["leakage"])
    assert [f.finding_id for f in a.findings] == [f.finding_id for f in b.findings]
    assert [f.model_dump() for f in a.findings] == [f.model_dump() for f in b.findings]
