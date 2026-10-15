"""Check registry, runner, ordering, validation of plugin output, and the --fail-on gate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import structlog
from conftest import dataset_from, make_tables

from datacopilot.audit import engine, registry
from datacopilot.audit.context import CheckContext
from datacopilot.audit.engine import gate_failed, run_audit
from datacopilot.audit.findings import make_finding
from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.schema import Dataset, Finding, PluginSpec, Severity

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"


def spec(name: str, **kw: Any) -> PluginSpec:
    base: dict[str, Any] = {
        "name": name,
        "kind": "check",
        "license": "Apache-2.0",
        "extra": None,
        "gpu_memory_gb": None,
        "version": "1",
    }
    return PluginSpec(**{**base, **kw})


Produce = Callable[[Dataset, CheckContext], Sequence[Finding]]


class FakeCheck:
    def __init__(self, name: str, produce: Produce | None = None) -> None:
        self.spec = spec(name)
        self._produce = produce or (lambda ds, ctx: [])

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        return list(self._produce(ds, ctx))


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    obj: object = None
    loads: list[str] = field(default_factory=list)

    def load(self) -> object:
        self.loads.append(self.name)
        return self.obj


def install(monkeypatch: pytest.MonkeyPatch, *checks: FakeCheck) -> list[FakeEntryPoint]:
    eps = [FakeEntryPoint(c.spec.name, f"m:{c.spec.name}", c) for c in checks]
    monkeypatch.setattr(registry, "_entry_points", lambda: list(eps))
    return eps


def ctx(tmp_path: Path, **kw: Any) -> CheckContext:
    return CheckContext(run_id=RUN, root=tmp_path, cache_dir=tmp_path / "cache", **kw)


@pytest.fixture
def ds() -> Dataset:
    images, anns, cats = make_tables()
    return dataset_from(images, anns, cats)


def emit(
    severity: Severity, check: str, image: int | None = 0, rule: str = "r", **extra: Any
) -> Finding:
    image_id = None if image is None else f"{image:016x}"
    return make_finding(RUN, check, severity, image_id=image_id, payload={"rule": rule, **extra})


# ---------------------------------------------------------------- registry
def test_registry_resolves_by_name_and_loads_only_the_requested_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = FakeCheck("alpha"), FakeCheck("beta")
    eps = install(monkeypatch, a, b)
    assert registry.get_check("alpha") is a
    assert eps[0].loads == ["alpha"] and eps[1].loads == []


def test_registry_unknown_name_lists_available(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeCheck("alpha"))
    with pytest.raises(FormatError, match="unknown check 'nope'.*alpha"):
        registry.get_check("nope")


def test_registry_rejects_duplicate_names_from_two_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eps = [
        FakeEntryPoint("dup", "pkg_a:p", FakeCheck("dup")),
        FakeEntryPoint("dup", "pkg_b:p", FakeCheck("dup")),
    ]
    monkeypatch.setattr(registry, "_entry_points", lambda: eps)
    with pytest.raises(FormatError, match="duplicate check name 'dup'.*pkg_a:p.*pkg_b:p"):
        registry.get_check("dup")


def test_registry_rejects_wrong_kind_and_mismatched_spec_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_kind = FakeCheck("one")
    wrong_kind.spec = spec("one", kind="format")
    renamed = FakeCheck("two")
    renamed.spec = spec("other")
    install(monkeypatch, wrong_kind)
    monkeypatch.setattr(
        registry,
        "_entry_points",
        lambda: [
            FakeEntryPoint("one", "m:one", wrong_kind),
            FakeEntryPoint("two", "m:two", renamed),
        ],
    )
    with pytest.raises(FormatError, match="PluginSpec"):
        registry.get_check("one")
    with pytest.raises(FormatError, match="different spec name"):
        registry.get_check("two")


def test_registry_orders_builtin_checks_first_then_alphabetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch, *(FakeCheck(n) for n in ["zeta", "boxes", "alpha", "integrity", "imbalance"])
    )
    assert registry.check_names() == ["integrity", "imbalance", "boxes", "alpha", "zeta"]


# ------------------------------------------------------------------ runner
def test_runner_runs_selected_checks_in_registry_order(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    seen: list[str] = []

    def recorder(name: str) -> FakeCheck:
        return FakeCheck(name, lambda d, c: seen.append(name) or [])

    install(monkeypatch, recorder("boxes"), recorder("integrity"), recorder("zeta"))
    run = run_audit(ds, ["zeta", "boxes", "integrity"], ctx(tmp_path))
    assert seen == ["integrity", "boxes", "zeta"] and run.checks == ("integrity", "boxes", "zeta")
    assert list(run.timings_ms) == ["integrity", "boxes", "zeta"]


def test_all_selects_every_registered_check(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("integrity"), FakeCheck("boxes"))
    assert run_audit(ds, "all", ctx(tmp_path)).checks == ("integrity", "boxes")


def test_unknown_or_empty_selection_is_rejected(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("boxes"))
    with pytest.raises(FormatError, match="unknown check"):
        run_audit(ds, ["nope"], ctx(tmp_path))
    with pytest.raises(ValueError, match="no checks"):
        run_audit(ds, [], ctx(tmp_path))


def test_runner_orders_findings_deterministically_and_counts_severities(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    def produce(d: Dataset, c: CheckContext) -> list[Finding]:
        return [
            emit("INFO", "boxes", 2),
            emit("CRITICAL", "boxes", 1, rule="a"),
            emit("WARN", "boxes", 1, rule="b"),
            emit("CRITICAL", "boxes", 0),
        ]

    install(monkeypatch, FakeCheck("boxes", produce))
    run = run_audit(ds, ["boxes"], ctx(tmp_path))
    assert [(f.severity, f.image_id) for f in run.findings] == [
        ("CRITICAL", f"{0:016x}"),
        ("CRITICAL", f"{1:016x}"),
        ("WARN", f"{1:016x}"),
        ("INFO", f"{2:016x}"),
    ]
    assert run.counts.model_dump() == {"critical": 2, "warn": 1, "info": 1}


def test_same_input_gives_identical_findings_regardless_of_check_output_order(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    fs = [emit("WARN", "boxes", n) for n in range(3)]
    install(monkeypatch, FakeCheck("boxes", lambda d, c: fs))
    first = run_audit(ds, ["boxes"], ctx(tmp_path))
    install(monkeypatch, FakeCheck("boxes", lambda d, c: list(reversed(fs))))
    second = run_audit(ds, ["boxes"], ctx(tmp_path))
    assert first.findings == second.findings


def test_identical_keys_collapse_to_the_most_severe_finding(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    fs = [emit("INFO", "boxes"), emit("CRITICAL", "boxes"), emit("WARN", "boxes")]
    install(monkeypatch, FakeCheck("boxes", lambda d, c: fs))
    (only,) = run_audit(ds, ["boxes"], ctx(tmp_path)).findings
    assert only.severity == "CRITICAL"


def test_identical_findings_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("boxes", lambda d, c: [emit("WARN", "boxes")] * 3))
    assert len(run_audit(ds, ["boxes"], ctx(tmp_path)).findings) == 1


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (lambda: make_finding("99999999-2222-4333-8444-555555555555", "boxes", "WARN"), "run_id"),
        (lambda: make_finding(RUN, "integrity", "WARN"), "another check"),
        (
            lambda: make_finding(RUN, "boxes", "WARN", image_id="f" * 16),
            "unknown image_id",
        ),
        (lambda: make_finding(RUN, "boxes", "WARN", ann_id="nope"), "unknown ann_id"),
    ],
)
def test_plugin_output_is_validated(
    monkeypatch: pytest.MonkeyPatch,
    ds: Dataset,
    tmp_path: Path,
    bad: Callable[[], Finding],
    message: str,
) -> None:
    install(monkeypatch, FakeCheck("boxes", lambda d, c: [bad()]))
    with pytest.raises(ValidationError, match=message):
        run_audit(ds, ["boxes"], ctx(tmp_path))


def test_plugin_returning_non_findings_is_rejected(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("boxes", lambda d, c: [{"severity": "WARN"}]))  # type: ignore[list-item,return-value]
    with pytest.raises(ValidationError, match="boxes"):
        run_audit(ds, ["boxes"], ctx(tmp_path))


def test_findings_per_check_are_capped_and_the_cap_keeps_the_gate_correct(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    monkeypatch.setattr(engine, "MAX_FINDINGS_PER_CHECK", 5)
    fs = [emit("INFO", "boxes", n % 3, rule=f"i{n}") for n in range(20)]
    fs.append(emit("CRITICAL", "boxes", 0, rule="c"))
    install(monkeypatch, FakeCheck("boxes", lambda d, c: fs))
    run = run_audit(ds, ["boxes"], ctx(tmp_path), fail_on=["critical"])
    assert len(run.findings) == 5
    assert run.findings[0].severity == "CRITICAL" and not run.passed
    (marker,) = [f for f in run.findings if f.payload.get("rule") == "findings_truncated"]
    assert marker.payload == {"rule": "findings_truncated", "dropped": 17}
    assert marker.severity == "INFO"
    assert list(run.findings) == sorted(run.findings, key=engine._sort_key)  # marker is sorted in


def test_cap_boundary_exactly_at_the_limit_is_not_truncated(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    monkeypatch.setattr(engine, "MAX_FINDINGS_PER_CHECK", 5)
    fs = [emit("WARN", "boxes", rule=f"r{n}") for n in range(5)]
    install(monkeypatch, FakeCheck("boxes", lambda d, c: fs))
    run = run_audit(ds, ["boxes"], ctx(tmp_path))
    assert len(run.findings) == 5
    assert all(f.payload.get("rule") != "findings_truncated" for f in run.findings)


def test_truncation_marker_takes_the_highest_dropped_severity(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    monkeypatch.setattr(engine, "MAX_FINDINGS_PER_CHECK", 3)
    fs = [emit("CRITICAL", "boxes", rule=f"c{n}") for n in range(6)]
    install(monkeypatch, FakeCheck("boxes", lambda d, c: fs))
    run = run_audit(ds, ["boxes"], ctx(tmp_path))
    assert run.findings[-1].severity == "CRITICAL" and run.counts.critical == 3


def test_runner_logs_one_event_per_check_and_one_completion(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(
        monkeypatch,
        FakeCheck("integrity", lambda d, c: [emit("WARN", "integrity")]),
        FakeCheck("boxes"),
    )
    with structlog.testing.capture_logs() as logs:
        run = run_audit(ds, "all", ctx(tmp_path), fail_on=["warn"])
    events = [(e["event"], e.get("check_name")) for e in logs]
    assert events == [
        ("audit_check_completed", "integrity"),
        ("audit_check_completed", "boxes"),
        ("audit_completed", None),
    ]
    first = logs[0]
    assert first["findings_count"] == 1 and isinstance(first["duration_ms"], int)
    done = logs[-1]
    assert done["run_id"] == RUN and done["dataset_version"] == ds.manifest.version
    assert (done["critical"], done["warn"], done["info"], done["passed"]) == (0, 1, 0, False)
    assert run.passed is False
    assert "file_path" not in str(logs) and "/data/" not in str(logs)


# -------------------------------------------------------------------- gate
def make_run(monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path, *fs: Finding) -> Any:
    install(monkeypatch, FakeCheck("boxes", lambda d, c: list(fs)), FakeCheck("integrity"))
    return run_audit(ds, "all", ctx(tmp_path))


@pytest.mark.parametrize(
    ("fail_on", "severity", "expected"),
    [
        (["critical"], "CRITICAL", True),
        (["critical"], "WARN", False),
        (["warn"], "CRITICAL", True),  # a severity name is a threshold: that level and above
        (["warn"], "WARN", True),
        (["warn"], "INFO", False),
        (["info"], "INFO", True),
        (["CRITICAL", "Warn"], "WARN", True),  # case-insensitive
        ([], "CRITICAL", False),
        (["boxes"], "INFO", True),  # a check name matches any severity of that check
        (["integrity"], "CRITICAL", False),  # a different check
        (["integrity", "info"], "INFO", True),
    ],
)
def test_gate_semantics(
    monkeypatch: pytest.MonkeyPatch,
    ds: Dataset,
    tmp_path: Path,
    fail_on: list[str],
    severity: Severity,
    expected: bool,
) -> None:
    run = make_run(monkeypatch, ds, tmp_path, emit(severity, "boxes"))
    assert gate_failed(run, fail_on) is expected


def test_gate_with_no_findings_passes_even_with_the_strictest_level(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    run = make_run(monkeypatch, ds, tmp_path)
    assert gate_failed(run, ["info", "boxes"]) is False


def test_gate_rejects_unknown_tokens_instead_of_never_firing(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    run = make_run(monkeypatch, ds, tmp_path)
    with pytest.raises(ValueError, match="unknown --fail-on value 'crit'"):
        gate_failed(run, ["crit"])
    with pytest.raises(ValueError, match="'leakage'.*not available"):
        gate_failed(run, ["leakage"])
    with pytest.raises(ValueError, match="empty"):
        gate_failed(run, ["critical", " "])


def test_fail_on_naming_a_check_that_did_not_run_is_rejected(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("boxes"), FakeCheck("integrity"))
    with pytest.raises(ValueError, match="'integrity' is not among the checks"):
        run_audit(ds, ["boxes"], ctx(tmp_path), fail_on=["integrity"])


def test_run_records_the_gate_and_passed_flag(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("boxes", lambda d, c: [emit("WARN", "boxes")]))
    failed = run_audit(ds, ["boxes"], ctx(tmp_path), fail_on=["warn", "CRITICAL"])
    passed = run_audit(ds, ["boxes"], ctx(tmp_path), fail_on=["critical"])
    assert failed.fail_on == ("warn", "critical") and failed.passed is False
    assert passed.passed is True and passed.run_id == RUN
    assert failed.dataset_version == ds.manifest.version


# ----------------------------------------------------------------- summary
def test_run_carries_a_dataset_summary_for_the_reports(
    monkeypatch: pytest.MonkeyPatch, ds: Dataset, tmp_path: Path
) -> None:
    install(monkeypatch, FakeCheck("boxes"))
    summary = run_audit(ds, ["boxes"], ctx(tmp_path, include_model_annotations=True)).summary
    assert summary is not None
    assert (summary.task, summary.image_count, summary.annotation_count) == ("detection", 3, 4)
    assert summary.excluded_model_annotations == 0
    default = run_audit(ds, ["boxes"], ctx(tmp_path)).summary  # the model row is left out
    assert default is not None
    assert (default.annotation_count, default.excluded_model_annotations) == (3, 1)
    assert summary.split_images == {"train": 1, "val": 1, "unassigned": 1}
    assert [(c.id, c.name) for c in summary.classes] == [(1, "cat"), (2, "dog")]
    assert summary.class_counts == {
        "train": {"1": 1, "2": 1},
        "val": {"1": 1},
        "unassigned": {"2": 1},
    }
    assert summary.classes_omitted is False


def test_summary_omits_per_class_detail_beyond_the_class_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datacopilot.audit import summary as summary_mod
    from datacopilot.core.schema import Category

    monkeypatch.setattr(summary_mod, "MAX_SUMMARY_CLASSES", 2)
    images, anns, _ = make_tables()
    cats = [Category(id=i, name=f"c{i}") for i in (1, 2, 3)]
    big = dataset_from(images, anns, cats)
    install(monkeypatch, FakeCheck("boxes"))
    s = run_audit(big, ["boxes"], ctx(tmp_path, include_model_annotations=True)).summary
    assert s is not None and s.classes_omitted and s.classes == () and s.class_counts == {}
    assert s.annotation_count == 4


def test_summary_names_are_truncated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from datacopilot.core.schema import Category

    images, anns, _ = make_tables()
    cats = [Category(id=1, name="x" * 5000), Category(id=2, name="dog")]
    install(monkeypatch, FakeCheck("boxes"))
    s = run_audit(dataset_from(images, anns, cats), ["boxes"], ctx(tmp_path)).summary
    assert s is not None and len(s.classes[0].name) == 200
