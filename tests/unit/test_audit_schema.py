"""Finding and AuditRun invariants (docs/architecture/03-data-model.md, invariants 13 and 15)."""

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from datacopilot.core.schema import MAX_PAYLOAD_BYTES, AuditRun, Counts, Finding
from datacopilot.core.version import finding_id

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
VERSION = "a" * 32


def finding(
    severity: str = "WARN",
    *,
    check: str = "boxes",
    image_id: str | None = "0" * 16,
    ann_id: str | None = None,
    payload: dict[str, Any] | None = None,
    run_id: str = RUN,
) -> Finding:
    payload = {"rule": "tiny"} if payload is None else payload
    return Finding(
        finding_id=finding_id(check, image_id, ann_id, payload),
        run_id=run_id,
        check=check,
        severity=severity,  # type: ignore[arg-type]
        image_id=image_id,
        ann_id=ann_id,
        payload=payload,
    )


def run_of(*findings: Finding, **kw: Any) -> AuditRun:
    counts = Counts(
        critical=sum(f.severity == "CRITICAL" for f in findings),
        warn=sum(f.severity == "WARN" for f in findings),
        info=sum(f.severity == "INFO" for f in findings),
    )
    base: dict[str, Any] = {
        "run_id": RUN,
        "dataset_version": VERSION,
        "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "checks": ("boxes",),
        "counts": counts,
        "passed": True,
        "findings": findings,
    }
    return AuditRun(**{**base, **kw})


def test_finding_id_is_deterministic_and_independent_of_run_and_severity() -> None:
    a = finding("WARN", run_id=RUN)
    b = finding("CRITICAL", run_id="99999999-2222-4333-8444-555555555555")
    assert a.finding_id == b.finding_id and len(a.finding_id) == 16


def test_finding_id_depends_on_check_target_and_payload() -> None:
    ids = {
        finding().finding_id,
        finding(check="integrity").finding_id,
        finding(image_id="1" * 16).finding_id,
        finding(ann_id="x").finding_id,
        finding(payload={"rule": "other"}).finding_id,
    }
    assert len(ids) == 5


def test_finding_rejects_a_forged_finding_id() -> None:
    good = finding()
    with pytest.raises(ValidationError, match="finding_id"):
        Finding(**{**good.model_dump(), "finding_id": "f" * 16})


@pytest.mark.parametrize("severity", ["error", "critical", "", "INFO "])
def test_finding_rejects_unknown_severity(severity: str) -> None:
    with pytest.raises(ValidationError):
        finding(severity)


@pytest.mark.parametrize("check", ["Boxes", "1x", "a-b", "", "a" * 0])
def test_finding_rejects_bad_check_names(check: str) -> None:
    with pytest.raises(ValidationError):
        finding(check=check)


def test_finding_rejects_non_finite_payload_numbers() -> None:
    with pytest.raises(ValidationError):
        Finding(
            finding_id="0" * 16,
            run_id=RUN,
            check="boxes",
            severity="WARN",
            payload={"iou": float("nan")},
        )


def test_finding_rejects_oversized_and_too_deep_payloads() -> None:
    with pytest.raises(ValidationError, match="payload"):
        finding(payload={"blob": "x" * (MAX_PAYLOAD_BYTES + 1)})
    deep: Any = {"k": 1}
    for _ in range(40):
        deep = {"k": deep}
    with pytest.raises(ValidationError, match="payload"):
        finding(payload=deep)
    with pytest.raises(ValidationError, match="payload"):
        finding(payload={"items": list(range(MAX_PAYLOAD_BYTES))})


def test_finding_payload_at_the_size_limit_is_accepted() -> None:
    finding(payload={"blob": "x" * 4000, "items": list(range(1000))})


def test_run_counts_must_equal_findings_per_severity() -> None:
    f = finding("WARN")
    with pytest.raises(ValidationError, match="counts"):
        run_of(f, counts=Counts(critical=0, warn=2, info=0))
    assert run_of(f).counts == Counts(critical=0, warn=1, info=0)


def test_run_rejects_findings_owned_by_another_run() -> None:
    foreign = finding(run_id="99999999-2222-4333-8444-555555555555")
    with pytest.raises(ValidationError, match="run_id"):
        run_of(foreign)


def test_run_rejects_duplicate_finding_ids() -> None:
    f = finding()
    with pytest.raises(ValidationError, match="duplicate"):
        run_of(f, f, counts=Counts(critical=0, warn=2, info=0))


@pytest.mark.parametrize("run_id", ["../x", "not-a-uuid", RUN.upper(), RUN + "0", ""])
def test_run_id_must_be_a_lowercase_uuid(run_id: str) -> None:
    with pytest.raises(ValidationError):
        run_of(run_id=run_id, findings=())


def test_run_round_trips_through_json() -> None:
    run = run_of(finding("INFO"), finding("WARN", image_id="1" * 16))
    again = AuditRun.model_validate_json(run.model_dump_json())
    assert again == run


@pytest.mark.parametrize("value", ["../x", "a b", "", "x" * 300, "a\nb", "name\u202e"])
def test_audit_run_bounds_the_embedder_label(value: str) -> None:
    from datetime import datetime, timezone

    from datacopilot.core.schema import AuditRun, Counts

    with pytest.raises(ValueError):
        AuditRun(
            run_id=RUN,
            dataset_version="a" * 32,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            checks=(),
            embedder=value,
            counts=Counts(critical=0, warn=0, info=0),
            passed=True,
        )
