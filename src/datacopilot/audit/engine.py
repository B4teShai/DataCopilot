"""Runner and gate: run checks in registry order, validate their output, evaluate ``--fail-on``.

Events: ``audit_check_completed`` per check and ``audit_completed`` once. Plugin output is
untrusted (ADR-0005 rule 5): every finding is re-validated, must belong to this run and check,
and may only reference images and annotations of the audited dataset.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import partial
from typing import Literal, TypeVar

from datacopilot.audit import registry
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.summary import build_summary
from datacopilot.core.errors import DataCopilotError, ValidationError
from datacopilot.core.schema import SEVERITIES, AuditRun, Counts, Dataset, Finding, Severity
from datacopilot.logging import get_logger

MAX_FINDINGS_PER_CHECK = 50_000
"""Findings kept per check. Findings are sorted most severe first before the cut, so a truncated
check still reports every severity it found; one ``findings_truncated`` finding carries the
number dropped and the highest severity among them, so a gate never changes because of the cap."""

T = TypeVar("T")
_SEVERITY_NAMES = tuple(s.lower() for s in SEVERITIES)
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def parse_fail_on(tokens: Sequence[str]) -> tuple[str, ...]:
    """Normalise ``--fail-on`` tokens to lower case and reject anything that could never match.

    Raises:
        ValueError: empty token, unknown name, or a documented check that is not built yet.
    """
    available = registry.check_names()
    out: list[str] = []
    lowered = [raw.strip().lower() for raw in tokens]
    if "none" in lowered:
        if len(lowered) > 1:
            raise ValueError("--fail-on none cannot be combined with other values")
        return ()
    for raw in tokens:
        token = raw.strip().lower()
        if not token:
            raise ValueError("empty --fail-on value")
        if token not in _SEVERITY_NAMES and token not in available:
            if token in registry.CANONICAL_ORDER:
                raise ValueError(f"check {token!r} is not available in this version")
            raise ValueError(
                f"unknown --fail-on value {token!r}; use "
                + ", ".join(_SEVERITY_NAMES)
                + " or a check name ("
                + ", ".join(available)
                + ")"
            )
        out.append(token)
    return tuple(out)


def _matches(finding: Finding, severities: tuple[str, ...], checks: frozenset[str]) -> bool:
    if finding.check in checks:
        return True
    threshold = max((_RANK[s.upper()] for s in severities), default=None)  # type: ignore[index]
    return threshold is not None and _RANK[finding.severity] <= threshold


def gate_failed(run: AuditRun, fail_on: Sequence[str]) -> bool:
    """True when ``run`` has a finding matched by ``fail_on`` (exit code 1).

    A severity name matches that severity and every more severe one (``warn`` fails on WARN and
    CRITICAL); a check name matches every finding of that check.

    Raises:
        ValueError: for tokens :func:`parse_fail_on` rejects.
    """
    tokens = parse_fail_on(fail_on)
    severities = tuple(t for t in tokens if t in _SEVERITY_NAMES)
    checks = frozenset(t for t in tokens if t not in _SEVERITY_NAMES)
    return any(_matches(f, severities, checks) for f in run.findings)


def _sort_key(f: Finding) -> tuple[int, str, str, str]:
    return (_RANK.get(f.severity, len(_RANK)), f.image_id or "", f.ann_id or "", f.finding_id)


def _finalise(name: str, raw: Sequence[Finding], ds: Dataset, ctx: CheckContext) -> list[Finding]:
    """Deduplicate, order, cap and re-validate one check's output."""
    for item in raw:
        if not isinstance(item, Finding):
            raise ValidationError(f"check {name!r} returned a {type(item).__name__}, not a Finding")
    unique: dict[str, Finding] = {}
    for f in sorted(raw, key=_sort_key):  # most severe first, so it wins a key collision
        unique.setdefault(f.finding_id, f)
    ordered = sorted(unique.values(), key=_sort_key)
    if len(ordered) > MAX_FINDINGS_PER_CHECK:
        kept, dropped = ordered[: MAX_FINDINGS_PER_CHECK - 1], ordered[MAX_FINDINGS_PER_CHECK - 1 :]
        top: Severity = min((f.severity for f in dropped), key=lambda s: _RANK.get(s, 99))
        kept.append(
            make_finding(
                ctx.run_id,
                name,
                top,
                payload={"rule": "findings_truncated", "dropped": len(dropped)},
            )
        )
        ordered = sorted(kept, key=_sort_key)  # the marker takes its place by severity
    images = {i.image_id for i in ds.images}
    anns = {a.ann_id for a in ds.annotations} if any(f.ann_id for f in ordered) else set()
    checked: list[Finding] = []
    for f in ordered:
        try:
            valid = Finding.model_validate(f.model_dump())
        except ValueError as exc:
            raise ValidationError(f"check {name!r} returned an invalid finding: {exc}") from None
        if valid.run_id != ctx.run_id:
            raise ValidationError(f"check {name!r} returned a finding with another run_id")
        if valid.check != name:
            raise ValidationError(f"check {name!r} returned a finding of another check")
        if valid.image_id is not None and valid.image_id not in images:
            raise ValidationError(f"check {name!r} returned a finding for unknown image_id")
        if valid.ann_id is not None and valid.ann_id not in anns:
            raise ValidationError(f"check {name!r} returned a finding for unknown ann_id")
        checked.append(valid)
    return checked


def _guarded(name: str, call: Callable[[], T]) -> T:
    """Run plugin code; anything but an ordinary exception or Ctrl-C fails the audit.

    ``SystemExit`` (even with code 0), ``GeneratorExit`` and the like must never end the process
    with a success status in the middle of a gate; ordinary exceptions already reach the error
    boundary (exit 1) and Ctrl-C stays an interrupt (exit 130).
    """
    try:
        return call()
    except (Exception, KeyboardInterrupt):
        raise
    except BaseException as exc:
        raise DataCopilotError(f"check {name!r} aborted the audit ({type(exc).__name__})") from None


def select_checks(
    checks: Sequence[str] | Literal["all"], *, with_embedder: bool = False
) -> list[str]:
    """Resolve a selection to registered names in run order.

    ``"all"`` includes the embedding checks (``registry.EMBEDDING_CHECKS``) only with an embedder.

    Raises:
        ValueError: an empty selection, or an embedding check without an embedder.
        FormatError: an unknown name.
    """
    available = registry.check_names()
    if checks == "all":
        return [n for n in available if with_embedder or n not in registry.EMBEDDING_CHECKS]
    if not checks:
        raise ValueError("no checks selected")
    for name in checks:
        if name not in available:
            registry.get_check(name)  # raises FormatError naming the available checks
        if name in registry.EMBEDDING_CHECKS and not with_embedder:
            raise ValueError(f"check {name!r} needs an embedder (--embedder NAME)")
    wanted = set(checks)
    return [n for n in available if n in wanted]


def not_in_run_message(token: str, with_embedder: bool) -> str:
    """Usage error for a ``--fail-on`` check name that is not among the checks being run."""
    hint = (
        " (it needs --embedder)" if token in registry.EMBEDDING_CHECKS and not with_embedder else ""
    )
    return f"--fail-on {token!r} is not among the checks being run{hint}"


def run_audit(
    ds: Dataset,
    checks: Sequence[str] | Literal["all"],
    ctx: CheckContext,
    fail_on: Sequence[str] = (),
) -> AuditRun:
    """Run ``checks`` over ``ds`` and return the run (not yet persisted).

    Raises:
        ValueError: empty selection, or a ``fail_on`` token that cannot match (unknown, not
            built, or naming a check that is not part of this run).
        FormatError: unknown check or broken plugin.
        ValidationError: a check returned findings that are invalid or foreign to this dataset.
    """
    if not ds.images:
        raise ValidationError("dataset has no images; there is nothing to audit")
    excluded = 0
    if not ctx.include_model_annotations:
        # unreviewed model output is not ground truth; accepted proposals (human-reviewed) stay
        kept = tuple(a for a in ds.annotations if a.provenance != "model")
        excluded = len(ds.annotations) - len(kept)
        if excluded:
            ds = Dataset(ds.manifest, ds.images, kept, ds.root)
    with_embedder = ctx.embeddings is not None
    names = select_checks(checks, with_embedder=with_embedder)
    tokens = parse_fail_on(fail_on)
    for token in tokens:
        if token not in _SEVERITY_NAMES and token not in names:
            raise ValueError(not_in_run_message(token, with_embedder))
    ctx = ctx.model_copy(update={"incomplete_is_critical": bool(tokens)})
    log = get_logger(__name__)
    if not hasattr(os, "O_NOFOLLOW"):
        log.warning("audit_nofollow_unavailable")  # symlink races cannot be ruled out here
    started = datetime.now(timezone.utc)
    findings: list[Finding] = []
    timings: dict[str, int] = {}
    for name in names:
        check = _guarded(name, partial(registry.get_check, name))
        t0 = time.perf_counter()
        produced = _finalise(name, _guarded(name, partial(check.run, ds, ctx)), ds, ctx)
        timings[name] = int((time.perf_counter() - t0) * 1000)
        findings.extend(produced)
        log.info(
            "audit_check_completed",
            check_name=name,
            findings_count=len(produced),
            duration_ms=timings[name],
        )
    tally = Counter(f.severity for f in findings)
    draft = AuditRun(
        run_id=ctx.run_id,
        dataset_version=ds.manifest.version,
        started_at=started,
        checks=tuple(names),
        embedder=ctx.embedder,
        counts=Counts(critical=tally["CRITICAL"], warn=tally["WARN"], info=tally["INFO"]),
        passed=True,
        fail_on=tokens,
        timings_ms=timings,
        summary=build_summary(ds, excluded),
        findings=tuple(findings),
    )
    run = draft.model_copy(update={"passed": not gate_failed(draft, tokens)})
    log.info(
        "audit_completed",
        run_id=run.run_id,
        dataset_version=run.dataset_version,
        critical=run.counts.critical,
        warn=run.counts.warn,
        info=run.counts.info,
        passed=run.passed,
    )
    return run
