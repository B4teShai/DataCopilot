"""Construction of :class:`~datacopilot.core.schema.Finding` rows with their deterministic id."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from datacopilot.core.schema import Finding, Severity
from datacopilot.core.version import finding_id

MAX_PAYLOAD_STRING_CHARS = 512
"""Strings in a finding payload are cut to this length (an ellipsis marks the cut)."""
_MAX_CLIP_DEPTH = 8


def _clip(value: JsonValue, depth: int = 0) -> JsonValue:
    if isinstance(value, str):
        if len(value) <= MAX_PAYLOAD_STRING_CHARS:
            return value
        return value[: MAX_PAYLOAD_STRING_CHARS - 1] + "…"
    if depth >= _MAX_CLIP_DEPTH:
        return value if not isinstance(value, (list, dict)) else "…"
    if isinstance(value, list):
        return [_clip(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _clip(v, depth + 1) for k, v in value.items()}
    return value


def make_finding(
    run_id: str,
    check: str,
    severity: Severity,
    *,
    image_id: str | None = None,
    ann_id: str | None = None,
    payload: Mapping[str, JsonValue] | None = None,
) -> Finding:
    """Build a finding whose ``finding_id`` is derived from check, target and payload.

    Raises:
        pydantic.ValidationError: for an invalid check name, id, or an unbounded payload.
    """
    data = {k: _clip(v) for k, v in (payload or {}).items()}
    return Finding(
        finding_id=finding_id(check, image_id, ann_id, data),
        run_id=run_id,
        check=check,
        severity=severity,
        image_id=image_id,
        ann_id=ann_id,
        payload=data,
    )
