"""JSON report: the stored :class:`AuditRun`, serialised deterministically."""

from __future__ import annotations

import json

from datacopilot.core.schema import AuditRun


def render_json(run: AuditRun) -> str:
    """Sorted keys, two-space indent, ASCII only (non-ASCII becomes ``\\uXXXX``), no NaN."""
    return (
        json.dumps(run.to_row(), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    )
