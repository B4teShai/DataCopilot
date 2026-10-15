"""Report generator (application layer): JSON, static HTML and SARIF from an ``AuditRun``.

Event: ``report_written`` (``format``, ``path`` = the file name only) per file. Files are written
to a temporary sibling and renamed into place, so a failed write leaves no partial report and a
symlink at the target is replaced rather than followed.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from datacopilot.core.errors import DataCopilotError
from datacopilot.core.schema import AuditRun
from datacopilot.logging import get_logger
from datacopilot.report import html_report, json_report, sarif_report

__all__ = ["render"]


def _write(path: Path, text: str, fmt: str) -> None:
    staged = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("wb") as handle:
            handle.write(text.encode("utf-8", errors="replace"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except OSError:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass  # the directory itself is unusable; the original failure is what matters
        raise DataCopilotError(f"cannot write {fmt} report {path.name}") from None
    get_logger(__name__).info("report_written", format=fmt, path=path.name)


def render(
    run: AuditRun, *, html: Path | None, json: Path | None, sarif: Path | None
) -> list[Path]:
    """Write the requested reports and return their paths (html, json, sarif order).

    Raises:
        DataCopilotError: a report file cannot be written (nothing partial is left behind).
    """
    written: list[Path] = []
    for fmt, target, make in (
        ("html", html, html_report.render_html),
        ("json", json, json_report.render_json),
        ("sarif", sarif, sarif_report.render_sarif),
    ):
        if target is not None:
            _write(target, make(run), fmt)
            written.append(target)
    return written
