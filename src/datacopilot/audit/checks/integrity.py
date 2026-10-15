"""``integrity``: every image row points at a readable file whose header matches the row.

Per image, without decoding pixels: the file exists and is a regular file, is not empty or over
``CheckConfig.max_image_bytes``, its header parses (PNG, JPEG, GIF, BMP), and the size it declares
(after EXIF orientation) equals ``Image.width`` x ``Image.height``. A non-trivial EXIF orientation
is noted as INFO. Remote URIs are not fetched; they are counted in one INFO finding.

Rules (``payload["rule"]``): ``file_missing``, ``unsafe_path``, ``not_a_regular_file``,
``unreadable``, ``empty_file``, ``invalid_header`` and ``size_mismatch`` (CRITICAL);
``file_too_large`` (WARN); ``exif_orientation`` (INFO); ``audit_incomplete`` (CRITICAL, WARN
without a gate: the time budget ``max_check_seconds`` ended the check before every image was
examined); ``remote_not_checked`` and
``external_not_checked`` (WARN, CRITICAL when more than ``max_unchecked_fraction`` of the images
were not checked or ``require_local_images`` is set; the latter counts images whose real path is
outside the dataset root, which are not touched unless ``allow_external_images`` is set). An
unrecognised file format is ``invalid_header``: ingest accepted every image, so a file that no
longer parses has been damaged or replaced.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from pydantic import JsonValue

from datacopilot.audit import pool
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.paths import display_name, resolve_image_path, unchecked_severity
from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Dataset, Finding, Image, PluginSpec, Severity
from datacopilot.io import files

NAME = "integrity"


def _inspect(image: Image, ctx: CheckContext) -> list[Finding]:
    """Findings of one local image (remote paths are counted by the caller)."""

    def found(severity: Severity, rule: str, **payload: JsonValue) -> list[Finding]:
        return [
            make_finding(
                ctx.run_id,
                NAME,
                severity,
                image_id=image.image_id,
                payload={"rule": rule, "file_name": display_name(image.file_path), **payload},
            )
        ]

    resolved = resolve_image_path(
        image.file_path, ctx.root, allow_external=ctx.allow_external_images
    )
    if resolved.kind == "unsafe" or resolved.path is None:
        return found("CRITICAL", "unsafe_path")
    path: Path = resolved.path
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return found("CRITICAL", "file_missing")
    except OSError:
        return found("CRITICAL", "unreadable")
    if not stat.S_ISREG(info.st_mode):
        return found("CRITICAL", "not_a_regular_file")
    if info.st_size == 0:
        return found("CRITICAL", "empty_file")
    if info.st_size > ctx.config.max_image_bytes:
        return found("WARN", "file_too_large", bytes=info.st_size)
    if not os.access(path, os.R_OK):
        return found("CRITICAL", "unreadable")
    try:
        header = files.image_header(path, nofollow=True)
    except FormatError:  # includes an unrecognised format: ingest accepted this file, so it changed
        return found("CRITICAL", "invalid_header")
    out: list[Finding] = []
    actual = header.oriented_size
    if actual != (image.width, image.height):
        out += found(
            "CRITICAL",
            "size_mismatch",
            declared=[image.width, image.height],
            actual=[actual[0], actual[1]],
        )
    if header.orientation != 1:
        out += found("INFO", "exif_orientation", orientation=header.orientation)
    return out


class IntegrityCheck:
    spec = PluginSpec(
        name=NAME,
        kind="check",
        license="Apache-2.0",
        extra=None,
        gpu_memory_gb=None,
        version="1",
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        local: list[Image] = []
        skipped = {"remote": 0, "external": 0}
        for image in ds.images:
            kind = resolve_image_path(
                image.file_path, ctx.root, allow_external=ctx.allow_external_images
            ).kind
            if kind in skipped:
                skipped[kind] += 1
            else:
                local.append(image)
        findings: list[Finding] = []
        done = 0
        deadline = time.monotonic() + ctx.config.max_check_seconds
        for result in pool.bounded_map(
            lambda i: _inspect(i, ctx), local, workers=ctx.workers, deadline=deadline
        ):
            findings += result
            done += 1
        if done < len(local):
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "CRITICAL" if ctx.incomplete_is_critical else "WARN",
                    payload={
                        "rule": "audit_incomplete",
                        "reason": "time_budget",
                        "checked": done,
                        "total": len(local),
                    },
                )
            )
        unchecked = sum(skipped.values())
        severity = unchecked_severity(
            unchecked,
            len(ds.images),
            max_fraction=ctx.config.max_unchecked_fraction,
            require_local=ctx.require_local_images,
        )
        for skip, count in skipped.items():
            if count:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        severity,
                        payload={
                            "rule": f"{skip}_not_checked",
                            "count": count,
                            "total": len(ds.images),
                        },
                    )
                )
        return findings


plugin = IntegrityCheck()
