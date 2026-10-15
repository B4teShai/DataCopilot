"""Application layer: the ``ingest`` and ``export`` use cases.

Formats are reached only through :mod:`datacopilot.io.registry`; tables are persisted by the
store. Events logged here: ``dataset_ingested``, ``dataset_exported`` and, per dropped or
skipped column, ``conversion_loss``.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.schema import Dataset, DatasetManifest, validate_tables
from datacopilot.io import registry
from datacopilot.io.report import ConversionReport, enforce_strict
from datacopilot.logging import get_logger
from datacopilot.store import DatasetStore, StoreLimits

DCP_FORMAT = "dcp"


@dataclass(frozen=True)
class IngestResult:
    path: Path
    manifest: DatasetManifest
    report: ConversionReport


def parse_group_by(pattern: str) -> re.Pattern[str]:
    """Compile a ``--group-by`` pattern (optional ``re:`` prefix).

    Raises:
        ValueError: if the regular expression is invalid.
    """
    text = pattern[3:] if pattern.startswith("re:") else pattern
    try:
        return re.compile(text)
    except re.error as exc:
        raise ValueError(f"invalid --group-by regular expression: {exc.msg}") from None


def _group_of(pattern: re.Pattern[str], relative_path: str) -> str | None:
    match = pattern.search(relative_path)
    if match is None:
        return None
    if "g" in pattern.groupindex:
        value = match.group("g")
    elif pattern.groups:
        value = match.group(1)
    else:
        value = match.group(0)
    return value or None


def _log_loss(report: ConversionReport) -> None:
    log = get_logger(__name__)
    for kind, counts in (("dropped", report.dropped), ("skipped", report.skipped)):
        for column, count in sorted(counts.items()):
            log.warning(
                "conversion_loss",
                source_format=report.source_format,
                target_format=report.target_format,
                column=column,
                count=count,
                kind=kind,
            )


def ingest(
    source: str | Path,
    format: str,  # noqa: A002  # public parameter name documented in the architecture
    out: str | Path,
    *,
    group_by: str | None = None,
    name: str | None = None,
    limits: StoreLimits | None = None,
    strict: bool = False,
) -> IngestResult:
    """Read ``source`` with the named reader, validate, and write a new ``dataset.dcp``.

    With ``strict`` any value or row the reader could not carry into the data model (an
    unsupported shape, a skipped file) fails the ingest before anything is written.

    Raises:
        FormatError, ValidationError, StoreError: nothing is written when any of them occurs.
        ConversionLossError: ``strict`` and the reader dropped or skipped anything.
        ValueError: for an invalid ``group_by`` expression.
    """
    started = time.perf_counter()
    pattern = parse_group_by(group_by) if group_by else None
    reader = registry.get_reader(format)
    path = Path(source)
    result = reader.read(path)
    if not result.images:
        raise FormatError("no images found in the input")
    images = list(result.images)
    if pattern is not None:
        root = (path if path.is_dir() else path.parent).resolve()
        images = [
            i.model_copy(
                update={"group_id": _group_of(pattern, os.path.relpath(i.file_path, root))}
            )
            for i in images
        ]
    seen: dict[str, str] = {}
    for image in images:
        label = Path(image.file_path).name
        if image.image_id in seen:
            raise ValidationError(
                f"images {seen[image.image_id]!r} and {label!r} have identical content "
                f"(image_id {image.image_id}); remove or deduplicate one before ingesting"
            )
        seen[image.image_id] = label
    validate_tables(images, result.annotations, result.categories, task=result.task)
    rows = len(images) + len(result.annotations)
    report = ConversionReport(
        source_format=format,
        target_format=DCP_FORMAT,
        rows_read=rows,
        rows_written=rows,
        dropped=result.dropped,
        skipped=result.skipped,
    )
    if strict and not report.is_lossless():
        _log_loss(report)
    enforce_strict(report, strict=strict)
    store = DatasetStore.create(
        out,
        images,
        result.annotations,
        result.categories,
        name=name or result.name,
        task=result.task,
        limits=limits,
    )
    manifest = store.manifest()
    _log_loss(report)
    get_logger(__name__).info(
        "dataset_ingested",
        format=format,
        image_count=len(images),
        annotation_count=len(result.annotations),
        dataset_version=manifest.version,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return IngestResult(Path(out), manifest, report)


def export(ds: Dataset, format: str, out: str | Path, *, strict: bool = False) -> ConversionReport:  # noqa: A002
    """Write ``ds`` in the named format into ``out`` (absent or an empty directory).

    The writer runs in a sibling temporary directory that is renamed into place only after
    ``strict`` is satisfied, so a failure leaves no partial output.

    Raises:
        FormatError: unknown or write-incapable format, unsafe or non-empty output.
        ConversionLossError: ``strict`` and the conversion dropped or skipped anything.
    """
    started = time.perf_counter()
    writer = registry.get_writer(format)
    target = Path(out)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FormatError(f"output {target.name} already exists and is not an empty directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        report = writer.write(ds, staging)
        report = report.model_copy(update={"source_format": DCP_FORMAT})
        _log_loss(report)
        enforce_strict(report, strict=strict)
        if target.exists():
            target.rmdir()
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    get_logger(__name__).info(
        "dataset_exported",
        format=format,
        image_count=len(ds.images),
        annotation_count=len(ds.annotations),
        dataset_version=ds.manifest.version,
        dropped_count=report.total_dropped,
        skipped_count=sum(report.skipped.values()),
        strict=strict,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return report
