"""Application layer: the ``ingest`` and ``export`` use cases.

Formats are reached only through :mod:`datacopilot.io.registry`; tables are persisted by the
store. Events logged here: ``dataset_ingested``, ``dataset_exported`` and, per dropped or
skipped column, ``conversion_loss``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from datacopilot.audit.paths import resolve_image_path
from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.schema import (
    SCHEMA_VERSION,
    Annotation,
    Category,
    Dataset,
    DatasetManifest,
    Image,
    Task,
    validate_tables,
)
from datacopilot.core.version import models_version
from datacopilot.io import files, registry
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


@dataclass(frozen=True)
class _Source:
    """A validated read of a source dataset, before it is stored or wrapped."""

    root: Path
    name: str
    task: Task
    images: list[Image]
    annotations: list[Annotation]
    categories: list[Category]
    report: ConversionReport


def _read_source(
    source: str | Path,
    format: str,  # noqa: A002
    *,
    group_by: str | None,
    name: str | None,
    strict: bool,
    options: Mapping[str, object] | None,
) -> _Source:
    pattern = parse_group_by(group_by) if group_by else None
    reader: Any = registry.get_reader(format)
    path = Path(source)
    try:
        result = reader.read(path, **options) if options else reader.read(path)
    except PydanticValidationError:  # a reader let a row-level validation error escape
        raise FormatError(f"{format}: the input contains a value the data model rejects") from None
    if not result.images:
        raise FormatError("no images found in the input")
    images = list(result.images)
    root = (path if path.is_dir() else path.parent).resolve()
    if pattern is not None:
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
    try:
        report = ConversionReport(
            source_format=format,
            target_format=DCP_FORMAT,
            rows_read=rows,
            rows_written=rows,
            dropped=result.dropped,
            skipped=result.skipped,
        )
    except PydanticValidationError:
        raise FormatError(f"{format}: the reader produced an invalid loss report") from None
    if strict and not report.is_lossless():
        _log_loss(report)
    enforce_strict(report, strict=strict)
    return _Source(
        root=root,
        name=name or result.name,
        task=result.task,
        images=images,
        annotations=list(result.annotations),
        categories=list(result.categories),
        report=report,
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
    reader_options: Mapping[str, object] | None = None,
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
    read = _read_source(
        source, format, group_by=group_by, name=name, strict=strict, options=reader_options
    )
    store = DatasetStore.create(
        out,
        read.images,
        read.annotations,
        read.categories,
        name=read.name,
        task=read.task,
        limits=limits,
    )
    manifest = store.manifest()
    _log_loss(read.report)
    get_logger(__name__).info(
        "dataset_ingested",
        format=format,
        image_count=len(read.images),
        annotation_count=len(read.annotations),
        dataset_version=manifest.version,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return IngestResult(Path(out), manifest, read.report)


def load_dataset(
    source: str | Path,
    format: str,  # noqa: A002
    *,
    group_by: str | None = None,
    name: str | None = None,
    strict: bool = False,
    reader_options: Mapping[str, object] | None = None,
) -> tuple[Dataset, ConversionReport]:
    """Read ``source`` into a validated in-memory :class:`Dataset` without writing anything.

    The dataset has the same content version as :func:`ingest` would store for it and its ``root``
    is the source directory.

    Raises:
        FormatError, ValidationError, ConversionLossError, ValueError: as :func:`ingest`.
    """
    read = _read_source(
        source, format, group_by=group_by, name=name, strict=strict, options=reader_options
    )
    manifest = DatasetManifest(
        version=models_version(
            read.images, read.annotations, read.categories, schema_version=SCHEMA_VERSION
        ),
        schema_version=SCHEMA_VERSION,
        name=read.name,
        task=read.task,
        categories=tuple(sorted(read.categories, key=lambda c: c.id)),
        image_count=len(read.images),
        annotation_count=len(read.annotations),
        created_at=datetime.now(timezone.utc),
    )
    _log_loss(read.report)
    get_logger(__name__).info(
        "dataset_loaded",
        format=format,
        image_count=len(read.images),
        annotation_count=len(read.annotations),
        dataset_version=manifest.version,
    )
    dataset = Dataset(manifest, tuple(read.images), tuple(read.annotations), read.root)
    return dataset, read.report


MAX_EXPORT_IMAGE_BYTES = 1024**3
"""Largest image file an export will copy (1 GiB); the file is hashed before it is copied."""
_HASH_CHUNK = 1024 * 1024


def _verified_images(ds: Dataset, *, allow_external: bool = False) -> Dataset:
    """``ds`` with every image path replaced by the verified real path of its file.

    A dataset directory is untrusted input: ``Image.file_path`` can name any file on the machine
    (``/home/x/.ssh/id_rsa``, a symlink, ``..``). Before a writer copies anything, each path is
    resolved inside ``ds.root`` (outside paths, remote URIs and unsafe names are refused), opened
    without following a symlink, bounded in size, and its sha256 must start with ``image_id``. The
    whole export is refused if any image fails; nothing is written then.
    """
    if ds.root is None:
        raise FormatError("the dataset has no root directory: its images cannot be verified")
    checked: list[Image] = []
    bad = 0
    for image in ds.images:
        resolved = resolve_image_path(image.file_path, ds.root, allow_external=allow_external)
        if resolved.kind != "local" or resolved.path is None:
            bad += 1
            continue
        digest = hashlib.sha256()
        total = 0
        try:
            with files.open_regular(resolved.path, nofollow=True) as handle:
                while chunk := handle.read(_HASH_CHUNK):
                    total += len(chunk)
                    if total > MAX_EXPORT_IMAGE_BYTES:
                        break
                    digest.update(chunk)
        except (FormatError, OSError):
            bad += 1
            continue
        if total > MAX_EXPORT_IMAGE_BYTES or digest.hexdigest()[:16] != image.image_id:
            bad += 1
            continue
        checked.append(image.model_copy(update={"file_path": str(resolved.path)}))
    if bad:
        raise FormatError(
            f"{bad} image file(s) are missing, outside the dataset directory, changed since "
            "ingest, too large or not regular files: nothing was exported"
        )
    return Dataset(ds.manifest, tuple(checked), ds.annotations, ds.root)


def export(
    ds: Dataset,
    format: str,  # noqa: A002
    out: str | Path,
    *,
    strict: bool = False,
    allow_external_images: bool = False,
) -> ConversionReport:
    """Write ``ds`` in the named format into ``out`` (absent or an empty directory).

    Every image is verified first (see :func:`_verified_images`), so an export can never copy a
    file that is not the dataset's own picture. Images that lie outside the dataset's directory are
    refused unless ``allow_external_images`` is set (their content must still match ``image_id``).

    The writer runs in a sibling temporary directory that is renamed into place only after
    ``strict`` is satisfied, so a failure leaves no partial output.

    Raises:
        FormatError: unknown or write-incapable format, unsafe or non-empty output.
        ConversionLossError: ``strict`` and the conversion dropped or skipped anything.
    """
    started = time.perf_counter()
    writer = registry.get_writer(format)
    ds = _verified_images(ds, allow_external=allow_external_images)
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
