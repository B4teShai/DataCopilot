"""Atomic, append-only reads and writes of a ``dataset.dcp`` directory (ADR-0006).

Layout and crash rules: docs/architecture/03-data-model.md, "On-disk layout". A new dataset is
built in a sibling ``<name>.tmp-<uuid>`` directory and renamed into place; a new version of an
existing dataset is built in ``versions/<version>.tmp-<uuid>``, renamed, and only then are the
top-level tables and ``manifest.json`` replaced by rename. Readers ignore ``*.tmp-*``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError as PydanticValidationError

from datacopilot.core.errors import FormatError, StoreError, ValidationError
from datacopilot.core.proposals import ProposalSet
from datacopilot.core.schema import (
    SCHEMA_VERSION,
    Annotation,
    AuditRun,
    Category,
    Dataset,
    DatasetManifest,
    Image,
    Task,
    validate_tables,
)
from datacopilot.core.version import extend_lineage, models_version
from datacopilot.io import files
from datacopilot.settings import DEFAULT_MAX_TABLE_BYTES, DEFAULT_MAX_TABLE_ROWS
from datacopilot.store import arrow

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
"""A manifest holds the category list only; 16 MiB is far beyond any real one."""
MAX_FILE_BYTES = 1024**3
"""Compressed size cap per Parquet file (1 GiB). A 5M-row annotations table is 60 to 300 MB."""
MAX_AUDIT_BYTES = 256 * 1024 * 1024
"""Largest ``audits/<run_id>.json`` that is read back (a run capped at 50,000 findings per check
stays far below this)."""
MAX_PROPOSAL_BYTES = 128 * 1024 * 1024
"""Largest ``proposals/<set_id>.json`` that is read back (a full set of 200,000 proposals is about
70 MB)."""
BATCH_ROWS = 10_000
"""Rows converted to model objects at a time, so Arrow buffers are released incrementally."""


@dataclass(frozen=True)
class StoreLimits:
    """Resource limits applied to a table before any of its data is read.

    The row and uncompressed-size limits come from ``DATACOPILOT_MAX_TABLE_ROWS`` and
    ``DATACOPILOT_MAX_TABLE_BYTES`` (see settings.py); the defaults fit 5M annotations.
    """

    max_rows: int = DEFAULT_MAX_TABLE_ROWS
    max_uncompressed_bytes: int = DEFAULT_MAX_TABLE_BYTES
    max_file_bytes: int = MAX_FILE_BYTES


AUDITS_DIR = "audits"
PROPOSALS_DIR = "proposals"
_SET_ID = re.compile(r"^[0-9a-f]{32}$")
_VERSION = _SET_ID
_RUN_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
TABLES = ("images.parquet", "annotations.parquet", "categories.parquet")
MANIFEST = "manifest.json"

T = TypeVar("T")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tmp_name(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _write_tables(directory: Path, tables: dict[str, pa.Table], manifest: DatasetManifest) -> None:
    for name, table in tables.items():
        pq.write_table(table, directory / name, compression="zstd")
        _fsync_file(directory / name)
    (directory / MANIFEST).write_text(
        json.dumps(manifest.to_row(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _fsync_file(directory / MANIFEST)
    _fsync_dir(directory)


def _arrow_tables(
    images: Sequence[Image], annotations: Sequence[Annotation], categories: Sequence[Category]
) -> dict[str, pa.Table]:
    return {
        "images.parquet": arrow.images_to_table(images),
        "annotations.parquet": arrow.annotations_to_table(annotations),
        "categories.parquet": arrow.categories_to_table(categories),
    }


def _read_table(
    path: Path,
    expected: pa.Schema,
    convert: Callable[[pa.Table], list[T]],
    limits: StoreLimits,
) -> list[T]:
    """Read one table in batches after checking file size, schema and Parquet metadata.

    Every check uses the file footer only; no row data is decoded until all of them pass.
    """
    name = path.name
    try:
        with _open_regular(path) as handle:
            return _read_open_table(handle, name, expected, convert, limits)
    except StoreError:
        raise
    except (OSError, pa.ArrowException, PydanticValidationError, TypeError, ValueError) as exc:
        raise StoreError(f"cannot read {name}: {type(exc).__name__}") from exc


def _open_regular(path: Path) -> BinaryIO:
    """Open a store file without following symlinks; only regular files are accepted."""
    try:
        return files.open_regular(path, nofollow=True)
    except FormatError as exc:
        text = str(exc)
        kind = "is not a regular file" if "not a regular file" in text else "cannot be read"
        raise StoreError(f"{path.name} {kind}") from None


def _read_open_table(
    handle: BinaryIO,
    name: str,
    expected: pa.Schema,
    convert: Callable[[pa.Table], list[T]],
    limits: StoreLimits,
) -> list[T]:
    try:
        size = os.fstat(handle.fileno()).st_size
        if size > limits.max_file_bytes:
            raise StoreError(
                f"{name}: file size {size} bytes exceeds the limit of {limits.max_file_bytes}"
            )
        pf = pq.ParquetFile(handle)
        meta = pf.metadata
        if meta.num_rows > limits.max_rows:
            raise StoreError(
                f"{name}: {meta.num_rows} rows exceeds the limit of {limits.max_rows} "
                "(raise DATACOPILOT_MAX_TABLE_ROWS to allow it)"
            )
        raw = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        if raw > limits.max_uncompressed_bytes:
            raise StoreError(
                f"{name}: {raw} uncompressed bytes exceeds the limit of "
                f"{limits.max_uncompressed_bytes} (raise DATACOPILOT_MAX_TABLE_BYTES to allow it)"
            )
        if not pf.schema_arrow.equals(expected):
            raise StoreError(f"{name}: columns or types do not match the expected schema")
        rows: list[T] = []
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            rows.extend(convert(pa.Table.from_batches([batch], schema=expected)))
        return rows
    except StoreError:
        raise
    except (OSError, pa.ArrowException, PydanticValidationError, TypeError, ValueError) as exc:
        raise StoreError(f"cannot read {name}: {type(exc).__name__}") from exc


def _read_manifest(path: Path) -> DatasetManifest:
    file = path / MANIFEST
    try:
        handle = files.open_regular(file, nofollow=True)
    except FormatError as exc:
        if "not a regular file" in str(exc):
            raise StoreError(f"{MANIFEST} is not a regular file") from None
        raise StoreError(f"{path} is not a dataset: {MANIFEST} is missing") from None
    try:
        with handle:
            if os.fstat(handle.fileno()).st_size > MAX_MANIFEST_BYTES:
                raise StoreError(f"{MANIFEST} is too large")
            data = handle.read(MAX_MANIFEST_BYTES + 1)
        if len(data) > MAX_MANIFEST_BYTES:
            raise StoreError(f"{MANIFEST} is too large")
        raw = json.loads(data.decode("utf-8"))
        manifest = DatasetManifest.model_validate(raw)
    except (OSError, ValueError) as exc:  # JSONDecodeError and pydantic errors are ValueErrors
        raise StoreError(f"{MANIFEST} is invalid: {type(exc).__name__}") from None
    if manifest.schema_version > SCHEMA_VERSION:
        raise StoreError(
            f"dataset has schema_version {manifest.schema_version}; this package supports up to "
            f"{SCHEMA_VERSION}. Upgrade datacopilot."
        )
    return manifest


class DatasetStore:
    """Handle to an existing ``dataset.dcp`` directory."""

    def __init__(
        self, path: Path, manifest: DatasetManifest, limits: StoreLimits | None = None
    ) -> None:
        self.path = path
        self._manifest = manifest
        self.limits = limits or StoreLimits()

    @classmethod
    def open(cls, path: str | Path, *, limits: StoreLimits | None = None) -> DatasetStore:
        """Open an existing dataset and validate its manifest (tables are read lazily)."""
        root = Path(path)
        try:
            info = os.lstat(root)
        except OSError:
            raise StoreError(f"{root} is not a dataset directory") from None
        if stat.S_ISLNK(info.st_mode):
            # a link can be re-pointed (or point outside the checkout) between runs; callers
            # resolve the dataset's parent to find the images, so only a real directory is opened
            raise StoreError(f"{root} is a symlink; pass the dataset directory itself")
        if not stat.S_ISDIR(info.st_mode):
            raise StoreError(f"{root} is not a dataset directory")
        return cls(root, _read_manifest(root), limits)

    @classmethod
    def create(
        cls,
        path: str | Path,
        images: Sequence[Image],
        annotations: Sequence[Annotation],
        categories: Sequence[Category],
        *,
        name: str,
        task: Task,
        limits: StoreLimits | None = None,
        lineage: Sequence[str] = (),
    ) -> DatasetStore:
        """Write a brand-new dataset atomically; refuses to overwrite anything.

        ``lineage`` names the versions this dataset was derived from, oldest first (``apply`` passes
        its parent's lineage plus the parent's version); the parent itself is never touched.

        Raises:
            StoreError: if ``path`` exists.
            ValidationError: if the rows violate an invariant or ``lineage`` is not a list of
                distinct versions that excludes the new one (nothing is written).
        """
        root = Path(path)
        if root.exists():
            raise StoreError(f"{root} already exists; refusing to overwrite")
        validate_tables(images, annotations, categories, task=task)
        version = models_version(images, annotations, categories, schema_version=SCHEMA_VERSION)
        if (
            not all(isinstance(v, str) and _VERSION.fullmatch(v) for v in lineage)
            or len(set(lineage)) != len(lineage)
            or version in lineage
        ):
            raise ValidationError("lineage must list distinct versions and not the new version")
        manifest = DatasetManifest(
            version=version,
            schema_version=SCHEMA_VERSION,
            name=name,
            task=task,
            categories=tuple(sorted(categories, key=lambda c: c.id)),
            image_count=len(images),
            annotation_count=len(annotations),
            lineage=tuple(lineage),
            created_at=datetime.now(timezone.utc),
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_name(root)
        try:
            tmp.mkdir()
            _write_tables(tmp, _arrow_tables(images, annotations, categories), manifest)
            snapshot = tmp / "versions" / version
            snapshot.mkdir(parents=True)
            for file in (*TABLES, MANIFEST):
                shutil.copy2(tmp / file, snapshot / file)
            _fsync_dir(snapshot)
            _fsync_dir(tmp / "versions")
            if root.exists():
                raise StoreError(f"{root} already exists; refusing to overwrite")
            os.rename(tmp, root)
            _fsync_dir(root.parent)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return cls(root, manifest, limits)

    def manifest(self) -> DatasetManifest:
        return self._manifest

    def write_version(
        self,
        images: Sequence[Image],
        annotations: Sequence[Annotation],
        categories: Sequence[Category],
    ) -> DatasetManifest:
        """Append a new version: snapshot first, then replace the top-level files by rename.

        Raises:
            ValidationError: invariant violation, or content identical to the current version.
            StoreError: a snapshot of this content already exists in ``versions/``.
        """
        parent = self._manifest
        validate_tables(images, annotations, categories, task=parent.task)
        version = models_version(images, annotations, categories, schema_version=SCHEMA_VERSION)
        lineage = extend_lineage(parent, version)
        snapshot = self.path / "versions" / version
        if snapshot.exists():
            raise StoreError(f"version {version} already exists in this dataset")
        manifest = DatasetManifest(
            version=version,
            schema_version=SCHEMA_VERSION,
            name=parent.name,
            task=parent.task,
            categories=tuple(sorted(categories, key=lambda c: c.id)),
            image_count=len(images),
            annotation_count=len(annotations),
            lineage=lineage,
            created_at=datetime.now(timezone.utc),
        )
        tables = _arrow_tables(images, annotations, categories)
        tmp = _tmp_name(snapshot)
        try:
            tmp.mkdir(parents=True)
            _write_tables(tmp, tables, manifest)
            os.rename(tmp, snapshot)
            _fsync_dir(snapshot.parent)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        for file in (*TABLES, MANIFEST):  # manifest last: it names the version the tables belong to
            staged = _tmp_name(self.path / file)
            shutil.copy2(snapshot / file, staged)
            _fsync_file(staged)
            os.replace(staged, self.path / file)
        _fsync_dir(self.path)
        self._manifest = manifest
        return manifest

    def read(self, *, verify: bool = True) -> Dataset:
        """Load the current version.

        With ``verify`` (default) the tables are validated against the invariants and the
        recomputed content hash must equal ``manifest.version`` (invariant 9).

        Raises:
            StoreError: unreadable tables or a hash/manifest mismatch.
        """
        limits = self.limits
        images = _read_table(
            self.path / "images.parquet", arrow.IMAGES_SCHEMA, arrow.table_to_images, limits
        )
        annotations = _read_table(
            self.path / "annotations.parquet",
            arrow.ANNOTATIONS_SCHEMA,
            arrow.table_to_annotations,
            limits,
        )
        categories = _read_table(
            self.path / "categories.parquet",
            arrow.CATEGORIES_SCHEMA,
            arrow.table_to_categories,
            limits,
        )
        manifest = self._manifest
        for field_name, claimed, actual in (
            ("image_count", manifest.image_count, len(images)),
            ("annotation_count", manifest.annotation_count, len(annotations)),
        ):
            if claimed != actual:
                raise StoreError(
                    f"manifest {field_name} claims {claimed} but the table has {actual} rows"
                )
        if verify:
            validate_tables(images, annotations, categories, task=manifest.task)
            recomputed = models_version(
                images, annotations, categories, schema_version=manifest.schema_version
            )
            expected_categories = sorted(manifest.categories, key=lambda c: c.id)
            if recomputed != manifest.version or sorted(categories, key=lambda c: c.id) != list(
                expected_categories
            ):
                raise StoreError(
                    f"tables in {self.path} do not match manifest version {manifest.version}"
                )
        return Dataset(manifest, tuple(images), tuple(annotations), self.path.resolve().parent)

    def _audit_path(self, run_id: str) -> Path:
        """Path of ``audits/<run_id>.json``; ``run_id`` must be a lowercase UUID (no separators)."""
        if not _RUN_ID.fullmatch(run_id):
            raise StoreError("run_id must be a lowercase UUID")
        directory = self.path / AUDITS_DIR
        if directory.is_symlink():
            raise StoreError(f"{AUDITS_DIR} must be a directory inside the dataset, not a symlink")
        return directory / f"{run_id}.json"

    def write_audit(self, run: AuditRun) -> Path:
        """Persist ``run`` as ``audits/<run_id>.json``: temporary file, fsync, rename.

        Audits are derived data: the dataset version and its tables are not touched.

        Raises:
            StoreError: the run id is not a UUID, the file already exists, or ``audits`` is a
                symlink.
        """
        target = self._audit_path(run.run_id)
        if target.exists():
            raise StoreError(f"audit {run.run_id} already exists; refusing to overwrite")
        target.parent.mkdir(exist_ok=True)
        staged = _tmp_name(target)
        text = json.dumps(run.to_row(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        try:
            staged.write_text(text, encoding="ascii")
            _fsync_file(staged)
            os.replace(staged, target)
            _fsync_dir(target.parent)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return target

    def read_audit(self, run_id: str) -> AuditRun:
        """Load and validate a stored audit run.

        Raises:
            StoreError: invalid id, missing, oversized or malformed file.
        """
        file = self._audit_path(run_id)
        try:
            handle = files.open_regular(file, nofollow=True)
        except FormatError as exc:
            if "not a regular file" in str(exc):
                raise StoreError(f"audit {run_id} is not a regular file") from None
            raise StoreError(f"no audit {run_id} in this dataset") from None
        try:
            with handle:
                size = os.fstat(handle.fileno()).st_size
                if size > MAX_AUDIT_BYTES:
                    raise StoreError(f"audit {run_id} is too large ({size} bytes)")
                data = handle.read(MAX_AUDIT_BYTES + 1)
            if len(data) > MAX_AUDIT_BYTES:
                raise StoreError(f"audit {run_id} is too large")
            return AuditRun.model_validate_json(data)
        except (OSError, ValueError) as exc:
            raise StoreError(f"audit {run_id} is invalid: {type(exc).__name__}") from None

    def _proposal_path(self, set_id: str) -> Path:
        """Path of ``proposals/<set_id>.json``; ``set_id`` must be 32 lowercase hex digits."""
        if not _SET_ID.fullmatch(set_id):
            raise StoreError("a set id is 32 lowercase hex digits")
        directory = self.path / PROPOSALS_DIR
        if directory.is_symlink():
            raise StoreError(
                f"{PROPOSALS_DIR} must be a directory inside the dataset, not a symlink"
            )
        return directory / f"{set_id}.json"

    def write_proposals(self, ps: ProposalSet) -> Path:
        """Persist ``ps`` as ``proposals/<set_id>.json``: temporary file, fsync, rename.

        A set is content-addressed, so writing the same set again changes nothing. Proposals are
        derived data: the dataset version and its tables are not touched.

        Raises:
            StoreError: ``proposals`` is a symlink, or the directory cannot be written.
        """
        target = self._proposal_path(ps.set_id)
        if target.exists():
            return target
        target.parent.mkdir(exist_ok=True)
        staged = _tmp_name(target)
        try:
            staged.write_text(ps.model_dump_json() + "\n", encoding="ascii")
            _fsync_file(staged)
            os.replace(staged, target)
            _fsync_dir(target.parent)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return target

    def read_proposals(self, set_id: str) -> ProposalSet:
        """Load a stored set; its content must hash to the id it is stored (and asked for) under.

        Raises:
            StoreError: invalid id, missing, oversized, not a regular file, malformed or tampered.
        """
        file = self._proposal_path(set_id)
        try:
            handle = files.open_regular(file, nofollow=True)
        except FormatError as exc:
            if "not a regular file" in str(exc) or file.is_symlink():
                raise StoreError(f"proposal set {set_id} is not a regular file") from None
            raise StoreError(f"no proposal set {set_id} in this dataset") from None
        try:
            with handle:
                if os.fstat(handle.fileno()).st_size > MAX_PROPOSAL_BYTES:
                    raise StoreError(f"proposal set {set_id} is too large")
                data = handle.read(MAX_PROPOSAL_BYTES + 1)
            if len(data) > MAX_PROPOSAL_BYTES:
                raise StoreError(f"proposal set {set_id} is too large")
            ps = ProposalSet.model_validate_json(data)
        except (OSError, ValueError) as exc:
            raise StoreError(f"proposal set {set_id} is invalid: {type(exc).__name__}") from None
        if ps.set_id != set_id:
            raise StoreError(f"proposal set {set_id} does not match the id stored in the file")
        return ps
