"""The public Python API: ``load``, ``audit`` and ``export``.

Three thin functions over the use cases the command line uses, with the same validation, limits
and fail-closed behaviour. Everything importable from the top-level ``datacopilot`` package and
listed in ``datacopilot.__all__`` is the public API; anything else (including this module's
helpers and the sub-packages) may change in any release. Before 1.0 a minor version may break the
API; ``CHANGELOG.md`` says so, a deprecation warning precedes a removal when possible
(:mod:`datacopilot.deprecation`) and ``tests/unit/test_api_surface.py`` fails on any accidental
change of a name or signature.

Library behaviour: nothing is printed to standard output, nothing is written next to a dataset
that you did not ask for, and logs are the same JSON records as the command line, on standard
error. If no request id is bound, a call configures logging for its duration and clears it
afterwards; to control logging yourself call :func:`datacopilot.logging.configure_logging` first.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from datacopilot import convert
from datacopilot.audit import engine
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.audit.embedding import DEFAULT_DTYPE, DEFAULT_EMBED_BATCH, PipelineProvider
from datacopilot.audit.registry import EMBEDDING_CHECKS
from datacopilot.audit.usecase import check_threshold, embedder_label, resolve_embedder
from datacopilot.core.schema import AuditRun, Dataset
from datacopilot.io.report import ConversionReport
from datacopilot.logging import clear_request_context, configure_logging, current_request_id
from datacopilot.settings import Settings, load_settings
from datacopilot.store import DatasetStore, StoreLimits

__all__ = ["audit", "export", "load"]

DCP_FORMAT = "dcp"


@contextmanager
def _request(settings: Settings) -> Iterator[None]:
    """Bind a request id (and JSON logging) for one call unless the caller already did."""
    if current_request_id() is not None:
        yield
        return
    configure_logging("datacopilot", settings)
    try:
        yield
    finally:
        clear_request_context()


def _limits(settings: Settings) -> StoreLimits:
    return StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )


def _is_dcp(path: Path) -> bool:
    return path.suffix == ".dcp"


def _read_dcp(path: Path, settings: Settings) -> Dataset:
    ds = DatasetStore.open(path, limits=_limits(settings)).read()
    return Dataset(ds.manifest, ds.images, ds.annotations, path.resolve().parent)


def load(
    path: str | Path,
    format: str | None = None,  # noqa: A002  # same name as the command line option
    *,
    group_by: str | None = None,
    strict: bool = False,
) -> Dataset:
    """Read a dataset into memory; nothing is written.

    Args:
        path: a source dataset (a directory or annotation file) or a ``.dcp`` directory.
        format: the source format (see ``datacopilot formats``); required for a source dataset,
            omitted or ``"dcp"`` for a ``.dcp`` directory.
        group_by: regular expression over the image path whose group ``g`` (or 1) is the group id
            (source datasets only).
        strict: fail with :class:`~datacopilot.core.errors.ConversionLossError` if the reader
            skipped or dropped anything (an unsupported shape, a missing image).

    Returns:
        The validated dataset; ``Dataset.root`` is the directory its images must lie in.

    Raises:
        ValueError: ``format`` missing or inconsistent with ``path``; an invalid ``group_by``.
        FormatError, ValidationError, StoreError, ConversionLossError: unreadable, invalid or
            (with ``strict``) lossy input; all are ``DataCopilotError``.
    """
    target = Path(path)
    settings = load_settings()
    with _request(settings):
        if _is_dcp(target):
            if format not in (None, DCP_FORMAT):
                raise ValueError("a .dcp directory is read with format 'dcp' or without a format")
            return _read_dcp(target, settings)
        if format is None or format == DCP_FORMAT:
            raise ValueError("format is required for a source dataset (see `datacopilot formats`)")
        dataset, _ = convert.load_dataset(target, format, group_by=group_by, strict=strict)
        return dataset


def export(
    dataset: Dataset | str | Path,
    format: str,  # noqa: A002
    out: str | Path,
    *,
    strict: bool = False,
) -> ConversionReport:
    """Write ``dataset`` in another format into ``out`` (absent or an empty directory).

    Args:
        dataset: a loaded dataset or the path of a ``.dcp`` directory.
        format: the target format (see ``datacopilot formats``).
        out: output directory.
        strict: fail, writing nothing, if any field cannot be carried over.

    Returns:
        The :class:`~datacopilot.io.report.ConversionReport`: every dropped value and skipped row,
        counted.

    Raises:
        FormatError: unknown or write-incapable format, unsafe or non-empty output.
        ConversionLossError: ``strict`` and the conversion lost something.
    """
    settings = load_settings()
    with _request(settings):
        ds = _read_dcp(Path(dataset), settings) if isinstance(dataset, (str, Path)) else dataset
        return convert.export(ds, format, out, strict=strict)


def audit(
    dataset: Dataset | str | Path,
    checks: Sequence[str] | Literal["all"] = "all",
    *,
    fail_on: Sequence[str] = ("critical",),
    embedder: str | None = None,
    embed_batch: int | None = None,
    similarity_threshold: float | None = None,
    allow_test_embedder: bool = False,
    root: str | Path | None = None,
    workers: int | None = None,
    require_local_images: bool = False,
    include_model_annotations: bool = False,
) -> AuditRun:
    """Run the audit checks over ``dataset`` and return the run; nothing is written.

    The hash cache is not used and images outside ``root`` are never read. The checks, limits and
    gate evaluation are those of the command line, but a failed gate is **recorded** in
    ``AuditRun.passed`` and does not raise (the command line turns it into exit code 1): call
    ``run.raise_for_gate()`` where a failure must stop the program.

    Args:
        dataset: a loaded dataset or the path of a ``.dcp`` directory (whose embedding cache is
            then used with an embedder).
        checks: ``"all"`` or check names; the embedding checks (``near_duplicates``, ``leakage``,
            ``outliers``) are included in ``"all"`` only with an ``embedder``.
        fail_on: ``critical``, ``warn``, ``info`` (that severity and above) and/or check names;
            ``()`` applies no gate.
        embedder: an embedder name (``dinov2_small``, ``dinov2_base``, ``siglip2_base``, ``hash``);
            the license guard (``DATACOPILOT_ALLOWED_LICENSES``) runs before any work.
        embed_batch: images per embedder call.
        similarity_threshold: cosine similarity above 0 and at most 1 for near-duplicates and
            leakage (default 0.95); needs ``embedder``.
        allow_test_embedder: permit a test-only embedder (``hash``) to drive a gate.
        root: directory the image paths must lie in; defaults to ``Dataset.root`` (set by
            :func:`load`) or the parent of a ``.dcp`` directory.
        workers: threads for the file checks (default: ``DATACOPILOT_WORKERS`` or the CPU count).
        require_local_images: treat any image that could not be examined as CRITICAL; use it
            wherever the result gates something (CI), because the default tolerates a minority of
            remote or external images as a WARN.
        include_model_annotations: also audit rows with ``provenance == "model"`` (for example
            Label Studio predictions); by default they are left out of every check and counted in
            ``run.summary.excluded_model_annotations``.

    Raises:
        ValueError: invalid ``checks``, ``fail_on``, ``embedder`` or threshold.
        LicenseError: the embedder's license is not allowed.
        FormatError, ValidationError, ModelError: unknown check, empty dataset, unusable model.
    """
    settings = load_settings()
    with _request(settings):
        dcp_dir: Path | None = None
        if isinstance(dataset, (str, Path)):
            dcp_dir = Path(dataset).resolve()
            ds = _read_dcp(Path(dataset), settings)
        else:
            ds = dataset
        names = engine.select_checks(
            "all" if checks == "all" else list(checks), with_embedder=embedder is not None
        )
        if embedder is None and (embed_batch is not None or similarity_threshold is not None):
            raise ValueError("embed_batch and similarity_threshold need an embedder")
        if embedder is not None and not set(names) & EMBEDDING_CHECKS:
            raise ValueError("an embedder is given but no selected check uses embeddings")
        check_threshold(similarity_threshold)
        gate = engine.parse_fail_on(list(fail_on))
        for token in gate:
            if token not in ("critical", "warn", "info") and token not in names:
                raise ValueError(engine.not_in_run_message(token, embedder is not None))
        provider = None
        created = None
        if embedder is not None:
            created = resolve_embedder(embedder, settings.allowed_licenses)
            if getattr(created, "test_only", False) and gate and not allow_test_embedder:
                raise ValueError(
                    f"{embedder!r} is test-only and is refused for gating; pass "
                    "allow_test_embedder=True or fail_on=()"
                )
            provider = PipelineProvider(
                created,
                batch_size=embed_batch if embed_batch is not None else DEFAULT_EMBED_BATCH,
                dtype=DEFAULT_DTYPE,
                use_cache=dcp_dir is not None,
            )
        base = Path(root) if root is not None else ds.root
        ctx = CheckContext(
            run_id=current_request_id() or str(uuid.uuid4()),
            root=(base if base is not None else Path.cwd()).resolve(),
            cache_dir=settings.cache_dir,
            workers=workers if workers is not None else settings.workers,
            use_hash_cache=False,
            cache_key=settings.cache_key,
            in_ci=settings.ci,
            require_local_images=require_local_images,
            include_model_annotations=include_model_annotations,
            dataset_dir=dcp_dir,
            embedder=None if created is None else embedder_label(created),
            embeddings=provider,
            config=(
                CheckConfig()
                if similarity_threshold is None
                else CheckConfig(
                    near_duplicate_cosine=similarity_threshold, leakage_cosine=similarity_threshold
                )
            ),
        )
        return engine.run_audit(ds, names, ctx, gate)
