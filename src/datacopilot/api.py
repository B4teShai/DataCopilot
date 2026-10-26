"""The public Python API: ``load``, ``audit``, ``export``, ``split``, ``propose`` and ``apply``.

Six thin functions over the use cases the command line uses, with the same validation, limits
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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from datacopilot import convert
from datacopilot.audit import engine
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.audit.embedding import DEFAULT_DTYPE, DEFAULT_EMBED_BATCH, PipelineProvider
from datacopilot.audit.registry import EMBEDDING_CHECKS
from datacopilot.audit.usecase import check_threshold, embedder_label, resolve_embedder
from datacopilot.core.proposals import ProposalSet
from datacopilot.core.schema import AuditRun, Dataset, DatasetManifest, SplitReport
from datacopilot.io.report import ConversionReport
from datacopilot.logging import clear_request_context, configure_logging, current_request_id
from datacopilot.proposals import run as proposal_run
from datacopilot.proposals import usecase as proposal_usecase
from datacopilot.settings import Settings, load_settings
from datacopilot.splits import usecase as split_usecase
from datacopilot.store import DatasetStore, StoreLimits

__all__ = ["apply", "audit", "export", "load", "propose", "split"]

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
            ``label_noise``, ``outliers``) are included in ``"all"`` only with an ``embedder``.
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


def propose(
    dataset: str | Path,
    proposer: str = "grounding_dino",
    *,
    classes: Sequence[str] | None = None,
    min_score: float = proposal_run.DEFAULT_MIN_SCORE,
    class_min_score: Mapping[str, float] | None = None,
    time_budget_seconds: float = proposal_run.DEFAULT_TIME_BUDGET_SECONDS,
) -> ProposalSet:
    """Find boxes the dataset is probably missing and store them as a proposal set.

    Nothing is applied and the dataset's version is not changed: the set is written to the
    dataset directory's ``proposals/`` and returned. Review it (``datacopilot review export`` and
    ``review import``), then :func:`apply` the reviewed set.

    Args:
        dataset: the path of a ``.dcp`` directory (the set has to be stored somewhere).
        proposer: ``grounding_dino`` (needs the extra ``datacopilot[vlm]``; the license guard
            runs first) or ``test_boxes``, a test double.
        classes: class names to look for (default: all).
        min_score: lowest confidence kept, above 0 and at most 1.
        class_min_score: ``class name -> lowest confidence`` for single classes.
        time_budget_seconds: wall time after which the run stops; the set is then marked
            incomplete (``ProposalSet.incomplete``) and ``apply`` refuses it unless asked.

    Raises:
        ValueError: an unknown proposer or class, an invalid option, or an in-memory dataset.
        LicenseError: the model's license is not in ``DATACOPILOT_ALLOWED_LICENSES``.
        FormatError, ValidationError, StoreError, ModelError: unusable proposer, dataset or model.
    """
    if isinstance(dataset, Dataset):
        raise ValueError("propose needs the path of a .dcp directory: the set is stored there")
    settings = load_settings()
    with _request(settings):
        ps, _ = proposal_usecase.propose_dataset(
            Path(dataset),
            proposer,
            settings=settings,
            classes=classes,
            min_score=min_score,
            class_min_score=class_min_score,
            time_budget_seconds=time_budget_seconds,
        )
        return ps


def apply(
    dataset: str | Path,
    proposals: str,
    out: str | Path,
    *,
    allow_incomplete: bool = False,
    allow_test_proposer: bool = False,
) -> DatasetManifest:
    """Write the accepted proposals of a reviewed set as a NEW dataset; the original is untouched.

    The new dataset records the old version in ``lineage``; each new box has provenance
    ``proposal_accepted``. Pending and rejected proposals are never applied.

    Args:
        dataset: the path of the ``.dcp`` directory the set was made for.
        proposals: the set id (``ProposalSet.set_id`` of a reviewed set).
        out: a new ``.dcp`` path outside ``dataset``.
        allow_incomplete: also apply a set from a run that did not reach every image.
        allow_test_proposer: also apply a set made by a test double such as ``test_boxes``.

    Returns:
        The new dataset's manifest.

    Raises:
        ValidationError: a set for another version, made by another model than the pinned one,
            incomplete (without ``allow_incomplete``), with nothing accepted, or an output inside
            the source.
        LicenseError, FormatError: the model is not allowed or its proposer not installed.
        StoreError: unknown or tampered set, or the output exists.
    """
    settings = load_settings()
    with _request(settings):
        return proposal_usecase.apply_set(
            Path(dataset),
            proposals,
            Path(out),
            settings=settings,
            allow_incomplete=allow_incomplete,
            allow_test_proposer=allow_test_proposer,
        )


def split(
    dataset: str | Path,
    out: str | Path,
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    by_group: str = "group_id",
    near_duplicates: bool = False,
    embedder: str | None = None,
    similarity_threshold: float | None = None,
    stratify: Literal["category"] | None = None,
    seed: int = 0,
    allow_test_embedder: bool = False,
) -> SplitReport:
    """Re-split a ``.dcp`` dataset by group and write a NEW dataset; the original is untouched.

    Images with the same ``by_group`` key and, with ``near_duplicates``, near copies (every pair
    whose embeddings reach the threshold) form groups; whole groups are assigned, largest first,
    to the split that keeps the image counts (and with ``stratify="category"`` the class counts)
    closest to ``ratios``. The same input, ratios and seed give the same assignment whatever the
    row order. The new dataset records the old version in ``lineage``.

    Args:
        dataset: the path of a ``.dcp`` directory.
        out: a new ``.dcp`` path outside ``dataset``.
        ratios: train, val and test fractions that sum to 1.
        by_group: ``group_id``, ``source.<name>`` or ``re:<pattern>`` over the file name (refused
            unless it is bounded and cannot backtrack badly); the new ``group_id`` is the key. An
            existing ``group_id`` always stays together too.
        near_duplicates: also keep near copies together; needs ``embedder``, and every image must
            be embedded (otherwise the split is refused).
        embedder: an embedder name (the license guard runs first).
        similarity_threshold: cosine similarity of near copies, above 0 and at most 1 (0.95).
        stratify: ``"category"`` balances each class across the splits too.
        seed: breaks ties between groups of equal size.
        allow_test_embedder: permit a test-only embedder (``hash``, which compares file bytes,
            not pictures); refused otherwise.

    Returns:
        The :class:`~datacopilot.core.schema.SplitReport`: the new version, the groups, and the
        deviation from the ratios per split and per class.

    Raises:
        ValueError: bad ratios, key, seed or threshold, near duplicates without an embedder, or a
            test-only embedder without ``allow_test_embedder``.
        ValidationError: an image without an embedding, too many near-copy pairs, a split equal to
            the current or an earlier version, or an output inside the source or in a directory
            that does not hold the images.
        StoreError: the output exists or the dataset is unreadable.
        LicenseError, FormatError: the embedder is not allowed or unusable.
    """
    if stratify not in (None, "category"):
        raise ValueError("stratify must be None or 'category'")
    settings = load_settings()
    with _request(settings):
        return split_usecase.split_dataset(
            Path(dataset),
            Path(out),
            settings=settings,
            ratios=ratios,
            group_key=by_group,
            near_duplicates=near_duplicates,
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            stratify=stratify == "category",
            seed=seed,
            allow_test_embedder=allow_test_embedder,
        )
