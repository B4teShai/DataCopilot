"""Command-line entry point."""

from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from datacopilot import __version__
from datacopilot.core.errors import DataCopilotError
from datacopilot.logging import (
    clear_request_context,
    configure_logging,
    current_request_id,
    get_logger,
)
from datacopilot.settings import Settings, SettingsError, load_settings

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, help="DataCopilot")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"datacopilot {__version__}")
        raise typer.Exit()


_settings_error: str | None = None
"""Set by the root callback for ``doctor`` only, which reports a bad environment instead of
refusing to start (every other command exits 2 at once)."""


@app.callback()
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    global _settings_error  # noqa: PLW0603
    _settings_error = None
    try:
        settings = load_settings()
    except SettingsError as exc:
        if ctx.invoked_subcommand != "doctor":
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None
        _settings_error = str(exc)
        settings = Settings()
    configure_logging("datacopilot", settings)


def _run_doctor() -> dict[str, Any]:
    """Diagnose the installation, log one ``doctor_completed`` record and return the report."""
    from datacopilot import doctor as diagnosis

    error = _settings_error
    report = diagnosis.diagnose(None if error else load_settings(), settings_error=error)
    get_logger("datacopilot.cli").info(
        "doctor_completed",
        status=report["status"],
        package_version=__version__,
        python_version=platform.python_version(),
    )
    return report


@app.command()
def doctor(
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the report as one JSON object.")
    ] = False,
) -> None:
    """Check the installation: versions, extras, settings, cache directory, FAISS and GPU.

    Prints the status (`ok`, `warn` or `fail`) and one line per section, or with --json one JSON
    object (schema 1). Never prints secrets. Exit codes: 0 ok or warn; 1 a core dependency is
    missing; 2 an invalid DATACOPILOT_* setting (reported in the output).
    """
    from datacopilot import doctor as diagnosis

    report = _run_doctor()
    typer.echo(diagnosis.to_json(report) if json_out else diagnosis.to_text(report))
    code = diagnosis.exit_code(report)
    if code:
        raise typer.Exit(code)


FormatOption = Annotated[
    str, typer.Option("--format", "-f", help="Format name; see `datacopilot formats`.")
]


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Dataset root (or annotation file) to read.")],
    format: FormatOption,  # noqa: A002
    output: Annotated[Path, typer.Option("--output", "-o", help="New dataset directory (*.dcp).")],
    group_by: Annotated[
        str | None,
        typer.Option(
            "--group-by", help="Regex over the image path; group `g` (or 1) is the group id."
        ),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Dataset name (default: input name).")] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict", help="Fail (nothing is written) if the reader skipped or dropped anything."
        ),
    ] = False,
    keep_task_data: Annotated[
        bool,
        typer.Option(
            "--keep-task-data",
            help="label_studio only: also keep the task data fields other than `image` (they "
            "are dropped and counted by default because they can hold personal data or URLs).",
        ),
    ] = False,
) -> None:
    """Convert a dataset into a versioned dataset.dcp directory and print its version.

    Exit codes: 0 success; 1 unreadable or invalid input, unknown format, output exists, or
    --strict with skipped or dropped input (nothing is written); 2 bad option or configuration.
    """
    from datacopilot import convert
    from datacopilot.store import StoreLimits

    if output.suffix != ".dcp":
        raise typer.BadParameter("must end with .dcp", param_hint="--output")
    if keep_task_data and format != "label_studio":
        raise typer.BadParameter(
            "only valid with --format label_studio", param_hint="--keep-task-data"
        )
    if group_by is not None:
        try:
            convert.parse_group_by(group_by)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--group-by") from None
    settings = load_settings()
    limits = StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )
    result = convert.ingest(
        path,
        format,
        output,
        group_by=group_by,
        name=name,
        limits=limits,
        strict=strict,
        reader_options={"keep_task_data": True} if keep_task_data else None,
    )
    typer.echo(result.manifest.version)


@app.command()
def export(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    format: FormatOption,  # noqa: A002
    output: Annotated[Path, typer.Option("--output", "-o", help="Empty or new output directory.")],
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail if any field cannot be carried to the format.")
    ] = False,
    allow_external_images: Annotated[
        bool,
        typer.Option(
            "--allow-external-images",
            help="Also copy images whose real path is outside the dataset's parent directory "
            "(each must still match its image_id). Do not use on datasets you do not trust.",
        ),
    ] = False,
) -> None:
    """Write a dataset.dcp in another format and print the output directory.

    Every image is verified before it is copied: it must lie inside the directory that holds the
    dataset (unless --allow-external-images), not be a symlink to elsewhere and match its image_id;
    otherwise nothing is exported.

    Exit codes: 0 success; 1 unreadable dataset, unknown format, non-empty output, or
    --strict with dropped fields (nothing is written); 2 bad option or configuration.
    """
    from datacopilot import convert
    from datacopilot.store import DatasetStore, StoreLimits

    settings = load_settings()
    limits = StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )
    ds = DatasetStore.open(dataset, limits=limits).read()
    convert.export(ds, format, output, strict=strict, allow_external_images=allow_external_images)
    typer.echo(str(output))


def _split_option(value: str, option: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    if not all(parts):
        raise typer.BadParameter("empty value in the list", param_hint=option)
    return parts


MAX_EMBED_BATCH_OPTION = 1024


def _resolve_embedder(name: str, allowed_licenses: frozenset[str]) -> Any:
    """Create the named embedder after the license guard; usage errors exit 2, license and
    missing-extra errors exit 1. No weights are touched here (the model loads on first use)."""
    from datacopilot.audit.usecase import resolve_embedder

    try:
        return resolve_embedder(name, allowed_licenses)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--embedder") from None


def _check_threshold(value: float | None) -> None:
    from datacopilot.audit.usecase import check_threshold

    try:
        check_threshold(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--similarity-threshold") from None


@app.command()
def audit(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    checks: Annotated[
        str, typer.Option("--checks", help="`all` or a comma-separated list of check names.")
    ] = "all",
    fail_on: Annotated[
        str | None,
        typer.Option(
            "--fail-on",
            help=(
                "Comma-separated gate: critical, warn, info (that severity and above) and/or "
                "check names; exit 1 if any finding matches. Default: critical. `none` applies "
                "no gate."
            ),
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="HTML report path (default ./report.html); JSON and SARIF go next to it.",
        ),
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON report path.")] = None,
    sarif: Annotated[Path | None, typer.Option("--sarif", help="SARIF 2.1.0 report path.")] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, max=256, help="Threads for file checks (default: CPUs)."),
    ] = None,
    hash_cache: Annotated[
        bool | None,
        typer.Option(
            "--hash-cache/--no-hash-cache",
            help="Use the file-hash cache (default: on). CI gates should pass --no-hash-cache.",
        ),
    ] = None,
    require_local_images: Annotated[
        bool,
        typer.Option(
            "--require-local-images",
            help="Treat any image the audit could not examine (remote, external) as CRITICAL.",
        ),
    ] = False,
    allow_external_images: Annotated[
        bool,
        typer.Option(
            "--allow-external-images",
            help="Also read images whose real path is outside the dataset's parent directory.",
        ),
    ] = False,
    no_store_audit: Annotated[
        bool,
        typer.Option(
            "--no-store-audit",
            help="Do not write audits/<run_id>.json into the dataset (read-only checkouts).",
        ),
    ] = False,
    allow_external_images_in_ci: Annotated[
        bool,
        typer.Option(
            "--allow-external-images-in-ci",
            help="Confirm --allow-external-images when the CI environment variable is set.",
        ),
    ] = False,
    embedder: Annotated[
        str | None,
        typer.Option(
            "--embedder",
            help=(
                "Embedder name (for example dinov2_base, siglip2_base; needs the [embed] extra): "
                "adds the near_duplicates, leakage, label_noise and outliers checks to "
                "`--checks all`."
            ),
        ),
    ] = None,
    embed_batch: Annotated[
        int | None,
        typer.Option(
            "--embed-batch", min=1, max=MAX_EMBED_BATCH_OPTION, help="Images per embedder call."
        ),
    ] = None,
    similarity_threshold: Annotated[
        float | None,
        typer.Option(
            "--similarity-threshold",
            help="Cosine similarity (above 0, at most 1) from which images are near-duplicates "
            "and cross-split pairs are leakage (default 0.95).",
        ),
    ] = None,
    embedding_cache: Annotated[
        bool,
        typer.Option(
            "--embedding-cache/--no-embedding-cache",
            help="Keep embeddings in dataset.dcp/embeddings (authenticated; default: on).",
        ),
    ] = True,
    allow_test_embedder: Annotated[
        bool,
        typer.Option(
            "--allow-test-embedder",
            help="Allow a test-only embedder such as `hash` (it compares file bytes, not pictures) "
            "to drive a gate. Without it such an embedder is refused unless `--fail-on none`.",
        ),
    ] = False,
    require_cache_key: Annotated[
        bool,
        typer.Option(
            "--require-cache-key",
            help="Fail (exit 2) unless DATACOPILOT_CACHE_KEY is set, whether or not CI is "
            "detected (only the CI variable is looked at). Use it where the cache directory is "
            "shared or restored from elsewhere.",
        ),
    ] = False,
    include_model_annotations: Annotated[
        bool,
        typer.Option(
            "--include-model-annotations",
            help="Also audit annotations whose provenance is `model` (for example Label Studio "
            "predictions). By default they are left out of every check, because model output is "
            "not ground truth, and the number left out is reported.",
        ),
    ] = False,
    show_group_labels: Annotated[
        bool,
        typer.Option(
            "--show-group-labels",
            help="Show group ids in leakage findings (default: only a hashed reference; group ids "
            "come from file paths and can carry names).",
        ),
    ] = False,
) -> None:
    """Audit a dataset.dcp, write reports and exit with the gate's verdict.

    Prints one line (`passed|failed critical=N warn=N info=N run_id=ID`) and stores the run as
    audits/<run_id>.json. The default gate is `--fail-on critical`; `--fail-on none` applies no
    gate. Exit codes: 0 gate passed (or no gate); 1 a --fail-on match, an empty dataset or an
    unreadable dataset, an image the leakage check could not embed (under a gate), an embedder
    that is not allowed (DATACOPILOT_ALLOWED_LICENSES) or fails
    to load; 2 bad option or configuration (unknown check, bad --fail-on, bad value).
    """
    from datacopilot.audit import engine
    from datacopilot.audit.context import CheckConfig, CheckContext
    from datacopilot.core.errors import FormatError
    from datacopilot.core.schema import TEST_ONLY_SUFFIX
    from datacopilot.report import render
    from datacopilot.store import DatasetStore, StoreLimits

    settings = load_settings()
    try:
        selected = "all" if checks.strip() == "all" else _split_option(checks, "--checks")
        names = engine.select_checks(selected, with_embedder=embedder is not None)
        if embedder is None and (embed_batch is not None or similarity_threshold is not None):
            raise ValueError("--embed-batch and --similarity-threshold need --embedder")
        if embedder is not None and not set(names) & registry_embedding_checks():
            raise ValueError("--embedder is given but no selected check uses embeddings")
        _check_threshold(similarity_threshold)
        if require_cache_key and (embedder is None or settings.cache_key is None):
            raise ValueError("--require-cache-key needs --embedder and DATACOPILOT_CACHE_KEY")
        gate = engine.parse_fail_on(
            _split_option("critical" if fail_on is None else fail_on, "--fail-on")
        )
        for token in gate:
            if token not in ("critical", "warn", "info") and token not in names:
                raise ValueError(engine.not_in_run_message(token, embedder is not None))
    except FormatError as exc:
        raise typer.BadParameter(str(exc), param_hint="--checks") from None
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--fail-on/--checks") from None
    html_path = report if report is not None else Path("report.html")
    targets = {
        "html": html_path,
        "json": json_out if json_out is not None else html_path.with_suffix(".json"),
        "sarif": sarif if sarif is not None else html_path.with_suffix(".sarif"),
    }
    seen: dict[str, str] = {}
    for fmt, target in targets.items():
        key = os.path.realpath(target)
        if sys.platform in ("darwin", "win32"):  # case-insensitive by default
            key = key.casefold()
        if key in seen:
            raise typer.BadParameter(
                f"the {seen[key]} and {fmt} reports would be written to the same file",
                param_hint="--report/--json/--sarif",
            )
        seen[key] = fmt
    if allow_external_images:
        if settings.ci and not allow_external_images_in_ci:
            raise typer.BadParameter(
                "--allow-external-images is refused in CI unless --allow-external-images-in-ci "
                "is also given (a hostile dataset could point the audit at files on the runner)",
                param_hint="--allow-external-images",
            )
        typer.echo(
            "warning: --allow-external-images reads image files outside the dataset's parent "
            "directory; do not use it on datasets you do not trust",
            err=True,
        )
    limits = StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )
    provider = None
    created = None
    if embedder is not None:  # the license guard runs before the dataset is even read
        from datacopilot.audit.embedding import DEFAULT_DTYPE, DEFAULT_EMBED_BATCH, PipelineProvider

        created = _resolve_embedder(embedder, settings.allowed_licenses)
        if getattr(created, "test_only", False):
            typer.echo(
                f"warning: embedder {embedder!r} is a test double that compares file bytes, not "
                "pictures; its verdicts say nothing about look-alike images",
                err=True,
            )
            if gate and not allow_test_embedder:
                raise typer.BadParameter(
                    f"{embedder!r} is test-only and is refused for gating; pass "
                    "--allow-test-embedder or use --fail-on none",
                    param_hint="--embedder",
                )
        provider = PipelineProvider(
            created,
            batch_size=embed_batch if embed_batch is not None else DEFAULT_EMBED_BATCH,
            dtype=DEFAULT_DTYPE,
            use_cache=embedding_cache,
        )
    store = DatasetStore.open(dataset, limits=limits)
    ds = store.read()
    run_id = current_request_id()
    if run_id is None:  # configure_logging always binds one; never invent a second identity
        raise RuntimeError("request_id is not bound")
    ctx = CheckContext(
        run_id=run_id,
        root=dataset.resolve().parent,
        cache_dir=settings.cache_dir,
        workers=workers if workers is not None else settings.workers,
        use_hash_cache=settings.hash_cache if hash_cache is None else hash_cache,
        cache_key=settings.cache_key,
        in_ci=settings.ci,
        allow_external_images=allow_external_images,
        require_local_images=require_local_images,
        dataset_dir=dataset.resolve(),
        show_group_labels=show_group_labels,
        include_model_annotations=include_model_annotations,
        embedder=None
        if created is None
        else (
            f"{created.model_id}@{created.revision[:12]}"
            + (TEST_ONLY_SUFFIX if getattr(created, "test_only", False) else "")
        ),
        embeddings=provider,
        config=(
            CheckConfig()
            if similarity_threshold is None
            else CheckConfig(
                near_duplicate_cosine=similarity_threshold, leakage_cosine=similarity_threshold
            )
        ),
    )
    result = engine.run_audit(ds, names, ctx, gate)
    render(result, html=targets["html"], json=targets["json"], sarif=targets["sarif"])
    if not no_store_audit:
        try:
            store.write_audit(result)
        except OSError as exc:  # a read-only checkout must not hide the verdict
            get_logger("datacopilot.cli").warning("audit_not_stored", reason=type(exc).__name__)
    counts = result.counts
    typer.echo(
        f"{'passed' if result.passed else 'failed'} critical={counts.critical} "
        f"warn={counts.warn} info={counts.info} run_id={result.run_id}"
    )
    excluded = result.summary.excluded_model_annotations if result.summary is not None else 0
    if excluded:
        typer.echo(
            f"note: {excluded} model-generated annotation{'s' if excluded != 1 else ''} "
            "(provenance: model) excluded from every check; pass --include-model-annotations "
            "to audit them",
            err=True,
        )
    if not result.passed:
        raise typer.Exit(1)


def registry_embedding_checks() -> frozenset[str]:
    from datacopilot.audit.registry import EMBEDDING_CHECKS

    return EMBEDDING_CHECKS


@app.command()
def embed(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    embedder: Annotated[
        str, typer.Option("--embedder", help="Embedder name (needs the [embed] extra).")
    ],
    embed_batch: Annotated[
        int | None,
        typer.Option(
            "--embed-batch", min=1, max=MAX_EMBED_BATCH_OPTION, help="Images per embedder call."
        ),
    ] = None,
    embedding_cache: Annotated[
        bool,
        typer.Option(
            "--embedding-cache/--no-embedding-cache",
            help="Store the embeddings in dataset.dcp/embeddings (default: on).",
        ),
    ] = True,
    require_cache_key: Annotated[
        bool,
        typer.Option(
            "--require-cache-key",
            help="Fail (exit 2) unless DATACOPILOT_CACHE_KEY is set.",
        ),
    ] = False,
) -> None:
    """Compute and store the embeddings of every image so that `audit --embedder` can reuse them.

    Resumable: shards already stored (and verified) are kept. Prints one line
    (`embedded=N total=T cached=C missing=M model=ID@REV run_id=ID`). Exit codes: 0 every image
    has an embedding and the cache was written; 1 any image could not be embedded, the embedder
    failed, the budget ended the run, the cache could not be written, an embedder that is not
    allowed or an unreadable dataset; 2 bad option, or a cache that cannot be used (for example
    `CI` set without `DATACOPILOT_CACHE_KEY`, or `--no-embedding-cache`): nothing would be stored.
    """
    from datacopilot.audit.context import CheckContext
    from datacopilot.audit.embedding import DEFAULT_DTYPE, DEFAULT_EMBED_BATCH, PipelineProvider
    from datacopilot.store import DatasetStore, StoreLimits

    settings = load_settings()
    if require_cache_key and settings.cache_key is None:
        raise typer.BadParameter(
            "DATACOPILOT_CACHE_KEY is not set", param_hint="--require-cache-key"
        )
    created = _resolve_embedder(embedder, settings.allowed_licenses)
    if getattr(created, "test_only", False):
        typer.echo(
            f"warning: embedder {embedder!r} is a test double that compares file bytes, not "
            "pictures",
            err=True,
        )
    provider = PipelineProvider(
        created,
        batch_size=embed_batch if embed_batch is not None else DEFAULT_EMBED_BATCH,
        dtype=DEFAULT_DTYPE,
        use_cache=embedding_cache,
    )
    limits = StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )
    ds = DatasetStore.open(dataset, limits=limits).read()
    run_id = current_request_id()
    if run_id is None:
        raise RuntimeError("request_id is not bound")
    ctx = CheckContext(
        run_id=run_id,
        root=dataset.resolve().parent,
        cache_dir=settings.cache_dir,
        workers=settings.workers,
        cache_key=settings.cache_key,
        in_ci=settings.ci,
        dataset_dir=dataset.resolve(),
        embedder=f"{created.model_id}@{created.revision[:12]}",
    )
    reason = provider.preflight(ctx)
    if reason is not None:
        raise typer.BadParameter(
            f"the embedding cache cannot be used ({reason}); nothing would be stored",
            param_hint="--embedder",
        )
    es = provider.get(ds, ctx)
    typer.echo(
        f"embedded={len(es.ids)} total={es.total} cached={es.cache_hits} "
        f"missing={es.missing_count} model={created.model_id}@{created.revision[:12]} "
        f"run_id={run_id}"
    )
    problems: list[str] = []
    if es.error is not None:
        problems.append(
            f"the embedder failed ({es.error})"
            + (f": {es.error_detail}" if es.error_detail else "")
        )
    if es.missing:
        reasons = ", ".join(f"{r}={len(v)}" for r, v in sorted(es.missing.items()))
        problems.append(f"images without an embedding: {reasons}")
    if es.cache_unwritable:
        problems.append("embeddings could not be stored in the dataset directory")
    if problems:
        for line in problems:
            typer.echo(f"error: {line}", err=True)
        raise typer.Exit(1)


_SET_ID = re.compile(r"^[0-9a-f]{32}$")


def _set_id(value: str) -> str:
    if not _SET_ID.fullmatch(value):
        raise typer.BadParameter(
            "a proposal set id is 32 lower-case hex digits (printed by `propose`)",
            param_hint="--proposals",
        )
    return value


def _threshold_pairs(values: list[str] | None) -> dict[str, float]:
    found: dict[str, float] = {}
    for item in values or []:
        name, sep, number = item.rpartition("=")
        try:
            score = float(number)
        except ValueError:
            score = float("nan")
        if not (sep and name and score == score):
            raise typer.BadParameter("use CLASS=SCORE", param_hint="--min-score-for")
        found[name] = score
    return found


@app.command()
def propose(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    proposer: Annotated[
        str,
        typer.Option(
            "--proposer",
            help="Proposer name; grounding_dino needs the [vlm] extra (and downloads its model "
            "once), test_boxes is a test double.",
        ),
    ] = "grounding_dino",
    classes: Annotated[
        str | None,
        typer.Option("--classes", help="Comma-separated class names to look for (default: all)."),
    ] = None,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Lowest confidence kept (above 0, at most 1)."),
    ] = 0.35,
    min_score_for: Annotated[
        list[str] | None,
        typer.Option(
            "--min-score-for",
            help="CLASS=SCORE: a lowest confidence for one class; repeat for more classes.",
        ),
    ] = None,
    time_budget: Annotated[
        float,
        typer.Option(
            "--time-budget", help="Seconds after which the run stops with an incomplete set."
        ),
    ] = 3600.0,
) -> None:
    """Find boxes a dataset is probably missing and store them as a proposal set.

    Nothing is applied and the dataset is not changed. The model is asked about every class (or
    --classes) in every image. A box is proposed only if the image has no box of that class at the
    same place already (IoU 0.5). Prints one line
    (`proposed=N images=DONE/TOTAL set=ID model=ID@REV run_id=ID`); the set id is what `review` and
    `apply` take. Exit codes: 0 the run reached every image; 1 the run is incomplete (the time
    budget ended it, an image could not be read; the partial set is stored and `apply` refuses it
    without --allow-incomplete), a model that is not allowed (DATACOPILOT_ALLOWED_LICENSES) or
    fails, a missing extra, or an unreadable or non-detection dataset; 2 bad option.
    """
    from datacopilot.proposals import usecase
    from datacopilot.proposals.run import OptionError

    settings = load_settings()
    names = None if classes is None else _split_option(classes, "--classes")
    thresholds = _threshold_pairs(min_score_for)
    try:
        ps, test_only = usecase.propose_dataset(
            dataset,
            proposer,
            settings=settings,
            classes=names,
            min_score=min_score,
            class_min_score=thresholds,
            time_budget_seconds=time_budget,
        )
    except OptionError as exc:  # only what was wrong before the model ran is a usage error
        raise typer.BadParameter(str(exc), param_hint="--proposer/--classes/--min-score") from None
    run_id = current_request_id()
    if test_only:
        typer.echo(
            f"warning: proposer {proposer!r} is a test double: its boxes say nothing about the "
            "pictures",
            err=True,
        )
    typer.echo(
        f"proposed={len(ps.proposals)} images={ps.images_done}/{ps.images_total} set={ps.set_id} "
        f"model={ps.model_id}@{ps.model_revision[:12]} run_id={run_id}"
    )
    if ps.incomplete:
        typer.echo(
            f"error: the run is incomplete ({ps.incomplete_reason}: {ps.images_done} of "
            f"{ps.images_total} images); `apply` needs --allow-incomplete for this set",
            err=True,
        )
        raise typer.Exit(1)


review_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Send proposals to an annotation tool and read the decisions back.",
)
app.add_typer(review_app, name="review")


@review_app.command("export")
def review_export(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    proposals: Annotated[
        str, typer.Option("--proposals", help="Proposal set id (from `propose`).")
    ],
    to: Annotated[str, typer.Option("--to", help="Annotation tool; currently supports `cvat`.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Empty or new output directory.")],
) -> None:
    """Write the proposals as a separate layer of a CVAT dataset, next to the dataset's own boxes.

    Open the output in CVAT (import it as "CVAT for images 1.1"). The proposals are boxes whose
    label is the class name plus `__proposed`; each has a `decision` attribute: type `accepted` or
    `rejected` (empty means undecided) and move the box first to correct it. Deleting a box decides
    nothing. Prints the output directory. Exit codes: 0 success; 1 unreadable dataset, an unknown,
    tampered or mismatching set, a non-empty output; 2 bad option.
    """
    from datacopilot.proposals import usecase

    _set_id(proposals)
    if to != "cvat":
        raise typer.BadParameter("only `cvat` is built", param_hint="--to")
    usecase.export_review_set(dataset, proposals, output, settings=load_settings())
    typer.echo(str(output))


@review_app.command("import")
def review_import(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    directory: Annotated[Path, typer.Argument(help="The reviewed CVAT export (a directory).")],
    proposals: Annotated[
        str, typer.Option("--proposals", help="Proposal set id that was exported.")
    ],
    source: Annotated[
        str, typer.Option("--from", help="Annotation tool; currently supports `cvat`.")
    ],
) -> None:
    """Read the decisions of a reviewed CVAT export into a new, reviewed proposal set.

    The original set is not changed: a new set names it as its parent, and its id is printed. Only
    the proposal layer is read; edits to the dataset's own boxes are ignored and counted. A decision
    file that names an unknown, repeated or foreign proposal, changes a class, uses an invalid
    decision or moves an accepted box outside its image is refused as a whole. Prints
    `accepted=N rejected=N pending=N edited=N edited_pending=N ignored=N skipped=N set=ID`. Exit
    codes: 0 success; 1 unreadable dataset or file, a refused decision file or no decision in it;
    2 bad option.
    """
    from datacopilot.proposals import usecase

    _set_id(proposals)
    if source != "cvat":
        raise typer.BadParameter("only `cvat` is built", param_hint="--from")
    reviewed, summary = usecase.import_review_set(
        dataset, proposals, directory, settings=load_settings()
    )
    typer.echo(
        f"accepted={summary.accepted} rejected={summary.rejected} pending={summary.pending} "
        f"edited={summary.edited} edited_pending={summary.edited_pending} "
        f"ignored={summary.ignored_boxes} skipped={summary.skipped_shapes} set={reviewed.set_id}"
    )
    if summary.edited_pending:
        typer.echo(
            f"note: {summary.edited_pending} proposal box(es) were moved but have no decision; "
            "they stay pending",
            err=True,
        )
    if summary.skipped_shapes:
        typer.echo(
            f"note: {summary.skipped_shapes} shape(s) of a kind that is not read (polygons, "
            "tracks, ...) were skipped",
            err=True,
        )
    if summary.ignored_boxes:
        typer.echo(
            f"note: {summary.ignored_boxes} box(es) of the dataset's own layer were ignored "
            "(labels are never edited by a review)",
            err=True,
        )


@app.command()
def apply(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    proposals: Annotated[
        str, typer.Option("--proposals", help="Reviewed proposal set id (from `review import`).")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="New dataset directory (*.dcp).")],
    allow_incomplete: Annotated[
        bool,
        typer.Option(
            "--allow-incomplete",
            help="Also apply a set from a run that did not reach every image.",
        ),
    ] = False,
    allow_test_proposer: Annotated[
        bool,
        typer.Option(
            "--allow-test-proposer",
            help="Also apply a set made by a test double such as test_boxes (its boxes say "
            "nothing about the pictures).",
        ),
    ] = False,
) -> None:
    """Write the accepted proposals as a NEW dataset; the original is not changed.

    Only accepted proposals of a REVIEWED set (from `review import`) are applied (pending and
    rejected ones never are); the reviewed set and the one it came from are copied into the new
    dataset. The new dataset
    records the old version in its lineage, and each new box has provenance `proposal_accepted`
    with the proposal and set ids. Refused, writing nothing: a set made for another dataset version
    or by a model that is not the pinned one or not allowed (DATACOPILOT_ALLOWED_LICENSES), a
    tampered set, an incomplete run (without --allow-incomplete), a set with nothing accepted, an
    output that exists or lies inside the dataset. Prints the new version. Exit codes: 0 success; 1
    refused or unreadable; 2 bad option.
    """
    from datacopilot.proposals import usecase

    _set_id(proposals)
    if output.suffix != ".dcp":
        raise typer.BadParameter("must end with .dcp", param_hint="--output")
    manifest = usecase.apply_set(
        dataset,
        proposals,
        output,
        settings=load_settings(),
        allow_incomplete=allow_incomplete,
        allow_test_proposer=allow_test_proposer,
    )
    typer.echo(manifest.version)


def _write_new_file(path: Path, text: str) -> None:
    """Create ``path`` exclusively, never following a symlink, and write ``text`` (UTF-8)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


@app.command()
def split(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="New dataset directory (*.dcp).")],
    ratios: Annotated[
        str, typer.Option("--ratios", help="Train, val and test fractions that sum to 1.")
    ] = "0.8,0.1,0.1",
    by_group: Annotated[
        str,
        typer.Option(
            "--by-group",
            help="Images with the same key stay together: `group_id` (default), "
            "`source.<name>` or `re:<pattern>` over the file name (a bounded, safe pattern; the "
            "named group `g` or group 1 is the key). The new version's group_id is the key.",
        ),
    ] = "group_id",
    near_dup: Annotated[
        bool,
        typer.Option(
            "--near-dup",
            help="Also keep near copies together: every image pair whose embeddings reach "
            "--similarity-threshold (needs --embedder; every image must be embedded).",
        ),
    ] = False,
    embedder: Annotated[
        str | None, typer.Option("--embedder", help="Embedder for --near-dup.")
    ] = None,
    similarity_threshold: Annotated[
        float | None,
        typer.Option(
            "--similarity-threshold", help="Cosine similarity of near copies (default 0.95)."
        ),
    ] = None,
    embed_batch: Annotated[
        int | None,
        typer.Option(
            "--embed-batch", min=1, max=MAX_EMBED_BATCH_OPTION, help="Images per embedder call."
        ),
    ] = None,
    stratify: Annotated[
        str | None,
        typer.Option("--stratify", help="`category`: also balance each class across splits."),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help="Breaks ties between groups of equal size (0 to 2^32-1).")
    ] = 0,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json",
            help="Write the split report as JSON to this new file (outside the dataset; an "
            "existing file or a symlink is refused).",
        ),
    ] = None,
    allow_test_embedder: Annotated[
        bool,
        typer.Option(
            "--allow-test-embedder",
            help="Allow a test-only embedder such as `hash` (it compares file bytes, not "
            "pictures) for --near-dup; refused without it.",
        ),
    ] = False,
) -> None:
    """Re-split a dataset by group and write it as a NEW dataset; the original is not changed.

    Groups are images with the same --by-group key and, with --near-dup, near copies (every pair
    at or above the threshold). Whole groups are assigned, largest first, to the split that keeps
    the image counts (and with --stratify category the class counts) closest to --ratios; the same
    input, ratios and seed give the same assignment whatever the row order. The new dataset records
    the old version in its lineage. Prints the new version, a summary line and one line per split
    (`<split> images=N target=T deviation=D`). Exit codes: 0 written; 1 refused (an image that
    could not be embedded, too many near-copy pairs, a split identical to the current one, an
    output that exists or lies inside the dataset) or unreadable; 2 bad option.
    """
    from datacopilot.splits import assign, groupkey
    from datacopilot.splits import usecase as split_usecase
    from datacopilot.store import ensure_outside

    if output.suffix != ".dcp":
        raise typer.BadParameter("must end with .dcp", param_hint="--output")
    try:
        parsed = assign.parse_ratios(ratios)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ratios") from None
    try:
        groupkey.parse(by_group)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--by-group") from None
    if stratify not in (None, "category"):
        raise typer.BadParameter("only `category` is supported", param_hint="--stratify")
    if near_dup and embedder is None:
        raise typer.BadParameter("--near-dup needs --embedder", param_hint="--near-dup")
    if not near_dup and (embedder is not None or similarity_threshold is not None):
        raise typer.BadParameter(
            "--embedder and --similarity-threshold need --near-dup", param_hint="--embedder"
        )
    if json_out is not None:
        try:
            ensure_outside(json_out, dataset)
        except DataCopilotError:
            raise typer.BadParameter(
                "must lie outside the dataset directory", param_hint="--json"
            ) from None
        if os.path.lexists(json_out):
            raise typer.BadParameter(
                "already exists (or is a symlink); choose a new file", param_hint="--json"
            )
    settings = load_settings()
    if embedder is not None:
        created = _resolve_embedder(embedder, settings.allowed_licenses)
        if getattr(created, "test_only", False):
            if not allow_test_embedder:
                raise typer.BadParameter(
                    f"{embedder!r} is test-only; pass --allow-test-embedder to use it",
                    param_hint="--embedder",
                )
            typer.echo(
                f"warning: embedder {embedder!r} is a test double that compares file bytes, not "
                "pictures",
                err=True,
            )
    try:
        report = split_usecase.split_dataset(
            dataset,
            output,
            settings=settings,
            ratios=parsed,
            group_key=by_group,
            near_duplicates=near_dup,
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            embed_batch=embed_batch,
            stratify=stratify == "category",
            seed=seed,
            allow_test_embedder=allow_test_embedder,
        )
    except ValueError as exc:  # the use case's usage errors (seed, threshold)
        raise typer.BadParameter(str(exc)) from None
    if json_out is not None:
        _write_new_file(json_out, report.model_dump_json(indent=2) + "\n")
    typer.echo(report.dataset_version)
    typer.echo(
        f"images={report.image_count} groups={report.group_count} "
        f"moved={report.moved_images} near_duplicate_pairs={report.near_duplicate_pairs} "
        f"groups_across_splits={report.groups_across_splits} "
        f"max_class_deviation={report.max_class_deviation}"
    )
    for name, share in report.splits.items():
        typer.echo(
            f"{name} images={share.images} target={share.target_images:g} "
            f"deviation={share.deviation:+.4f}"
        )


@app.command()
def formats() -> None:
    """List registered formats as name, capabilities, license and extra (tab separated)."""
    from datacopilot.io import registry

    for info in registry.list_formats():
        caps = ",".join(c for c, ok in (("read", info.can_read), ("write", info.can_write)) if ok)
        typer.echo("\t".join([info.spec.name, caps, info.spec.license, info.spec.extra or "-"]))


def _usage_exit_code(exc: Exception) -> int | None:
    """Exit code for Click-style usage errors, else None.

    Duck-typed because recent Typer vendors its own Click exception classes, so
    ``isinstance(exc, click.ClickException)`` misses them.
    """
    code = getattr(exc, "exit_code", None)
    if isinstance(code, int) and code != 0 and callable(getattr(exc, "show", None)):
        return code
    return None


def _one_line(exc: Exception) -> str:
    """Error text safe for a terminal and for CI logs: control characters are escaped and a
    leading ``::`` (a GitHub Actions workflow command) is removed."""
    from datacopilot.report.text import clean_text

    return clean_text(str(exc), 600).lstrip(": \t")


def _os_error_line(exc: OSError) -> str:
    """What failed and why, without the directory (a path can carry a user name)."""
    reason = exc.strerror or type(exc).__name__
    name = re.sub(r"\.tmp-[0-9a-f]+$", "", Path(str(exc.filename)).name) if exc.filename else ""
    return _one_line(OSError(f"{reason}: {name}" if name else reason))


def main() -> NoReturn:
    argv = sys.argv[1:]
    command = argv[0] if argv and not argv[0].startswith("-") else None
    code = 1  # a verdict of success is only set after the command has returned normally
    try:
        result = app(standalone_mode=False)
        code = result if isinstance(result, int) else 0
    except SystemExit:  # never let a stray sys.exit(0) in a command or plugin pass as success
        typer.echo("error: the command exited unexpectedly", err=True)
        code = 1
    except typer.Abort:
        code = 130
    except Exception as exc:  # error boundary: logged once here, nowhere else
        usage_code = _usage_exit_code(exc)
        if usage_code is None:
            if isinstance(exc, DataCopilotError):
                typer.echo(f"error: {_one_line(exc)}", err=True)
            elif isinstance(exc, OSError):  # unwritable output, unreadable input, full disk
                typer.echo(f"error: {_os_error_line(exc)}", err=True)
            get_logger("datacopilot.cli").error("command_failed", command=command, exc_info=exc)
            code = 1
        else:
            exc.show()  # type: ignore[attr-defined]
            code = usage_code
    finally:
        clear_request_context()
    raise SystemExit(code)
