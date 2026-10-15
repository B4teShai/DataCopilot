"""Command-line entry point."""

from __future__ import annotations

import math
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
from datacopilot.settings import SettingsError, load_settings

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, help="DataCopilot")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"datacopilot {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    try:
        settings = load_settings()
    except SettingsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    configure_logging("datacopilot", settings)


def _run_doctor() -> None:
    get_logger("datacopilot.cli").info(
        "doctor_completed",
        status="ok",
        package_version=__version__,
        python_version=platform.python_version(),
    )


@app.command()
def doctor() -> None:
    """Check that the installation works."""
    _run_doctor()
    typer.echo("ok")


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
) -> None:
    """Convert a dataset into a versioned dataset.dcp directory and print its version.

    Exit codes: 0 success; 1 unreadable or invalid input, unknown format, output exists;
    2 bad option or configuration.
    """
    from datacopilot import convert
    from datacopilot.store import StoreLimits

    if output.suffix != ".dcp":
        raise typer.BadParameter("must end with .dcp", param_hint="--output")
    if group_by is not None:
        try:
            convert.parse_group_by(group_by)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--group-by") from None
    settings = load_settings()
    limits = StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )
    result = convert.ingest(path, format, output, group_by=group_by, name=name, limits=limits)
    typer.echo(result.manifest.version)


@app.command()
def export(
    dataset: Annotated[Path, typer.Argument(help="A dataset.dcp directory.")],
    format: FormatOption,  # noqa: A002
    output: Annotated[Path, typer.Option("--output", "-o", help="Empty or new output directory.")],
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail if any field cannot be carried to the format.")
    ] = False,
) -> None:
    """Write a dataset.dcp in another format and print the output directory.

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
    convert.export(ds, format, output, strict=strict)
    typer.echo(str(output))


def _split_option(value: str, option: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    if not all(parts):
        raise typer.BadParameter("empty value in the list", param_hint=option)
    return parts


_EMBEDDER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_EMBED_BATCH_OPTION = 1024


def _resolve_embedder(name: str, allowed_licenses: frozenset[str]) -> Any:
    """Create the named embedder after the license guard; usage errors exit 2, license and
    missing-extra errors exit 1. No weights are touched here (the model loads on first use)."""
    from datacopilot.embed import registry as embed_registry

    if not _EMBEDDER_NAME.fullmatch(name):
        raise typer.BadParameter(
            "use lower-case letters, digits and underscores", param_hint="--embedder"
        )
    available = embed_registry.embedder_names()
    if name not in available:
        raise typer.BadParameter(
            f"unknown embedder; available: {', '.join(available) or 'none'}",
            param_hint="--embedder",
        )
    return embed_registry.get_embedder(name, allowed_licenses)


def _check_threshold(value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and 0.0 < value <= 1.0):
        raise typer.BadParameter(
            "must be above 0 and at most 1", param_hint="--similarity-threshold"
        )


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
                "adds the near_duplicates, leakage and outliers checks to `--checks all`."
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
            get_logger("datacopilot.cli").error("command_failed", command=command, exc_info=exc)
            code = 1
        else:
            exc.show()  # type: ignore[attr-defined]
            code = usage_code
    finally:
        clear_request_context()
    raise SystemExit(code)
