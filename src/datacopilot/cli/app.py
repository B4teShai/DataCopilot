"""Command-line entry point."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Annotated, NoReturn

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
) -> None:
    """Audit a dataset.dcp, write reports and exit with the gate's verdict.

    Prints one line (`passed|failed critical=N warn=N info=N run_id=ID`) and stores the run as
    audits/<run_id>.json. The default gate is `--fail-on critical`; `--fail-on none` applies no
    gate. Exit codes: 0 gate passed (or no gate); 1 a --fail-on match, an empty dataset or an
    unreadable dataset; 2 bad option or configuration (unknown check, bad --fail-on, bad value).
    """
    from datacopilot.audit import engine
    from datacopilot.audit.context import CheckContext
    from datacopilot.core.errors import FormatError
    from datacopilot.report import render
    from datacopilot.store import DatasetStore, StoreLimits

    settings = load_settings()
    try:
        selected = "all" if checks.strip() == "all" else _split_option(checks, "--checks")
        names = engine.select_checks(selected)
        gate = engine.parse_fail_on(
            _split_option("critical" if fail_on is None else fail_on, "--fail-on")
        )
        for token in gate:
            if token not in ("critical", "warn", "info") and token not in names:
                raise ValueError(f"--fail-on {token!r} is not among the checks being run")
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
