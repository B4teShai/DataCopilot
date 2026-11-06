"""MCP server: dataset facts, audits and proposals as tools for coding agents, over stdio.

Start it with ``datacopilot mcp`` (extra ``datacopilot[mcp]``). It wraps the public API
(:mod:`datacopilot.api`) and the dataset store with the same validation and fail-closed behaviour
as the command line.

Safety rules, enforced here and not left to the caller:

* **Read-only by default.** Tools that create a dataset or a proposal set (``ingest``,
  ``propose``, ``apply_proposals``, ``split``) are registered only when
  ``DATACOPILOT_MCP_ALLOW_WRITE=1``. None of them changes or deletes an existing dataset version.
* **Path roots.** Every path argument must resolve (symlinks followed) inside one of
  ``DATACOPILOT_MCP_ROOTS`` (``os.pathsep``-separated; default: the working directory at start).
  Relative paths are taken from the first root.
* **Summaries, never rows.** Results carry counts, versions and at most ``top_k`` findings (50 at
  most) with payloads cut to 300 characters; no image or annotation table is returned.
* **Logging.** Each tool call gets its own ``request_id``; logs are the usual JSON records on
  standard error (standard output is the protocol). A failure is logged once, here.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from datacopilot import api, convert
from datacopilot.core.errors import DataCopilotError
from datacopilot.core.proposals import ProposalSet
from datacopilot.core.schema import AuditRun, DatasetManifest, Finding, Summary
from datacopilot.logging import clear_request_context, configure_logging, get_logger
from datacopilot.proposals.run import DEFAULT_MIN_SCORE, DEFAULT_TIME_BUDGET_SECONDS
from datacopilot.report.text import clean_text
from datacopilot.settings import Settings, load_settings
from datacopilot.store import DatasetStore, StoreLimits

if TYPE_CHECKING:  # the mcp package is the extra datacopilot[mcp]: imported when serving only
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

__all__ = ["ROOTS_ENV", "WRITE_ENV", "create_server", "main"]

WRITE_ENV = "DATACOPILOT_MCP_ALLOW_WRITE"
ROOTS_ENV = "DATACOPILOT_MCP_ROOTS"
MAX_TOP_K = 50
MAX_LISTED = 20
PAYLOAD_CHARS = 300
_RANK = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
_RUN_FILE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.json$")
_SET_FILE = re.compile(r"^[0-9a-f]{32}\.json$")

DatasetArg = Annotated[str, Field(description="Path of a dataset.dcp directory.")]
TopK = Annotated[int, Field(ge=1, le=MAX_TOP_K, description="Findings to return, worst first.")]
NewDcp = Annotated[str, Field(description="New .dcp path; must not exist.")]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _env_roots() -> tuple[Path, ...]:
    raw = os.environ.get(ROOTS_ENV, "")
    parts = [p for p in raw.split(os.pathsep) if p.strip()] or [os.getcwd()]
    return tuple(Path(p).expanduser().resolve() for p in parts)


class Refused(ValueError):
    """A tool argument the server will not act on (a path outside the roots, a missing dataset)."""


class Paths:
    """Resolve tool path arguments and refuse anything outside the allowed roots."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        if not roots:
            raise ValueError("at least one root is required")
        self.roots = roots

    def _resolve(self, value: str) -> Path:
        if not value or "\x00" in value:
            raise Refused("empty or invalid path")
        path = Path(value).expanduser()
        resolved = (path if path.is_absolute() else self.roots[0] / path).resolve()
        if not any(resolved == r or r in resolved.parents for r in self.roots):
            raise Refused(f"{value!r} is outside the allowed roots (set {ROOTS_ENV})")
        return resolved

    def existing(self, value: str) -> Path:
        path = self._resolve(value)
        if not path.exists():
            raise Refused(f"{value!r} does not exist")
        return path

    def dataset(self, value: str) -> Path:
        path = self.existing(value)
        if path.suffix != ".dcp" or not path.is_dir():
            raise Refused(f"{value!r} is not a .dcp directory (use ingest to create one)")
        return path

    def new_dcp(self, value: str) -> Path:
        path = self._resolve(value)
        if path.suffix != ".dcp":
            raise Refused("the output must end with .dcp")
        if path.exists():
            raise Refused(f"{value!r} already exists; datasets are never overwritten")
        return path


@contextmanager
def _call(tool: str) -> Iterator[Settings]:
    """One tool call: a fresh request id, library errors as tool errors, one log on failure."""
    from mcp.server.mcpserver.exceptions import ToolError

    settings = load_settings()
    configure_logging("datacopilot", settings)
    log = get_logger("datacopilot.mcp")
    try:
        yield settings
    except Refused as exc:
        log.warning("tool_refused", tool=tool, error_type="Refused")
        raise ToolError(str(exc)) from None
    except (DataCopilotError, ValueError) as exc:  # refused input or data: the caller can fix it
        log.warning("tool_refused", tool=tool, error_type=type(exc).__name__)
        raise ToolError(f"{type(exc).__name__}: {clean_text(str(exc), 600)}") from None
    except OSError as exc:  # says what failed, not where: a path can carry a user name
        log.error("tool_failed", tool=tool, exc_info=exc)
        raise ToolError(f"OSError: {exc.strerror or type(exc).__name__}") from None
    except Exception as exc:  # error boundary: logged once here, no internals to the caller
        log.error("tool_failed", tool=tool, exc_info=exc)
        raise ToolError(f"internal error ({type(exc).__name__}); see the server log") from None
    finally:
        clear_request_context()


def _limits(settings: Settings) -> StoreLimits:
    return StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )


def _manifest(m: DatasetManifest) -> dict[str, Any]:
    return {
        "name": m.name,
        "version": m.version,
        "task": m.task,
        "image_count": m.image_count,
        "annotation_count": m.annotation_count,
        "category_count": len(m.categories),
        "categories": [c.name for c in m.categories[: MAX_LISTED * 5]],
        "lineage": list(m.lineage),
        "created_at": m.created_at.isoformat(),
    }


def _finding(f: Finding) -> dict[str, Any]:
    payload = json.dumps(f.payload, sort_keys=True, default=str, ensure_ascii=True)
    if len(payload) > PAYLOAD_CHARS:
        payload = payload[:PAYLOAD_CHARS] + "...(truncated)"
    return {
        "finding_id": f.finding_id,
        "check": f.check,
        "severity": f.severity,
        "image_id": f.image_id,
        "ann_id": f.ann_id,
        "payload": payload,
    }


def _summary(s: Summary | None) -> dict[str, Any] | None:
    if s is None:
        return None
    per_class: Counter[str] = Counter()
    for counts in s.class_counts.values():
        per_class.update(counts)
    names = {str(c.id): c.name for c in s.classes}
    rarest = sorted(per_class.items(), key=lambda kv: (kv[1], kv[0]))[:10]
    return {
        "image_count": s.image_count,
        "annotation_count": s.annotation_count,
        "split_images": dict(s.split_images),
        "class_count": len(s.classes),
        "classes_omitted": s.classes_omitted,
        "rarest_classes": [{"class": names.get(k, k), "annotations": n} for k, n in rarest],
        "excluded_model_annotations": s.excluded_model_annotations,
    }


def audit_summary_of(run: AuditRun, top_k: int) -> dict[str, Any]:
    """Bounded view of an audit run: verdict, counts per check, the worst ``top_k`` findings."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    by_check = Counter((f.check, f.severity) for f in run.findings)
    worst = sorted(run.findings, key=lambda f: (_RANK.get(f.severity, 3), f.check, f.finding_id))
    return {
        "run_id": run.run_id,
        "dataset_version": run.dataset_version,
        "started_at": run.started_at.isoformat(),
        "passed": run.passed,
        "fail_on": list(run.fail_on),
        "checks": list(run.checks),
        "embedder": run.embedder,
        "counts": run.counts.to_row(),
        "finding_total": len(run.findings),
        "by_check": [
            {"check": c, "severity": s, "count": n} for (c, s), n in by_check.most_common()
        ],
        "top_findings": [_finding(f) for f in worst[:top_k]],
        "truncated": len(run.findings) > top_k,
        "summary": _summary(run.summary),
    }


def _proposal_set(ps: ProposalSet) -> dict[str, Any]:
    return {
        "set_id": ps.set_id,
        "dataset_version": ps.dataset_version,
        "proposer": ps.proposer,
        "model_id": ps.model_id,
        "model_revision": ps.model_revision,
        "min_score": ps.min_score,
        "images_done": ps.images_done,
        "images_total": ps.images_total,
        "incomplete_reason": ps.incomplete_reason,
        "test_only": ps.test_only,
        "status": dict(Counter(p.status for p in ps.proposals)),
    }


def _stored(directory: Path, pattern: re.Pattern[str]) -> list[str]:
    """Ids of stored records, newest file first; symlinks and stray names are ignored."""
    if not directory.is_dir() or directory.is_symlink():
        return []
    files = [p for p in directory.iterdir() if pattern.match(p.name) and not p.is_symlink()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]


def _register_read(server: MCPServer, paths: Paths, read: ToolAnnotations) -> None:
    @server.tool(title="List dataset formats", annotations=read)
    def list_formats() -> dict[str, Any]:
        """List the dataset formats DataCopilot can read or write, with license and extra.

        Use the name as ``format`` for ``ingest``."""
        from datacopilot.io import registry

        with _call("list_formats"):
            return {
                "formats": [
                    {
                        "name": i.spec.name,
                        "read": i.can_read,
                        "write": i.can_write,
                        "license": i.spec.license,
                        "extra": i.spec.extra,
                    }
                    for i in registry.list_formats()
                ]
            }

    @server.tool(title="Dataset info", annotations=read)
    def dataset_info(dataset: DatasetArg) -> dict[str, Any]:
        """Describe a .dcp dataset: manifest (version, counts, classes, lineage), its stored audits
        (verdict and counts, newest first) and its proposal sets (progress and review status).

        Reads metadata only; for findings use ``audit_summary``."""
        with _call("dataset_info") as settings:
            path = paths.dataset(dataset)
            store = DatasetStore.open(path, limits=_limits(settings))
            audits = []
            for run_id in _stored(path / "audits", _RUN_FILE)[:MAX_LISTED]:
                run = store.read_audit(run_id)
                audits.append(
                    {
                        "run_id": run.run_id,
                        "started_at": run.started_at.isoformat(),
                        "dataset_version": run.dataset_version,
                        "passed": run.passed,
                        "counts": run.counts.to_row(),
                    }
                )
            sets = [
                _proposal_set(store.read_proposals(set_id))
                for set_id in _stored(path / "proposals", _SET_FILE)[:MAX_LISTED]
            ]
            return {
                "manifest": _manifest(store.manifest()),
                "audits": audits,
                "proposal_sets": sets,
            }

    @server.tool(title="Stored audit summary", annotations=read)
    def audit_summary(
        dataset: DatasetArg,
        run_id: Annotated[
            str | None,
            Field(
                description="Audit run id (from dataset_info); default: the newest stored audit."
            ),
        ] = None,
        top_k: TopK = 10,
    ) -> dict[str, Any]:
        """Summarise a STORED audit of a dataset: gate verdict, counts per check and severity,
        the worst findings (payloads truncated) and class balance. Audits are stored by
        ``datacopilot audit`` on the command line; to audit now without storing it, use
        ``run_audit``."""
        with _call("audit_summary") as settings:
            path = paths.dataset(dataset)
            store = DatasetStore.open(path, limits=_limits(settings))
            if run_id is None:
                stored = _stored(path / "audits", _RUN_FILE)
                if not stored:
                    raise Refused("no stored audit in this dataset; use run_audit")
                run_id = stored[0]
            return audit_summary_of(store.read_audit(run_id.strip().lower()), top_k)

    @server.tool(title="Run an audit", annotations=read)
    def run_audit(
        dataset: DatasetArg,
        checks: Annotated[
            list[str] | None, Field(description="Check names; default: every check that applies.")
        ] = None,
        fail_on: Annotated[
            list[str],
            Field(description="critical, warn, info (that severity and above) and/or check names."),
        ] = ["critical"],  # noqa: B006  # pydantic copies defaults
        embedder: Annotated[
            str | None,
            Field(
                description="dinov2_small, dinov2_base or siglip2_base enables near_duplicates, "
                "leakage, label_noise and outliers (extra datacopilot[embed]; may download the "
                "model and fill the dataset's embedding cache)."
            ),
        ] = None,
        similarity_threshold: Annotated[float | None, Field(gt=0, le=1)] = None,
        require_local_images: bool = False,
        include_model_annotations: bool = False,
        top_k: TopK = 10,
    ) -> dict[str, Any]:
        """Run the audit checks on a .dcp dataset now and summarise the result; the run is NOT
        stored. ``passed`` is the gate verdict for ``fail_on``. Use ``audit_summary`` for an audit
        that was already stored."""
        with _call("run_audit"):
            path = paths.dataset(dataset)
            run = api.audit(
                path,
                checks if checks else "all",
                fail_on=tuple(fail_on),
                embedder=embedder,
                similarity_threshold=similarity_threshold,
                require_local_images=require_local_images,
                include_model_annotations=include_model_annotations,
            )
            return audit_summary_of(run, top_k)


def _register_write(server: MCPServer, paths: Paths, create: ToolAnnotations) -> None:
    @server.tool(title="Ingest a dataset", annotations=create)
    def ingest(
        source: Annotated[str, Field(description="Dataset root or annotation file to read.")],
        format: Annotated[str, Field(description="Source format name (see list_formats).")],  # noqa: A002
        output: NewDcp,
        group_by: Annotated[
            str | None,
            Field(description="Regex over the image path; group `g` (or 1) is the group id."),
        ] = None,
        name: str | None = None,
        strict: Annotated[
            bool, Field(description="Fail, writing nothing, if the reader dropped anything.")
        ] = True,
    ) -> dict[str, Any]:
        """Convert a source dataset into a NEW versioned .dcp directory and return its manifest and
        what the conversion dropped or skipped. Label Studio task data is always dropped."""
        with _call("ingest") as settings:
            src = paths.existing(source)
            out = paths.new_dcp(output)
            result = convert.ingest(
                src, format, out, group_by=group_by, name=name, limits=_limits(settings),
                strict=strict,
            )  # fmt: skip
            r = result.report
            return {
                "path": str(result.path),
                "manifest": _manifest(result.manifest),
                "conversion": {
                    "rows_read": r.rows_read,
                    "rows_written": r.rows_written,
                    "dropped": dict(r.dropped),
                    "skipped": dict(r.skipped),
                },
            }

    @server.tool(title="Propose missing boxes", annotations=create)
    def propose(
        dataset: DatasetArg,
        proposer: Annotated[
            Literal["grounding_dino"], Field(description="Needs the extra datacopilot[vlm].")
        ] = "grounding_dino",
        classes: Annotated[list[str] | None, Field(description="Class names; default all.")] = None,
        min_score: Annotated[float, Field(gt=0, le=1)] = DEFAULT_MIN_SCORE,
        time_budget_seconds: Annotated[float, Field(gt=0, le=86_400)] = DEFAULT_TIME_BUDGET_SECONDS,
    ) -> dict[str, Any]:
        """Find boxes the dataset is probably missing and store them as a proposal set in the
        dataset directory. Nothing is applied: review the set (``datacopilot review export`` /
        ``review import``), then ``apply_proposals``. May download the model weights."""
        with _call("propose"):
            path = paths.dataset(dataset)
            ps = api.propose(
                path, proposer, classes=classes, min_score=min_score,
                time_budget_seconds=time_budget_seconds,
            )  # fmt: skip
            return _proposal_set(ps)

    @server.tool(title="Apply reviewed proposals", annotations=create)
    def apply_proposals(
        dataset: DatasetArg,
        set_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$", description="Reviewed set id.")],
        output: NewDcp,
        allow_incomplete: bool = False,
    ) -> dict[str, Any]:
        """Write the ACCEPTED proposals of a reviewed set as a NEW dataset version (lineage records
        the parent); pending and rejected proposals are never applied."""
        with _call("apply_proposals"):
            path = paths.dataset(dataset)
            out = paths.new_dcp(output)
            manifest = api.apply(path, set_id, out, allow_incomplete=allow_incomplete)
            return {"path": str(out), "manifest": _manifest(manifest)}

    @server.tool(title="Re-split a dataset", annotations=create)
    def split(
        dataset: DatasetArg,
        output: NewDcp,
        ratios: Annotated[
            list[float], Field(min_length=3, max_length=3, description="train, val, test; sum 1.")
        ] = [0.8, 0.1, 0.1],  # noqa: B006
        by_group: Annotated[
            str, Field(description="group_id, source.<name> or re:<pattern> over the file name.")
        ] = "group_id",
        near_duplicates: Annotated[
            bool, Field(description="Keep near copies together; needs embedder.")
        ] = False,
        embedder: str | None = None,
        similarity_threshold: Annotated[float | None, Field(gt=0, le=1)] = None,
        stratify_by_category: bool = False,
        seed: Annotated[int, Field(ge=0)] = 0,
    ) -> dict[str, Any]:
        """Re-split a dataset by group (no group spans two splits) into a NEW dataset version and
        report the deviation from the ratios. Use it to fix leakage between splits."""
        with _call("split"):
            path = paths.dataset(dataset)
            out = paths.new_dcp(output)
            report = api.split(
                path, out, ratios=tuple(ratios), by_group=by_group,
                near_duplicates=near_duplicates, embedder=embedder,
                similarity_threshold=similarity_threshold,
                stratify="category" if stratify_by_category else None, seed=seed,
            )  # fmt: skip
            row = report.model_dump(mode="json", exclude={"classes"})
            return {"path": str(out), "report": row}


INSTRUCTIONS = (
    "DataCopilot: audit and fix detection/classification datasets stored as versioned .dcp "
    "directories. Start with dataset_info, then audit_summary (stored audit) or run_audit. "
    "Paths must lie under the configured roots. Tools return summaries, never dataset rows. "
    "Tools that create datasets are present only when the server was started with "
    f"{WRITE_ENV}=1; they never modify an existing version."
)


def create_server(
    *, roots: tuple[Path, ...] | None = None, allow_write: bool | None = None
) -> MCPServer:
    """Build the server; ``None`` means: read ``DATACOPILOT_MCP_ROOTS`` or
    ``DATACOPILOT_MCP_ALLOW_WRITE``."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    from datacopilot import __version__

    read = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )
    create = ToolAnnotations(  # writes a NEW dataset or set; never touches an existing version
        read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
    )
    paths = Paths(roots if roots is not None else _env_roots())
    server = MCPServer(
        "datacopilot", version=__version__, instructions=INSTRUCTIONS, log_level="WARNING"
    )
    _register_read(server, paths, read)
    if allow_write if allow_write is not None else _env_flag(WRITE_ENV):
        _register_write(server, paths, create)
    return server


def main(run: Callable[[MCPServer], None] = lambda s: s.run("stdio")) -> None:
    """Serve over standard input and output until the client disconnects."""
    run(create_server())
