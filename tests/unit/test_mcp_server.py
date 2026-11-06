"""The MCP server (``datacopilot mcp``): tool surface, path roots, read/write split, bounded output.

Every call goes through a real client session (in memory), so tool errors are seen the way an
agent sees them: a result with ``isError``, not an exception.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from fixtures import synth  # noqa: E402
from mcp.client import Client  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import datacopilot as dc  # noqa: E402
from datacopilot import convert  # noqa: E402
from datacopilot.audit.findings import make_finding  # noqa: E402
from datacopilot.cli.app import app  # noqa: E402
from datacopilot.core.schema import AuditRun, Counts  # noqa: E402
from datacopilot.mcp_server import PAYLOAD_CHARS, audit_summary_of, create_server  # noqa: E402
from datacopilot.store import DatasetStore  # noqa: E402

READ_TOOLS = {"list_formats", "dataset_info", "audit_summary", "run_audit"}
WRITE_TOOLS = {"ingest", "propose", "apply_proposals", "split"}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("DATACOPILOT_ENV", "DATACOPILOT_ALLOWED_LICENSES", "DATACOPILOT_MCP_ALLOW_WRITE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def dcp(root: Path) -> Path:
    return convert.ingest(synth.build_coco(root / "coco"), "coco", root / "a.dcp").path


def tools(server: Any) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(server) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return asyncio.run(go())


def call(server: Any, name: str, **args: Any) -> tuple[bool, Any]:
    """(is_error, structured result or error text)."""

    async def go() -> tuple[bool, Any]:
        async with Client(server) as client:
            result = await client.call_tool(name, args)
            if result.is_error:
                return True, result.content[0].text
            return False, result.structured_content

    return asyncio.run(go())


# ---------------------------------------------------------------- surface
def test_read_only_by_default_and_every_tool_is_annotated(root: Path) -> None:
    found = tools(create_server(roots=(root,)))
    assert set(found) == READ_TOOLS
    for tool in found.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.title and tool.description


def test_write_tools_need_the_flag_and_never_claim_to_destroy(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATACOPILOT_MCP_ALLOW_WRITE", "1")
    monkeypatch.setenv("DATACOPILOT_MCP_ROOTS", str(root))
    found = tools(create_server())
    assert set(found) == READ_TOOLS | WRITE_TOOLS
    for name in WRITE_TOOLS:
        assert found[name].annotations.read_only_hint is False
        assert found[name].annotations.destructive_hint is False


def test_a_write_tool_is_not_callable_without_the_flag(root: Path, dcp: Path) -> None:
    error, text = call(create_server(roots=(root,)), "split", dataset=str(dcp), output="b.dcp")
    assert error and "split" in text
    assert not (root / "b.dcp").exists()


# ---------------------------------------------------------------- paths
def test_paths_outside_the_roots_are_refused(root: Path, tmp_path: Path) -> None:
    outside = convert.ingest(synth.build_coco(tmp_path / "o"), "coco", tmp_path / "o.dcp").path
    server = create_server(roots=(root,))
    error, text = call(server, "dataset_info", dataset=str(outside))
    assert error and "outside the allowed roots" in text
    (root / "link.dcp").symlink_to(outside)  # a symlink inside a root does not widen it
    error, text = call(server, "dataset_info", dataset="link.dcp")
    assert error and "outside the allowed roots" in text
    error, text = call(server, "dataset_info", dataset="../o.dcp")
    assert error and "outside the allowed roots" in text


def test_only_dcp_directories_are_datasets(root: Path) -> None:
    (root / "x").mkdir()
    error, text = call(create_server(roots=(root,)), "dataset_info", dataset="x")
    assert error and "not a .dcp directory" in text
    error, text = call(create_server(roots=(root,)), "dataset_info", dataset="missing.dcp")
    assert error and "does not exist" in text


# ---------------------------------------------------------------- read tools
def test_list_formats_names_the_readers() -> None:
    error, out = call(create_server(roots=(Path.cwd(),)), "list_formats")
    assert not error
    names = {f["name"] for f in out["formats"]}
    assert {"coco", "yolo", "voc"} <= names


def test_dataset_info_lists_manifest_audits_and_proposal_sets(root: Path, dcp: Path) -> None:
    run = dc.audit(dcp)
    DatasetStore.open(dcp).write_audit(run)
    ps = dc.propose(dcp, "test_boxes")
    error, out = call(create_server(roots=(root,)), "dataset_info", dataset="a.dcp")
    assert not error
    manifest = DatasetStore.open(dcp).manifest()
    assert out["manifest"]["version"] == manifest.version
    assert out["manifest"]["image_count"] == manifest.image_count
    assert [a["run_id"] for a in out["audits"]] == [run.run_id]
    assert out["audits"][0]["passed"] is run.passed
    [stored] = out["proposal_sets"]
    assert stored["set_id"] == ps.set_id and stored["test_only"] is True
    assert sum(stored["status"].values()) == len(ps.proposals)


def test_audit_summary_reads_the_newest_stored_audit(root: Path, dcp: Path) -> None:
    server = create_server(roots=(root,))
    error, text = call(server, "audit_summary", dataset=str(dcp))
    assert error and "no stored audit" in text
    run = dc.audit(dcp)
    DatasetStore.open(dcp).write_audit(run)
    error, out = call(server, "audit_summary", dataset=str(dcp), top_k=1)
    assert not error
    assert out["run_id"] == run.run_id and out["passed"] is run.passed
    assert out["counts"] == run.counts.to_row()
    assert len(out["top_findings"]) <= 1
    assert out["summary"]["image_count"] == run.summary.image_count  # type: ignore[union-attr]
    error, text = call(
        server, "audit_summary", dataset=str(dcp), run_id="00000000-0000-4000-8000-000000000000"
    )
    assert error and "no audit" in text


def test_run_audit_gates_and_stores_nothing(root: Path, dcp: Path) -> None:
    error, out = call(
        create_server(roots=(root,)), "run_audit", dataset=str(dcp), fail_on=["info"], top_k=50
    )
    assert not error
    assert out["passed"] is (out["finding_total"] == 0)
    assert out["fail_on"] == ["info"]
    assert not (dcp / "audits").exists()


def test_run_audit_reports_bad_options_as_tool_errors(root: Path, dcp: Path) -> None:
    error, text = call(create_server(roots=(root,)), "run_audit", dataset=str(dcp), checks=["nope"])
    assert error and "nope" in text
    error, text = call(create_server(roots=(root,)), "run_audit", dataset=str(dcp), top_k=500)
    assert error  # schema: at most 50


def test_audit_summary_is_bounded_and_worst_first() -> None:
    run_id = "12345678-1234-4234-8234-123456789012"
    findings = tuple(
        make_finding(
            run_id,
            "label_noise" if i else "missing_image",
            "INFO" if i else "CRITICAL",
            payload={"text": "x" * 2000, "i": i},
        )
        for i in range(80)
    )
    run = AuditRun(
        run_id=run_id,
        dataset_version="0" * 32,
        started_at="2026-11-06T00:00:00Z",  # type: ignore[arg-type]
        checks=("missing_image", "label_noise"),
        counts=Counts(critical=1, warn=0, info=79),
        passed=False,
        findings=findings,
    )
    out = audit_summary_of(run, 500)
    assert len(out["top_findings"]) == 50 and out["truncated"] is True
    assert out["top_findings"][0]["severity"] == "CRITICAL"
    assert all(len(f["payload"]) <= PAYLOAD_CHARS + 20 for f in out["top_findings"])
    assert {"check": "label_noise", "severity": "INFO", "count": 79} in out["by_check"]
    json.dumps(out)  # plain JSON


# ---------------------------------------------------------------- write tools
def test_ingest_creates_a_new_dataset_and_never_overwrites(root: Path) -> None:
    src = synth.build_coco(root / "coco")
    server = create_server(roots=(root,), allow_write=True)
    error, out = call(server, "ingest", source=str(src), format="coco", output="n.dcp")
    assert not error
    assert out["manifest"]["image_count"] == 4 and (root / "n.dcp").is_dir()
    assert out["conversion"]["rows_read"] >= 4
    error, text = call(server, "ingest", source=str(src), format="coco", output="n.dcp")
    assert error and "already exists" in text
    error, text = call(server, "ingest", source=str(src), format="coco", output="n.txt")
    assert error and ".dcp" in text


def test_split_writes_a_new_version_with_lineage(root: Path, dcp: Path) -> None:
    error, out = call(
        create_server(roots=(root,), allow_write=True),
        "split",
        dataset=str(dcp),
        output="s.dcp",
        ratios=[0.5, 0.25, 0.25],
    )
    assert not error
    parent = DatasetStore.open(dcp).manifest().version
    assert out["report"]["parent_version"] == parent
    assert parent in DatasetStore.open(root / "s.dcp").manifest().lineage


def test_apply_refuses_a_test_set_and_propose_needs_its_extra(root: Path, dcp: Path) -> None:
    ps = dc.propose(dcp, "test_boxes")
    server = create_server(roots=(root,), allow_write=True)
    error, text = call(
        server, "apply_proposals", dataset=str(dcp), set_id=ps.set_id, output="p.dcp"
    )
    assert error and not (root / "p.dcp").exists()
    if __import__("importlib.util").util.find_spec("transformers") is None:
        error, text = call(server, "propose", dataset=str(dcp))
        assert error and "vlm" in text


# ---------------------------------------------------------------- command line
def test_the_command_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "mcp" else real(name, *a)
    )
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 2
    assert "datacopilot[mcp]" in result.output


def test_the_command_serves_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[Any] = []
    monkeypatch.setattr("datacopilot.mcp_server.main", lambda: served.append(True))
    assert CliRunner().invoke(app, ["mcp"]).exit_code == 0
    assert served == [True]


def test_an_unexpected_failure_reaches_the_client_without_internals(
    root: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("/home/someone/private/path")

    monkeypatch.setattr("datacopilot.api.audit", boom)
    error, text = call(create_server(roots=(root,)), "run_audit", dataset=str(dcp))
    assert error and "internal error (RuntimeError)" in text
    assert "someone" not in text
