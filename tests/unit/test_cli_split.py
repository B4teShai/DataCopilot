"""``datacopilot split`` through the real CLI path: exit codes, output, leakage check after."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_cli_embedding import write_coco

from datacopilot.cli import app as cli
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CI", "DATACOPILOT_CACHE_KEY", "DATACOPILOT_ALLOWED_LICENSES", "DATACOPILOT_ENV"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))
    yield
    clear_request_context()


def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


@pytest.fixture
def leaky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_coco(tmp_path / "src", leak=True, extra_train=20)
    assert (
        run(monkeypatch, "ingest", tmp_path / "src", "--format", "coco", "-o", tmp_path / "ds.dcp")
        == 0
    )
    return tmp_path / "ds.dcp"


def leakage_gate(monkeypatch: pytest.MonkeyPatch, path: Path, tmp_path: Path) -> int:
    return run(
        monkeypatch, "audit", path, "--checks", "leakage", "--embedder", "hash",
        "--allow-test-embedder", "--fail-on", "leakage", "--no-hash-cache",
        "--report", tmp_path / f"{path.stem}.html",
    )  # fmt: skip


def test_split_by_near_copies_then_the_leakage_gate_passes(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert leakage_gate(monkeypatch, leaky, tmp_path) == 1  # the planted leak fails the gate
    capsys.readouterr()
    out = tmp_path / "split.dcp"
    code = run(
        monkeypatch, "split", leaky, "-o", out, "--ratios", "0.7,0.15,0.15", "--near-dup",
        "--embedder", "hash", "--allow-test-embedder", "--stratify", "category", "--seed", "3",
        "--json", tmp_path / "split.json",
    )  # fmt: skip
    printed = capsys.readouterr()
    assert code == 0, printed.err
    lines = printed.out.splitlines()
    assert lines[0] == DatasetStore.open(out).manifest().version
    assert lines[1].startswith("images=24 groups=") and "near_duplicate_pairs=" in lines[1]
    assert [ln.split()[0] for ln in lines[2:5]] == ["train", "val", "test"]
    report = json.loads((tmp_path / "split.json").read_text())
    assert report["groups_across_splits"] == 0 and report["near_duplicate_pairs"] >= 1
    assert "embedder 'hash' is a test double" in printed.err
    assert leakage_gate(monkeypatch, out, tmp_path) == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--ratios", "0.5,0.5"],
        ["--ratios", "0.5,0.6,0.1"],
        ["--by-group", "re:(a+)+"],
        ["--by-group", "folder"],
        ["--near-dup"],
        ["--embedder", "hash"],
        ["--similarity-threshold", "1.5", "--near-dup", "--embedder", "hash"],
        ["--stratify", "size"],
        ["--near-dup", "--embedder", "hash"],  # a test double needs --allow-test-embedder
        ["--seed", "-1"],
    ],
)
def test_bad_options_are_usage_errors(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    assert run(monkeypatch, "split", leaky, "-o", tmp_path / "s.dcp", *args) == 2
    assert not (tmp_path / "s.dcp").exists()


def test_the_output_must_be_a_new_dcp_directory(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "split", leaky, "-o", tmp_path / "plain") == 2
    (tmp_path / "taken.dcp").mkdir()
    capsys.readouterr()
    assert run(monkeypatch, "split", leaky, "-o", tmp_path / "taken.dcp") == 1
    assert "already exists" in capsys.readouterr().err
    assert run(monkeypatch, "split", leaky, "-o", leaky / "inner.dcp") == 1


def test_an_unembeddable_image_refuses_the_split_with_exit_1(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "src" / "images" / "train3.png").unlink()
    capsys.readouterr()
    code = run(
        monkeypatch, "split", leaky, "-o", tmp_path / "s.dcp", "--near-dup", "--embedder", "hash",
        "--allow-test-embedder",
    )  # fmt: skip
    assert code == 1 and "file_missing=1" in capsys.readouterr().err
    assert not (tmp_path / "s.dcp").exists()


def test_the_json_report_is_never_written_into_the_dataset_or_over_a_file(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = sorted(p.name for p in leaky.rglob("*"))
    assert (
        run(monkeypatch, "split", leaky, "-o", tmp_path / "a.dcp", "--json", leaky / "r.json") == 2
    )
    assert sorted(p.name for p in leaky.rglob("*")) == before and not (tmp_path / "a.dcp").exists()
    (tmp_path / "taken.json").write_text("keep")
    assert (
        run(
            monkeypatch, "split", leaky, "-o", tmp_path / "b.dcp", "--json", tmp_path / "taken.json"
        )
        == 2
    )
    assert (tmp_path / "taken.json").read_text() == "keep"
    (tmp_path / "link.json").symlink_to(tmp_path / "target.json")
    assert (
        run(monkeypatch, "split", leaky, "-o", tmp_path / "c.dcp", "--json", tmp_path / "link.json")
        == 2
    )
    assert not (tmp_path / "target.json").exists()
