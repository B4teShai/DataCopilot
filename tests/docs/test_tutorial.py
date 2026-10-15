"""Every command and script of the tutorial is executed and its output compared with the page."""

from __future__ import annotations

import os
import re
import shlex
import subprocess  # noqa: S404  # runs the documented commands
import sys
from pathlib import Path

import pytest
from mdblocks import Block, blocks

DOCS = Path(__file__).resolve().parents[2] / "docs"
TUTORIAL = DOCS / "tutorial.md"
RUNNABLE = [TUTORIAL, DOCS / "guide" / "python-api.md"]
PLACEHOLDERS = {
    "<uuid>": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    "<hex32>": r"[0-9a-f]{32}",
    "<number>": r"[0-9]+",
    "...": r".*",
}
_TOKEN = re.compile("(" + "|".join(re.escape(k) for k in PLACEHOLDERS) + ")")
_EXIT = re.compile(r"\s+#\s*exit\s+(\d+)\s*$")


def _pattern(expected: str) -> re.Pattern[str]:
    parts = [PLACEHOLDERS.get(p, re.escape(p)) for p in _TOKEN.split(expected)]
    return re.compile("".join(parts))


def _normal(line: str) -> str:
    return " ".join(line.split())


def _env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DATACOPILOT_") and k != "CI"}
    env.update(
        HOME=str(home),
        XDG_CACHE_HOME=str(home / "xdg"),
        DATACOPILOT_CACHE_DIR=str(home / "cache"),
        PYTHONIOENCODING="utf-8",
    )
    return env


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=180, check=False
    )


def _check(expected: list[str], actual: list[str], where: str) -> None:
    assert len(expected) == len(actual), f"{where}: expected {expected!r}, got {actual!r}"
    for want, got in zip(expected, actual, strict=True):
        assert _pattern(_normal(want)).fullmatch(_normal(got)), f"{where}: {want!r} != {got!r}"


def _human_lines(done: subprocess.CompletedProcess[str]) -> list[str]:
    stderr = [ln for ln in done.stderr.splitlines() if ln.strip() and not ln.startswith("{")]
    return [ln for ln in done.stdout.splitlines() if ln.strip()] + stderr


def test_every_command_block_of_the_tutorial_is_marked() -> None:
    from mdblocks import COMMAND_LANGUAGES

    unmarked = [b.where for b in blocks(TUTORIAL) if b.lang in COMMAND_LANGUAGES and not b.marker]
    assert unmarked == []


@pytest.mark.parametrize("page", RUNNABLE, ids=lambda p: p.name)
def test_the_page_runs_and_its_output_is_what_it_shows(page: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    env = _env(tmp_path / "home")
    found = blocks(page)
    executed = 0
    for index, block in enumerate(found):
        if block.marker != "run":
            continue
        executed += 1
        if block.lang == "console":
            _run_console(block, work, env)
        else:
            assert block.lang == "python", block.where
            done = _run([sys.executable, "-c", block.text], work, env)
            assert done.returncode == 0, f"{block.where}: {done.stderr[-800:]}"
            following = found[index + 1] if index + 1 < len(found) else None
            expected = (
                following.text.splitlines() if following and following.marker == "output" else []
            )
            _check(
                [ln for ln in expected if ln.strip()],
                [ln for ln in done.stdout.splitlines() if ln.strip()],
                block.where,
            )
    assert executed >= 1
    if page == TUTORIAL:
        assert executed >= 10
        assert (work / "shop.dcp" / "manifest.json").is_file()
        assert (work / "report.sarif").is_file()


def _run_console(block: Block, work: Path, env: dict[str, str]) -> None:
    commands: list[tuple[str, int, list[str]]] = []
    for line in block.text.splitlines():
        if line.startswith("$ "):
            text = line[2:]
            exit_match = _EXIT.search(text)
            code = int(exit_match.group(1)) if exit_match else 0
            commands.append((_EXIT.sub("", text), code, []))
        elif commands and line.strip():
            commands[-1][2].append(line)
    assert commands, block.where
    for text, code, expected in commands:
        argv = shlex.split(text)
        assert argv[0] in ("datacopilot", "python"), f"{block.where}: {argv[0]}"
        argv = (
            [sys.executable, "-m", "datacopilot", *argv[1:]]
            if argv[0] == "datacopilot"
            else [sys.executable, *argv[1:]]
        )
        done = _run(argv, work, env)
        assert done.returncode == code, f"{block.where}: {text}: exit {done.returncode}"
        _check(expected, _human_lines(done), f"{block.where}: {text}")
