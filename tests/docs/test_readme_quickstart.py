"""The README's quick start is executed against the tutorial's generated demo dataset."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

from mdblocks import blocks
from test_tutorial import _check, _env, _human_lines, _run

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
TUTORIAL = ROOT / "docs" / "tutorial.md"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
EXIT = re.compile(r"exit (\d+)")


def _commands() -> list[tuple[list[str], int, list[str]]]:
    console = [b for b in blocks(README) if b.lang == "console"]
    assert len(console) == 1, "the quick start is exactly one console block"
    commands: list[tuple[list[str], int, list[str]]] = []
    for line in console[0].text.splitlines():
        if line.startswith("$ "):
            command, _, comment = line[2:].partition("#")
            found = EXIT.search(comment)
            commands.append((shlex.split(command), 1 if found else 0, []))
        elif line.strip():
            commands[-1][2].append(UUID.sub("<uuid>", HEX32.sub("<hex32>", line.strip())))
    return commands


def test_the_quick_start_is_short() -> None:
    commands = _commands()
    assert 4 <= len(commands) <= 5
    assert {argv[1] for argv, _, _ in commands} == {"ingest", "export", "audit"}


def test_the_quick_start_runs_and_prints_what_the_readme_shows(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    env = _env(tmp_path / "home")
    generator = next(b for b in blocks(TUTORIAL) if b.marker == "run" and b.lang == "python")
    done = _run([sys.executable, "-c", generator.text], work, env)
    assert done.returncode == 0, done.stderr[-800:]
    assert (work / "shop" / "annotations" / "train.json").is_file()  # what the README promises
    for argv, code, expected in _commands():
        assert argv[0] == "datacopilot"
        done = _run([sys.executable, "-m", "datacopilot", *argv[1:]], work, env)
        assert done.returncode == code, f"{argv}: exit {done.returncode}"
        _check(expected, _human_lines(done), " ".join(argv))
    assert (work / "shop_yolo" / "data.yaml").is_file()


def test_the_readme_names_the_demo_dataset_the_tutorial_generates() -> None:
    assert "`shop/`" in README.read_text(encoding="utf-8")
    assert 'root = Path("shop")' in TUTORIAL.read_text(encoding="utf-8")
