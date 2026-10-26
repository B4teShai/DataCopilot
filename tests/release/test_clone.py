"""A fresh clone installs and runs: nothing is published, so a clone is how everyone gets it.

The real run (``-m clone``) clones the committed HEAD, installs it with pip into an empty virtual
environment and runs ``--version``, ``doctor`` and the README quick start through
``scripts/check_clone.sh``. It needs the network for the dependencies and takes a minute or two, so
it is off by default; the shape of the script is checked on every run.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: S404  # runs the clone check script
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_clone.sh"
BASH = shutil.which("bash") or "/bin/bash"


def test_the_script_is_executable_strict_and_keeps_caches_inside_its_work_directory() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert os.access(SCRIPT, os.X_OK)
    assert "set -euo pipefail" in text
    for variable in ("HOME", "PIP_CACHE_DIR", "DATACOPILOT_CACHE_DIR", "XDG_CACHE_HOME"):
        assert f'{variable}="$work/' in text, variable
    assert 'pip install --quiet "$work/src"' in text  # the clone, not a package index name
    assert "--version" in text and "doctor" in text


def test_the_script_refuses_a_bad_call(tmp_path: Path) -> None:
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    for argv in (["a", "b", "c"], [sys.executable, str(tmp_path / "busy")]):
        done = subprocess.run(  # noqa: S603
            [BASH, str(SCRIPT), *argv], capture_output=True, text=True, timeout=60, check=False
        )
        assert done.returncode == 2, done.stderr


@pytest.mark.clone
def test_a_fresh_clone_installs_and_runs_the_readme_quick_start(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    python = os.environ.get("DATACOPILOT_CLONE_PYTHON", sys.executable)
    done = subprocess.run(  # noqa: S603
        [BASH, str(SCRIPT), python, str(tmp_path / "work")],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "ok: clone, install, --version, doctor and" in done.stdout
