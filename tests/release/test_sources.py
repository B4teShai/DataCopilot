"""The source tree that gets packaged: regular files only, and exactly the committed manifest."""

from __future__ import annotations

import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ALLOWED_SUFFIXES = {".py", ".typed", ".j2"}


def walk() -> tuple[list[str], list[str]]:
    files, odd = [], []
    for directory, subdirs, names in os.walk(SRC, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d != "__pycache__")
        for name in subdirs + names:
            path = Path(directory) / name
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                odd.append(str(path.relative_to(SRC)))
        files += [str((Path(directory) / n).relative_to(SRC)) for n in names if n != ".DS_Store"]
    return sorted(f for f in files if not f.endswith(".pyc")), odd


def test_source_tree_has_no_symlink_or_special_file() -> None:
    _, odd = walk()
    assert odd == []


def test_source_tree_matches_the_wheel_manifest() -> None:
    files, _ = walk()
    manifest = (ROOT / "tests/release/wheel_manifest.txt").read_text().split()
    assert files == manifest, "update tests/release/wheel_manifest.txt after reviewing the change"
    assert {Path(f).suffix for f in files} <= ALLOWED_SUFFIXES
