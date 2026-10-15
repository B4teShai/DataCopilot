"""The build environment is created outside the lock file, so every requirement of it is pinned."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
EXACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.!+_-]*(?:\s*;.*)?$")


def test_every_build_requirement_is_an_exact_pin() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires = config["build-system"]["requires"]
    assert requires and all(EXACT.match(r) for r in requires), requires
    assert any(r.startswith("hatchling==") for r in requires)
    assert config["build-system"]["build-backend"] == "hatchling.build"


def test_the_release_workflow_builds_with_the_pinned_backend_not_a_floating_one() -> None:
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "hatchling" not in text and "--no-build-isolation" not in text
    assert "uv build" in text
