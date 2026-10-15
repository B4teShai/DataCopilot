"""The README is the package index's long description: it must read correctly on PyPI and on GitHub,
before and after the release is published."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
STATUS = "> **Release status:**"


def test_every_link_is_absolute_so_it_works_on_the_package_index() -> None:
    for target in LINK.findall(README):
        assert target.startswith(("https://", "#", "mailto:")), target


def test_no_placeholder_or_unfinished_text() -> None:
    for needle in ("<repository-url>", "TODO", "FIXME", "XXX", "lorem", "your-org"):
        assert needle not in README, needle
    assert "http://" not in README


def test_there_is_at_most_one_release_status_line_and_it_is_the_only_unpublished_claim() -> None:
    lines = [ln for ln in README.splitlines() if ln.startswith(STATUS)]
    assert len(lines) <= 1
    claims = [
        ln for ln in README.splitlines()
        if re.search(r"not (yet )?(been )?published|unpublished", ln, re.I)
    ]  # fmt: skip
    assert all(ln.startswith(STATUS) for ln in claims), claims


def test_the_install_command_is_the_package_name() -> None:
    assert "pip install datacopilot\n" in README


def test_every_extra_the_readme_names_exists_and_none_is_empty() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]
    assert extras and all(extras.values()), "an empty extra installs nothing"
    for name in re.findall(r"datacopilot\[([a-z0-9_,-]+)\]", README):
        assert set(name.split(",")) <= set(extras), name


def test_the_readme_does_not_claim_what_is_not_built() -> None:
    assert "DuckDB" not in README.split("## Planned usage")[0]
    core = ("click", "jinja2", "numpy", "pyarrow", "pydantic", "structlog", "typer")
    section = README.split("## Install")[1].split("## Usage")[0]
    for name in core:
        assert name in section, name
    assert "core dependencies only" not in section


def test_the_install_page_carries_the_release_status_line_exactly_when_the_readme_does() -> None:
    page = (ROOT / "docs/guide/install.md").read_text(encoding="utf-8")
    assert page.count(STATUS) == README.count(STATUS)
