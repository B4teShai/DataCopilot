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
    assert "DuckDB" not in README
    for planned in ("datacopilot split", "dc.split", "FiftyOne", "SAM 3", "## Planned"):
        assert planned not in README, planned
    for built in ("datacopilot propose", "dc.propose", "dc.apply"):  # built in Phase 5
        assert built in README, built


def test_the_install_page_names_the_core_dependencies_and_no_heavy_one() -> None:
    page = (ROOT / "docs/guide/install.md").read_text(encoding="utf-8")
    for name in ("click", "jinja2", "numpy", "pyarrow", "pydantic", "structlog", "typer"):
        assert name in page, name
    assert "core dependencies only" not in page


def _prose(text: str) -> str:
    """The README without fenced code, HTML comments and URLs: what a reader actually reads."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"\]\(https?://[^)\s]+\)", "]", text)


def test_the_readme_stays_short_enough_to_read_in_a_minute() -> None:
    assert len(README.splitlines()) <= 90
    assert len(_prose(README).split()) <= 450


def test_the_readme_has_the_sections_a_newcomer_looks_for_in_order() -> None:
    headings = [ln for ln in README.splitlines() if ln.startswith("#")]
    assert headings == ["# DataCopilot", "## What is it?", "## Quick start", "## What you can do",
                        "## Learn more", "## License"]  # fmt: skip
    assert README.splitlines()[2].startswith("DataCopilot finds and fixes mistakes")


def test_the_readme_shows_exactly_one_picture_and_it_is_the_overview() -> None:
    images = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", README)
    assert len(images) == 1 and images[0].endswith("/docs/diagrams/png/overview.png"), images
    assert (ROOT / "docs/diagrams/png/overview.png").is_file()
    assert "<img" not in README


def test_the_readme_embeds_no_architecture_diagram() -> None:
    for name in ("context", "containers", "components", "data-model", "flow-a", "flow-b", "runtime",
                 "system-context"):  # fmt: skip
        assert f"diagrams/png/{name}.png" not in README, name


def test_the_readme_avoids_the_jargon_a_newcomer_cannot_parse() -> None:
    prose = _prose(README).lower()
    for word in ("headless", "sarif", "manifest.version", "hmac", "lineage", "quarantine",
                 "embedding cache", "trust policy", "gate"):  # fmt: skip
        assert word not in prose, word


def test_the_install_page_carries_the_release_status_line_exactly_when_the_readme_does() -> None:
    page = (ROOT / "docs/guide/install.md").read_text(encoding="utf-8")
    assert page.count(STATUS) == README.count(STATUS)
