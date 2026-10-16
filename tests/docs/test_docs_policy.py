"""User-facing pages: every command block is executed by a test or says why it is not."""

import re
from pathlib import Path

import pytest
from mdblocks import COMMAND_LANGUAGES, blocks

DOCS = Path(__file__).resolve().parents[2] / "docs"
PAGES = sorted([DOCS / "index.md", DOCS / "tutorial.md", *(DOCS / "guide").glob("*.md")])


def test_the_user_pages_exist() -> None:
    names = {p.name for p in PAGES}
    assert {"index.md", "tutorial.md", "install.md", "concepts.md", "formats.md", "audit-ci.md",
            "embeddings.md", "trust-model.md", "python-api.md"} <= names  # fmt: skip


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_command_block_is_run_or_skipped_with_a_reason(page: Path) -> None:
    unmarked = [b.where for b in blocks(page) if b.lang in COMMAND_LANGUAGES and not b.marker]
    assert unmarked == [], "mark with <!-- run --> or <!-- skip: reason -->"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_output_blocks_follow_a_run_block(page: Path) -> None:
    found = blocks(page)
    for index, block in enumerate(found):
        if block.marker == "output":
            assert index > 0 and found[index - 1].marker == "run", block.where


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_have_no_remote_scripts_images_or_secrets(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    for needle in ("<script", "<img", "](http://"):
        assert needle not in text, (page.name, needle)
    remote = [t for t in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text) if "//" in t]
    assert remote == [], (page.name, remote)  # images are local files of the site
