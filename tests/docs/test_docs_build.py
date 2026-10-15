"""The site builds in strict mode (broken links, anchors and nav entries fail the build) and loads
nothing from anywhere but itself. Needs the `docs` dependency group: skipped without it."""

from __future__ import annotations

import re
import subprocess  # noqa: S404  # runs the documented build command
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs")
pytest.importorskip("material")

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ATTR = re.compile(
    r"""<(?:script|img|source|iframe|link)\b[^>]*\b(?:src|href)=["'](?:https?:)?//""", re.I
)
REMOTE_CSS = re.compile(r"""(?:@import\s+(?:url\()?["']?|url\(["']?)(?:https?:)?//""", re.I)


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--clean", "-d", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    return out


def test_the_strict_build_produces_the_pages(site: Path) -> None:
    for page in ("index.html", "tutorial/index.html", "reference/cli/index.html",
                 "guide/trust-model/index.html", "architecture/03-data-model/index.html",
                 "engineering/release/index.html"):  # fmt: skip
        assert (site / page).is_file(), page


def test_no_page_loads_a_remote_script_style_font_or_image(site: Path) -> None:
    bad = []
    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        if REMOTE_ATTR.search(text) and not re.search(r'rel="(?:canonical|alternate)"', text):
            bad.append(path.name)
        for match in REMOTE_ATTR.finditer(text):
            if "canonical" not in match.group(0) and 'rel="alternate"' not in match.group(0):
                bad.append(f"{path.relative_to(site)}: {match.group(0)[:80]}")
    for path in site.rglob("*.css"):
        if REMOTE_CSS.search(path.read_text(encoding="utf-8")):
            bad.append(str(path.relative_to(site)))
    assert bad == []


def test_no_analytics_fonts_or_trackers(site: Path) -> None:
    forbidden = (
        "fonts.googleapis",
        "fonts.gstatic",
        "google-analytics",
        "googletagmanager",
        "gtag(",
    )
    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert not [f for f in forbidden if f in text], path.name


def test_the_built_site_is_not_committed() -> None:
    assert "/site/" in (ROOT / ".gitignore").read_text().splitlines()
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files", "site"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,  # noqa: S607
    )
    assert tracked.stdout.strip() == ""


def test_the_dependency_group_pins_exact_versions() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    block = text.split("docs = [", 1)[1].split("]", 1)[0]
    pins = [ln.strip().strip(",").strip('"') for ln in block.splitlines() if "==" in ln]
    assert len(pins) >= 3 and all("==" in p and ">" not in p for p in pins)
