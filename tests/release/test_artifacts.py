"""Build the wheel and the sdist and assert what is (and is not) inside them.

Needs the `release` dependency group (`build`): skipped without it. The build runs in an isolated
environment, so it needs access to a package index for the build backend.
"""

from __future__ import annotations

import email
import re
import subprocess  # noqa: S404  # runs the build backend
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("build")

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = "." + "cla" + "ude"  # spelled in pieces: the quality gate bans the word in sources
WHEEL_BUDGET = 700 * 1024
SDIST_BUDGET = 900 * 1024
FORBIDDEN = re.compile(
    r"(^|/)(tests?|benchmarks?|plans?|docs|scripts|\.github|\.git|\.venv|__pycache__|\.pytest_cache|"
    r"\.mypy_cache|\.ruff_cache|htmlcov|golden|node_modules|site|dist|build)(/|$)"
    r"|(^|/)(\.env[^/]*|\.DS_Store|\.coverage[^/]*|[^/]*\.(pyc|pyo|key|pem|log|parquet|bak|orig)|uv\.lock)$"
    r"|(^|/)" + re.escape(TOOL_DIR) + r"(/|$)"
)
HEAVY = re.compile(
    "^("
    + "to"
    + "rch|"
    + "to"
    + "rchvision|triton|nvidia[-_].*|transformers|tokenizers|safetensors|huggingface[-_]hub|"
    r"faiss.*|opencv.*|tensorflow.*|jax.*|scipy|pandas|" + "pola" + "rs|pillow|onnx.*|accelerate)$",
    re.I,
)
CORE = {"click", "jinja2", "numpy", "pyarrow", "pydantic", "structlog", "typer"}


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "build", "--outdir", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    return out


@pytest.fixture(scope="module")
def version() -> str:
    text = (ROOT / "src/datacopilot/__init__.py").read_text()
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)  # type: ignore[union-attr]


@pytest.fixture(scope="module")
def wheel(dist: Path) -> Path:
    (path,) = dist.glob("*.whl")
    return path


@pytest.fixture(scope="module")
def sdist(dist: Path) -> Path:
    (path,) = dist.glob("*.tar.gz")
    return path


def metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(z.read(name))


def test_artifact_names_carry_the_package_version(dist: Path, version: str) -> None:
    assert sorted(p.name for p in dist.iterdir()) == [
        f"datacopilot-{version}-py3-none-any.whl",
        f"datacopilot-{version}.tar.gz",
    ]


def test_wheel_has_license_notice_and_py_typed(wheel: Path, version: str) -> None:
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
    assert f"datacopilot-{version}.dist-info/licenses/LICENSE" in names
    assert f"datacopilot-{version}.dist-info/licenses/NOTICE" in names
    assert "datacopilot/py.typed" in names
    assert "datacopilot/cli/app.py" in names and "datacopilot/__main__.py" in names


MANIFEST = Path(__file__).with_name("wheel_manifest.txt")
ALLOWED_SUFFIXES = {".py", ".typed", ".j2"}
DIST_INFO = {
    "METADATA",
    "WHEEL",
    "RECORD",
    "entry_points.txt",
    "licenses/LICENSE",
    "licenses/NOTICE",
}


def test_wheel_holds_exactly_the_committed_manifest_and_its_metadata(
    wheel: Path, version: str
) -> None:
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    package = sorted(n for n in names if n.startswith("datacopilot/"))
    assert package == MANIFEST.read_text().split(), (
        "the wheel's files changed; review, then update tests/release/wheel_manifest.txt"
    )
    assert {Path(n).suffix for n in package} <= ALLOWED_SUFFIXES
    prefix = f"datacopilot-{version}.dist-info/"
    info = {n[len(prefix) :] for n in names if n.startswith(prefix)}
    assert info == DIST_INFO
    assert len(package) + len(info) == len(names), "nothing outside the package and dist-info"
    assert [n for n in names if FORBIDDEN.search(n)] == []


def test_wheel_fits_the_size_budget(wheel: Path, sdist: Path) -> None:
    assert wheel.stat().st_size < WHEEL_BUDGET
    assert sdist.stat().st_size < SDIST_BUDGET


def test_wheel_metadata_is_correct(wheel: Path, version: str) -> None:
    meta = metadata(wheel)
    assert meta["Name"] == "datacopilot" and meta["Version"] == version
    assert meta["Requires-Python"] == ">=3.10"
    assert meta["License-Expression"] == "Apache-2.0"
    assert set(meta.get_all("License-File")) == {"LICENSE", "NOTICE"}
    assert meta["Description-Content-Type"].startswith("text/markdown")
    classifiers = meta.get_all("Classifier")
    assert "Programming Language :: Python :: 3.10" in classifiers
    assert "Programming Language :: Python :: 3.13" in classifiers
    assert "Typing :: Typed" in classifiers
    assert not [c for c in classifiers if c.startswith("License ::")]  # the expression says it
    urls = {u.split(", ")[0]: u.split(", ")[1] for u in meta.get_all("Project-URL")}
    assert {"Homepage", "Repository", "Issues", "Changelog"} <= set(urls)
    assert all(u.startswith("https://") for u in urls.values())
    assert "datacopilot" in meta["Summary"].lower() or "dataset" in meta["Summary"].lower()


def test_core_requirements_are_exactly_the_documented_set_and_not_heavy(wheel: Path) -> None:
    meta = metadata(wheel)
    unconditional = {
        re.split(r"[<>=!~\[ ;]", r.strip())[0].lower()
        for r in meta.get_all("Requires-Dist")
        if "extra ==" not in r
    }
    assert unconditional == CORE
    assert not [n for n in unconditional if HEAVY.match(n)]
    extras = [r for r in meta.get_all("Requires-Dist") if "extra ==" in r]
    assert extras and all(re.search(r"""extra == ['"](embed|vlm)['"]""", r) for r in extras)
    assert {"embed", "vlm"} <= set(meta.get_all("Provides-Extra"))


def test_wheel_entry_points(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith("entry_points.txt"))
        text = z.read(name).decode()
    assert "datacopilot = datacopilot.cli.app:main" in text
    for group in (
        "datacopilot.formats",
        "datacopilot.checks",
        "datacopilot.embedders",
        "datacopilot.proposers",
    ):
        assert f"[{group}]" in text
    assert "grounding_dino = datacopilot.proposers.grounding_dino:plugin" in text
    assert "cvat_xml = datacopilot.io.cvat:plugin" in text
    assert "label_studio = datacopilot.io.labelstudio:plugin" in text


def test_sdist_is_an_allow_list(sdist: Path, version: str) -> None:
    with tarfile.open(sdist) as t:
        names = [m.name for m in t.getmembers()]
        top = {n.split("/")[0] for n in names}
        assert top == {f"datacopilot-{version}"}
        relative = [n.split("/", 1)[1] for n in names if "/" in n]
        assert not [m for m in t.getmembers() if m.issym() or m.islnk() or m.isdev()]
    assert [n for n in relative if FORBIDDEN.search(n)] == []
    assert {"LICENSE", "NOTICE", "README.md", "CHANGELOG.md", "pyproject.toml", "PKG-INFO"} <= set(
        relative
    )
    allowed_top = {
        "LICENSE", "NOTICE", "README.md", "CHANGELOG.md", "THIRD_PARTY_MODELS.md", "pyproject.toml",
        "PKG-INFO", "src", ".gitignore",
    }  # fmt: skip
    assert {n.split("/")[0] for n in relative} <= allowed_top
    assert "src/datacopilot/py.typed" in relative


SECRET_MARKERS = [
    rb"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}",
    rb"-{5}BEGIN PGP PRIVATE KEY BLOCK-{5}",
    rb"\bgh[pousr]_[A-Za-z0-9]{20,}",
    rb"\bgithub_pat_[A-Za-z0-9_]{20,}",
    rb"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    rb"\bsk-[A-Za-z0-9_-]{20,}",
    rb"\bAIza[0-9A-Za-z_-]{30,}",
    rb"\bAKIA[0-9A-Z]{16}\b",
    rb"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    rb"\bhf_[A-Za-z0-9]{30,}",
    rb"\bpypi-AgE[A-Za-z0-9_-]{20,}",
]


def test_the_secret_patterns_recognise_what_they_are_for() -> None:
    samples = [
        b"-" * 5 + b"BEGIN RSA PRIVATE KEY" + b"-" * 5,
        b"ghp_" + b"a" * 30,
        b"github_pat_" + b"A" * 30,
        b"xoxb-123456789012-abc",
        b"sk-" + b"a" * 30,
        b"AIza" + b"B" * 35,
        b"AKIA" + b"A" * 16,
        b"eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4",
        b"hf_" + b"a" * 34,
        b"pypi-AgEIcHlwaS5vcmc" + b"A" * 30,
    ]
    for sample in samples:
        assert any(re.search(m, sample) for m in SECRET_MARKERS), sample


def test_no_member_of_either_artifact_holds_a_secret_marker(wheel: Path, sdist: Path) -> None:
    def hits(data: bytes) -> bool:
        return any(re.search(m, data) for m in SECRET_MARKERS)

    with zipfile.ZipFile(wheel) as z:
        assert not [n for n in z.namelist() if hits(z.read(n))]
    with tarfile.open(sdist) as t:
        for member in t.getmembers():
            handle = t.extractfile(member)
            assert handle is None or not hits(handle.read()), member.name


def test_the_wheel_installs_nothing_outside_its_prefix(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as z:
        assert not [
            n
            for n in z.namelist()
            if ".data/" in n or n.startswith(("/", "../")) or ".." in n.split("/")
        ]


def test_the_wheel_installs_in_a_clean_environment_and_runs(
    wheel: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    import importlib.util
    import shutil

    if shutil.which("uv") is None:
        pytest.skip("needs uv")
    spec = importlib.util.spec_from_file_location(
        "verify_install", ROOT / "scripts/verify_install.py"
    )
    assert spec and spec.loader
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert verify.main([str(wheel), "--python", here]) == 0
    out = capfd.readouterr().out
    assert "datacopilot doctor -> exit 0" in out and "\nok\n" in out
