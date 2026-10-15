import importlib.util
import re
import shutil
import subprocess  # noqa: S404  # runs the release helper scripts
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def tag():  # type: ignore[no-untyped-def]
    return module("check_release_tag")


def test_the_tag_must_equal_the_package_version(tag, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    init = tmp_path / "__init__.py"
    init.write_text('"""x."""\n\n__version__ = "1.2.3"\n')
    readme = tmp_path / "README.md"
    readme.write_text("# x\n")
    assert tag.main(["v1.2.3"], init, readme) == 0
    for wrong in ("v1.2.4", "1.2.3", "v1.2.3rc1", "v1.2", "", "v1.2.3 ", "refs/tags/v1.2.3"):
        assert tag.main([wrong], init, readme) == 1, wrong


def test_a_malformed_version_or_usage_is_refused(tag, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "0.0.1.dev0"\n')
    assert tag.main(["v0.0.1.dev0"], init) == 1  # a development version is never released
    init.write_text("no version here\n")
    assert tag.main(["v1.0.0"], init) == 1
    assert tag.main([], init) == 2 and tag.main(["a", "b"], init) == 2


def test_the_repository_version_is_readable_by_the_checker(tag) -> None:  # type: ignore[no-untyped-def]
    version = tag.read_version(ROOT / "src" / "datacopilot" / "__init__.py")
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:\.dev[0-9]+)?", version)


def test_the_script_runs_as_a_command(tmp_path: Path) -> None:
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(ROOT / "scripts" / "check_release_tag.py"), "v9.9.9"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1 and "error:" in done.stderr


def test_verify_install_needs_an_existing_wheel_and_known_pythons() -> None:
    verify = module("verify_install")
    assert verify.main(["/nonexistent/datacopilot-0-py3-none-any.whl"]) == 2
    assert verify.main([]) == 2


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_verify_install_reports_a_failing_environment(tmp_path: Path) -> None:
    verify = module("verify_install")
    bad = tmp_path / "datacopilot-0-py3-none-any.whl"
    bad.write_bytes(b"not a wheel")
    assert verify.main([str(bad), "--python", sys.version.split()[0].rsplit(".", 1)[0]]) == 1


def test_a_version_with_a_leading_zero_is_refused(tag, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    for version in ("01.0.0", "1.02.3", "1.0.00", "00.1.0"):
        init = tmp_path / "__init__.py"
        init.write_text(f'__version__ = "{version}"\n')
        assert tag.main([f"v{version}"], init) == 1, version  # PyPI would normalise it
    init.write_text('__version__ = "10.20.30"\n')
    (tmp_path / "README.md").write_text("# x\n")
    assert tag.main(["v10.20.30"], init, tmp_path / "README.md") == 0


def test_a_readme_that_still_says_not_published_blocks_the_release(tag, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "1.2.3"\n')
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n> **Release status:** not yet published.\n")
    assert tag.main(["v1.2.3"], init, readme) == 1
    readme.write_text("# x\n\n```bash\npip install datacopilot\n```\n")
    assert tag.main(["v1.2.3"], init, readme) == 0
    assert tag.main(["v1.2.3"], init, tmp_path / "missing.md") == 1
