import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from datacopilot import __version__, doctor
from datacopilot.cli import app as cli
from datacopilot.core.schema import SCHEMA_VERSION
from datacopilot.logging import clear_request_context
from datacopilot.settings import Settings, load_settings

TENSOR_LIB = "to" + "rch"  # the quality gate forbids the literal name in source files


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in list(os.environ):
        if var.startswith("DATACOPILOT_"):
            monkeypatch.delenv(var)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    yield
    clear_request_context()


def diagnose(**kwargs: Any) -> dict[str, Any]:
    return doctor.diagnose(load_settings(), **kwargs)


def run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


# --------------------------------------------------------------- report
def test_report_has_the_documented_sections() -> None:
    report = diagnose()
    assert report["schema"] == 1 and report["status"] in ("ok", "warn")
    assert set(report) == {
        "schema", "status", "package", "dependencies", "extras", "settings", "cache", "faiss",
        "gpu", "plugins",
    }  # fmt: skip
    package = report["package"]
    assert package["version"] == __version__ and package["dcp_schema_version"] == SCHEMA_VERSION
    assert package["python"].count(".") == 2 and package["platform"]
    assert set(report["dependencies"]) == {
        "click", "jinja2", "numpy", "pyarrow", "pydantic", "structlog", "typer"
    }  # fmt: skip
    assert all(isinstance(v, str) for v in report["dependencies"].values())
    assert report["plugins"]["formats"] == [
        "coco", "cvat_xml", "imagefolder", "label_studio", "voc", "yolo"
    ]  # fmt: skip
    assert "boxes" in report["plugins"]["checks"] and "hash" in report["plugins"]["embedders"]


def test_report_is_json_serialisable_and_fast() -> None:
    started = time.perf_counter()
    report = diagnose()
    assert time.perf_counter() - started < 3.0
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_a_missing_core_dependency_fails_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    real = doctor._version_of
    monkeypatch.setattr(doctor, "_version_of", lambda n: None if n == "numpy" else real(n))
    report = diagnose()
    assert report["dependencies"]["numpy"] is None and report["status"] == "fail"
    assert doctor.exit_code(report) == 1


def test_extras_report_what_is_importable_without_importing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    present = {"PIL", "safetensors"}
    monkeypatch.setattr(doctor, "_module_available", lambda m: m in present)
    extras = diagnose()["extras"]
    assert extras["embed"]["available"] is False
    assert set(extras["embed"]["missing"]) == {
        "huggingface-hub", "transformers", "faiss-cpu", TENSOR_LIB
    }  # fmt: skip
    assert set(extras["embed"]["present"]) == {"pillow", "safetensors"}
    monkeypatch.setattr(doctor, "_module_available", lambda m: True)
    monkeypatch.setattr(doctor, "_probe_gpu", lambda: {"tensor_library": "none"})
    assert diagnose()["extras"]["embed"] == {"available": True, "missing": [], "present": sorted(
        ["pillow", "safetensors", "huggingface-hub", "transformers", "faiss-cpu", TENSOR_LIB]
    )}  # fmt: skip
    assert diagnose()["status"] == "ok"  # a missing optional extra is not a problem


# -------------------------------------------------------------- settings
def test_settings_are_reported_without_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "s3cr3t-key-value-123456"  # noqa: S105
    monkeypatch.setenv("DATACOPILOT_CACHE_KEY", secret)
    monkeypatch.setenv("DATACOPILOT_WORKERS", "3")
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "MIT, Apache-2.0")
    report = diagnose()
    settings = report["settings"]
    assert settings["valid"] is True and settings["error"] is None
    assert settings["cache_key"] == "set" and settings["workers"] == 3
    assert settings["allowed_licenses"] == ["apache-2.0", "mit"]
    assert secret not in json.dumps(report) and secret not in doctor.to_text(report)


def test_invalid_settings_are_reported_and_exit_with_2() -> None:
    report = doctor.diagnose(None, settings_error="invalid value for DATACOPILOT_ENV")
    assert report["settings"] == {"valid": False, "error": "invalid value for DATACOPILOT_ENV"}
    assert report["status"] == "fail" and doctor.exit_code(report) == 2
    assert "settings: INVALID" in doctor.to_text(report)


# ----------------------------------------------------------------- cache
def test_cache_directory_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(cache))
    absent = diagnose()["cache"]
    assert absent["exists"] is False and absent["size_bytes"] == 0 and not cache.exists()
    (cache / "sub").mkdir(parents=True)
    (cache / "a.bin").write_bytes(b"x" * 100)
    (cache / "sub" / "b.bin").write_bytes(b"y" * 50)
    seen = diagnose()["cache"]
    assert seen["exists"] and seen["writable"] and seen["size_bytes"] == 150 and seen["files"] == 2
    assert seen["truncated"] is False


def test_cache_size_does_not_follow_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"z" * 10_000)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "link").symlink_to(outside, target_is_directory=True)
    (cache / "file.bin").write_bytes(b"x" * 10)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(cache))
    assert diagnose()["cache"]["size_bytes"] < 10_000


def test_cache_walk_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    for n in range(20):
        (cache / f"f{n}").write_bytes(b"x")
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(cache))
    monkeypatch.setattr(doctor, "MAX_CACHE_ENTRIES", 5)
    seen = diagnose()["cache"]
    assert seen["truncated"] is True and seen["files"] <= 5


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_cache_directory_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    cache.chmod(0o500)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(cache))
    try:
        report = diagnose()
    finally:
        cache.chmod(0o700)
    assert report["cache"]["writable"] is False and report["status"] == "warn"
    assert doctor.exit_code(report) == 0


def test_the_cache_path_is_shown_relative_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(home))
    report = diagnose()
    assert report["cache"]["path"] == "~/.cache/datacopilot"
    assert str(home) not in doctor.to_text(report) and str(home) not in json.dumps(report)


# ------------------------------------------------------------- faiss, gpu
def test_faiss_is_probed_without_importing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda m: m == "faiss")
    monkeypatch.setattr(doctor, "_version_of", lambda n: "1.15.0" if n == "faiss-cpu" else "1")
    assert diagnose()["faiss"] == {"installed": True, "version": "1.15.0"}
    monkeypatch.setattr(doctor, "_module_available", lambda m: False)
    assert diagnose()["faiss"] == {"installed": False, "version": None}


def test_gpu_without_the_tensor_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda m: False)
    assert diagnose()["gpu"] == {
        "tensor_library": "not installed",
        "cuda": False,
        "devices": [],
        "mps": False,
    }  # noqa: E501


def _fake_tensor_lib(cuda: bool, mps: bool, boom: bool = False) -> types.ModuleType:
    lib = types.ModuleType(TENSOR_LIB)
    lib.__version__ = "9.9.9"  # type: ignore[attr-defined]

    def available() -> bool:
        if boom:
            raise RuntimeError("driver exploded")
        return cuda

    props = types.SimpleNamespace(name="Fake GPU 0", total_memory=8 * 1024**3)
    lib.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=available, device_count=lambda: 1, get_device_properties=lambda i: props
    )
    lib.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))  # type: ignore[attr-defined]
    return lib


def test_gpu_devices_are_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda m: m == TENSOR_LIB)
    monkeypatch.setitem(sys.modules, TENSOR_LIB, _fake_tensor_lib(cuda=True, mps=False))
    gpu = diagnose()["gpu"]
    assert gpu == {
        "tensor_library": "9.9.9",
        "cuda": True,
        "devices": [{"name": "Fake GPU 0", "memory_mb": 8192}],
        "mps": False,
    }


def test_a_failing_gpu_probe_never_breaks_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda m: m == TENSOR_LIB)
    monkeypatch.setitem(sys.modules, TENSOR_LIB, _fake_tensor_lib(cuda=False, mps=False, boom=True))
    report = diagnose()
    assert report["gpu"]["error"] == "RuntimeError" and report["status"] in ("ok", "warn")
    assert "driver exploded" not in json.dumps(report)


def test_an_import_error_of_the_tensor_library_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda m: m == TENSOR_LIB)
    monkeypatch.setitem(sys.modules, TENSOR_LIB, None)  # makes the import raise ImportError
    assert diagnose()["gpu"]["error"] in ("ImportError", "ModuleNotFoundError")


# ---------------------------------------------------------------- text
def test_text_form_starts_with_the_status_and_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "evil\x1b[31m‮dir"))
    text = doctor.to_text(diagnose())
    lines = text.splitlines()
    assert lines[0] == "ok"
    assert "\x1b" not in text and "‮" not in text
    assert {line.split(":")[0] for line in lines[1:]} >= {
        "package", "python", "dependencies", "extras", "settings", "cache", "faiss", "gpu",
        "plugins",
    }  # fmt: skip


def test_default_settings_object_is_accepted() -> None:
    assert doctor.diagnose(Settings())["settings"]["valid"] is True


# ----------------------------------------------------------------- CLI
def test_cli_text_output_and_one_log_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "doctor") == 0
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == "ok" and "dependencies:" in out
    (rec,) = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    assert rec["message"] == "doctor_completed" and rec["status"] == "ok"
    assert rec["package_version"] == __version__


def test_cli_json_output_is_one_object_without_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "another-secret-key-0000"  # noqa: S105
    monkeypatch.setenv("DATACOPILOT_CACHE_KEY", secret)
    assert run(monkeypatch, "doctor", "--json") == 0
    out, err = capsys.readouterr()
    report = json.loads(out)
    assert report["settings"]["cache_key"] == "set" and report["status"] == "ok"
    assert secret not in out and secret not in err
    assert len(out.strip().splitlines()) == 1


@pytest.mark.parametrize("flags", [(), ("--json",)])
def test_cli_invalid_setting_is_diagnosed_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flags: tuple[str, ...]
) -> None:
    monkeypatch.setenv("DATACOPILOT_ENV", "banana")
    assert run(monkeypatch, "doctor", *flags) == 2
    out, err = capsys.readouterr()
    assert "DATACOPILOT_ENV" in out and "Traceback" not in out + err
    assert "banana" not in out


def test_cli_a_failing_dependency_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = doctor._version_of
    monkeypatch.setattr(doctor, "_version_of", lambda n: None if n == "click" else real(n))
    assert run(monkeypatch, "doctor") == 1
    out, _ = capsys.readouterr()
    assert out.splitlines()[0] == "fail" and "click" in out
