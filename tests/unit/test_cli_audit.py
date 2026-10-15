"""``datacopilot audit``: exit-code contract, events, reports, settings wiring."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from fixtures import synth
from fixtures.synth import png_bytes
from sarif_shape import validate_sarif

from datacopilot.cli import app as cli
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

REQUIRED = {"@timestamp", "level", "message", "service", "env", "request_id", "logger"}
ENV = ("DATACOPILOT_ENV", "DATACOPILOT_LOG_LEVEL", "DATACOPILOT_WORKERS", "DATACOPILOT_HASH_CACHE")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in (*ENV, "DATACOPILOT_CACHE_DIR", "DATACOPILOT_CACHE_KEY", "XDG_CACHE_HOME", "CI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))
    yield
    clear_request_context()


def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def logs(err: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in err.splitlines() if line.startswith("{")]


def ingest(monkeypatch: pytest.MonkeyPatch, source: Path, out: Path, fmt: str = "coco") -> None:
    assert run(monkeypatch, "ingest", source, "--format", fmt, "-o", out) == 0


@pytest.fixture
def dcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Path:
    coco = synth.build_coco(tmp_path / "coco")
    out = tmp_path / "ds.dcp"
    ingest(monkeypatch, coco, out)
    capsys.readouterr()
    return out


def reports(tmp_path: Path) -> list[str]:
    return [
        "--report",
        str(tmp_path / "r.html"),
        "--json",
        str(tmp_path / "r.json"),
        "--sarif",
        str(tmp_path / "r.sarif"),
    ]


# ------------------------------------------------------------ exit codes
def test_clean_dataset_passes_the_critical_gate_with_exit_0(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(monkeypatch, "audit", dcp, "--fail-on", "critical", *reports(tmp_path))
    out, err = capsys.readouterr()
    assert code == 0
    line = out.strip()
    assert out.count("\n") == 1 and re.fullmatch(
        r"passed critical=0 warn=\d+ info=\d+ run_id=[0-9a-f-]{36}", line
    ), line
    assert "Traceback" not in err


def test_a_matching_gate_exits_1_and_still_writes_every_report(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(monkeypatch, "audit", dcp, "--fail-on", "warn", *reports(tmp_path))
    out, err = capsys.readouterr()
    assert code == 1 and out.startswith("failed ")
    assert all((tmp_path / n).is_file() for n in ("r.html", "r.json", "r.sarif"))
    done = [r for r in logs(err) if r["message"] == "audit_completed"]
    assert len(done) == 1 and done[0]["passed"] is False
    assert not [r for r in logs(err) if r["message"] == "command_failed"]


def test_gate_on_a_check_name_matches_any_severity_of_that_check(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "audit", dcp, "--fail-on", "imbalance", *reports(tmp_path)) == 1
    assert run(monkeypatch, "audit", dcp, "--fail-on", "boxes", *reports(tmp_path)) == 0


def test_no_gate_means_exit_0_even_with_findings(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "coco" / "images" / "a.png").unlink()
    assert run(monkeypatch, "audit", dcp, "--fail-on", "none", *reports(tmp_path)) == 0
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 1  # the default gate is critical


def test_missing_image_is_critical_and_fails_a_critical_gate(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "coco" / "images" / "a.png").unlink()
    code = run(monkeypatch, "audit", dcp, "--fail-on", "critical", *reports(tmp_path))
    assert code == 1
    findings = json.loads((tmp_path / "r.json").read_text())["findings"]
    assert any(f["payload"]["rule"] == "file_missing" for f in findings)


@pytest.mark.parametrize(
    "args",
    [
        ["--checks", "nope"],
        ["--checks", ""],
        ["--fail-on", "crit"],
        ["--fail-on", "leakage"],  # documented but not built: must not be a gate that never fires
        ["--fail-on", ""],
        ["--checks", "boxes", "--fail-on", "integrity"],  # a check that is not part of the run
        ["--workers", "0"],
        ["--workers", "257"],
        ["--workers", "x"],
        ["--no-such-option"],
    ],
)
def test_bad_options_exit_2_before_touching_the_dataset(
    dcp: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
) -> None:
    assert run(monkeypatch, "audit", dcp, *args, *reports(tmp_path)) == 2
    out, err = capsys.readouterr()
    assert out == "" and "Traceback" not in err
    assert not (dcp / "audits").exists() and not (tmp_path / "r.html").exists()
    assert not any(r["message"] == "command_failed" for r in logs(err))


@pytest.mark.parametrize(
    ("var", "value"),
    [("DATACOPILOT_WORKERS", "abc"), ("DATACOPILOT_HASH_CACHE", "maybe")],
)
def test_invalid_environment_exits_2(
    dcp: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    var: str,
    value: str,
) -> None:
    monkeypatch.setenv(var, value)
    assert run(monkeypatch, "audit", dcp) == 2
    assert var in capsys.readouterr().err


def test_unreadable_datasets_exit_1_with_one_error_line(
    tmp_path: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "audit", tmp_path / "missing.dcp") == 1
    err = capsys.readouterr().err
    assert [line for line in err.splitlines() if line.startswith("error:")]
    (dcp / "images.parquet").write_bytes(b"not parquet")
    assert run(monkeypatch, "audit", dcp) == 1
    assert capsys.readouterr().out == ""


def test_dataset_that_is_not_a_directory_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "file.dcp"
    plain.write_text("x")
    assert run(monkeypatch, "audit", plain) == 1


# ------------------------------------------------------------- run record
def test_run_id_is_the_request_id_of_the_logs_and_the_stored_file_name(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    out, err = capsys.readouterr()
    run_id = out.split("run_id=")[1].strip()
    records = logs(err)
    assert {r["request_id"] for r in records} == {run_id}
    stored = DatasetStore.open(dcp).read_audit(run_id)
    assert (
        stored.run_id == run_id
        and stored.dataset_version == DatasetStore.open(dcp).manifest().version
    )
    assert json.loads((tmp_path / "r.json").read_text())["run_id"] == run_id
    assert (dcp / "audits" / f"{run_id}.json").is_file()


def test_audit_leaves_the_dataset_version_and_tables_untouched(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {p.name: p.read_bytes() for p in dcp.iterdir() if p.is_file()}
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    assert {p.name: p.read_bytes() for p in dcp.iterdir() if p.is_file()} == before


def test_events_are_logged_in_order_with_the_required_fields_and_no_paths(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    err = capsys.readouterr().err
    records = logs(err)
    assert all(REQUIRED <= r.keys() for r in records)
    assert [r["message"] for r in records] == [
        "audit_check_completed",
        "audit_check_completed",
        "audit_check_completed",
        "audit_check_completed",
        "audit_completed",
        "report_written",
        "report_written",
        "report_written",
    ]
    assert [r["check_name"] for r in records[:4]] == [
        "integrity",
        "duplicates",
        "imbalance",
        "boxes",
    ]
    assert [(r["format"], r["path"]) for r in records[5:]] == [
        ("html", "r.html"),
        ("json", "r.json"),
        ("sarif", "r.sarif"),
    ]
    assert str(tmp_path) not in err


# ----------------------------------------------------------------- outputs
def test_reports_are_valid_and_the_sarif_has_the_21_shape(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    validate_sarif(json.loads((tmp_path / "r.sarif").read_text()))
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")


def test_without_output_options_reports_land_next_to_report_html_in_the_cwd(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert run(monkeypatch, "audit", dcp) == 0
    assert sorted(p.name for p in work.iterdir()) == ["report.html", "report.json", "report.sarif"]


def test_report_option_derives_the_json_and_sarif_names(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "audit", dcp, "--report", tmp_path / "out" / "x.html") == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["x.html", "x.json", "x.sarif"]


def test_checks_option_selects_a_subset(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "audit", dcp, "--checks", "boxes,integrity", *reports(tmp_path)) == 0
    assert json.loads((tmp_path / "r.json").read_text())["checks"] == ["integrity", "boxes"]


def test_hostile_names_end_to_end_produce_safe_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "coco"
    name = "<img src=x onerror=alert(1)>&\"'.png"
    (root / "images").mkdir(parents=True)
    (root / "images" / name).write_bytes(png_bytes(20, 20, 1))
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": name, "width": 20, "height": 20}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 20, 0.5],
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "<script>alert(1)</script>"}],
            }
        )
    )
    ingest(monkeypatch, root, tmp_path / "h.dcp")
    assert run(monkeypatch, "audit", tmp_path / "h.dcp", *reports(tmp_path)) == 0
    page = (tmp_path / "r.html").read_text()
    assert "<script" not in page and "<img" not in page and "onerror=alert(1)>" not in page
    validate_sarif(json.loads((tmp_path / "r.sarif").read_text()))
    rules = {
        f["payload"]["rule"] for f in json.loads((tmp_path / "r.json").read_text())["findings"]
    }
    assert {"tiny_box", "extreme_aspect"} <= rules


# --------------------------------------------------------- settings wiring
def capture_context(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from datacopilot.audit import engine

    seen: list[Any] = []
    real = engine.run_audit

    def spy(ds: Any, checks: Any, ctx: Any, fail_on: Any = ()) -> Any:
        seen.append(ctx)
        return real(ds, checks, ctx, fail_on)

    monkeypatch.setattr(engine, "run_audit", spy)
    return seen


def test_workers_and_cache_dir_come_from_the_environment_and_the_option_wins(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture_context(monkeypatch)
    monkeypatch.setenv("DATACOPILOT_WORKERS", "3")
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    assert seen[-1].workers == 3 and seen[-1].cache_dir == tmp_path / "cache"
    assert run(monkeypatch, "audit", dcp, "--workers", "2", *reports(tmp_path)) == 0
    assert seen[-1].workers == 2


def test_hash_cache_setting_option_and_default_are_wired_to_the_check(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    assert run(monkeypatch, "audit", dcp, "--no-hash-cache", *reports(tmp_path)) == 0
    assert not cache.exists()  # the option disables the cache: nothing is read or created
    monkeypatch.setenv("DATACOPILOT_HASH_CACHE", "false")
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    assert not cache.exists()  # so does the environment setting
    assert run(monkeypatch, "audit", dcp, "--hash-cache", *reports(tmp_path)) == 0
    assert any(p.name.startswith("sha256-") for p in cache.iterdir())  # option beats the setting
    monkeypatch.delenv("DATACOPILOT_HASH_CACHE")
    for p in cache.iterdir():
        p.unlink()
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    assert any(p.name.startswith("sha256-") for p in cache.iterdir())  # default is on


def test_images_outside_the_dataset_root_need_an_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coco = synth.build_coco(tmp_path / "elsewhere" / "coco")
    nested = tmp_path / "datasets" / "ds.dcp"
    ingest(monkeypatch, coco, nested)
    assert run(monkeypatch, "audit", nested, "--fail-on", "critical", *reports(tmp_path)) == 1
    findings = json.loads((tmp_path / "r.json").read_text())["findings"]
    assert any(f["payload"]["rule"] == "external_not_checked" for f in findings)  # all unchecked
    assert run(monkeypatch, "audit", nested, "--allow-external-images", *reports(tmp_path)) == 0
    (coco / "images" / "a.png").unlink()
    code = run(
        monkeypatch,
        "audit",
        nested,
        "--allow-external-images",
        "--fail-on",
        "critical",
        *reports(tmp_path),
    )
    assert code == 1


def test_ci_without_a_cache_key_runs_without_a_cache_and_warns_once(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CI", "true")
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    err = capsys.readouterr().err
    warns = [r for r in logs(err) if r["message"] == "audit_cache_untrusted"]
    assert [w["reason"] for w in warns] == ["ci_without_key"]
    assert not (tmp_path / "cache").exists()


def test_ci_with_an_external_key_uses_the_cache_and_never_prints_the_key(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = "not-a-real-key-" + "x" * 20
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv("DATACOPILOT_CACHE_KEY", fake)
    assert run(monkeypatch, "audit", dcp, *reports(tmp_path)) == 0
    out, err = capsys.readouterr()
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["sha256-v2.cache"]
    everything = out + err + "".join(p.read_text() for p in tmp_path.glob("r.*"))
    assert fake not in everything
    assert not any(r["message"] == "audit_cache_untrusted" for r in logs(err))
