"""Report ordering and caps, SARIF identity, output collisions, error line, CI guard, text, BMP."""

from __future__ import annotations

import json
import struct
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from sarif_shape import validate_sarif
from test_report_formats import findings, make_run, parse

from datacopilot.cli import app as cli
from datacopilot.core.errors import DataCopilotError, FormatError
from datacopilot.io import files
from datacopilot.logging import clear_request_context
from datacopilot.report import html_report, sarif_report, text


def mixed(n_each: int = 4) -> Any:
    specs = []
    for sev in ("INFO", "WARN", "CRITICAL"):  # deliberately least severe first
        for n in range(n_each):
            specs.append(("boxes", sev, f"{n + 1:016x}", None, {"rule": f"{sev}{n}"}))
    return findings(*specs)


# ---------------------------------------------------------------- D1
def test_html_shows_the_most_severe_findings_and_counts_what_it_omits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(html_report, "MAX_HTML_FINDINGS", 5)
    page = html_report.render_html(make_run(mixed()))
    c = parse(page)
    sev_cells = [
        t
        for tag, a in c.tags
        if tag == "td" and (a.get("class") or "") in {"critical", "warn", "info"}
        for t in [a["class"]]
    ]
    assert sev_cells == ["critical"] * 4 + ["warn"]
    body = " ".join(" ".join(c.text).split())
    assert "Showing 5 of 12" in body and "omitted: 3 warn, 4 info" in body


# ---------------------------------------------------------------- D2
def test_sarif_run_identity_is_stable_and_the_run_id_is_the_correlation_guid() -> None:
    a = json.loads(sarif_report.render_sarif(make_run(mixed(1))))["runs"][0]
    assert a["automationDetails"]["id"] == "datacopilot/audit/"
    assert a["automationDetails"]["correlationGuid"] == "1111aaaa-2222-4333-8444-5555bbbb5555"


def test_sarif_results_are_sorted_by_severity_and_capped_with_a_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sarif_report, "MAX_SARIF_RESULTS", 5)
    doc = json.loads(sarif_report.render_sarif(make_run(mixed())))
    validate_sarif(doc)
    results = doc["runs"][0]["results"]
    assert [r["level"] for r in results] == ["error"] * 4 + ["warning"]
    assert len(results) == 5
    summary = results[-1]
    assert summary["ruleId"] == "datacopilot/results_truncated"
    assert "8" in summary["message"]["text"] and summary["level"] == "warning"


def test_sarif_does_not_invent_a_repository_path_for_images() -> None:
    fs = findings(
        ("integrity", "CRITICAL", "1" * 16, None, {"rule": "file_missing", "file_name": "a b.png"})
    )
    doc = json.loads(sarif_report.render_sarif(make_run(fs)))
    validate_sarif(doc)
    loc = doc["runs"][0]["results"][0]["locations"][0]
    art = loc["physicalLocation"]["artifactLocation"]
    assert art["uri"] == "a%20b.png" and art["uriBaseId"] == "DATASET_IMAGES"
    assert "DATASET_IMAGES" in doc["runs"][0]["originalUriBaseIds"]
    assert "images/" not in json.dumps(loc)


def test_sarif_findings_without_a_file_use_logical_locations_only() -> None:
    fs = findings(("imbalance", "WARN", None, None, {"rule": "long_tail"}))
    loc = json.loads(sarif_report.render_sarif(make_run(fs)))["runs"][0]["results"][0]["locations"][
        0
    ]
    assert "physicalLocation" not in loc and loc["logicalLocations"][0]["name"] == "dataset.dcp"


# ---------------------------------------------------------- D3-D5 (CLI)
@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CI", raising=False)
    yield
    clear_request_context()


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


@pytest.fixture
def dcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Path:
    from fixtures import synth

    coco = synth.build_coco(tmp_path / "coco")
    assert run_cli(monkeypatch, "ingest", coco, "--format", "coco", "-o", tmp_path / "ds.dcp") == 0
    capsys.readouterr()
    return tmp_path / "ds.dcp"


@pytest.mark.parametrize(
    "args",
    [
        ["--report", "x.out", "--json", "x.out"],
        ["--report", "x.out", "--sarif", "x.out"],
        ["--json", "x.out", "--sarif", "./x.out"],
        ["--report", "d/x.html", "--json", "d/../d/x.html"],
        ["--report", "r.html", "--json", "r.html"],
    ],
)
def test_colliding_output_paths_are_refused_before_anything_runs(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_cli(monkeypatch, "audit", dcp, *args) == 2
    assert not (dcp / "audits").exists() and not list(tmp_path.glob("x.out"))


def test_derived_names_that_collide_with_explicit_ones_are_refused(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o"
    assert (
        run_cli(monkeypatch, "audit", dcp, "--report", out / "x.html", "--sarif", out / "x.json")
        == 2
    )


def test_error_line_is_cleaned_and_cannot_inject_workflow_commands(
    dcp: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from datacopilot.audit import engine

    def boom(*_: Any, **__: Any) -> None:
        raise DataCopilotError("::error::set-output \x1b[31mred\x1b[0m\r\n::warning::second line")

    monkeypatch.setattr(engine, "run_audit", boom)
    assert run_cli(monkeypatch, "audit", dcp) == 1
    err = capsys.readouterr().err
    line = next(x for x in err.splitlines() if x.startswith("error:"))
    assert "\x1b" not in err.replace("\\u001b", "") and "\r" not in line
    assert not line.removeprefix("error:").lstrip().startswith("::")
    assert not any(x.lstrip().startswith("::") for x in err.splitlines())
    assert "\\u001b" in line and "\\u000d" in line


def test_external_images_are_refused_in_ci_unless_confirmed_and_always_warn(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CI", "true")
    out = ["--report", str(tmp_path / "r.html")]
    assert run_cli(monkeypatch, "audit", dcp, "--allow-external-images", *out) == 2
    assert "allow-external-images-in-ci" in capsys.readouterr().err
    assert (
        run_cli(
            monkeypatch,
            "audit",
            dcp,
            "--allow-external-images",
            "--allow-external-images-in-ci",
            *out,
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "warning:" in err and "outside the dataset" in err
    monkeypatch.delenv("CI")
    assert run_cli(monkeypatch, "audit", dcp, "--allow-external-images", *out) == 0
    assert "warning:" in capsys.readouterr().err


# ---------------------------------------------------------------- D6
def test_clean_text_escapes_every_format_and_control_character() -> None:
    bad = [
        cp
        for cp in range(0x110000)
        if unicodedata.category(chr(cp)) in {"Cc", "Cf", "Cs"} or cp in (0x061C, 0x180E)
    ]
    for cp in bad:
        out = text.clean_text(f"a{chr(cp)}b")
        assert out.startswith(("a\\u", "a\\U")) and out.endswith("b"), hex(cp)
        assert chr(cp) not in out
    assert text.clean_text("\U000e0041tag") == "\\U000e0041tag"
    assert text.clean_text("؜᠎") == "\\u061c\\u180e"


def test_clean_text_keeps_ordinary_text() -> None:
    assert text.clean_text("héllo 猫 🐈 -_.") == "héllo 猫 🐈 -_."


# ---------------------------------------------------------------- D7
def bmp(dib_size: int, body: bytes) -> bytes:
    return b"BM" + b"\x00" * 12 + struct.pack("<I", dib_size) + body + b"\x00" * 8


def test_bmp_core_header_uses_16_bit_dimensions(tmp_path: Path) -> None:
    p = tmp_path / "core.bmp"
    p.write_bytes(bmp(12, struct.pack("<HH", 300, 200)))
    assert files.image_size(p) == (300, 200)


@pytest.mark.parametrize("dib", [40, 52, 56, 64, 108, 124])
def test_bmp_info_headers_use_signed_32_bit_dimensions(tmp_path: Path, dib: int) -> None:
    p = tmp_path / "info.bmp"
    p.write_bytes(bmp(dib, struct.pack("<ii", 70_000, -50_000)))
    assert files.image_size(p) == (70_000, 50_000)


@pytest.mark.parametrize("dib", [0, 1, 7, 13, 39, 41, 125, 2**31])
def test_bmp_with_an_unknown_header_size_is_rejected(tmp_path: Path, dib: int) -> None:
    p = tmp_path / "bad.bmp"
    p.write_bytes(bmp(dib, struct.pack("<ii", 20, 20)))
    with pytest.raises(FormatError):
        files.image_size(p)


def test_truncation_marker_has_a_human_description() -> None:
    from datacopilot.report.rules import describe_rule

    assert "truncated" in describe_rule("boxes", "findings_truncated").lower()
    assert describe_rule("integrity", "audit_incomplete").startswith("The check ended")


# ------------------------------------------------------------------ E
def test_reports_are_written_even_when_the_dataset_is_read_only(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    dcp.chmod(0o555)
    try:
        if os.access(dcp, os.W_OK):
            pytest.skip("directory permissions are not enforced for this user")
        code = run_cli(monkeypatch, "audit", dcp, "--report", tmp_path / "r.html")
    finally:
        dcp.chmod(0o755)
    assert code == 0 and (tmp_path / "r.json").is_file()
    err = capsys.readouterr().err
    records = [json.loads(x) for x in err.splitlines() if x.startswith("{")]
    assert [r["level"] for r in records if r["message"] == "audit_not_stored"] == ["WARNING"]
    assert not (dcp / "audits").exists()


def test_no_store_audit_skips_writing_the_run(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        run_cli(monkeypatch, "audit", dcp, "--no-store-audit", "--report", tmp_path / "r.html") == 0
    )
    assert not (dcp / "audits").exists() and (tmp_path / "r.json").is_file()


def test_reports_are_written_before_the_run_is_stored(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datacopilot.store import DatasetStore

    seen: list[bool] = []
    real = DatasetStore.write_audit

    def spy(self: Any, run: Any) -> Path:
        seen.append((tmp_path / "r.sarif").is_file())
        return real(self, run)

    monkeypatch.setattr(DatasetStore, "write_audit", spy)
    assert run_cli(monkeypatch, "audit", dcp, "--report", tmp_path / "r.html") == 0
    assert seen == [True]


@pytest.mark.skipif(sys.platform not in ("darwin", "win32"), reason="case-insensitive file systems")
def test_output_paths_differing_only_in_case_collide_on_case_insensitive_systems(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        run_cli(
            monkeypatch,
            "audit",
            dcp,
            "--report",
            tmp_path / "x.html",
            "--json",
            tmp_path / "X.HTML",
        )
        == 2
    )


def test_missing_o_nofollow_is_reported_once_per_audit(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    if hasattr(os, "O_NOFOLLOW"):
        monkeypatch.delattr(os, "O_NOFOLLOW")
    assert run_cli(monkeypatch, "audit", dcp, "--report", tmp_path / "r.html") == 0
    records = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    assert [r["message"] for r in records].count("audit_nofollow_unavailable") == 1
