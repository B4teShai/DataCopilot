import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from fixtures import synth

from datacopilot.cli import app as cli
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

REQUIRED = {"@timestamp", "level", "message", "service", "env", "request_id", "logger"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    for var in ("DATACOPILOT_ENV", "DATACOPILOT_LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    yield
    clear_request_context()


def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def logs(err: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in err.splitlines() if line.startswith("{")]


def human(err: str) -> list[str]:
    return [line for line in err.splitlines() if not line.startswith("{") and line.strip()]


@pytest.fixture
def coco(tmp_path: Path) -> Path:
    return synth.build_coco(tmp_path / "coco")


@pytest.fixture
def dcp(tmp_path: Path, coco: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "ds.dcp"
    assert run(monkeypatch, "ingest", coco, "--format", "coco", "-o", out) == 0
    return out


# ------------------------------------------------------------------ ingest
def test_ingest_prints_the_version_and_logs_one_compliant_event(
    tmp_path: Path, coco: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ds.dcp"
    assert run(monkeypatch, "ingest", coco, "--format", "coco", "-o", out) == 0
    stdout, err = capsys.readouterr()
    version = stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{32}", version) and stdout.count("\n") == 1
    assert DatasetStore.open(out).manifest().version == version
    (rec,) = logs(err)
    assert (
        REQUIRED <= rec.keys() and rec["message"] == "dataset_ingested" and rec["level"] == "INFO"
    )
    assert (rec["format"], rec["image_count"], rec["annotation_count"]) == ("coco", 4, 4)
    assert rec["dataset_version"] == version and isinstance(rec["duration_ms"], int)
    assert str(tmp_path) not in err  # no paths in logs


def test_ingest_is_deterministic_across_runs(
    tmp_path: Path, coco: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    versions = []
    for n in range(2):
        assert run(monkeypatch, "ingest", coco, "-f", "coco", "-o", tmp_path / f"d{n}.dcp") == 0
        versions.append(capsys.readouterr().out.strip())
    assert versions[0] == versions[1]


def test_ingest_version_is_independent_of_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = synth.build_coco(tmp_path / "a")
    b = synth.build_coco(tmp_path / "b")
    for f in (b / "annotations").glob("*.json"):
        doc = json.loads(f.read_text())
        doc["images"].reverse()
        doc["annotations"].reverse()
        doc["categories"].reverse()
        f.write_text(json.dumps(doc))
    out = []
    for name, root in (("a", a), ("b", b)):
        assert run(monkeypatch, "ingest", root, "-f", "coco", "-o", tmp_path / f"{name}.dcp") == 0
        out.append(capsys.readouterr().out.strip())
    assert out[0] == out[1]


def test_ingest_group_by_regex_sets_group_id(
    tmp_path: Path, coco: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "g.dcp"
    pattern = r"re:^images/(?P<g>[ab])\.png$"
    assert run(monkeypatch, "ingest", coco, "-f", "coco", "--group-by", pattern, "-o", out) == 0
    ds = DatasetStore.open(out).read()
    assert {i.source["file_name"]: i.group_id for i in ds.images} == {
        "a.png": "a",
        "b.png": "b",
        "c.png": None,
        "d.png": None,
    }


def test_ingest_fails_with_exit_1_one_human_line_and_one_log_when_output_exists(
    tmp_path: Path,
    coco: Path,
    dcp: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capsys.readouterr()
    before = (dcp / "manifest.json").read_text()
    assert run(monkeypatch, "ingest", coco, "-f", "coco", "-o", dcp) == 1
    out, err = capsys.readouterr()
    assert out == ""
    (line,) = human(err)
    assert line.startswith("error:") and "already exists" in line and "Traceback" not in line
    (rec,) = logs(err)
    assert rec["message"] == "command_failed" and rec["command"] == "ingest"
    assert (dcp / "manifest.json").read_text() == before


def test_unwritable_output_directory_exits_1_with_one_human_error_line(
    tmp_path: Path,
    coco: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        if os.access(locked, os.W_OK):  # running as root: the directory is writable anyway
            pytest.skip("directory permissions are not enforced for this user")
        assert run(monkeypatch, "ingest", coco, "-f", "coco", "-o", locked / "out.dcp") == 1
    finally:
        locked.chmod(0o755)
    out, err = capsys.readouterr()
    assert out == ""
    (line,) = human(err)
    assert line.startswith("error:") and "ermission denied" in line
    assert str(tmp_path) not in line and "Traceback" not in line
    assert line == "error: Permission denied: out.dcp"  # no temporary-file suffix, no directory
    (rec,) = logs(err)
    assert rec["message"] == "command_failed" and rec["command"] == "ingest"


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (["-f", "nope"], "unknown format"),
        (["-f", "yolo"], "data.yaml"),
    ],
)
def test_ingest_format_errors_exit_1(
    tmp_path: Path,
    coco: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    needle: str,
) -> None:
    assert run(monkeypatch, "ingest", coco, *args, "-o", tmp_path / "x.dcp") == 1
    _, err = capsys.readouterr()
    assert needle in err and not (tmp_path / "x.dcp").exists()
    assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]


def test_ingest_of_a_missing_path_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run(monkeypatch, "ingest", tmp_path / "nope", "-f", "coco", "-o", tmp_path / "x.dcp") == 1
    )
    assert "does not exist" in capsys.readouterr().err


def test_ingest_invariant_violation_names_the_annotation_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = synth.build_coco(tmp_path / "bad")
    f = root / "annotations" / "instances_val.json"
    f.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
    f = root / "annotations" / "instances_train.json"
    doc = json.loads(f.read_text())
    doc["annotations"][0]["bbox"] = [60, 40, 20, 20]
    f.write_text(json.dumps(doc))
    assert run(monkeypatch, "ingest", root, "-f", "coco", "-o", tmp_path / "x.dcp") == 1
    _, err = capsys.readouterr()
    assert "ann_id 'instances_train:101'" in err and not (tmp_path / "x.dcp").exists()


def test_ingest_reports_identical_image_content_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = synth.build_coco(tmp_path / "dup")
    (root / "images" / "b.png").write_bytes((root / "images" / "a.png").read_bytes())
    for f in (root / "annotations").glob("*.json"):
        doc = json.loads(f.read_text())
        for im in doc["images"]:
            if im["file_name"] == "b.png":
                im["width"], im["height"] = 64, 48
        f.write_text(json.dumps(doc))
    assert run(monkeypatch, "ingest", root, "-f", "coco", "-o", tmp_path / "x.dcp") == 1
    err = capsys.readouterr().err
    assert "identical content" in err and "a.png" in err and "b.png" in err


def test_ingest_of_an_empty_dataset_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "e"
    root.mkdir()
    (root / "annotations.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []})
    )
    assert run(monkeypatch, "ingest", root, "-f", "coco", "-o", tmp_path / "x.dcp") == 1
    assert "no images" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["-o", "plain_dir"],  # missing .dcp suffix
        ["-o", "x.dcp", "--group-by", "(unclosed"],  # invalid regex
    ],
)
def test_ingest_usage_errors_exit_2(
    tmp_path: Path,
    coco: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
) -> None:
    argv = [
        "ingest",
        str(coco),
        "-f",
        "coco",
        *[str(tmp_path / a) if a.endswith(("dcp", "dir")) else a for a in extra],
    ]
    assert run(monkeypatch, *argv) == 2
    out, err = capsys.readouterr()
    assert out == "" and not any(line.startswith("{") for line in err.splitlines())


def test_ingest_without_format_exits_2(
    tmp_path: Path, coco: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "ingest", coco, "-o", tmp_path / "x.dcp") == 2


def test_ingest_logs_a_loss_event_for_each_dropped_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = synth.build_coco(tmp_path / "c")
    f = root / "annotations" / "instances_train.json"
    doc = json.loads(f.read_text())
    doc["categories"][0]["color"] = "red"
    f.write_text(json.dumps(doc))
    assert run(monkeypatch, "ingest", root, "-f", "coco", "-o", tmp_path / "x.dcp") == 0
    recs = logs(capsys.readouterr().err)
    loss = [r for r in recs if r["message"] == "conversion_loss"]
    assert [(r["column"], r["count"], r["level"]) for r in loss] == [
        ("categories.color", 1, "WARNING")
    ]
    assert loss[0]["source_format"] == "coco" and loss[0]["target_format"] == "dcp"
    assert recs[-1]["message"] == "dataset_ingested"


# ------------------------------------------------------------------ export
def test_export_coco_reproduces_the_annotations_exactly(
    tmp_path: Path,
    coco: Path,
    dcp: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capsys.readouterr()
    out = tmp_path / "exported"
    assert run(monkeypatch, "export", dcp, "-f", "coco", "-o", out, "--strict") == 0
    stdout, err = capsys.readouterr()
    assert stdout.strip() == str(out)
    (rec,) = logs(err)
    assert (
        rec["message"] == "dataset_exported" and rec["format"] == "coco" and rec["strict"] is True
    )
    assert (rec["image_count"], rec["annotation_count"], rec["dropped_count"]) == (4, 4, 0)
    assert rec["dataset_version"] == DatasetStore.open(dcp).manifest().version
    for split in ("train", "val", "test"):
        a = json.loads((coco / "annotations" / f"instances_{split}.json").read_text())
        b = json.loads((out / "annotations" / f"instances_{split}.json").read_text())
        assert a == b  # the synthetic fixture is written in the writer's own key order


def test_export_to_another_format_logs_every_loss_and_still_succeeds(
    tmp_path: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    out = tmp_path / "yolo_out"
    assert run(monkeypatch, "export", dcp, "-f", "yolo", "-o", out) == 0
    recs = logs(capsys.readouterr().err)
    loss = {r["column"]: r["count"] for r in recs if r["message"] == "conversion_loss"}
    assert loss["annotations.is_crowd"] == 1 and loss["categories.id"] == 3
    assert all(r["level"] == "WARNING" for r in recs if r["message"] == "conversion_loss")
    done = recs[-1]
    assert done["message"] == "dataset_exported" and done["dropped_count"] == sum(loss.values())
    assert (out / "data.yaml").is_file()


def test_export_strict_fails_on_loss_and_leaves_no_output(
    tmp_path: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    out = tmp_path / "yolo_out"
    assert run(monkeypatch, "export", dcp, "-f", "yolo", "-o", out, "--strict") == 1
    _, err = capsys.readouterr()
    assert "lossy" in err and "annotations.is_crowd=1" in err
    assert not out.exists() and not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]
    messages = [r["message"] for r in logs(err)]
    assert messages.count("command_failed") == 1 and "dataset_exported" not in messages


def test_export_refuses_a_non_empty_directory_but_accepts_an_empty_one(
    tmp_path: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("keep")
    assert run(monkeypatch, "export", dcp, "-f", "coco", "-o", full) == 1
    assert (full / "keep.txt").read_text() == "keep" and len(list(full.iterdir())) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(monkeypatch, "export", dcp, "-f", "coco", "-o", empty) == 0
    assert (empty / "annotations").is_dir()


def test_export_errors_exit_1(
    tmp_path: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run(monkeypatch, "export", tmp_path / "nope.dcp", "-f", "coco", "-o", tmp_path / "o") == 1
    )
    assert run(monkeypatch, "export", dcp, "-f", "nope", "-o", tmp_path / "o") == 1
    assert (
        run(monkeypatch, "export", dcp, "-f", "imagefolder", "-o", tmp_path / "o") == 1
    )  # detection data
    assert "classification" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_export_usage_errors_exit_2(
    tmp_path: Path, dcp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "export", dcp, "-o", tmp_path / "o") == 2
    assert run(monkeypatch, "export", dcp, "-f", "coco") == 2


def test_table_limits_from_settings_apply_and_invalid_values_exit_2(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    monkeypatch.setenv("DATACOPILOT_MAX_TABLE_ROWS", "2")
    assert run(monkeypatch, "export", dcp, "-f", "coco", "-o", tmp_path / "o") == 1
    assert "limit of 2" in capsys.readouterr().err
    monkeypatch.setenv("DATACOPILOT_MAX_TABLE_ROWS", "0")
    assert run(monkeypatch, "export", dcp, "-f", "coco", "-o", tmp_path / "o") == 2


# ------------------------------------------------------------------ formats
def test_formats_lists_every_builtin_format_with_capabilities(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "formats") == 0
    out, err = capsys.readouterr()
    rows = [line.split("\t") for line in out.splitlines()]
    assert [r[0] for r in rows] == [
        "coco",
        "cvat_xml",
        "imagefolder",
        "label_studio",
        "voc",
        "yolo",
    ]
    assert all(r[1] == "read,write" and r[2] == "Apache-2.0" for r in rows)
    assert logs(err) == []


def test_formats_help_documents_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for cmd in ("ingest", "export"):
        assert run(monkeypatch, cmd, "--help") == 0
        assert "Exit codes" in capsys.readouterr().out


def test_ingest_strict_fails_on_skipped_input_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = synth.build_cvat_xml(tmp_path / "src")
    first = src / "annotations" / "train.xml"
    first.write_text(
        first.read_text().replace(
            "</image>", '<polygon label="cat" points="1,1;5,5;9,1"/></image>', 1
        )
    )
    capsys.readouterr()
    assert run(monkeypatch, "ingest", src, "-f", "cvat_xml", "-o", tmp_path / "ok.dcp") == 0
    capsys.readouterr()
    code = run(monkeypatch, "ingest", src, "-f", "cvat_xml", "-o", tmp_path / "s.dcp", "--strict")
    assert code == 1
    assert "annotations.unsupported.polygon=1" in capsys.readouterr().err
    assert not (tmp_path / "s.dcp").exists()


def test_an_unsafe_group_by_pattern_is_refused_before_anything_is_read() -> None:
    from datacopilot import convert

    with pytest.raises(ValueError, match="unsafe --group-by pattern"):
        convert.parse_group_by("re:^(a+)+$")
    assert convert.parse_group_by(r"re:^images/(?P<g>[^/.]+)").groups == 1
