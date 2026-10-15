import json
from pathlib import Path

import pytest
from conftest import table_rows
from fixtures import synth

import datacopilot as dc
from datacopilot import convert
from datacopilot.core.errors import ConversionLossError, FormatError, LicenseError
from datacopilot.core.schema import AuditRun, Dataset
from datacopilot.io.report import ConversionReport
from datacopilot.logging import current_request_id
from datacopilot.store import DatasetStore


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("DATACOPILOT_ENV", "DATACOPILOT_LOG_LEVEL", "DATACOPILOT_ALLOWED_LICENSES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def coco(tmp_path: Path) -> Path:
    return synth.build_coco(tmp_path / "coco")


# ---------------------------------------------------------------- load
def test_load_reads_a_source_dataset_in_memory(coco: Path, tmp_path: Path) -> None:
    ds = dc.load(coco, format="coco")
    assert isinstance(ds, Dataset) and ds.manifest.task == "detection"
    assert ds.manifest.image_count == 4 and ds.manifest.annotation_count == 4
    assert not list(tmp_path.glob("*.dcp"))  # nothing was written
    stored = convert.ingest(coco, "coco", tmp_path / "a.dcp")
    assert ds.manifest.version == stored.manifest.version  # same content, same version


def test_load_accepts_a_dcp_directory_and_verifies_it(coco: Path, tmp_path: Path) -> None:
    convert.ingest(coco, "coco", tmp_path / "a.dcp")
    ds = dc.load(tmp_path / "a.dcp")
    assert ds == DatasetStore.open(tmp_path / "a.dcp").read()
    assert dc.load(str(tmp_path / "a.dcp"), format="dcp").manifest == ds.manifest


def test_load_group_by_sets_group_ids(coco: Path) -> None:
    ds = dc.load(coco, format="coco", group_by=r"re:^images/(?P<g>[^/.]+)")
    assert {i.group_id for i in ds.images} <= {"a", "b", "c", "d", None}


def test_load_strict_fails_on_a_lossy_read(tmp_path: Path) -> None:
    root = synth.build_cvat_xml(tmp_path / "cvat")
    train = root / "annotations" / "train.xml"
    train.write_text(
        train.read_text().replace(
            "</image>", '<polygon label="cat" points="1,1;5,5;9,1"/></image>', 1
        )
    )
    assert dc.load(root, format="cvat_xml").manifest.image_count == 4
    with pytest.raises(ConversionLossError, match="polygon"):
        dc.load(root, format="cvat_xml", strict=True)


def test_load_requires_a_format_for_a_source_dataset_and_rejects_unknown_ones(
    coco: Path,
) -> None:
    with pytest.raises(ValueError, match="format is required"):
        dc.load(coco)
    with pytest.raises(FormatError, match="unknown format"):
        dc.load(coco, format="nope")
    with pytest.raises(ValueError, match="dcp"):
        dc.load(coco.with_suffix(".dcp"), format="coco")


def test_load_errors_are_the_documented_exceptions(tmp_path: Path) -> None:
    with pytest.raises(FormatError):
        dc.load(tmp_path / "missing", format="coco")
    assert issubclass(FormatError, dc.DataCopilotError)


# --------------------------------------------------------------- export
def test_export_writes_a_format_and_returns_the_report(coco: Path, tmp_path: Path) -> None:
    ds = dc.load(coco, format="coco")
    report = dc.export(ds, "yolo", tmp_path / "yolo")
    assert isinstance(report, ConversionReport) and report.total_dropped > 0
    assert (tmp_path / "yolo" / "data.yaml").is_file()
    again = dc.load(tmp_path / "yolo", format="yolo")
    assert again.manifest.annotation_count == ds.manifest.annotation_count


def test_export_strict_raises_and_writes_nothing(coco: Path, tmp_path: Path) -> None:
    ds = dc.load(coco, format="coco")
    with pytest.raises(ConversionLossError):
        dc.export(ds, "yolo", tmp_path / "yolo", strict=True)
    assert not (tmp_path / "yolo").exists()


def test_export_from_a_dcp_path_round_trips(coco: Path, tmp_path: Path) -> None:
    convert.ingest(coco, "coco", tmp_path / "a.dcp")
    report = dc.export(tmp_path / "a.dcp", "coco", tmp_path / "out", strict=True)
    assert report.is_lossless()
    ds = dc.load(tmp_path / "out", format="coco")
    first = dc.load(coco, format="coco")
    assert table_rows(ds.images, ds.annotations) == table_rows(first.images, first.annotations)


# ---------------------------------------------------------------- audit
def test_audit_returns_the_run_and_does_not_raise_on_a_failed_gate(
    coco: Path, tmp_path: Path
) -> None:
    ds = dc.load(coco, format="coco")
    run = dc.audit(ds, root=coco.parent)
    assert isinstance(run, AuditRun) and run.passed is (run.counts.critical == 0)
    assert run.dataset_version == ds.manifest.version
    assert run.checks == ("integrity", "duplicates", "imbalance", "boxes")
    strict = dc.audit(ds, root=coco.parent, fail_on=("info",))
    assert strict.passed is False  # imbalance info on the tiny fixture
    none = dc.audit(ds, root=coco.parent, fail_on=())
    assert none.passed is True and none.fail_on == ()


def test_audit_accepts_a_dcp_path_and_selected_checks(coco: Path, tmp_path: Path) -> None:
    convert.ingest(coco, "coco", tmp_path / "a.dcp")
    run = dc.audit(tmp_path / "a.dcp", checks=["boxes", "imbalance"])
    assert run.checks == ("imbalance", "boxes")
    assert not (tmp_path / "a.dcp" / "audits").exists()  # the API never writes into the dataset


def test_audit_validates_its_arguments(coco: Path) -> None:
    ds = dc.load(coco, format="coco")
    with pytest.raises(ValueError, match="unknown --fail-on"):
        dc.audit(ds, fail_on=("banana",))
    with pytest.raises(FormatError, match="unknown check"):
        dc.audit(ds, checks=["nope"])
    with pytest.raises(ValueError, match="needs an embedder"):
        dc.audit(ds, checks=["leakage"])
    with pytest.raises(ValueError, match="similarity.threshold"):
        dc.audit(ds, embedder="hash", allow_test_embedder=True, similarity_threshold=2.0)
    with pytest.raises(ValueError, match="similarity.threshold"):
        dc.audit(ds, similarity_threshold=0.9)


def test_audit_with_the_test_embedder_needs_explicit_permission_for_a_gate(
    tmp_path: Path,
) -> None:
    root = synth.build_coco(tmp_path / "coco")
    ds = dc.load(root, format="coco")
    with pytest.raises(ValueError, match="test-only"):
        dc.audit(ds, root=tmp_path, embedder="hash")
    run = dc.audit(
        ds, root=tmp_path, embedder="hash", allow_test_embedder=True, fail_on=("leakage",)
    )
    assert run.embedder is not None and run.embedder.endswith("-test-only")
    assert "leakage" in run.checks
    ungated = dc.audit(ds, root=tmp_path, embedder="hash", fail_on=())
    assert ungated.passed


def test_audit_applies_the_license_guard_before_any_work(
    coco: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "MIT")
    ds = dc.load(coco, format="coco")
    with pytest.raises(LicenseError):
        dc.audit(ds, embedder="dinov2_base")


def test_audit_refuses_an_unknown_embedder_and_an_invalid_name(coco: Path) -> None:
    ds = dc.load(coco, format="coco")
    with pytest.raises(ValueError, match="unknown embedder"):
        dc.audit(ds, embedder="nope")
    with pytest.raises(ValueError, match="lower-case"):
        dc.audit(ds, embedder="../x")


def test_audit_refuses_an_empty_dataset(tmp_path: Path) -> None:
    root = synth.build_coco(tmp_path / "coco")
    ds = dc.load(root, format="coco")
    empty = Dataset(ds.manifest, (), ())
    with pytest.raises(dc.DataCopilotError, match="no images"):
        dc.audit(empty)


# ------------------------------------------------------------- logging
def test_the_api_is_quiet_on_stdout_logs_json_to_stderr_and_cleans_up(
    coco: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = dc.load(coco, format="coco")
    dc.audit(ds, root=coco.parent)
    out, err = capsys.readouterr()
    assert out == "" and current_request_id() is None
    records = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    assert {"audit_completed"} <= {r["message"] for r in records}
    assert len({r["request_id"] for r in records}) == 2  # one request id per call
    assert all(
        {"@timestamp", "level", "message", "service", "env", "request_id", "logger"} <= r.keys()
        for r in records
    )


def test_an_existing_request_id_is_kept(coco: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datacopilot.logging import configure_logging
    from datacopilot.settings import load_settings

    rid = configure_logging("host", load_settings())
    try:
        dc.load(coco, format="coco")
        assert current_request_id() == rid
    finally:
        from datacopilot.logging import clear_request_context

        clear_request_context()
    err = capsys.readouterr().err
    assert all(
        json.loads(line)["request_id"] == rid for line in err.splitlines() if line.startswith("{")
    )
