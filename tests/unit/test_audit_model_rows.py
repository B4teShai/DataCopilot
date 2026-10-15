"""Model-generated rows are not ground truth: audits exclude them unless asked, and say so."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import dataset_from
from fixtures import synth

import datacopilot as dc
from datacopilot.audit.context import CheckContext
from datacopilot.audit.engine import run_audit
from datacopilot.cli import app as cli
from datacopilot.core.errors import GateFailedError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.report import html_report, render

RUN = "00000000-0000-4000-8000-000000000000"


def dataset(tmp_path: Path):  # type: ignore[no-untyped-def]
    synth.write_png(tmp_path / "i.png", 64, 48, 1)
    synth.write_png(tmp_path / "j.png", 64, 48, 2)
    images = [
        Image(image_id=f"{n:016x}", file_path=str(tmp_path / f), width=64, height=48, split="train")
        for n, f in ((1, "i.png"), (2, "j.png"))
    ]
    anns = [
        Annotation(ann_id="h1", image_id=images[0].image_id, category_id=0, kind="bbox",
                   bbox_xyxy_abs=(8.0, 6.0, 40.0, 30.0)),
        # a model box that the boxes check would call tiny, and a model class nobody labelled
        Annotation(ann_id="m1", image_id=images[0].image_id, category_id=1, kind="bbox",
                   bbox_xyxy_abs=(1.0, 1.0, 1.5, 1.5), provenance="model", score=0.4),
        Annotation(ann_id="m2", image_id=images[1].image_id, category_id=1, kind="bbox",
                   bbox_xyxy_abs=(2.0, 2.0, 30.0, 30.0), provenance="model"),
        Annotation(ann_id="a1", image_id=images[1].image_id, category_id=0, kind="bbox",
                   bbox_xyxy_abs=(3.0, 3.0, 30.0, 30.0), provenance="proposal_accepted"),
    ]  # fmt: skip
    cats = [Category(id=0, name="cup"), Category(id=1, name="ghost")]
    return dataset_from(images, anns, cats)


def ctx(tmp_path: Path, **extra: object) -> CheckContext:
    return CheckContext(run_id=RUN, root=tmp_path, cache_dir=tmp_path / "c", workers=1,
                        use_hash_cache=False, **extra)  # type: ignore[arg-type]  # fmt: skip


def test_model_rows_are_excluded_from_every_check_by_default(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    run = run_audit(ds, ["boxes", "imbalance"], ctx(tmp_path), ())
    assert run.summary is not None and run.summary.excluded_model_annotations == 2
    assert run.summary.annotation_count == 2  # the human row and the accepted proposal
    rules = {(f.check, str(f.payload.get("rule"))) for f in run.findings}
    assert ("boxes", "tiny_box") not in rules
    assert not any(f.ann_id in ("m1", "m2") for f in run.findings)
    assert ("imbalance", "class_unused") in rules  # the class only the model used is unused


def test_they_can_be_included_on_request(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    run = run_audit(ds, ["boxes"], ctx(tmp_path, include_model_annotations=True), ())
    assert run.summary is not None and run.summary.excluded_model_annotations == 0
    assert any(f.ann_id == "m1" for f in run.findings)  # the tiny model box is reported


def test_a_dataset_without_model_rows_is_audited_as_before(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    human = type(ds)(
        ds.manifest, ds.images, tuple(a for a in ds.annotations if a.provenance == "human")
    )
    run = run_audit(human, ["boxes"], ctx(tmp_path), ())
    assert run.summary is not None and run.summary.excluded_model_annotations == 0


def test_the_html_report_says_how_many_rows_were_excluded(tmp_path: Path) -> None:
    run = run_audit(dataset(tmp_path), ["boxes"], ctx(tmp_path), ())
    page = html_report.render_html(run)
    assert "2 model-generated annotations were excluded" in page
    assert "--include-model-annotations" in page
    included = run_audit(
        dataset(tmp_path), ["boxes"], ctx(tmp_path, include_model_annotations=True), ()
    )
    assert "were excluded" not in html_report.render_html(included)
    render(run, html=tmp_path / "r.html", json=tmp_path / "r.json", sarif=tmp_path / "r.sarif")
    assert (
        json.loads((tmp_path / "r.json").read_text())["summary"]["excluded_model_annotations"] == 2
    )


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def label_studio_with_predictions(tmp_path: Path) -> Path:
    root = tmp_path / "ls"
    synth.write_png(root / "images" / "a.png", 64, 48, 1)
    box = synth.ls_rect("cat", (8, 6, 40, 30), (64, 48), "r")
    task = {
        "id": 1,
        "data": {"image": "images/a.png"},
        "annotations": [{"id": 5, "result": [box]}],
        "predictions": [{"result": [synth.ls_rect("dog", (1, 1, 5, 5), (64, 48), "p")]}],
    }
    (root / "annotations.json").write_text(json.dumps([task]))
    return root


def test_the_command_line_excludes_model_rows_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = label_studio_with_predictions(tmp_path)
    assert run_cli(monkeypatch, "ingest", root, "-f", "label_studio", "-o", tmp_path / "a.dcp") == 0
    capsys.readouterr()
    report = tmp_path / "r.html"
    assert (
        run_cli(monkeypatch, "audit", tmp_path / "a.dcp", "--no-hash-cache", "--report", report)
        == 0
    )
    out, err = capsys.readouterr()
    assert out.startswith("passed") and len(out.strip().splitlines()) == 1
    assert "1 model-generated annotation" in err and "--include-model-annotations" in err
    assert "excluded" in report.read_text()
    assert run_cli(monkeypatch, "audit", tmp_path / "a.dcp", "--no-hash-cache", "--report", report,
                   "--include-model-annotations") == 0  # fmt: skip
    assert "excluded" not in capsys.readouterr().err


def test_the_api_has_the_same_default_and_the_same_option(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    assert (
        dc.audit(ds, ["boxes"], fail_on=(), root=tmp_path).summary.excluded_model_annotations == 2
    )  # type: ignore[union-attr]
    run = dc.audit(ds, ["boxes"], fail_on=(), root=tmp_path, include_model_annotations=True)
    assert run.summary is not None and run.summary.excluded_model_annotations == 0


# F14: the gate is recorded, then raised on request
def test_raise_for_gate_turns_a_failed_gate_into_an_exception(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    failed = dc.audit(ds, ["boxes", "integrity"], fail_on=("warn",), root=tmp_path / "elsewhere")
    assert failed.passed is False
    with pytest.raises(GateFailedError, match=r"critical=\d+ warn=\d+ info=\d+"):
        failed.raise_for_gate()
    ok = dc.audit(ds, ["boxes"], fail_on=(), root=tmp_path)
    assert ok.passed and ok.raise_for_gate() is None
    assert issubclass(GateFailedError, dc.DataCopilotError)
    assert dc.GateFailedError is GateFailedError


def test_require_local_images_makes_unexamined_images_critical(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    # the images are outside the stated root: with a gate that is a CRITICAL finding either way
    run = dc.audit(ds, ["integrity"], root=tmp_path / "other", require_local_images=True)
    assert run.counts.critical >= 1 and run.passed is False
