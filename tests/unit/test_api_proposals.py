"""``dc.propose`` and ``dc.apply``: the library side of the propose, review, apply flow."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import synth

import datacopilot as dc
from datacopilot.core.errors import LicenseError, ValidationError
from datacopilot.core.proposals import ProposalSet
from datacopilot.store import DatasetStore


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("DATACOPILOT_ALLOWED_LICENSES", "DATACOPILOT_CACHE_DIR", "CI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def dcp(tmp_path: Path) -> Path:
    dc.load(synth.build_coco(tmp_path / "coco"), "coco")  # reads fine
    from datacopilot import convert

    convert.ingest(tmp_path / "coco", "coco", tmp_path / "ds.dcp")
    return tmp_path / "ds.dcp"


def test_the_names_are_public() -> None:
    assert {"propose", "apply", "ProposalSet"} <= set(dc.__all__)
    assert dc.ProposalSet is ProposalSet


def test_propose_returns_the_stored_set_and_prints_nothing(
    dcp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ps = dc.propose(dcp, "test_boxes")
    assert isinstance(ps, ProposalSet) and ps.proposer == "test_boxes" and not ps.incomplete
    assert DatasetStore.open(dcp).read_proposals(ps.set_id) == ps
    assert capsys.readouterr().out == ""


def test_propose_validates_its_options(dcp: Path) -> None:
    for kw in ({"min_score": 0.0}, {"classes": ["unicorn"]}, {"time_budget_seconds": -1.0},
               {"class_min_score": {"cat": 2.0}}):  # fmt: skip
        with pytest.raises(ValueError):
            dc.propose(dcp, "test_boxes", **kw)
    with pytest.raises(ValueError, match="unknown proposer"):
        dc.propose(dcp, "nope")


def test_propose_wants_a_stored_dataset_not_an_in_memory_one(dcp: Path) -> None:
    with pytest.raises(ValueError, match=r"\.dcp"):
        dc.propose(dc.load(dcp), "test_boxes")  # type: ignore[arg-type]


def test_a_disallowed_license_raises(dcp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "mit")
    with pytest.raises(LicenseError):
        dc.propose(dcp, "test_boxes")


def reviewed(dcp: Path, ps: ProposalSet) -> ProposalSet:
    done = ps.derive([p.model_copy(update={"status": "accepted"}) for p in ps.proposals])
    DatasetStore.open(dcp).write_proposals(done)
    return done


def test_apply_writes_a_new_dataset_and_returns_its_manifest(dcp: Path, tmp_path: Path) -> None:
    ps = dc.propose(dcp, "test_boxes")
    done = reviewed(dcp, ps)
    manifest = dc.apply(dcp, done.set_id, tmp_path / "new.dcp")
    parent = DatasetStore.open(dcp).manifest()
    assert manifest.lineage == (parent.version,)
    assert manifest.annotation_count == parent.annotation_count + len(ps.proposals)
    loaded = dc.load(tmp_path / "new.dcp")
    assert {a.provenance for a in loaded.annotations} >= {"human", "proposal_accepted"}


def test_apply_refuses_an_unreviewed_set(dcp: Path, tmp_path: Path) -> None:
    ps = dc.propose(dcp, "test_boxes")
    with pytest.raises(ValidationError, match="no accepted"):
        dc.apply(dcp, ps.set_id, tmp_path / "new.dcp")
    assert not (tmp_path / "new.dcp").exists()


def test_apply_checks_the_license_again(
    dcp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    done = reviewed(dcp, dc.propose(dcp, "test_boxes"))
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "mit")
    with pytest.raises(LicenseError):
        dc.apply(dcp, done.set_id, tmp_path / "new.dcp")


def test_the_functions_do_not_write_to_standard_output(
    dcp: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    done = reviewed(dcp, dc.propose(dcp, "test_boxes"))
    dc.apply(dcp, done.set_id, tmp_path / "n.dcp")
    assert capsys.readouterr().out == ""
