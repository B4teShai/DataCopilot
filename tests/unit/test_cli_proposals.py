"""propose, review export, review import and apply through the real CLI path (``cli.main``).

``test_boxes`` is the dependency-free proposer: its boxes mean nothing about pictures, but the whole
flow (store, review round trip, new dataset with lineage, exit codes, settings) runs for real.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fixtures import synth
from fixtures.reviewxml import decide, xml_files

from datacopilot.cli import app as cli
from datacopilot.core.proposals import ProposalSet
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

ENV = (
    "DATACOPILOT_ENV",
    "DATACOPILOT_LOG_LEVEL",
    "DATACOPILOT_WORKERS",
    "DATACOPILOT_HASH_CACHE",
    "DATACOPILOT_ALLOWED_LICENSES",
    "DATACOPILOT_CACHE_KEY",
    "DATACOPILOT_CACHE_DIR",
    "XDG_CACHE_HOME",
    "CI",
)


def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    clean_env(monkeypatch, tmp_path)
    yield
    clear_request_context()


def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def logs(err: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in err.splitlines() if line.startswith("{")]


def messages(err: str) -> list[str]:
    return [line for line in err.splitlines() if line and not line.startswith("{")]


@pytest.fixture
def dcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = synth.build_coco(tmp_path / "coco")
    assert run(monkeypatch, "ingest", root, "--format", "coco", "-o", tmp_path / "ds.dcp") == 0
    return tmp_path / "ds.dcp"


def propose(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    *extra: str,
) -> tuple[int, str, str]:
    capsys.readouterr()
    code = run(monkeypatch, "propose", dcp, "--proposer", "test_boxes", *extra)
    out, err = capsys.readouterr()
    return code, out, err


def set_id_of(out: str) -> str:
    match = re.search(r"\bset=([0-9a-f]{32})\b", out)
    assert match, out
    return match.group(1)


def digest_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ----------------------------------------------------------------------------- propose
def test_propose_stores_a_set_and_prints_one_summary_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dcp: Path
) -> None:
    code, out, err = propose(monkeypatch, capsys, dcp)
    assert code == 0
    (line,) = out.splitlines()
    match = re.fullmatch(
        r"proposed=(\d+) images=4/4 set=([0-9a-f]{32}) model=test_boxes@1 run_id=[0-9a-f-]{36}",
        line,
    )
    assert match and int(match.group(1)) >= 1
    ps = DatasetStore.open(dcp).read_proposals(match.group(2))
    assert len(ps.proposals) == int(match.group(1)) and ps.proposer == "test_boxes"
    assert ps.dataset_version == DatasetStore.open(dcp).manifest().version
    assert any("test double" in m for m in messages(err))  # the warning for a test proposer
    events = {r["message"]: r for r in logs(err)}
    assert events["proposal_set_written"]["set_id"] == ps.set_id
    assert events["proposer_started"]["image_count"] == 4


def test_propose_is_deterministic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dcp: Path
) -> None:
    a = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    b = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    assert a == b


def test_propose_never_changes_the_dataset_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dcp: Path
) -> None:
    before = {k: v for k, v in digest_tree(dcp).items() if not k.startswith("proposals")}
    propose(monkeypatch, capsys, dcp)
    after = {k: v for k, v in digest_tree(dcp).items() if not k.startswith("proposals")}
    assert before == after


def test_classes_and_thresholds_reach_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dcp: Path
) -> None:
    code, out, _ = propose(
        monkeypatch, capsys, dcp, "--classes", "cat,dog", "--min-score", "0.6",
        "--min-score-for", "cat=0.9",
    )  # fmt: skip
    assert code == 0
    ps = DatasetStore.open(dcp).read_proposals(set_id_of(out))
    assert ps.min_score == 0.6 and ps.class_min_score == {"cat": 0.9}
    assert {p.prompt for p in ps.proposals} <= {"cat", "dog"}
    assert all(p.score >= (0.9 if p.prompt == "cat" else 0.6) for p in ps.proposals)


@pytest.mark.parametrize(
    "extra",
    [
        ("--min-score", "0"),
        ("--min-score", "1.5"),
        ("--min-score", "nan"),
        ("--min-score-for", "cat"),
        ("--min-score-for", "cat=abc"),
        ("--min-score-for", "cat=2"),
        ("--min-score-for", "unicorn=0.5"),
        ("--classes", "unicorn"),
        ("--classes", "cat,,dog"),
        ("--time-budget", "0"),
        ("--time-budget", "-5"),
        ("--time-budget", "inf"),
        ("--proposer", "../etc"),
        ("--proposer", "no_such_proposer"),
    ],
)
def test_bad_options_exit_2_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    extra: tuple[str, ...],
) -> None:
    capsys.readouterr()
    code = run(monkeypatch, "propose", dcp, "--proposer", "test_boxes", *extra)
    err = capsys.readouterr().err
    assert code in (1, 2) and "Traceback" not in "".join(messages(err))
    if "no_such_proposer" not in extra and "../etc" not in extra:
        assert code == 2
    assert not (dcp / "proposals").exists() or not list((dcp / "proposals").iterdir())


def test_a_disallowed_license_exits_1_before_the_dataset_is_read(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "mit")
    capsys.readouterr()
    code = run(monkeypatch, "propose", tmp_path / "missing.dcp", "--proposer", "grounding_dino")
    err = capsys.readouterr().err
    assert code == 1 and "DATACOPILOT_ALLOWED_LICENSES" in err and "not a dataset" not in err


def test_a_missing_optional_dependency_exits_1_and_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
) -> None:
    pytest.importorskip("PIL")  # present in the dev environment; the detector needs transformers
    try:
        import transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("the [vlm] libraries are installed here")
    capsys.readouterr()
    code = run(monkeypatch, "propose", dcp, "--proposer", "grounding_dino")
    err = capsys.readouterr().err
    assert code == 1 and "vlm" in err and "Traceback" not in "".join(messages(err))


def test_a_time_budget_that_ends_the_run_exits_1_but_keeps_the_partial_set(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
) -> None:
    from datacopilot.proposals import run as run_module

    ticks = iter(range(1000))
    fake = types.SimpleNamespace(monotonic=lambda: float(next(ticks)))
    monkeypatch.setattr(run_module, "time", fake)
    code, out, err = propose(monkeypatch, capsys, dcp, "--time-budget", "1.5")
    assert code == 1
    ps = DatasetStore.open(dcp).read_proposals(set_id_of(out))
    assert ps.incomplete_reason == "time_budget" and ps.images_done < ps.images_total
    assert any("incomplete" in m and "--allow-incomplete" in m for m in messages(err))


def test_a_classification_dataset_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = synth.build_imagefolder(tmp_path / "folder")
    assert (
        run(monkeypatch, "ingest", root, "--format", "imagefolder", "-o", tmp_path / "c.dcp") == 0
    )
    code, _, err = propose(monkeypatch, capsys, tmp_path / "c.dcp")
    assert code == 1 and "detection" in err


# ----------------------------------------------------------------------------- the whole flow
def test_propose_review_apply_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    _, out, _ = propose(monkeypatch, capsys, dcp)
    set_id = set_id_of(out)
    parent = DatasetStore.open(dcp).manifest()

    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    assert capsys.readouterr().out.strip() == str(review_dir)
    assert (review_dir / "images").is_dir() and xml_files(review_dir)

    # a reviewer accepts everything they can see, except one box they move first
    ids = [p.proposal_id for p in DatasetStore.open(dcp).read_proposals(set_id).proposals]
    decide(review_dir, {pid: "accepted" for pid in ids})
    before = digest_tree(dcp)
    assert run(monkeypatch, "review", "import", dcp, review_dir, "--proposals", set_id,
               "--from", "cvat") == 0  # fmt: skip
    line = capsys.readouterr().out.strip()
    reviewed_id = set_id_of(line)
    assert f"accepted={len(ids)} rejected=0 pending=0" in line and reviewed_id != set_id
    reviewed = DatasetStore.open(dcp).read_proposals(reviewed_id)
    assert reviewed.parent_set_id == set_id
    after = digest_tree(dcp)
    assert {k for k in after if k not in before} == {f"proposals/{reviewed_id}.json"}
    assert all(after[k] == before[k] for k in before)  # nothing that existed was touched

    new = tmp_path / "new.dcp"
    assert run(monkeypatch, "apply", dcp, "--proposals", reviewed_id, "-o", new) == 0
    version = capsys.readouterr().out.strip()
    child = DatasetStore.open(new)
    assert child.manifest().version == version and child.manifest().lineage == (parent.version,)
    rows = child.read().annotations
    added = [a for a in rows if a.provenance == "proposal_accepted"]
    assert len(added) == len(ids) and len(rows) == parent.annotation_count + len(ids)
    assert all(a.source["set_id"] == reviewed_id for a in added)
    assert digest_tree(dcp) == after  # the parent is exactly as it was before the apply

    # the new dataset is a dataset like any other
    assert run(monkeypatch, "audit", new, "--no-hash-cache", "--fail-on", "none",
               "--report", tmp_path / "r.html") == 0  # fmt: skip
    capsys.readouterr()
    assert run(monkeypatch, "export", new, "--format", "yolo", "-o", tmp_path / "yolo") == 0


def test_apply_refuses_what_nobody_accepted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    capsys.readouterr()
    code = run(monkeypatch, "apply", dcp, "--proposals", set_id, "-o", tmp_path / "new.dcp")
    err = capsys.readouterr().err
    assert code == 1 and "no accepted" in err and not (tmp_path / "new.dcp").exists()


def test_apply_refuses_an_incomplete_set_unless_told_to(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    from datacopilot.proposals import run as run_module

    ticks = iter(range(1000))
    fake = types.SimpleNamespace(monotonic=lambda: float(next(ticks)))
    monkeypatch.setattr(run_module, "time", fake)
    _, out, _ = propose(monkeypatch, capsys, dcp, "--time-budget", "2.5")
    partial_id = set_id_of(out)
    monkeypatch.undo()  # the real clock again
    clean_env(monkeypatch, tmp_path)
    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", partial_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    ps = DatasetStore.open(dcp).read_proposals(partial_id)
    if not ps.proposals:
        pytest.skip("this partial run proposed nothing")
    decide(review_dir, {p.proposal_id: "accepted" for p in ps.proposals})
    capsys.readouterr()
    assert run(monkeypatch, "review", "import", dcp, review_dir, "--proposals", partial_id,
               "--from", "cvat") == 0  # fmt: skip
    reviewed_id = set_id_of(capsys.readouterr().out)
    assert run(monkeypatch, "apply", dcp, "--proposals", reviewed_id, "-o", tmp_path / "n.dcp") == 1
    assert "incomplete" in capsys.readouterr().err and not (tmp_path / "n.dcp").exists()
    assert run(monkeypatch, "apply", dcp, "--proposals", reviewed_id, "-o", tmp_path / "n.dcp",
               "--allow-incomplete") == 0  # fmt: skip
    assert (tmp_path / "n.dcp").is_dir()


@pytest.mark.parametrize("bad", ["", "latest", "../x", "A" * 32, "a" * 31])
def test_a_malformed_set_id_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
    bad: str,
) -> None:
    capsys.readouterr()
    for command in (
        ["review", "export", dcp, "--proposals", bad, "--to", "cvat", "-o", tmp_path / "r"],
        ["apply", dcp, "--proposals", bad, "-o", tmp_path / "n.dcp"],
    ):
        assert run(monkeypatch, *command) == 2
        assert "Traceback" not in "".join(messages(capsys.readouterr().err))


def test_an_unknown_set_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    capsys.readouterr()
    code = run(monkeypatch, "apply", dcp, "--proposals", "a" * 32, "-o", tmp_path / "n.dcp")
    assert code == 1 and "no proposal set" in capsys.readouterr().err


def test_a_tampered_set_is_refused_by_every_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    path = dcp / "proposals" / f"{set_id}.json"
    doc = json.loads(path.read_text())
    for proposal in doc["proposals"]:
        proposal["status"] = "accepted"  # a forged review
    path.write_text(json.dumps(doc))
    capsys.readouterr()
    for command in (
        ["review", "export", dcp, "--proposals", set_id, "--to", "cvat", "-o", tmp_path / "r"],
        ["apply", dcp, "--proposals", set_id, "-o", tmp_path / "n.dcp"],
    ):
        assert run(monkeypatch, *command) == 1
        assert "invalid" in capsys.readouterr().err
    assert not (tmp_path / "n.dcp").exists() and not (tmp_path / "r").exists()


def test_a_set_for_another_version_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    other = synth.build_coco(tmp_path / "coco2", with_extras=False)
    (other / "images" / "extra.png").write_bytes(synth.png_bytes(30, 30, 99))
    first = json.loads((other / "annotations" / "instances_train.json").read_text())
    first["images"].append({"id": 99, "file_name": "extra.png", "width": 30, "height": 30})
    (other / "annotations" / "instances_train.json").write_text(json.dumps(first))
    assert run(monkeypatch, "ingest", other, "--format", "coco", "-o", tmp_path / "v2.dcp") == 0
    DatasetStore.open(tmp_path / "v2.dcp").write_proposals(
        DatasetStore.open(dcp).read_proposals(set_id)
    )
    capsys.readouterr()
    code = run(monkeypatch, "apply", tmp_path / "v2.dcp", "--proposals", set_id,
               "-o", tmp_path / "n.dcp")  # fmt: skip
    assert code == 1 and "another dataset version" in capsys.readouterr().err
    assert not (tmp_path / "n.dcp").exists()


def test_apply_refuses_a_model_whose_license_is_no_longer_allowed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    _, out, _ = propose(monkeypatch, capsys, dcp)
    set_id = set_id_of(out)
    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    decide(
        review_dir,
        {
            p.proposal_id: "accepted"
            for p in DatasetStore.open(dcp).read_proposals(set_id).proposals
        },
    )
    capsys.readouterr()
    assert run(monkeypatch, "review", "import", dcp, review_dir, "--proposals", set_id,
               "--from", "cvat") == 0  # fmt: skip
    reviewed_id = set_id_of(capsys.readouterr().out)
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "mit")
    assert run(monkeypatch, "apply", dcp, "--proposals", reviewed_id, "-o", tmp_path / "n.dcp") == 1
    assert "DATACOPILOT_ALLOWED_LICENSES" in capsys.readouterr().err
    assert not (tmp_path / "n.dcp").exists()


def test_apply_to_an_existing_or_inner_output_exits_1_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    before = digest_tree(dcp)
    (tmp_path / "taken.dcp").mkdir()
    capsys.readouterr()
    for target in (tmp_path / "taken.dcp", dcp / "inner.dcp", dcp):
        assert run(monkeypatch, "apply", dcp, "--proposals", set_id, "-o", target) in (1, 2)
        assert "Traceback" not in "".join(messages(capsys.readouterr().err))
    assert digest_tree(dcp) == before


def test_apply_output_must_end_in_dcp(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dcp: Path, tmp_path: Path
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    capsys.readouterr()
    assert run(monkeypatch, "apply", dcp, "--proposals", set_id, "-o", tmp_path / "new") == 2


def test_review_import_with_an_unknown_target_or_empty_file_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    capsys.readouterr()
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "fiftyone",
               "-o", tmp_path / "x") == 2  # fmt: skip
    assert run(monkeypatch, "review", "import", dcp, review_dir, "--proposals", set_id,
               "--from", "fiftyone") == 2  # fmt: skip
    capsys.readouterr()
    assert (
        run(
            monkeypatch,
            "review",
            "import",
            dcp,
            review_dir,
            "--proposals",
            set_id,
            "--from",
            "cvat",
        )
        == 1
    )  # no decisions yet
    assert "no decisions" in capsys.readouterr().err


def test_review_import_reports_boxes_that_were_moved_without_a_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    from fixtures.reviewxml import move

    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    ps = DatasetStore.open(dcp).read_proposals(set_id)
    if len(ps.proposals) < 2:
        pytest.skip("not enough proposals in this deterministic run")
    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    first, second = ps.proposals[0], ps.proposals[1]
    decide(review_dir, {second.proposal_id: "rejected"})
    move(review_dir, first.proposal_id, 0.5)
    capsys.readouterr()
    assert run(monkeypatch, "review", "import", dcp, review_dir, "--proposals", set_id,
               "--from", "cvat") == 0  # fmt: skip
    out, err = capsys.readouterr()
    assert "edited_pending=1" in out
    assert any("moved" in m and "no decision" in m for m in messages(err))


def test_the_decision_file_of_another_dataset_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    decide(
        review_dir,
        {
            p.proposal_id: "accepted"
            for p in DatasetStore.open(dcp).read_proposals(set_id).proposals
        },
    )
    # a different dataset with its own proposals
    root = synth.build_voc(tmp_path / "voc")
    assert run(monkeypatch, "ingest", root, "--format", "voc", "-o", tmp_path / "voc.dcp") == 0
    other_id = set_id_of(propose(monkeypatch, capsys, tmp_path / "voc.dcp")[1])
    capsys.readouterr()
    code = run(monkeypatch, "review", "import", tmp_path / "voc.dcp", review_dir,
               "--proposals", other_id, "--from", "cvat")  # fmt: skip
    assert code == 1 and "unknown proposal" in capsys.readouterr().err
    assert DatasetStore.open(tmp_path / "voc.dcp").read_proposals(other_id).dataset_version != (
        DatasetStore.open(dcp).manifest().version
    )


def test_importing_a_review_adds_a_set_and_never_changes_the_one_it_came_from(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dcp: Path,
    tmp_path: Path,
) -> None:
    set_id = set_id_of(propose(monkeypatch, capsys, dcp)[1])
    review_dir = tmp_path / "review"
    assert run(monkeypatch, "review", "export", dcp, "--proposals", set_id, "--to", "cvat",
               "-o", review_dir) == 0  # fmt: skip
    ps = DatasetStore.open(dcp).read_proposals(set_id)
    decide(review_dir, {ps.proposals[0].proposal_id: "accepted"})
    original = (dcp / "proposals" / f"{set_id}.json").read_bytes()
    capsys.readouterr()
    assert run(monkeypatch, "review", "import", dcp, review_dir, "--proposals", set_id,
               "--from", "cvat") == 0  # fmt: skip
    reviewed = DatasetStore.open(dcp).read_proposals(set_id_of(capsys.readouterr().out))
    assert isinstance(reviewed, ProposalSet) and reviewed.parent_set_id == set_id
    assert {p.proposal_id for p in reviewed.proposals} == {p.proposal_id for p in ps.proposals}
    assert (dcp / "proposals" / f"{set_id}.json").read_bytes() == original
