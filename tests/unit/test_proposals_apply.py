"""apply: accepted proposals become a NEW dataset with lineage; nothing existing is ever touched."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import structlog
from fixtures.auditdata import Img, build

from datacopilot.core.errors import FormatError, LicenseError, StoreError, ValidationError
from datacopilot.core.proposals import Proposal, ProposalSet
from datacopilot.core.schema import Dataset, DatasetManifest
from datacopilot.models import registry as models
from datacopilot.proposals.apply import apply_proposals
from datacopilot.store import DatasetStore

ALLOWED = frozenset({"apache-2.0", "mit", "bsd-3-clause"})
SRC = Path(__file__).resolve().parents[2] / "src" / "datacopilot"


@pytest.fixture(autouse=True)
def _captured_logs() -> structlog.testing.LogCapture:
    with structlog.testing.capture_logs() as logs:
        yield logs


def stored(tmp_path: Path) -> tuple[Path, Dataset]:
    ds = build(
        tmp_path,
        [
            Img("a.png", 64, 48, boxes=[(0, 8.0, 6.0, 40.0, 30.0)]),
            Img("b.png", 80, 60, split="val", boxes=[(1, 5.0, 5.0, 30.0, 30.0)]),
        ],
    )
    path = tmp_path / "src.dcp"
    DatasetStore.create(
        path, ds.images, ds.annotations, ds.categories, name="demo", task="detection"
    )
    return path, DatasetStore.open(path).read()


def make_set(
    ds: Dataset,
    statuses: tuple[str, ...] = ("accepted", "rejected", "pending"),
    first_id: int = 0,
    parent_set_id: str | None = "1" * 32,
    **kw: object,
) -> ProposalSet:
    a, b = (next(i for i in ds.images if i.file_path.endswith(n)) for n in ("a.png", "b.png"))
    spec = [
        (a, 1, "cat", (41.0, 31.0, 60.0, 46.0)),
        (a, 3, "bird", (1.0, 1.0, 20.0, 20.0)),
        (b, 2, "dog", (40.0, 30.0, 70.0, 55.0)),
    ]
    props = [
        Proposal(
            proposal_id=f"{first_id + n:016x}",
            image_id=img.image_id,
            category_id=cat,
            bbox_xyxy_abs=box,
            score=0.9 - n / 10,
            prompt=name,
            status=statuses[n],  # type: ignore[arg-type]
        )
        for n, (img, cat, name, box) in enumerate(spec)
    ]
    fields: dict[str, object] = {
        "dataset_version": ds.manifest.version,
        "proposer": "test_boxes",
        "model_id": "test_boxes",
        "model_revision": "1",
        "weights": {},
        "prompt_hash": "d" * 64,
        "min_score": 0.35,
        "class_min_score": {},
        "overlap_iou": 0.5,
        "images_total": 2,
        "images_done": 2,
        "incomplete_reason": None,
        "rejected": {},
        "test_only": False,
        "parent_set_id": parent_set_id,
        "proposals": props,
    }
    return ProposalSet.build(**{**fields, **kw})  # type: ignore[arg-type]


def digest_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def go(ds: Dataset, ps: ProposalSet, out: Path, **kw: object) -> DatasetManifest:
    """apply_proposals with the defaults most tests want: the test double allowed, a source."""
    kw.setdefault("allowed_licenses", ALLOWED)
    kw.setdefault("source", out.parent / "src.dcp")
    kw.setdefault("allow_test_proposer", True)
    return apply_proposals(ds, ps, out, **kw)  # type: ignore[arg-type]


def run(tmp_path: Path, ds: Dataset, ps: ProposalSet, **kw: object) -> Path:
    out = tmp_path / "new.dcp"
    go(ds, ps, out, **kw)
    return out


# ----------------------------------------------------------------------------- what is written
def test_only_accepted_proposals_become_annotations_with_provenance(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    ps = make_set(ds)
    new = DatasetStore.open(run(tmp_path, ds, ps)).read()
    added = [a for a in new.annotations if a.provenance == "proposal_accepted"]
    assert len(new.annotations) == len(ds.annotations) + 1 and len(added) == 1
    (row,) = added
    assert row.ann_id == "proposal:" + f"{0:016x}" and row.kind == "bbox"
    assert row.category_id == 1 and row.bbox_xyxy_abs == (41.0, 31.0, 60.0, 46.0)
    assert row.score == 0.9
    assert row.source == {
        "proposal_id": f"{0:016x}",
        "set_id": ps.set_id,
        "model_id": "test_boxes",
        "model_revision": "1",
        "edited": False,
    }
    assert {a.ann_id for a in ds.annotations} <= {a.ann_id for a in new.annotations}
    assert new.images == ds.images and new.categories == ds.categories


def test_an_edited_box_is_the_one_that_is_applied(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    ps = make_set(ds)
    edited = ps.derive(
        [
            p.model_copy(update={"edited": True, "bbox_xyxy_abs": (40.0, 30.0, 60.0, 46.0)})
            if p.status == "accepted"
            else p
            for p in ps.proposals
        ]
    )
    new = DatasetStore.open(run(tmp_path, ds, edited)).read()
    (row,) = [a for a in new.annotations if a.provenance == "proposal_accepted"]
    assert row.bbox_xyxy_abs == (40.0, 30.0, 60.0, 46.0)
    assert row.score is None and row.source["edited"] is True  # the reviewer's box, not the model's


def test_the_new_version_records_its_parent_in_the_lineage(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    manifest = go(ds, make_set(ds), tmp_path / "new.dcp")
    assert manifest.lineage == (ds.manifest.version,)
    assert manifest.version != ds.manifest.version
    assert manifest.annotation_count == ds.manifest.annotation_count + 1
    again = DatasetStore.open(tmp_path / "new.dcp").manifest()
    assert again == manifest
    # a second generation extends the chain
    second_set = make_set(
        DatasetStore.open(tmp_path / "new.dcp").read(),
        statuses=("accepted", "accepted", "rejected"),
        first_id=100,
    )
    second = go(
        DatasetStore.open(tmp_path / "new.dcp").read(),
        second_set,
        tmp_path / "third.dcp",
    )
    assert second.lineage == (ds.manifest.version, manifest.version)


def test_nothing_in_the_source_dataset_changes(tmp_path: Path) -> None:
    path, ds = stored(tmp_path)
    store = DatasetStore.open(path)
    store.write_proposals(make_set(ds))
    before = digest_tree(path)
    run(tmp_path, ds, make_set(ds), source=path)
    assert digest_tree(path) == before
    assert DatasetStore.open(path).read().manifest.version == ds.manifest.version


def test_the_same_inputs_give_the_same_version(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    ps = make_set(ds)
    a = go(ds, ps, tmp_path / "a.dcp")
    b = go(ds, ps, tmp_path / "b.dcp")
    assert a.version == b.version


def test_applying_logs_the_versions_and_the_count(
    tmp_path: Path, _captured_logs: list[dict[str, object]]
) -> None:
    _, ds = stored(tmp_path)
    ps = make_set(ds)
    manifest = go(ds, ps, tmp_path / "new.dcp")
    (event,) = [e for e in _captured_logs if e["event"] == "proposals_applied"]
    assert event["parent_version"] == ds.manifest.version
    assert event["dataset_version"] == manifest.version
    assert event["applied_count"] == 1 and event["set_id"] == ps.set_id
    assert str(tmp_path) not in repr(_captured_logs)


# ----------------------------------------------------------------------------- append only
def test_no_code_path_in_the_proposal_use_cases_can_rewrite_an_existing_version() -> None:
    for path in (SRC / "proposals").glob("*.py"):
        text = path.read_text()
        assert "write_version" not in text, path.name
        assert not re.search(
            r"os\.(replace|rename|remove|unlink)|shutil\.(rmtree|move|copy)", text
        ), path.name
    assert "DatasetStore.create(" in (SRC / "proposals" / "apply.py").read_text()


@pytest.mark.parametrize("where", ["same", "inside", "parent"])
def test_the_output_must_be_a_new_directory_outside_the_source(tmp_path: Path, where: str) -> None:
    path, ds = stored(tmp_path)
    out = {"same": path, "inside": path / "new.dcp", "parent": tmp_path}[where]
    before = digest_tree(path)
    with pytest.raises((ValidationError, StoreError)):
        go(ds, make_set(ds), out, allowed_licenses=ALLOWED, source=path)
    assert digest_tree(path) == before and not (path / "new.dcp").exists()


def test_a_symlink_into_the_source_does_not_get_around_the_guard(tmp_path: Path) -> None:
    path, ds = stored(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises((ValidationError, StoreError)):
        go(ds, make_set(ds), link / "new.dcp", allowed_licenses=ALLOWED, source=path)
    assert not (path / "new.dcp").exists()


def test_an_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    (tmp_path / "new.dcp").mkdir()
    (tmp_path / "new.dcp" / "keep.txt").write_text("x")
    with pytest.raises(StoreError, match="already exists"):
        run(tmp_path, ds, make_set(ds))
    assert (tmp_path / "new.dcp" / "keep.txt").read_text() == "x"


# ----------------------------------------------------------------------------- refusals
def test_a_set_for_another_version_is_refused(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    with pytest.raises(ValidationError, match="another dataset version"):
        run(tmp_path, ds, make_set(ds, dataset_version="e" * 32))
    assert not (tmp_path / "new.dcp").exists()


def test_a_set_without_an_accepted_proposal_is_refused(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    with pytest.raises(ValidationError, match="no accepted"):
        run(tmp_path, ds, make_set(ds, statuses=("pending", "rejected", "pending")))
    assert not (tmp_path / "new.dcp").exists()


def test_an_incomplete_set_is_refused_unless_asked_for(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    partial = make_set(ds, images_done=1, incomplete_reason="time_budget")
    with pytest.raises(ValidationError, match="incomplete"):
        run(tmp_path, ds, partial)
    assert not (tmp_path / "new.dcp").exists()
    run(tmp_path, ds, partial, allow_incomplete=True)
    assert (tmp_path / "new.dcp").is_dir()


def test_a_proposer_that_is_not_installed_is_refused(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    with pytest.raises(FormatError, match="unknown proposer"):
        run(tmp_path, ds, make_set(ds, proposer="ghost"))


def test_a_model_that_is_not_on_the_license_allow_list_is_refused(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    with pytest.raises(LicenseError, match="DATACOPILOT_ALLOWED_LICENSES"):
        go(ds, make_set(ds), tmp_path / "new.dcp", allowed_licenses=frozenset({"mit"}))
    assert not (tmp_path / "new.dcp").exists()
    with pytest.raises(LicenseError):
        go(
            ds,
            make_set(ds, proposer="grounding_dino", model_id="grounding_dino"),
            tmp_path / "new2.dcp",
            allowed_licenses=frozenset({"mit"}),
        )


def test_a_registered_model_must_be_the_pinned_revision_with_the_pinned_files(
    tmp_path: Path,
) -> None:
    _, ds = stored(tmp_path)
    card = models.get_model("grounding_dino", ALLOWED)
    good = make_set(
        ds,
        proposer="grounding_dino",
        model_id=card.model_id,
        model_revision=card.revision,
        weights=card.pinned_files(),
    )
    run(tmp_path, ds, good)
    for change in (
        {"model_revision": "other-revision"},
        {"weights": {**card.pinned_files(), "model.safetensors": "0" * 64}},
        {"weights": {}},
        {"model_id": "something_else"},
    ):
        fields = {
            "proposer": "grounding_dino",
            "model_id": card.model_id,
            "model_revision": card.revision,
            "weights": card.pinned_files(),
            **change,
        }
        with pytest.raises(ValidationError, match="pinned"):
            go(
                ds,
                make_set(ds, **fields),
                tmp_path / f"x{len(str(change))}.dcp",
            )


def test_a_hostile_set_cannot_smuggle_a_box_outside_an_image(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    a = next(i for i in ds.images if i.file_path.endswith("a.png"))
    bad = Proposal(
        proposal_id="a" * 16,
        image_id=a.image_id,
        category_id=1,
        bbox_xyxy_abs=(1.0, 1.0, 640.0, 480.0),
        score=0.5,
        prompt="cat",
        status="accepted",
    )
    ps = make_set(ds)
    hostile = ProposalSet.build(
        **{**ps.model_dump(exclude={"set_id", "proposals"}), "proposals": [bad]}
    )
    with pytest.raises(ValidationError, match="outside"):
        run(tmp_path, ds, hostile)


def test_a_proposal_id_that_collides_with_an_existing_annotation_is_refused(tmp_path: Path) -> None:
    from fixtures.auditdata import dataset

    _, ds = stored(tmp_path)
    clash = ds.annotations[0].model_copy(update={"ann_id": "proposal:" + f"{0:016x}"})
    dup = dataset(ds.images, [clash, *ds.annotations[1:]], ds.categories, root=tmp_path)
    with pytest.raises(ValidationError, match="duplicate"):
        go(dup, make_set(dup), tmp_path / "new.dcp")
    assert not (tmp_path / "new.dcp").exists()


# ----------------------------------------------------------------------------- trust in the set
def test_the_source_directory_is_required(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    with pytest.raises(TypeError):
        apply_proposals(  # type: ignore[call-arg]
            ds,
            make_set(ds),
            tmp_path / "new.dcp",
            allowed_licenses=ALLOWED,
            allow_test_proposer=True,
        )


def test_a_set_that_never_went_through_review_import_is_refused(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    forged = make_set(ds, parent_set_id=None)  # hand-written: accepted, with a valid digest
    with pytest.raises(ValidationError, match="review import"):
        run(tmp_path, ds, forged)
    assert not (tmp_path / "new.dcp").exists()


def reviewed_pair(ds: Dataset) -> tuple[ProposalSet, ProposalSet]:
    fresh = make_set(ds, statuses=("pending", "pending", "pending"), parent_set_id=None)
    done = fresh.derive([p.model_copy(update={"status": "accepted"}) for p in fresh.proposals[:1]]
                        + list(fresh.proposals[1:]))  # fmt: skip
    return fresh, done


def test_a_review_that_matches_its_parent_is_applied_and_both_sets_travel(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    fresh, done = reviewed_pair(ds)
    out = tmp_path / "new.dcp"
    go(ds, done, out, parent=fresh)
    store = DatasetStore.open(out)
    assert (
        store.read_proposals(done.set_id) == done
    )  # the provenance survives the source's deletion
    assert store.read_proposals(fresh.set_id) == fresh


def test_a_review_that_is_not_derived_from_the_given_parent_is_refused(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    fresh, done = reviewed_pair(ds)
    other = make_set(
        ds, statuses=("pending", "pending", "pending"), parent_set_id=None, min_score=0.5
    )
    with pytest.raises(ValidationError, match="not a review of its parent"):
        go(ds, done, tmp_path / "a.dcp", parent=other)
    # a forged review: same parent id, but a proposal the parent never had
    forged = ProposalSet.build(
        **{**done.model_dump(exclude={"set_id", "proposals"}),
           "proposals": [done.proposals[0].model_copy(update={"score": 0.5}), *done.proposals[1:]]}
    )  # fmt: skip
    with pytest.raises(ValidationError, match="not a review of its parent"):
        go(ds, forged, tmp_path / "b.dcp", parent=fresh)


def test_a_test_double_is_refused_unless_the_caller_allows_it(tmp_path: Path) -> None:
    _, ds = stored(tmp_path)
    with pytest.raises(ValidationError, match="test double"):
        go(ds, make_set(ds), tmp_path / "a.dcp", allow_test_proposer=False)
    go(ds, make_set(ds), tmp_path / "b.dcp", allow_test_proposer=True)
    # a set that claims not to be a test double, from the proposer that is one, is still refused
    claim = make_set(ds, test_only=False)
    assert claim.proposer == "test_boxes"
    with pytest.raises(ValidationError, match="test double"):
        go(ds, claim, tmp_path / "c.dcp", allow_test_proposer=False)
    flagged = make_set(ds, test_only=True)
    with pytest.raises(ValidationError, match="test double"):
        go(ds, flagged, tmp_path / "d.dcp", allow_test_proposer=False)


def test_a_set_cannot_claim_the_identity_of_a_registered_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    from datacopilot.proposals import apply as apply_module

    _, ds = stored(tmp_path)
    with pytest.raises(ValidationError, match="pinned"):
        go(ds, make_set(ds, model_id="grounding_dino"), tmp_path / "a.dcp")  # proposer test_boxes
    # a third-party proposer that names a first-party model as its own
    plugin = types.SimpleNamespace(spec=None, test_only=False)
    monkeypatch.setattr(apply_module, "check_proposer", lambda name, allowed: plugin)
    with pytest.raises(ValidationError, match="pinned"):
        go(ds, make_set(ds, proposer="thirdparty", model_id="dinov2_base"), tmp_path / "b.dcp")
    # and an honest third-party identity is fine
    go(ds, make_set(ds, proposer="thirdparty", model_id="their_model"), tmp_path / "c.dcp")


def test_a_set_whose_parent_is_missing_from_the_dataset_is_refused_by_the_use_case(
    tmp_path: Path,
) -> None:
    from datacopilot.proposals import usecase
    from datacopilot.settings import Settings

    path, ds = stored(tmp_path)
    _, done = reviewed_pair(ds)
    DatasetStore.open(path).write_proposals(done)  # the parent was deleted (or never copied)
    settings = Settings(allowed_licenses=ALLOWED, cache_dir=tmp_path / "cache")
    with pytest.raises(ValidationError, match="not in the dataset"):
        usecase.apply_set(path, done.set_id, tmp_path / "n.dcp", settings=settings,
                          allow_test_proposer=True)  # fmt: skip


# ----------------------------------------------------------------------------- output elsewhere
def stored_relative(tmp_path: Path) -> tuple[Path, Dataset]:
    """A dataset at ``proj/src.dcp`` whose rows name their images relative to ``proj/``."""
    proj = tmp_path / "proj"
    ds = build(
        proj,
        [
            Img("a.png", 64, 48, boxes=[(0, 8.0, 6.0, 40.0, 30.0)], path="images/a.png"),
            Img("b.png", 80, 60, split="val", boxes=[(1, 5.0, 5.0, 30.0, 30.0)],
                path="images/b.png"),
        ],
    )  # fmt: skip
    path = proj / "src.dcp"
    DatasetStore.create(
        path, ds.images, ds.annotations, ds.categories, name="demo", task="detection"
    )
    return path, DatasetStore.open(path).read()


def test_an_output_in_another_directory_still_finds_the_images(tmp_path: Path) -> None:
    import datacopilot as dc

    source, ds = stored_relative(tmp_path)
    out = tmp_path / "new.dcp"  # one level up: the relative paths must gain "proj/"
    go(ds, make_set(ds), out, source=source)
    new = DatasetStore.open(out).read()
    assert sorted(i.file_path for i in new.images) == ["proj/images/a.png", "proj/images/b.png"]
    run = dc.audit(out, ["integrity"], fail_on=("warn",))
    assert run.passed and run.findings == ()


def test_an_output_whose_directory_does_not_hold_the_images_is_refused(tmp_path: Path) -> None:
    source, ds = stored_relative(tmp_path)
    out = tmp_path / "elsewhere" / "new.dcp"
    with pytest.raises(ValidationError, match="outside"):
        go(ds, make_set(ds), out, source=source)
    assert not out.exists()
