"""The review round trip through CVAT: proposals out as their own layer, decisions back in.

The decisions file is untrusted: unknown, duplicated or foreign ids, changed classes, invalid
decisions and boxes outside the image are refused, never ignored or half applied.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET  # noqa: S405  # tests read the XML this package wrote
from collections.abc import Callable
from pathlib import Path

import pytest
import structlog
from fixtures.auditdata import Img, build, dataset
from fixtures.reviewxml import decide, edit, move, proposal_boxes, set_attribute, xml_files

from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.proposals import Proposal, ProposalSet
from datacopilot.core.schema import Annotation, Category, Dataset
from datacopilot.io import registry
from datacopilot.proposals import review
from datacopilot.proposals.review import ReviewSummary, export_review, import_review


@pytest.fixture(autouse=True)
def _captured_logs() -> structlog.testing.LogCapture:
    with structlog.testing.capture_logs() as logs:
        yield logs


def make_dataset(tmp_path: Path) -> Dataset:
    return build(
        tmp_path,
        [
            Img("a.png", 64, 48, boxes=[(0, 8.0, 6.0, 40.0, 30.0)]),
            Img("b.png", 80, 60, split="val", boxes=[(1, 5.0, 5.0, 30.0, 30.0)]),
        ],
    )


def make_set(ds: Dataset, **kw: object) -> ProposalSet:
    a, b = (next(i for i in ds.images if i.file_path.endswith(n)) for n in ("a.png", "b.png"))
    props = [
        Proposal(
            proposal_id=f"{n:016x}",
            image_id=img.image_id,
            category_id=cat,
            bbox_xyxy_abs=box,
            score=0.9 - n / 10,
            prompt=name,
        )
        for n, (img, cat, name, box) in enumerate(
            [
                (a, 1, "cat", (41.0, 31.0, 60.0, 46.0)),
                (a, 3, "bird", (1.0, 1.0, 20.0, 20.0)),
                (b, 2, "dog", (40.0, 30.0, 70.0, 55.0)),
                (b, 1, "cat", (50.0, 5.0, 79.0, 28.0)),
            ]
        )
    ]
    fields: dict[str, object] = {
        "dataset_version": ds.manifest.version,
        "proposer": "fake",
        "model_id": "fake_model",
        "model_revision": "r1",
        "weights": {},
        "prompt_hash": "d" * 64,
        "min_score": 0.35,
        "class_min_score": {},
        "overlap_iou": 0.5,
        "images_total": 2,
        "images_done": 2,
        "incomplete_reason": None,
        "rejected": {},
        "proposals": props,
    }
    return ProposalSet.build(**{**fields, **kw})  # type: ignore[arg-type]


def exported(tmp_path: Path) -> tuple[Dataset, ProposalSet, Path]:
    ds = make_dataset(tmp_path)
    ps = make_set(ds)
    out = tmp_path / "review"
    export_review(ds, ps, out)
    return ds, ps, out


def read_back(directory: Path) -> tuple[list[Annotation], list[Category]]:
    result = registry.get_reader("cvat_xml").read(directory)
    return list(result.annotations), list(result.categories)


IDS = [f"{n:016x}" for n in range(4)]


# ----------------------------------------------------------------------------- export
def test_export_writes_the_originals_and_the_proposals_as_a_separate_layer(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    anns, cats = read_back(out)
    names = {c.id: c.name for c in cats}
    layer = [a for a in anns if names[a.category_id].endswith(review.SUFFIX)]
    human = [a for a in anns if not names[a.category_id].endswith(review.SUFFIX)]
    assert len(human) == 2 and len(layer) == 4
    assert {names[a.category_id] for a in layer} == {
        f"{n}{review.SUFFIX}" for n in ("cat", "bird", "dog")
    }
    assert {a.attributes["proposal_id"] for a in layer} == set(IDS)
    assert {a.attributes["decision"] for a in layer} == {""}
    scores = {a.attributes["proposal_id"]: a.attributes["score"] for a in layer}
    assert scores[IDS[0]] == "0.900" and all(isinstance(v, str) for v in scores.values())
    assert sorted(a.bbox_xyxy_abs for a in human) == sorted(  # type: ignore[type-var]
        a.bbox_xyxy_abs for a in ds.annotations if a.bbox_xyxy_abs
    )
    assert all(a.provenance == "model" for a in layer)  # CVAT's "auto"


def test_the_images_travel_with_the_file_so_the_reviewer_sees_them(tmp_path: Path) -> None:
    _, _, out = exported(tmp_path)
    assert sorted(p.name for p in (out / "images").iterdir()) == ["a.png", "b.png"]


def test_a_reviewed_set_is_exported_with_its_decisions_filled_in(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    ps = make_set(ds)
    reviewed = ps.derive(
        [
            p.model_copy(update={"status": s})
            for p, s in zip(
                ps.proposals, ("accepted", "rejected", "pending", "pending"), strict=True
            )
        ]
    )
    export_review(ds, reviewed, tmp_path / "again")
    anns, _ = read_back(tmp_path / "again")
    got = {
        a.attributes["proposal_id"]: a.attributes["decision"]
        for a in anns
        if "proposal_id" in a.attributes
    }
    assert sorted(got.values()) == ["", "", "accepted", "rejected"]


def test_a_class_that_already_ends_in_the_layer_suffix_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    cats = [*ds.categories[:2], Category(id=3, name="bird" + review.SUFFIX)]
    clash = dataset(ds.images, ds.annotations, cats)
    ps = make_set(clash)
    with pytest.raises(ValidationError, match="suffix"):
        export_review(clash, ps, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_set_for_another_dataset_version_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path)
    ps = make_set(ds, dataset_version="e" * 32)
    with pytest.raises(ValidationError, match="another dataset version"):
        export_review(ds, ps, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_empty_set_has_nothing_to_review(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    with pytest.raises(ValidationError, match="no proposals"):
        export_review(ds, make_set(ds, proposals=[]), tmp_path / "out")


def test_an_existing_output_directory_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    with pytest.raises(FormatError):
        export_review(ds, make_set(ds), tmp_path / "out")
    assert (tmp_path / "out" / "keep.txt").read_text() == "x"


def test_class_names_with_markup_characters_survive_the_round_trip(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    cats = [
        Category(id=1, name="fish & chips"),
        Category(id=2, name="dog"),
        Category(id=3, name="bird"),
    ]
    ds2 = dataset(ds.images, [], cats)
    export_review(ds2, make_set(ds2), tmp_path / "out")
    _, got = read_back(tmp_path / "out")
    assert "fish & chips" + review.SUFFIX in {c.name for c in got}


def test_a_proposal_id_that_collides_with_an_annotation_id_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    first = ds.annotations[0]
    taken = first.model_copy(update={"ann_id": "proposal:" + IDS[0]})
    clash = dataset(ds.images, [taken, *ds.annotations[1:]], ds.categories)
    with pytest.raises(ValidationError):
        export_review(clash, make_set(clash), tmp_path / "out")


# ----------------------------------------------------------------------------- import
def test_a_round_trip_with_everything_accepted_changes_nothing_but_the_status(
    tmp_path: Path,
) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {pid: "accepted" for pid in IDS})
    reviewed, summary = import_review(ds, ps, out)
    assert summary == ReviewSummary(4, 0, 0, 0, 0, 2)
    assert reviewed.parent_set_id == ps.set_id and reviewed.dataset_version == ps.dataset_version
    assert [(p.proposal_id, p.bbox_xyxy_abs, p.score) for p in reviewed.proposals] == [
        (p.proposal_id, p.bbox_xyxy_abs, p.score) for p in ps.proposals
    ]
    assert all(p.status == "accepted" and not p.edited for p in reviewed.proposals)
    assert reviewed.set_id != ps.set_id


def test_accepted_rejected_and_untouched_proposals_get_their_own_status(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted", IDS[1]: "rejected", IDS[2]: ""})
    reviewed, summary = import_review(ds, ps, out)
    status = {p.proposal_id: p.status for p in reviewed.proposals}
    assert status == {IDS[0]: "accepted", IDS[1]: "rejected", IDS[2]: "pending", IDS[3]: "pending"}
    assert (summary.accepted, summary.rejected, summary.pending) == (1, 1, 2)


def test_decisions_are_read_without_regard_to_case_and_spaces(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "  Accepted ", IDS[1]: "REJECTED", IDS[2]: "pending"})
    reviewed, _ = import_review(ds, ps, out)
    status = {p.proposal_id: p.status for p in reviewed.proposals}
    assert status[IDS[0]] == "accepted" and status[IDS[1]] == "rejected"
    assert status[IDS[2]] == "pending"


def test_a_moved_box_that_is_accepted_is_an_edited_proposal(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})
    move(out, IDS[0], -3.0)
    reviewed, summary = import_review(ds, ps, out)
    edited = next(p for p in reviewed.proposals if p.proposal_id == IDS[0])
    assert edited.edited and edited.status == "accepted"
    assert edited.bbox_xyxy_abs[:2] == (38.0, 28.0) and edited.bbox_xyxy_abs[2:] == (60.0, 46.0)
    assert (summary.accepted, summary.edited) == (1, 1)


def test_a_moved_box_that_is_rejected_keeps_its_original_box(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "rejected"})
    move(out, IDS[0], 2.0)
    reviewed, _ = import_review(ds, ps, out)
    one = next(p for p in reviewed.proposals if p.proposal_id == IDS[0])
    assert (
        one.status == "rejected"
        and not one.edited
        and one.bbox_xyxy_abs == (41.0, 31.0, 60.0, 46.0)
    )


def test_a_moved_box_without_a_decision_stays_pending_and_is_reported(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[1]: "accepted"})
    move(out, IDS[0], 2.0)
    reviewed, summary = import_review(ds, ps, out)
    one = next(p for p in reviewed.proposals if p.proposal_id == IDS[0])
    assert one.status == "pending" and one.bbox_xyxy_abs == (41.0, 31.0, 60.0, 46.0)
    assert summary.edited_pending == 1


def test_a_movement_within_the_tolerance_is_not_an_edit(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})
    move(out, IDS[0], review.EDIT_TOLERANCE_PX / 2)
    reviewed, summary = import_review(ds, ps, out)
    assert (
        summary.edited == 0
        and not next(p for p in reviewed.proposals if p.proposal_id == IDS[0]).edited
    )


def test_proposals_missing_from_the_file_keep_the_status_they_had(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    ps = make_set(ds)
    prior = ps.derive(
        [p.model_copy(update={"status": "accepted"}) if p.proposal_id == IDS[3] else p
         for p in ps.proposals]
    )  # fmt: skip
    out = tmp_path / "r"
    export_review(ds, prior, out)

    def drop(root: ET.Element) -> None:
        for pid, box in proposal_boxes(root).items():
            if pid == IDS[3]:
                for parent in root.iter():
                    if box in list(parent):
                        parent.remove(box)

    edit(out, drop)
    decide(out, {IDS[0]: "rejected"})
    reviewed, _ = import_review(ds, prior, out)
    status = {p.proposal_id: p.status for p in reviewed.proposals}
    assert status[IDS[3]] == "accepted" and status[IDS[0]] == "rejected"


def test_edits_to_the_human_layer_are_ignored_and_counted(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})

    def wreck(root: ET.Element) -> None:
        for box in root.iter("box"):
            if not any(a.get("name") == "proposal_id" for a in box.findall("attribute")):
                box.set("xtl", "1.0")
                box.set("label", "bird")

    edit(out, wreck)
    reviewed, summary = import_review(ds, ps, out)
    assert summary.ignored_boxes == 2 and summary.accepted == 1
    assert len(reviewed.proposals) == 4


def test_a_file_without_a_single_decision_is_refused(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    with pytest.raises(ValidationError, match="no decisions"):
        import_review(ds, ps, out)


def test_importing_changes_neither_the_set_nor_the_dataset(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})
    before = ps.model_dump_json()
    import_review(ds, ps, out)
    assert ps.model_dump_json() == before and all(p.status == "pending" for p in ps.proposals)


def test_the_import_logs_counts_only(
    tmp_path: Path, _captured_logs: list[dict[str, object]]
) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted", IDS[1]: "rejected"})
    import_review(ds, ps, out)
    events = {e["event"]: e for e in _captured_logs}
    assert events["review_exported"]["proposal_count"] == 4
    imported = events["review_imported"]
    assert (imported["accepted"], imported["rejected"], imported["pending"]) == (1, 1, 2)
    assert str(tmp_path) not in repr(_captured_logs)


# ----------------------------------------------------------------------------- hostile files
def refused(tmp_path: Path, match: str, tamper: Callable[[Path], None]) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})
    tamper(out)
    with pytest.raises(ValidationError, match=match):
        import_review(ds, ps, out)


def test_an_unknown_proposal_id_is_refused(tmp_path: Path) -> None:
    refused(
        tmp_path,
        "unknown proposal",
        lambda out: set_attribute(out, IDS[1], "proposal_id", "f" * 16),
    )


@pytest.mark.parametrize("bad", ["", "xyz", "../../etc", "A" * 16, "f" * 17, "<script>"])
def test_a_malformed_proposal_id_is_refused_without_being_echoed(tmp_path: Path, bad: str) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})
    set_attribute(out, IDS[1], "proposal_id", bad)
    with pytest.raises(ValidationError, match="proposal_id") as err:
        import_review(ds, ps, out)
    assert bad not in str(err.value) or bad == ""


def test_a_missing_proposal_id_on_the_layer_is_refused(tmp_path: Path) -> None:
    refused(tmp_path, "proposal_id", lambda out: set_attribute(out, IDS[1], "proposal_id", None))


def test_a_duplicated_decision_is_refused(tmp_path: Path) -> None:
    def duplicate(out: Path) -> None:
        def change(root: ET.Element) -> None:
            for pid, box in proposal_boxes(root).items():
                if pid == IDS[0]:
                    for parent in root.iter():
                        if box in list(parent):
                            parent.append(copy.deepcopy(box))
                            return

        edit(out, change)

    refused(tmp_path, "twice", duplicate)


@pytest.mark.parametrize("bad", ["approve", "yes", "accepted; rejected", "1", "<b>", "accepted\nx"])
def test_an_invalid_decision_is_refused(tmp_path: Path, bad: str) -> None:
    refused(tmp_path, "decision", lambda out: set_attribute(out, IDS[1], "decision", bad))


def test_an_enormous_decision_text_is_refused_by_the_reader(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    set_attribute(out, IDS[1], "decision", "x" * 5000)
    with pytest.raises(FormatError, match="too long"):
        import_review(ds, ps, out)


def test_a_proposal_that_was_moved_to_another_class_is_refused(tmp_path: Path) -> None:
    def relabel(out: Path) -> None:
        def change(root: ET.Element) -> None:
            box = proposal_boxes(root).get(IDS[0])
            if box is not None:
                box.set("label", "dog" + review.SUFFIX)

        edit(out, change)

    refused(tmp_path, "class", relabel)


def test_a_proposal_box_given_a_normal_label_is_refused(tmp_path: Path) -> None:
    def relabel(out: Path) -> None:
        def change(root: ET.Element) -> None:
            box = proposal_boxes(root).get(IDS[0])
            if box is not None:
                box.set("label", "cat")

        edit(out, change)

    refused(tmp_path, "layer", relabel)


def test_a_new_box_drawn_in_the_proposal_layer_is_refused(tmp_path: Path) -> None:
    def draw(out: Path) -> None:
        def change(root: ET.Element) -> None:
            for box in root.iter("box"):
                if box.get("label", "").endswith(review.SUFFIX):
                    clone = copy.deepcopy(box)
                    for attr in clone.findall("attribute"):
                        clone.remove(attr)
                    for parent in root.iter():
                        if box in list(parent):
                            parent.append(clone)
                            return

        edit(out, change)

    refused(tmp_path, "proposal_id", draw)


def test_a_decision_for_a_proposal_of_another_image_is_refused(tmp_path: Path) -> None:
    def swap(out: Path) -> None:
        def change(root: ET.Element) -> None:
            boxes = proposal_boxes(root)
            if IDS[0] in boxes:  # image a.png: give its box the id of a proposal of b.png
                for attr in boxes[IDS[0]].findall("attribute"):
                    if attr.get("name") == "proposal_id":
                        attr.text = IDS[3]

        edit(out, change)

    refused(tmp_path, "another image", swap)


def test_a_box_edited_outside_its_image_is_refused(tmp_path: Path) -> None:
    def push(out: Path) -> None:
        def change(root: ET.Element) -> None:
            box = proposal_boxes(root).get(IDS[0])
            if box is not None:
                box.set("xbr", "64.5")

        edit(out, change)

    refused(tmp_path, "outside", push)


def test_decisions_of_another_dataset_are_refused(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = build(other_dir, [Img("z.png", 70, 70, seed=77), Img("y.png", 60, 60, seed=78)])
    a, b = sorted(other.images, key=lambda i: i.image_id)
    foreign = ProposalSet.build(
        **{
            **{k: getattr(ps, k) for k in (
                "proposer", "model_id", "model_revision", "weights", "prompt_hash", "min_score",
                "class_min_score", "overlap_iou", "images_total", "images_done",
                "incomplete_reason", "rejected")},  # fmt: skip
            "dataset_version": other.manifest.version,
            "proposals": [
                Proposal(proposal_id="9" * 16, image_id=a.image_id, category_id=1,
                         bbox_xyxy_abs=(1.0, 1.0, 9.0, 9.0), score=0.5, prompt="cat"),
            ],
        }
    )  # fmt: skip
    with pytest.raises(ValidationError, match="unknown proposal"):
        import_review(other, foreign, out)
    with pytest.raises(ValidationError, match="another dataset version"):
        import_review(other, ps, out)


@pytest.mark.parametrize(
    "document",
    [
        '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><annotations/>',
        "<annotations><image",
        "not xml at all",
        "<annotations>" + "<x>" * 100 + "</annotations>",
    ],
)
def test_a_hostile_xml_document_is_refused_by_the_reader(tmp_path: Path, document: str) -> None:
    ds, ps, out = exported(tmp_path)
    for path in xml_files(out):
        path.write_text(document)
    with pytest.raises((FormatError, ValidationError)):
        import_review(ds, ps, out)


def test_a_directory_without_a_cvat_file_is_refused(tmp_path: Path) -> None:
    ds, ps, _ = exported(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FormatError):
        import_review(ds, ps, empty)


def test_a_non_finite_coordinate_is_refused(tmp_path: Path) -> None:
    ds, ps, out = exported(tmp_path)
    decide(out, {IDS[0]: "accepted"})

    def poison(root: ET.Element) -> None:
        box = proposal_boxes(root).get(IDS[0])
        if box is not None:
            box.set("xtl", "nan")

    edit(out, poison)
    with pytest.raises(FormatError):
        import_review(ds, ps, out)
