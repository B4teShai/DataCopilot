"""Proposal and ProposalSet: bounded, validated and content-addressed (03-data-model.md)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from datacopilot.core import proposals as pr
from datacopilot.core.proposals import Proposal, ProposalSet

IMG_A = "a" * 16
IMG_B = "b" * 16
VERSION = "c" * 32
SHA = "d" * 64


def proposal(pid: str = "1" * 16, **kw: Any) -> Proposal:
    base: dict[str, Any] = {
        "proposal_id": pid,
        "image_id": IMG_A,
        "category_id": 1,
        "bbox_xyxy_abs": (10.0, 10.0, 60.0, 40.0),
        "score": 0.8,
        "prompt": "cat",
    }
    return Proposal(**{**base, **kw})


def build(proposals: list[Proposal] | None = None, **kw: Any) -> ProposalSet:
    base: dict[str, Any] = {
        "dataset_version": VERSION,
        "proposer": "fake",
        "model_id": "fake_model",
        "model_revision": "r1",
        "weights": {"model.safetensors": SHA},
        "prompt_hash": SHA,
        "min_score": 0.35,
        "class_min_score": {},
        "overlap_iou": 0.5,
        "images_total": 2,
        "images_done": 2,
        "incomplete_reason": None,
        "rejected": {},
        "test_only": False,
        "proposals": proposals if proposals is not None else [proposal()],
    }
    return ProposalSet.build(**{**base, **kw})


# --- identity ------------------------------------------------------------------------------------
def test_the_set_id_is_the_digest_of_the_content_and_is_deterministic() -> None:
    a = build([proposal("1" * 16), proposal("2" * 16, image_id=IMG_B)])
    b = build([proposal("2" * 16, image_id=IMG_B), proposal("1" * 16)])
    assert a.set_id == b.set_id and len(a.set_id) == 32
    assert [p.proposal_id for p in a.proposals] == [p.proposal_id for p in b.proposals]


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_version": "e" * 32},
        {"model_revision": "r2"},
        {"min_score": 0.5},
        {"proposals": [proposal(score=0.81)]},
        {"parent_set_id": "f" * 32},
    ],
)
def test_any_field_change_changes_the_id(change: dict[str, Any]) -> None:
    assert build(**change).set_id != build().set_id


def test_a_tampered_file_is_refused_on_load() -> None:
    doc = json.loads(build().model_dump_json())
    doc["proposals"][0]["score"] = 0.99
    with pytest.raises(ValueError, match="digest"):
        ProposalSet.model_validate(doc)


def test_a_forged_set_id_is_refused() -> None:
    doc = json.loads(build().model_dump_json())
    doc["set_id"] = "0" * 32
    with pytest.raises(ValueError, match="digest"):
        ProposalSet.model_validate(doc)


def test_the_round_trip_through_json_is_the_identity() -> None:
    original = build([proposal("1" * 16), proposal("2" * 16, image_id=IMG_B, status="accepted")])
    assert ProposalSet.model_validate_json(original.model_dump_json()) == original


def test_extra_fields_are_refused() -> None:
    doc = json.loads(build().model_dump_json())
    doc["proposals"][0]["evil"] = 1
    with pytest.raises(ValueError):
        ProposalSet.model_validate(doc)
    doc = json.loads(build().model_dump_json())
    doc["evil"] = 1
    with pytest.raises(ValueError):
        ProposalSet.model_validate(doc)


# --- one proposal --------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "xyz", "A" * 16, "a" * 15, "a" * 17, "../" + "a" * 13])
def test_ids_are_lower_hex(bad: str) -> None:
    with pytest.raises(ValueError):
        proposal(bad)
    with pytest.raises(ValueError):
        proposal(image_id=bad)


@pytest.mark.parametrize("score", [0.0, -0.1, 1.01, math.nan, math.inf, -math.inf])
def test_the_score_is_finite_and_in_zero_to_one(score: float) -> None:
    with pytest.raises(ValueError):
        proposal(score=score)


@pytest.mark.parametrize(
    "box",
    [
        (math.nan, 0.0, 5.0, 5.0),
        (0.0, 0.0, math.inf, 5.0),
        (5.0, 0.0, 5.0, 5.0),
        (6.0, 0.0, 5.0, 5.0),
        (0.0, 5.0, 5.0, 5.0),
        (-1.0, 0.0, 5.0, 5.0),
        (0.0, 0.0, 5.0),
    ],
)
def test_a_box_is_finite_non_negative_and_has_positive_extent(box: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        proposal(bbox_xyxy_abs=box)


@pytest.mark.parametrize("kind", ["change_class", "delete_box", "refine_box", "flag_ambiguous", ""])
def test_only_add_box_exists_in_this_phase(kind: str) -> None:
    with pytest.raises(ValueError):
        proposal(kind=kind)


@pytest.mark.parametrize(
    "prompt", ["", "x" * (pr.MAX_PROMPT_CHARS + 1), "a\nb", "a\x00b", "a‮b", "a\x1b[31mb"]
)
def test_the_prompt_is_one_bounded_printable_line(prompt: str) -> None:
    with pytest.raises(ValueError):
        proposal(prompt=prompt)


def test_the_prompt_at_the_limit_is_accepted() -> None:
    assert proposal(prompt="x" * pr.MAX_PROMPT_CHARS).prompt == "x" * pr.MAX_PROMPT_CHARS


def test_an_edited_proposal_must_be_accepted() -> None:
    assert proposal(status="accepted", edited=True).edited
    for status in ("pending", "rejected"):
        with pytest.raises(ValueError):
            proposal(status=status, edited=True)


def test_unknown_status_is_refused() -> None:
    with pytest.raises(ValueError):
        proposal(status="approved")


# --- the set -------------------------------------------------------------------------------------
def test_duplicate_proposal_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build([proposal("1" * 16), proposal("1" * 16, image_id=IMG_B)])


def test_the_number_of_proposals_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "MAX_PROPOSALS", 2)
    build([proposal("1" * 16), proposal("2" * 16)])
    with pytest.raises(ValueError, match="at most"):
        build([proposal("1" * 16), proposal("2" * 16), proposal("3" * 16)])


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_version": "xyz"},
        {"dataset_version": "C" * 32},
        {"proposer": "Bad Name"},
        {"proposer": ""},
        {"model_id": "x" * 65},
        {"model_id": "a b"},
        {"model_revision": ""},
        {"model_revision": "a/b"},
        {"model_revision": "r" * 65},
        {"weights": {"model.safetensors": "zz"}},
        {"weights": {"../evil": SHA}},
        {"weights": {f"w{i}.safetensors": SHA for i in range(17)}},
        {"prompt_hash": "zz"},
        {"min_score": 0.0},
        {"min_score": 1.5},
        {"min_score": math.nan},
        {"overlap_iou": 0.0},
        {"overlap_iou": math.nan},
        {"class_min_score": {"cat": 2.0}},
        {"class_min_score": {"c" * 65: 0.5}},
        {"class_min_score": {f"c{i}": 0.5 for i in range(pr.MAX_CLASS_THRESHOLDS + 1)}},
        {"images_total": -1},
        {"images_done": 3},
        {"incomplete_reason": "because"},
        {"rejected": {"x" * 65: 1}},
        {"rejected": {"a": -1}},
        {"rejected": {f"r{i}": 1 for i in range(pr.MAX_REJECTION_REASONS + 1)}},
        {"parent_set_id": "xyz"},
    ],
)
def test_set_fields_are_validated(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build(**change)


def test_a_run_that_did_not_reach_every_image_must_say_why() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        build(images_done=1)
    done = build(images_done=1, incomplete_reason="time_budget")
    assert done.incomplete and done.incomplete_reason == "time_budget"
    assert not build().incomplete


def test_a_complete_run_cannot_claim_a_reason() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        build(incomplete_reason="time_budget")


def test_derive_binds_the_reviewed_set_to_its_parent() -> None:
    parent = build([proposal("1" * 16), proposal("2" * 16, image_id=IMG_B)])
    reviewed = parent.derive(
        [parent.proposals[0].model_copy(update={"status": "accepted"}), parent.proposals[1]]
    )
    assert reviewed.parent_set_id == parent.set_id and reviewed.set_id != parent.set_id
    assert reviewed.dataset_version == parent.dataset_version
    assert [p.status for p in reviewed.proposals] == ["accepted", "pending"]


def test_derive_refuses_a_proposal_that_is_not_in_the_parent() -> None:
    parent = build()
    with pytest.raises(ValueError, match="not in the set"):
        parent.derive([proposal("9" * 16)])
    with pytest.raises(ValueError, match="same"):
        parent.derive([parent.proposals[0].model_copy(update={"category_id": 2})])


def test_accepted_returns_only_accepted_proposals_in_order() -> None:
    ps = build(
        [
            proposal("1" * 16, status="accepted"),
            proposal("2" * 16, image_id=IMG_B, status="rejected"),
            proposal("3" * 16, image_id=IMG_B, status="pending", score=0.5),
        ]
    )
    assert [p.proposal_id for p in ps.accepted()] == ["1" * 16]


# --- binding to a dataset ------------------------------------------------------------------------
def _dataset(version: str | None = None) -> Any:
    from datacopilot.core.schema import Category, Dataset, DatasetManifest, Image

    image = Image(image_id=IMG_A, file_path="/x/a.png", width=100, height=50)
    manifest = DatasetManifest(
        version=version or VERSION,
        schema_version=1,
        name="d",
        task="detection",
        categories=(Category(id=1, name="cat"),),
        image_count=1,
        annotation_count=0,
        created_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
    )
    return Dataset(manifest, (image,), ())


def test_a_set_that_fits_its_dataset_validates() -> None:
    from datacopilot.core.proposals import validate_against

    validate_against(build(), _dataset())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dataset_version": "e" * 32}, "another dataset version"),
        ({"proposals": [proposal(image_id=IMG_B)]}, "unknown image"),
        ({"proposals": [proposal(category_id=9)]}, "unknown class"),
        ({"proposals": [proposal(bbox_xyxy_abs=(10.0, 10.0, 100.5, 40.0))]}, "outside"),
        ({"proposals": [proposal(bbox_xyxy_abs=(10.0, 10.0, 60.0, 50.5))]}, "outside"),
    ],
)
def test_a_set_that_does_not_fit_its_dataset_is_refused(
    change: dict[str, Any], message: str
) -> None:
    from datacopilot.core.errors import ValidationError
    from datacopilot.core.proposals import validate_against

    with pytest.raises(ValidationError, match=message):
        validate_against(build(**change), _dataset())


def test_the_binding_error_lists_a_few_ids_and_counts_the_rest() -> None:
    from datacopilot.core.errors import ValidationError
    from datacopilot.core.proposals import validate_against

    many = [proposal(f"{i:016x}", image_id=IMG_B) for i in range(9)]
    with pytest.raises(ValidationError, match="and 4 more"):
        validate_against(build(many), _dataset())


def test_derive_validates_the_reviewed_proposals_again() -> None:
    parent = build()
    broken = parent.proposals[0].model_copy(update={"bbox_xyxy_abs": (9.0, 9.0, 3.0, 3.0)})
    with pytest.raises(ValueError, match="x_min"):
        parent.derive([broken])  # model_copy does not validate; derive must
    broken = parent.proposals[0].model_copy(update={"status": "pending", "edited": True})
    with pytest.raises(ValueError, match="accepted"):
        parent.derive([broken])
