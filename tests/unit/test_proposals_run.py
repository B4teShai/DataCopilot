"""propose(): what a model says is untrusted; what comes out is validated and bounded."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import structlog
from fixtures.auditdata import Img, build, dataset
from fixtures.synth import png_bytes

from datacopilot.core.errors import ModelError, ValidationError
from datacopilot.core.proposals import ProposalSet
from datacopilot.core.schema import Category, Dataset, Image
from datacopilot.embed.protocol import ImageDecodeError
from datacopilot.proposals import run as run_module
from datacopilot.proposals.run import propose
from datacopilot.proposers.protocol import Candidate

Script = Callable[[bytes, Sequence[str], tuple[int, int], float], list[Candidate]]


class FakeProposer:
    model_id = "fake_model"
    revision = "rev1"
    weights = {"model.safetensors": "d" * 64}

    def __init__(self, script: Script | None = None) -> None:
        self.script = script or (lambda *_: [])
        self.calls: list[dict[str, Any]] = []

    def detect(
        self, image: bytes, phrases: Sequence[str], size: tuple[int, int], *, min_score: float
    ) -> list[Candidate]:
        self.calls.append(
            {"image": image, "phrases": list(phrases), "size": size, "min": min_score}
        )
        return self.script(image, phrases, size, min_score)


def C(box: Sequence[float], label: str = "cat", score: float = 0.9) -> Candidate:
    return Candidate(tuple(box), label, score)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _captured_logs() -> Iterator[None]:
    with structlog.testing.capture_logs():
        yield


def two_images(tmp_path: Path) -> Dataset:
    return build(
        tmp_path,
        [
            Img("a.png", 64, 48, boxes=[(0, 8.0, 6.0, 40.0, 30.0)]),
            Img("b.png", 80, 60, split="val"),
        ],
    )


def data_of(ds: Dataset, name: str) -> bytes:
    return next(Path(i.file_path).read_bytes() for i in ds.images if i.file_path.endswith(name))


def run(ds: Dataset, tmp_path: Path, script: Script | None = None, **kw: Any) -> ProposalSet:
    return propose(ds, FakeProposer(script), proposer_name="fake", root=tmp_path, **kw)


def by_name(ds: Dataset, table: dict[str, list[Candidate]]) -> Script:
    mapping = {data_of(ds, name): found for name, found in table.items()}
    return lambda image, *_: list(mapping.get(image, []))


# ----------------------------------------------------------------------------- the result
def test_a_set_is_bound_to_the_dataset_and_the_model_and_sorted(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    script = by_name(
        ds,
        {
            "a.png": [C((40, 30, 60, 46), "dog", 0.6), C((1, 1, 20, 20), "cat", 0.8)],
            "b.png": [C((5, 5, 30, 30), "bird", 0.7)],
        },
    )
    ps = run(ds, tmp_path, script)
    assert ps.dataset_version == ds.manifest.version and ps.proposer == "fake"
    assert (ps.model_id, ps.model_revision) == ("fake_model", "rev1")
    assert ps.weights == FakeProposer.weights
    assert (ps.images_total, ps.images_done, ps.incomplete_reason) == (2, 2, None)
    assert ps.parent_set_id is None and not ps.incomplete
    assert {(p.prompt, p.category_id, p.status, p.kind) for p in ps.proposals} == {
        ("cat", 1, "pending", "add_box"),
        ("dog", 2, "pending", "add_box"),
        ("bird", 3, "pending", "add_box"),
    }
    keys = [(p.image_id, -p.score) for p in ps.proposals]
    assert keys == sorted(keys) and len(ps.proposals) == 3
    assert {p.bbox_xyxy_abs for p in ps.proposals} >= {(1.0, 1.0, 20.0, 20.0)}


def test_the_proposer_is_asked_about_names_and_aliases_with_the_dataset_image_size(
    tmp_path: Path,
) -> None:
    ds = two_images(tmp_path)
    cats = [
        Category(id=1, name="Cat", aliases=("Kitty", " house   cat ")),
        Category(id=2, name="dog"),
        Category(id=3, name="bird"),
    ]
    ds = dataset(ds.images, ds.annotations, cats)
    proposer = FakeProposer()
    ps = propose(ds, proposer, proposer_name="fake", root=tmp_path)
    phrases = ["bird", "cat", "dog", "house cat", "kitty"]
    assert all(c["phrases"] == phrases for c in proposer.calls)
    assert sorted(c["size"] for c in proposer.calls) == [(64, 48), (80, 60)]
    assert ps.prompt_hash == hashlib.sha256("\n".join(phrases).encode()).hexdigest()


def test_an_alias_maps_back_to_its_category(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    cats = [Category(id=1, name="cat", aliases=("kitty",)), Category(id=2, name="dog")]
    ds = dataset(ds.images, [], cats)
    ps = run(ds, tmp_path, by_name(ds, {"b.png": [C((5, 5, 30, 30), "kitty")]}))
    assert [(p.category_id, p.prompt) for p in ps.proposals] == [(1, "kitty")]


def test_classes_limit_the_prompt_and_unknown_classes_are_an_error(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    proposer = FakeProposer()
    propose(ds, proposer, proposer_name="fake", root=tmp_path, classes=["dog", "cat"])
    assert proposer.calls[0]["phrases"] == ["cat", "dog"]
    with pytest.raises(ValueError, match="unknown class"):
        run(ds, tmp_path, classes=["dog", "unicorn"])


def test_the_run_is_deterministic_whatever_order_the_model_answers_in(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    found = [C((1 + 5 * i, 1, 4 + 5 * i, 8), "cat", 0.5 + i / 100) for i in range(8)]

    def shuffled(seed: int) -> Script:
        def script(*_: Any) -> list[Candidate]:
            copy = list(found)
            random.Random(seed).shuffle(copy)
            return copy

        return script

    ids = {run(ds, tmp_path, shuffled(s)).set_id for s in range(4)}
    assert len(ids) == 1


# ----------------------------------------------------------------------------- thresholds
def test_min_score_and_per_class_thresholds_apply(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    script = by_name(
        ds,
        {
            "b.png": [
                C((1, 1, 9, 9), "cat", 0.4),
                C((20, 20, 30, 30), "dog", 0.4),
                C((40, 40, 50, 50), "bird", 0.34),
            ]
        },
    )
    ps = run(ds, tmp_path, script, min_score=0.35, class_min_score={"cat": 0.5, "bird": 0.3})
    assert sorted(p.prompt for p in ps.proposals) == ["bird", "dog"]
    assert (ps.min_score, ps.class_min_score) == (0.35, {"cat": 0.5, "bird": 0.3})


def test_the_proposer_is_asked_for_the_lowest_threshold_in_use(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    proposer = FakeProposer()
    propose(
        ds, proposer, proposer_name="fake", root=tmp_path, min_score=0.4,
        class_min_score={"dog": 0.25, "cat": 0.9},
    )  # fmt: skip
    assert {c["min"] for c in proposer.calls} == {0.25}


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_thresholds_are_validated(tmp_path: Path, bad: float) -> None:
    ds = two_images(tmp_path)
    with pytest.raises(ValueError, match="min_score"):
        run(ds, tmp_path, min_score=bad)
    with pytest.raises(ValueError, match="threshold"):
        run(ds, tmp_path, class_min_score={"cat": bad})
    with pytest.raises(ValueError, match="overlap"):
        run(ds, tmp_path, overlap_iou=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_the_time_budget_is_validated(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="time budget"):
        run(two_images(tmp_path), tmp_path, time_budget_seconds=bad)


def test_a_threshold_for_an_unknown_class_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown class"):
        run(two_images(tmp_path), tmp_path, class_min_score={"unicorn": 0.5})


def test_a_classification_dataset_cannot_be_proposed_for(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    classification = Dataset(
        ds.manifest.model_copy(update={"task": "classification"}), ds.images, (), None
    )
    with pytest.raises(ValidationError, match="detection"):
        run(classification, tmp_path)


# ----------------------------------------------------------------------------- hostile model output
@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (C((math.nan, 1, 9, 9)), "invalid_candidate"),
        (C((1, 1, math.inf, 9)), "invalid_candidate"),
        (C((9, 1, 1, 9)), "invalid_candidate"),
        (C((1, 1, 9, 9), score=math.nan), "invalid_candidate"),
        (C((1, 1, 9, 9), score=math.inf), "invalid_candidate"),
        (C((1, 1, 9, 9), score=1.5), "invalid_candidate"),
        (C((1, 1, 9, 9), score=-0.5), "invalid_candidate"),
        (C((1, 1, 9), "cat"), "invalid_candidate"),
        (C((1, 1, 9, 9), label="unicorn"), "unknown_label"),
        (C((1, 1, 9, 9), label=""), "unknown_label"),
        (C((1, 1, 9, 9), label="cat. dog"), "unknown_label"),
        (C((1, 1, 9, 9), label=7), "unknown_label"),  # type: ignore[arg-type]
        (C((100, 1, 140, 9)), "box_outside_image"),
        (C((-30, 1, 9, 9)), "box_outside_image"),
        (C((1, 1, 9, 500)), "box_outside_image"),
        (C((1, 1, 2, 9)), "box_too_small"),
        (C((1, 1, 9, 2.5)), "box_too_small"),
    ],
)
def test_an_invalid_candidate_is_dropped_and_counted_by_reason(
    tmp_path: Path, candidate: Candidate, reason: str
) -> None:
    ds = two_images(tmp_path)
    ps = run(ds, tmp_path, by_name(ds, {"b.png": [candidate, C((20, 20, 30, 30), "dog")]}))
    assert [p.prompt for p in ps.proposals] == ["dog"]
    assert ps.rejected == {reason: 1}


def test_a_box_that_overshoots_the_image_by_a_pixel_is_clipped_not_dropped(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    ps = run(ds, tmp_path, by_name(ds, {"b.png": [C((-0.5, 2, 80.7, 60.9))]}))
    (p,) = ps.proposals
    assert p.bbox_xyxy_abs == (0.0, 2.0, 80.0, 60.0) and ps.rejected == {}


def test_a_label_is_matched_after_normalisation(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    ps = run(ds, tmp_path, by_name(ds, {"b.png": [C((1, 1, 9, 9), "  CAT ")]}))
    assert [p.prompt for p in ps.proposals] == ["cat"]


def test_stored_boxes_are_rounded_to_two_decimals_and_inside_the_image(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    ps = run(ds, tmp_path, by_name(ds, {"b.png": [C((1.23456, 2.34567, 79.999, 59.999))]}))
    (p,) = ps.proposals
    assert p.bbox_xyxy_abs == (1.23, 2.35, 80.0, 60.0)


def test_one_image_cannot_flood_the_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "MAX_PROPOSALS_PER_IMAGE", 5)
    ds = two_images(tmp_path)
    many = [C((1 + 7 * i % 70, 1 + 5 * (i // 10), 5 + 7 * i % 70, 5 + 5 * (i // 10)),
              "dog", 0.5 + i / 1000) for i in range(30)]  # fmt: skip
    ps = run(ds, tmp_path, by_name(ds, {"b.png": many}))
    assert len(ps.proposals) == 5 and ps.rejected.get("image_cap", 0) > 0
    assert min(p.score for p in ps.proposals) >= 0.5 + 24 / 1000 - 1e-9  # the best ones stay


# ----------------------------------------------------------------------------- existing boxes
def test_a_box_that_already_exists_is_not_proposed_again(tmp_path: Path) -> None:
    ds = two_images(tmp_path)  # a.png has a cat at (8, 6, 40, 30)
    same = C((8, 6, 40, 30), "cat")
    shifted_a_bit = C((9, 7, 41, 31), "cat")  # IoU well above 0.5
    other_class = C((8, 6, 40, 30), "dog")  # same place, another class: not the same box
    far_away = C((41, 31, 60, 46), "cat")
    ps = run(ds, tmp_path, by_name(ds, {"a.png": [same, shifted_a_bit, other_class, far_away]}))
    assert sorted((p.prompt, p.bbox_xyxy_abs) for p in ps.proposals) == [
        ("cat", (41.0, 31.0, 60.0, 46.0)),
        ("dog", (8.0, 6.0, 40.0, 30.0)),
    ]
    assert ps.rejected == {"exists_already": 2}


def test_the_overlap_threshold_is_inclusive_and_recorded(tmp_path: Path) -> None:
    ds = two_images(tmp_path)  # existing cat (8, 6, 40, 30): area 32 x 24 = 768
    # a box of exactly half the area inside it: IoU 0.5
    half = C((8, 6, 40, 18), "cat")
    ps = run(ds, tmp_path, by_name(ds, {"a.png": [half]}), overlap_iou=0.5)
    assert ps.proposals == () and ps.rejected == {"exists_already": 1} and ps.overlap_iou == 0.5
    ps = run(ds, tmp_path, by_name(ds, {"a.png": [half]}), overlap_iou=0.51)
    assert len(ps.proposals) == 1


def test_duplicates_among_the_proposals_keep_the_best_per_class(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    ps = run(
        ds, tmp_path,
        by_name(ds, {"b.png": [C((10, 10, 30, 30), "cat", 0.6), C((11, 11, 31, 31), "cat", 0.9),
                               C((10, 10, 30, 30), "dog", 0.5)]}),
    )  # fmt: skip
    assert sorted((p.prompt, p.score) for p in ps.proposals) == [("cat", 0.9), ("dog", 0.5)]
    assert ps.rejected == {"duplicate_proposal": 1}


# ----------------------------------------------------------------------------- class names
@pytest.mark.parametrize(
    "name", ["a. b", "ünïcode", "日本語", "   ", "\u200b", "x" * 65, "<b>", "a;b", "cat.", "a\x00b"]
)
def test_a_class_name_that_could_rewrite_the_prompt_is_never_used(
    tmp_path: Path, name: str
) -> None:
    ds = two_images(tmp_path)
    cats = [Category(id=1, name="cat"), Category(id=2, name=name or "empty-name")]
    if not name.strip():
        cats[1] = Category(id=2, name="​")  # invisible: still not a prompt phrase
    ds = dataset(ds.images, [], cats)
    proposer = FakeProposer()
    ps = propose(ds, proposer, proposer_name="fake", root=tmp_path)
    assert proposer.calls[0]["phrases"] == ["cat"]
    assert ps.rejected["unusable_class_name"] == 1


def test_line_breaks_and_tabs_in_a_name_become_single_spaces_and_cannot_split_the_prompt(
    tmp_path: Path,
) -> None:
    ds = two_images(tmp_path)
    cats = [Category(id=1, name="cat"), Category(id=2, name="hot\n\tdog   stand")]
    ds = dataset(ds.images, [], cats)
    proposer = FakeProposer()
    propose(ds, proposer, proposer_name="fake", root=tmp_path)
    assert proposer.calls[0]["phrases"] == ["cat", "hot dog stand"]


def test_an_alias_that_is_unusable_is_skipped_but_its_class_stays(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    cats = [Category(id=1, name="cat", aliases=("a. b", "kitty")), Category(id=2, name="dog")]
    ds = dataset(ds.images, [], cats)
    proposer = FakeProposer()
    ps = propose(ds, proposer, proposer_name="fake", root=tmp_path)
    assert proposer.calls[0]["phrases"] == ["cat", "dog", "kitty"]
    assert ps.rejected["unusable_class_name"] == 1


def test_two_classes_with_the_same_phrase_do_not_both_get_it(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    cats = [Category(id=1, name="cat"), Category(id=2, name="dog", aliases=("CAT",))]
    ds = dataset(ds.images, [], cats)
    ps = run(ds, tmp_path, by_name(ds, {"b.png": [C((1, 1, 9, 9), "cat")]}))
    assert [p.category_id for p in ps.proposals] == [1] and ps.rejected["ambiguous_class_name"] == 1


def test_a_dataset_without_a_usable_class_name_is_an_error(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    ds = dataset(ds.images, [], [Category(id=1, name="日本語")])
    with pytest.raises(ValueError, match="usable class"):
        run(ds, tmp_path)


def test_too_many_phrases_ask_for_classes(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    cats = [Category(id=i + 1, name=f"c{i}") for i in range(run_module.MAX_PHRASES + 1)]
    ds = dataset(ds.images, [], cats)
    with pytest.raises(ValueError, match="--classes"):
        run(ds, tmp_path)
    ps = run(ds, tmp_path, classes=["c0", "c1"])
    assert ps.images_done == 2


# ----------------------------------------------------------------------------- unreadable images
def test_an_image_that_cannot_be_read_is_skipped_and_the_run_is_incomplete(tmp_path: Path) -> None:
    ds = build(
        tmp_path,
        [Img("a.png"), Img("b.png", write=False), Img("c.png", path="https://x.test/c.png")],
    )
    proposer = FakeProposer()
    ps = propose(ds, proposer, proposer_name="fake", root=tmp_path)
    assert len(proposer.calls) == 1
    assert (ps.images_total, ps.images_done, ps.incomplete_reason) == (3, 1, "images_skipped")


def test_an_image_outside_the_root_is_never_opened(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    ds = build(outside, [Img("a.png")])
    root = tmp_path / "root"
    root.mkdir()
    proposer = FakeProposer()
    ps = propose(ds, proposer, proposer_name="fake", root=root)
    assert proposer.calls == [] and ps.images_done == 0 and ps.incomplete_reason == "images_skipped"


def test_a_file_that_changed_since_ingest_is_not_given_to_the_model(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    Path(ds.images[0].file_path).write_bytes(png_bytes(64, 48, 99))
    proposer = FakeProposer()
    ps = propose(ds, proposer, proposer_name="fake", root=tmp_path)
    assert len(proposer.calls) == 1 and ps.images_done == 1


def test_an_image_the_model_cannot_decode_is_skipped(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    bad = data_of(ds, "a.png")

    def script(image: bytes, *_: Any) -> list[Candidate]:
        if image == bad:
            raise ImageDecodeError(0, "decode_failed")
        return [C((1, 1, 9, 9), "cat")]

    ps = run(ds, tmp_path, script)
    assert (ps.images_done, ps.incomplete_reason) == (1, "images_skipped")
    assert len(ps.proposals) == 1


def test_a_model_failure_ends_the_run_and_nothing_is_returned(tmp_path: Path) -> None:
    def boom(*_: Any) -> list[Candidate]:
        raise ModelError("the model failed")

    with pytest.raises(ModelError):
        run(two_images(tmp_path), tmp_path, boom)


# ----------------------------------------------------------------------------- budgets
def test_the_time_budget_ends_the_run_with_a_partial_set_that_says_so(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img(f"{n}.png", seed=n) for n in range(6)])
    ticks = itertools.count()
    ps = run(
        ds, tmp_path, by_name(ds, {"0.png": [C((1, 1, 9, 9), "cat")]}),
        time_budget_seconds=2.5, clock=lambda: float(next(ticks)),
    )  # fmt: skip
    assert ps.incomplete_reason == "time_budget" and ps.incomplete
    assert 0 < ps.images_done < ps.images_total == 6


def test_the_number_of_proposals_is_capped_and_the_run_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_module, "MAX_PROPOSALS", 3)
    ds = build(tmp_path, [Img(f"{n}.png", seed=n) for n in range(4)])
    every = lambda *_: [  # noqa: E731
        C((1, 1, 9, 9), "cat", 0.9),
        C((20, 20, 30, 30), "dog", 0.8),
    ]
    ps = run(ds, tmp_path, every)
    assert ps.incomplete_reason == "proposal_cap" and len(ps.proposals) <= 3
    assert ps.images_done < ps.images_total


# ----------------------------------------------------------------------------- logs
def test_the_run_logs_constant_events_without_names_or_paths(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    with structlog.testing.capture_logs() as logs:
        run(ds, tmp_path, by_name(ds, {"b.png": [C((1, 1, 9, 9), "unicorn")]}))
    events = {e["event"]: e for e in logs}
    assert events["proposer_started"]["image_count"] == 2
    assert events["proposer_started"]["model_id"] == "fake_model"
    rejected = events["proposal_candidates_rejected"]
    assert rejected["log_level"] == "warning" and rejected["reason"] == "unknown_label"
    assert rejected["count"] == 1
    assert "unicorn" not in repr(logs) and str(tmp_path) not in repr(logs)


def test_the_image_table_is_not_modified(tmp_path: Path) -> None:
    ds = two_images(tmp_path)
    before: tuple[Image, ...] = ds.images
    run(ds, tmp_path)
    assert ds.images == before
