"""Deterministic greedy assignment of groups to splits, with optional per-class balance."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from datacopilot.splits import assign


def groups(sizes: list[int], classes: list[dict[int, int]] | None = None) -> list[assign.Group]:
    out = []
    for n, size in enumerate(sizes):
        members = tuple(f"{n:04x}{k:012x}" for k in range(size))
        out.append(assign.Group.of(members, (classes or [{}] * len(sizes))[n]))
    return out


def split_sizes(found: assign.Assignment, gs: list[assign.Group]) -> Counter[str]:
    sizes: Counter[str] = Counter()
    for g in gs:
        sizes[found.split_of[g.key]] += len(g.members)
    return sizes


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.8,0.1,0.1", (0.8, 0.1, 0.1)), ("0.7, 0.3, 0", (0.7, 0.3, 0.0)), ("1,0,0", (1.0, 0, 0))],
)
def test_ratios_parse(text: str, expected: tuple[float, float, float]) -> None:
    assert assign.parse_ratios(text) == expected


@pytest.mark.parametrize(
    "text", ["0.8,0.1", "0.8,0.1,0.1,0", "0.5,0.5,0.5", "-0.1,0.6,0.5", "nan,0.5,0.5", "a,b,c", ""]
)
def test_bad_ratios_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match="ratios"):
        assign.parse_ratios(text)


def test_a_group_key_depends_only_on_its_members() -> None:
    a = assign.Group.of(("b" * 16, "a" * 16), {})
    b = assign.Group.of(("a" * 16, "b" * 16), {1: 2})
    assert a.key == b.key and a.members == ("a" * 16, "b" * 16)


def test_sizes_follow_the_ratios_and_every_group_is_whole() -> None:
    gs = groups([1] * 70 + [3] * 10)
    found = assign.assign(gs, (0.8, 0.1, 0.1), seed=0)
    sizes = split_sizes(found, gs)
    assert sum(sizes.values()) == 100
    assert abs(sizes["train"] - 80) <= 3 and abs(sizes["val"] - 10) <= 3
    assert set(found.split_of) == {g.key for g in gs}


def test_the_result_does_not_depend_on_input_order_and_repeats_exactly() -> None:
    gs = groups([1] * 40 + [2] * 20 + [5] * 4)
    first = assign.assign(gs, (0.7, 0.2, 0.1), seed=3)
    shuffled = list(gs)
    random.Random(1).shuffle(shuffled)
    assert assign.assign(shuffled, (0.7, 0.2, 0.1), seed=3) == first
    other = assign.assign(gs, (0.7, 0.2, 0.1), seed=4)
    assert other.split_of != first.split_of  # the seed breaks ties between equal groups


def test_a_zero_ratio_split_is_never_used() -> None:
    gs = groups([1] * 30)
    found = assign.assign(gs, (0.5, 0.5, 0.0), seed=0)
    assert "test" not in set(found.split_of.values())


def test_a_group_larger_than_a_target_still_lands_whole() -> None:
    gs = groups([50, 1, 1, 1])
    found = assign.assign(gs, (0.5, 0.25, 0.25), seed=0)
    assert found.split_of[gs[0].key] == "train"


def test_stratification_balances_classes_better_than_sizes_alone() -> None:
    rng = random.Random(0)
    classes = [{1: 1} if rng.random() < 0.85 else {2: 1} for _ in range(400)]
    gs = groups([1] * 400, classes)
    plain = assign.assign(gs, (0.8, 0.1, 0.1), seed=0)
    balanced = assign.assign(gs, (0.8, 0.1, 0.1), seed=0, stratify=True)
    worst = lambda a: max(abs(c.deviation) for s in a.classes.values() for c in s.values())  # noqa: E731
    assert worst(balanced) < worst(plain)
    assert worst(balanced) <= 0.02
    assert balanced.max_class_deviation == pytest.approx(worst(balanced))


def test_the_report_numbers_are_computed_from_the_assignment() -> None:
    gs = groups([1] * 10, [{1: 2}] * 5 + [{2: 1}] * 5)
    found = assign.assign(gs, (0.6, 0.2, 0.2), seed=0, stratify=True)
    total = sum(s.images for s in found.splits.values())
    assert total == 10
    for name, share in found.splits.items():
        assert share.target_images == pytest.approx(
            10 * {"train": 0.6, "val": 0.2, "test": 0.2}[name]
        )
        assert share.deviation == pytest.approx(
            share.images / 10 - share.target_images / 10, abs=1e-4
        )
    assert (
        sum(found.classes[s]["1"].annotations for s in found.classes if "1" in found.classes[s])
        == 10
    )
