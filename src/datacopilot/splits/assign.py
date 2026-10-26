"""Deterministic greedy assignment of whole groups to train, val and test.

Groups are placed largest first (by images, then annotations); groups of equal size are ordered by
``sha256(seed:key)``, where ``key`` is a digest of the sorted member ids, so the result depends on
the members, the ratios and the seed only, never on the order of the rows. Each group goes to the
split whose cost grows least:

    cost = (images_s / N - ratio_s)^2  +  mean over classes c of (annotations_s,c / A_c - ratio_s)^2

(the class term only with ``stratify``), computed incrementally for the one split that changes.
Ties go to train, then val, then test; a split with ratio 0 is never used. Splitting whole groups
cannot always meet the ratios (one group may be larger than a target, a class may live in one
group), so the result reports the deviation per split and per class instead of claiming balance.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from datacopilot.core.schema import ClassShare, SplitShare

SPLITS = ("train", "val", "test")
RATIO_TOLERANCE = 1e-6
"""How far the three ratios may sum from 1."""

Ratios = tuple[float, float, float]


@dataclass(frozen=True)
class Group:
    """Images that must share a split. ``key`` is the 16-hex digest of the sorted members."""

    key: str
    members: tuple[str, ...]
    classes: Mapping[int, int]

    @classmethod
    def of(cls, members: Sequence[str], classes: Mapping[int, int]) -> Group:
        ordered = tuple(sorted(members))
        key = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:16]
        return cls(key, ordered, dict(classes))


@dataclass(frozen=True)
class Assignment:
    split_of: dict[str, str]
    splits: dict[str, SplitShare]
    classes: dict[str, dict[str, ClassShare]]
    max_class_deviation: float | None


def check_ratios(values: Sequence[float]) -> Ratios:
    """Three finite fractions in [0, 1] that sum to 1 (within ``RATIO_TOLERANCE``).

    Raises:
        ValueError: anything else.
    """
    if len(values) != 3:
        raise ValueError("ratios must be three numbers: train, val, test")
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
        raise ValueError("ratios must be numbers between 0 and 1")
    if abs(sum(values) - 1.0) > RATIO_TOLERANCE:
        raise ValueError(f"ratios must sum to 1 (they sum to {sum(values):.6g})")
    return (float(values[0]), float(values[1]), float(values[2]))


def parse_ratios(text: str) -> Ratios:
    """``"0.8,0.1,0.1"`` to a checked triple (see :func:`check_ratios`)."""
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        raise ValueError("ratios must be three numbers such as 0.8,0.1,0.1") from None
    return check_ratios(values)


def _order(group: Group, seed: int) -> tuple[int, int, bytes]:
    tie = hashlib.sha256(f"{seed}:{group.key}".encode()).digest()
    return (-len(group.members), -sum(group.classes.values()), tie)


def assign(
    groups: Sequence[Group], ratios: Ratios, *, seed: int, stratify: bool = False
) -> Assignment:
    """Place every group in one split (see the module docstring) and measure the result."""
    images = sum(len(g.members) for g in groups)
    totals: dict[int, int] = {}
    for g in groups:
        for cid, count in sorted(g.classes.items()):
            totals[cid] = totals.get(cid, 0) + count
    totals = {c: n for c, n in totals.items() if n > 0}
    weight = 1.0 / len(totals) if stratify and totals else 0.0
    open_splits = [n for n, r in enumerate(ratios) if r > 0]
    placed = [0, 0, 0]
    per_class: list[dict[int, int]] = [{}, {}, {}]
    split_of: dict[str, str] = {}
    for group in sorted(groups, key=lambda g: _order(g, seed)):
        size = len(group.members)
        best, best_cost = open_splits[0], math.inf
        for s in open_splits:
            target = ratios[s] * images
            cost = ((placed[s] + size - target) ** 2 - (placed[s] - target) ** 2) / images**2
            if weight:
                for cid, count in sorted(group.classes.items()):  # a fixed float order
                    total = totals.get(cid)
                    if not total:
                        continue
                    have, aim = per_class[s].get(cid, 0), ratios[s] * total
                    cost += weight * ((have + count - aim) ** 2 - (have - aim) ** 2) / total**2
            if cost < best_cost:
                best, best_cost = s, cost
        placed[best] += size
        for cid, count in group.classes.items():
            per_class[best][cid] = per_class[best].get(cid, 0) + count
        split_of[group.key] = SPLITS[best]
    splits = {
        name: SplitShare(
            images=placed[s],
            target_images=round(ratios[s] * images, 4),
            deviation=round(placed[s] / images - ratios[s], 4) if images else 0.0,
        )
        for s, name in enumerate(SPLITS)
    }
    classes: dict[str, dict[str, ClassShare]] = {}
    worst: float | None = None
    for s, name in enumerate(SPLITS):
        row: dict[str, ClassShare] = {}
        for cid in sorted(totals):
            have = per_class[s].get(cid, 0)
            share = have / totals[cid]
            row[str(cid)] = ClassShare(
                annotations=have, share=round(share, 4), deviation=round(share - ratios[s], 4)
            )
            worst = max(worst or 0.0, abs(share - ratios[s]))
        classes[name] = row
    return Assignment(split_of, splits, classes, None if worst is None else round(worst, 4))
