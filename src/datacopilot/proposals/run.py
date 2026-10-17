"""``propose``: ask a proposer for boxes a dataset is probably missing, and keep only what survives.

What comes back from a model is untrusted. Every candidate is checked before it can become a
proposal: finite numbers, a score in (0, 1], a label that was one of the phrases asked about, a box
inside the image (a pixel of overshoot is clipped, more is dropped) and at least
:data:`MIN_BOX_SIDE_PX` wide and high. Dropped candidates are counted by constant reason in the set,
never silently. A candidate is not proposed when the image already has a box of the same class that
overlaps it by at least ``overlap_iou`` (it exists already), nor when a better candidate of the same
class overlaps it that much (non-maximum suppression); a box of another class in the same place is
a different statement and is kept.

Bounds: the images are read through the audit's contained, size-capped, hash-verified reader (an
image that cannot be read is skipped); at most :data:`MAX_PROPOSALS_PER_IMAGE` candidates per image
and :data:`MAX_PROPOSALS` per set; a time budget ends the run. A run that did not reach every image
is marked incomplete in its set and ``apply`` refuses it unless told otherwise. A model failure ends
the run with an error and nothing is stored.

Events: ``proposer_started``, ``proposal_candidates_rejected`` (WARN, per reason),
``propose_progress``.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from datacopilot.audit.context import CheckContext
from datacopilot.audit.embedding import read_verified
from datacopilot.core.errors import ValidationError
from datacopilot.core.geometry import clip_to_image, iou_xyxy
from datacopilot.core.proposals import MAX_PROPOSALS, IncompleteReason, Proposal, ProposalSet
from datacopilot.core.schema import Dataset
from datacopilot.embed.protocol import ImageDecodeError
from datacopilot.logging import get_logger
from datacopilot.proposers.protocol import MAX_PHRASES, Candidate, Proposer, normalize_phrase

__all__ = ["MAX_PHRASES", "MAX_PROPOSALS", "OptionError", "propose"]

DEFAULT_MIN_SCORE = 0.35
"""Confidence below which a candidate is dropped unless a class has its own threshold."""
DEFAULT_OVERLAP_IOU = 0.5
"""IoU from which a box of the same class counts as the same box (an existing one, or a better
candidate); inclusive."""
DEFAULT_TIME_BUDGET_SECONDS = 3600.0
MAX_PROPOSALS_PER_IMAGE = 100
"""Most candidates of one image that are considered (best score first); the rest is counted."""
MIN_BOX_SIDE_PX = 2.0
"""A candidate narrower or lower than this is noise (the audit calls such a label ``tiny_box``)."""
CLIP_TOLERANCE_PX = 1.0
"""A box that overshoots the image by at most this many pixels is clipped; more is dropped."""
PROGRESS_EVERY = 25
MAX_CLASS_NAME_CHARS = 64
"""Longest class name a stored set can hold as a threshold key."""
MAX_LABEL_CHARS = 256
"""Longest model label that is interpreted at all; a longer one is counted as unknown."""
SCORE_DECIMALS = 4
"""Scores are stored rounded to this many decimals, so a model rerun gives the same set id."""
_WARNED = ("invalid_candidate", "unknown_label", "ambiguous_label", "box_outside_image",
           "box_too_small", "unusable_class_name", "ambiguous_class_name")  # fmt: skip
_PUNCT = re.compile(r"\s*([-'/&])\s*")
_Box = tuple[float, float, float, float]


class OptionError(ValueError):
    """A usage mistake found before the model runs (the command line exits 2 for it)."""


def _printable_line(text: str) -> bool:
    return all(
        unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Zl", "Zp") and c.isprintable()
        for c in text
    )


def _squash(phrase: str) -> str:
    """``t - shirt`` and ``t-shirt`` are the same phrase: no spaces around ``- ' / &``."""
    return _PUNCT.sub(r"\1", phrase)


def _phrases(ds: Dataset, classes: Sequence[str] | None, rejected: Counter[str]) -> dict[str, int]:
    """``phrase -> category id`` for the selected classes, from their names and aliases.

    A phrase that two classes share is dropped for both (nobody is guessed) and counted.
    """
    categories = sorted(ds.categories, key=lambda c: c.id)
    if classes is not None:
        wanted = set(classes)
        unknown = sorted(wanted - {c.name for c in categories})
        if unknown:
            raise OptionError(f"unknown class {unknown[0][:40]!r}; see the dataset's categories")
        categories = [c for c in categories if c.name in wanted]
    owners: dict[str, set[int]] = {}
    for category in categories:
        for text in (category.name, *category.aliases):
            phrase = normalize_phrase(text)
            if phrase is None:
                rejected["unusable_class_name"] += 1
            else:
                owners.setdefault(phrase, set()).add(category.id)
    found: dict[str, int] = {}
    for phrase, ids in owners.items():
        if len(ids) > 1:
            rejected["ambiguous_class_name"] += len(ids)
        else:
            found[phrase] = next(iter(ids))
    if not found:
        raise OptionError("the dataset has no usable class names to ask a model about")
    if len(found) > MAX_PHRASES:
        raise OptionError(
            f"{len(found)} class names are too many to ask about at once (at most {MAX_PHRASES}); "
            "select some with --classes"
        )
    return found


class _Labels:
    """Turns what a model says into the class that was asked about, or says why it cannot.

    The decoder returns the words whose score passed its text threshold: usually one asked phrase,
    but also two classes in one label (``bicycle motorcycle``), a piece of a phrase
    (``bench dining``, ``##s snowboard``) and spacing around punctuation (``t - shirt``). A label is
    matched when it is an asked phrase or is made up entirely of asked phrases of ONE class
    (longest first); two classes in one label is ``ambiguous_label``, anything else (a fragment, an
    empty label) is ``unknown_label``. Nothing is guessed.
    """

    def __init__(self, phrase_class: Mapping[str, int]) -> None:
        self.by_phrase: dict[str, tuple[int, str]] = {}
        for phrase, category in phrase_class.items():
            self.by_phrase.setdefault(_squash(phrase), (category, phrase))
        self.by_words = {tuple(k.split(" ")): v for k, v in self.by_phrase.items()}
        self.longest = max(len(w) for w in self.by_words)

    def resolve(self, label: str) -> tuple[int, str] | str:
        text = _squash(" ".join(unicodedata.normalize("NFKC", label).lower().split()))
        exact = self.by_phrase.get(text)
        if exact is not None:
            return exact
        words = text.split(" ")
        covered = [False] * len(words)
        hits: list[tuple[int, str]] = []
        for size in range(min(self.longest, len(words)), 0, -1):
            for i in range(len(words) - size + 1):
                hit = self.by_words.get(tuple(words[i : i + size]))
                if hit is not None and not any(covered[i : i + size]):
                    covered[i : i + size] = [True] * size
                    hits.append(hit)
        if not hits or not all(covered):
            return "unknown_label"
        return hits[0] if len({h[0] for h in hits}) == 1 else "ambiguous_label"


def _check_options(
    min_score: float, class_min_score: Mapping[str, float], overlap_iou: float, budget: float
) -> None:
    if not (math.isfinite(min_score) and 0 < min_score <= 1):
        raise OptionError("min_score must be above 0 and at most 1")
    if not all(math.isfinite(v) and 0 < v <= 1 for v in class_min_score.values()):
        raise OptionError("a class threshold must be above 0 and at most 1")
    if not (math.isfinite(overlap_iou) and 0 < overlap_iou <= 1):
        raise OptionError("overlap_iou must be above 0 and at most 1")
    if not (math.isfinite(budget) and budget > 0):
        raise OptionError("the time budget must be a positive number of seconds")


def _valid(candidate: Candidate, width: int, height: int) -> tuple[_Box, float, str] | str:
    """``(box, score, label)`` of a trustworthy candidate, else the reason it is not."""
    try:
        box = tuple(float(v) for v in candidate.box)
        score = float(candidate.score)
    except (TypeError, ValueError):
        return "invalid_candidate"
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or not math.isfinite(score):
        return "invalid_candidate"
    if not (0 < score <= 1 and box[0] < box[2] and box[1] < box[3]):
        return "invalid_candidate"
    label = candidate.label
    if not isinstance(label, str) or len(label) > MAX_LABEL_CHARS:
        return "unknown_label"
    try:
        x1, y1, x2, y2 = clip_to_image(box, width, height, tol=CLIP_TOLERANCE_PX)
    except ValueError:
        return "box_outside_image"
    if x2 - x1 < MIN_BOX_SIDE_PX or y2 - y1 < MIN_BOX_SIDE_PX:
        return "box_too_small"
    return (x1, y1, x2, y2), score, label


def _proposal_id(
    version: str, model_id: str, revision: str, image_id: str, cat: int, box: _Box
) -> str:
    text = "\x1f".join([version, model_id, revision, image_id, str(cat), *map(repr, box)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def propose(
    ds: Dataset,
    proposer: Proposer,
    *,
    proposer_name: str,
    root: Path,
    classes: Sequence[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    class_min_score: Mapping[str, float] | None = None,
    overlap_iou: float = DEFAULT_OVERLAP_IOU,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    clock: Callable[[], float] | None = None,
) -> ProposalSet:
    """Run ``proposer`` over every image of ``ds`` and return the validated, ordered set.

    Args:
        ds: a detection dataset; the images must lie inside ``root``.
        proposer: the model; ``proposer_name`` is its registry name, recorded in the set.
        classes: names of the classes to look for (default: every class).
        min_score, class_min_score: confidence thresholds; ``class_min_score`` is keyed by class
            name and overrides ``min_score`` for that class.
        overlap_iou: IoU at which a box of the same class counts as the same box.
        time_budget_seconds: wall time after which the run stops with an incomplete set.
        clock: monotonic seconds (default ``time.monotonic``; a test seam).

    Raises:
        OptionError (a ``ValueError``): an invalid option, an unknown class, a threshold key the
            set cannot hold, or no usable class name; all found before the model runs.
        ValidationError: the dataset is not a detection dataset.
        ModelError: the model failed (nothing is returned).
    """
    thresholds = dict(class_min_score or {})
    _check_options(min_score, thresholds, overlap_iou, time_budget_seconds)
    if ds.manifest.task != "detection":
        raise ValidationError("propose needs a detection dataset (it suggests boxes)")
    log = get_logger(__name__)
    rejected: Counter[str] = Counter()
    phrase_class = _phrases(ds, classes, rejected)
    names = {c.id: c.name for c in ds.categories}
    selected = {names[cid] for cid in phrase_class.values()}
    if not set(thresholds) <= selected:
        raise OptionError("a threshold names an unknown class (or one that was not selected)")
    for name in thresholds:  # the set must be able to hold the key: found out before the model runs
        if not 0 < len(name) <= MAX_CLASS_NAME_CHARS or not _printable_line(name):
            raise OptionError(
                f"a class name used as a threshold key must be one printable line of at most "
                f"{MAX_CLASS_NAME_CHARS} characters"
            )
    cutoff = {cid: thresholds.get(names[cid], min_score) for cid in set(phrase_class.values())}
    phrases = sorted(phrase_class)
    labels = _Labels(phrase_class)
    prompt_hash = hashlib.sha256("\n".join(phrases).encode("utf-8")).hexdigest()
    floor = min([min_score, *thresholds.values()])
    existing: dict[tuple[str, int], list[_Box]] = {}
    for ann in ds.annotations:
        if ann.bbox_xyxy_abs is not None:
            existing.setdefault((ann.image_id, ann.category_id), []).append(ann.bbox_xyxy_abs)
    ctx = CheckContext(
        run_id=str(uuid.uuid4()), root=root.resolve(), cache_dir=Path(tempfile.gettempdir())
    )
    images = sorted(ds.images, key=lambda i: i.image_id)
    log.info(
        "proposer_started",
        proposer=proposer_name,
        model_id=proposer.model_id,
        model_revision=proposer.revision,
        image_count=len(images),
    )
    tick = clock or time.monotonic
    started = tick()
    proposals: list[Proposal] = []
    done = 0
    stopped: IncompleteReason | None = None
    for position, image in enumerate(images):
        if tick() - started > time_budget_seconds:
            stopped = "time_budget"
            break
        data, _digest, reason = read_verified(image, ctx)
        if data is None:
            log.warning("proposer_image_skipped", image_id=image.image_id, reason=reason)
            continue
        try:
            found = proposer.detect(data, phrases, (image.width, image.height), min_score=floor)
        except ImageDecodeError as exc:
            log.warning("proposer_image_skipped", image_id=image.image_id, reason=exc.reason)
            continue
        fresh = _image_proposals(
            ds, proposer, image.image_id, (image.width, image.height),
            found, labels, cutoff, existing, overlap_iou, rejected,
        )  # fmt: skip
        if len(proposals) + len(fresh) > MAX_PROPOSALS:
            stopped = "proposal_cap"
            break
        proposals.extend(fresh)
        done += 1
        if (position + 1) % PROGRESS_EVERY == 0:
            log.info("propose_progress", images_done=done, images_total=len(images))
    if stopped is None and done < len(images):
        stopped = "images_skipped"
    for reason in _WARNED:
        if rejected[reason]:
            log.warning("proposal_candidates_rejected", reason=reason, count=rejected[reason])
    return ProposalSet.build(
        proposals=proposals,
        dataset_version=ds.manifest.version,
        proposer=proposer_name,
        model_id=proposer.model_id,
        model_revision=proposer.revision,
        weights=dict(proposer.weights),
        prompt_hash=prompt_hash,
        min_score=min_score,
        class_min_score=thresholds,
        overlap_iou=overlap_iou,
        images_total=len(images),
        images_done=done,
        incomplete_reason=stopped,
        rejected={k: v for k, v in sorted(rejected.items()) if v},
        test_only=bool(getattr(proposer, "test_only", False)),
    )


def _image_proposals(
    ds: Dataset,
    proposer: Proposer,
    image_id: str,
    size: tuple[int, int],
    found: Sequence[Candidate],
    labels: _Labels,
    cutoff: Mapping[int, float],
    existing: Mapping[tuple[str, int], list[_Box]],
    overlap_iou: float,
    rejected: Counter[str],
) -> list[Proposal]:
    """The proposals of one image: validated, thresholded, suppressed, capped, in order."""
    valid: list[tuple[float, int, _Box, str]] = []
    for candidate in found:
        checked = _valid(candidate, *size)
        if isinstance(checked, str):
            rejected[checked] += 1
            continue
        box, score, label = checked
        match = labels.resolve(label)
        if isinstance(match, str):
            rejected[match] += 1
        elif score >= cutoff[match[0]]:
            valid.append((-score, match[0], box, match[1]))
    valid.sort()
    if len(valid) > MAX_PROPOSALS_PER_IMAGE:
        rejected["image_cap"] += len(valid) - MAX_PROPOSALS_PER_IMAGE
        del valid[MAX_PROPOSALS_PER_IMAGE:]
    kept: dict[int, list[_Box]] = {}
    out: list[Proposal] = []
    for negative, category, box, phrase in valid:
        candidate_box = np.asarray([box], dtype=np.float64)
        theirs = existing.get((image_id, category))
        if theirs and _overlaps(candidate_box, theirs, overlap_iou):
            rejected["exists_already"] += 1
        elif category in kept and _overlaps(candidate_box, kept[category], overlap_iou):
            rejected["duplicate_proposal"] += 1
        else:
            kept.setdefault(category, []).append(box)
            rounded = tuple(round(v, 2) for v in box)
            out.append(
                Proposal(
                    proposal_id=_proposal_id(
                        ds.manifest.version,
                        proposer.model_id,
                        proposer.revision,
                        image_id,
                        category,
                        rounded,  # type: ignore[arg-type]
                    ),  # fmt: skip
                    image_id=image_id,
                    category_id=category,
                    bbox_xyxy_abs=rounded,  # type: ignore[arg-type]
                    score=max(round(-negative, SCORE_DECIMALS), 10**-SCORE_DECIMALS),
                    prompt=phrase,
                )
            )
    return out


def _overlaps(box: NDArray[np.float64], others: Sequence[_Box], threshold: float) -> bool:
    iou = iou_xyxy(box, np.asarray(others, dtype=np.float64))
    return bool((iou >= threshold).any())
