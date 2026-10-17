"""Proposals: boxes a model thinks are missing, kept as data until a person has reviewed them.

A :class:`ProposalSet` is the result of one ``propose`` run (or of one review of such a run). It is
**content-addressed**: ``set_id`` is the first 32 hex digits of the sha256 of the canonical JSON of
everything else, so a file whose content was changed no longer matches its own id (checked whenever
a set is built or read) and nothing can be edited in place; a review writes a *new* set that names
its parent. A set is bound to the dataset version it was made for and to the model identity (id,
revision and the sha256 of the weights) that made it.

Everything in a set that a model, a reviewer or a file can influence is bounded and validated here
(finite numbers, positive boxes, one printable line of prompt text, counted rejections), so a
hostile proposal file or a hostile model cannot reach the store, a report or an annotation tool
unchecked.
Whether a box lies inside its image is a property of the dataset, not of the set: ``proposals/``
checks it against the images before a set is stored or applied.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

from datacopilot.core.errors import ValidationError
from datacopilot.core.schema import Dataset
from datacopilot.core.version import canonical_json

MAX_PROPOSALS = 200_000
"""Most proposals in one set (a proposal is about 350 bytes of JSON, so a set stays under 70 MB)."""
MAX_PROMPT_CHARS = 64
"""Longest ``Proposal.prompt``: the class name or alias that produced the box."""
MAX_CLASS_THRESHOLDS = 256
"""Most per-class score thresholds a set records."""
MAX_REJECTION_REASONS = 16
"""Most distinct reasons under which dropped model candidates are counted."""
MAX_WEIGHT_FILES = 16
"""Most weight files whose sha256 a set records."""

_HEX16 = r"^[0-9a-f]{16}$"
_HEX32 = r"^[0-9a-f]{32}$"
_HEX64 = r"^[0-9a-f]{64}$"
_WEIGHT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

Status = Literal["pending", "accepted", "rejected"]
IncompleteReason = Literal["time_budget", "images_skipped", "proposal_cap"]
Box = tuple[float, float, float, float]


def _one_printable_line(value: str) -> str:
    for char in value:
        if unicodedata.category(char) in ("Cc", "Cf", "Cs", "Zl", "Zp") or not char.isprintable():
            raise ValueError("the text must be one printable line")
    return value


def _positive_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("a box must satisfy 0 <= x_min < x_max and 0 <= y_min < y_max")
    return box


def _weights(value: dict[str, str]) -> dict[str, str]:
    if len(value) > MAX_WEIGHT_FILES:
        raise ValueError(f"at most {MAX_WEIGHT_FILES} weight files")
    for name, digest in value.items():
        if not _WEIGHT_NAME.fullmatch(name) or not re.fullmatch(_HEX64, digest):
            raise ValueError("weights map plain file names to sha256 hex digests")
    return value


def _thresholds(value: dict[str, float]) -> dict[str, float]:
    if len(value) > MAX_CLASS_THRESHOLDS:
        raise ValueError(f"at most {MAX_CLASS_THRESHOLDS} class thresholds")
    for name, score in value.items():
        if not 0 < len(name) <= MAX_PROMPT_CHARS or not 0 < score <= 1:
            raise ValueError("class thresholds map a class name to a score in (0, 1]")
        _one_printable_line(name)
    return value


def _reasons(value: dict[str, int]) -> dict[str, int]:
    if len(value) > MAX_REJECTION_REASONS:
        raise ValueError(f"at most {MAX_REJECTION_REASONS} rejection reasons")
    for reason in value:
        if not _REASON.fullmatch(reason):
            raise ValueError("a rejection reason is a short constant name")
    return value


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Proposal(_Model):
    """One suggested box. Only ``add_box`` exists in this phase.

    Attributes:
        proposal_id: 16 hex digits, unique in its set.
        kind: ``add_box``; ``change_class``, ``delete_box`` and ``refine_box`` are not built.
        image_id: the image the box was found in.
        category_id: the class (a category of the dataset).
        bbox_xyxy_abs: absolute pixel box; after a review it is the reviewer's box.
        score: the model's confidence, above 0 and at most 1.
        prompt: the class name or alias the model was asked about.
        status: ``pending`` until a review says ``accepted`` or ``rejected``.
        edited: the reviewer moved the box before accepting it.
    """

    proposal_id: str = Field(pattern=_HEX16)
    kind: Literal["add_box"] = "add_box"
    image_id: str = Field(pattern=_HEX16)
    category_id: int
    bbox_xyxy_abs: Annotated[Box, AfterValidator(_positive_box)]
    score: float = Field(gt=0, le=1)
    prompt: Annotated[
        str, Field(min_length=1, max_length=MAX_PROMPT_CHARS), AfterValidator(_one_printable_line)
    ]
    status: Status = "pending"
    edited: bool = False

    @model_validator(mode="after")
    def _edited_means_accepted(self) -> Proposal:
        if self.edited and self.status != "accepted":
            raise ValueError("only an accepted proposal can have an edited box")
        return self


def _order(p: Proposal) -> tuple[object, ...]:
    return (p.image_id, -p.score, p.category_id, p.bbox_xyxy_abs, p.proposal_id)


class ProposalSet(_Model):
    """The proposals of one run, bound to a dataset version and a model, content-addressed.

    Attributes:
        set_id: digest of every other field (32 hex digits); recomputed on every construction.
        dataset_version: the ``DatasetManifest.version`` the run read; ``apply`` refuses another.
        proposer: registry name of the proposer plugin.
        model_id, model_revision, weights: the model identity (``weights``: file name to sha256).
        prompt_hash: sha256 of the sorted prompt phrases the model was asked about.
        min_score, class_min_score: the thresholds the run used (``class_min_score`` is keyed by
            class name and overrides ``min_score`` for that class).
        overlap_iou: an existing box of the same class with at least this IoU suppresses a proposal.
        images_total, images_done: images in the dataset and images the model really ran on.
        incomplete_reason: set exactly when ``images_done < images_total``; ``apply`` refuses an
            incomplete set unless asked.
        rejected: model candidates dropped by validation, counted by constant reason.
        parent_set_id: the set a review was derived from (None for a fresh run).
        proposals: in canonical order (image, descending score, class, box, id).
    """

    set_id: str = Field(pattern=_HEX32)
    dataset_version: str = Field(pattern=_HEX32)
    proposer: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    model_revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    weights: Annotated[dict[str, str], AfterValidator(_weights)]
    prompt_hash: str = Field(pattern=_HEX64)
    min_score: float = Field(gt=0, le=1)
    class_min_score: Annotated[dict[str, float], AfterValidator(_thresholds)]
    overlap_iou: float = Field(gt=0, le=1)
    images_total: int = Field(ge=0)
    images_done: int = Field(ge=0)
    incomplete_reason: IncompleteReason | None
    rejected: Annotated[dict[str, Annotated[int, Field(ge=0)]], AfterValidator(_reasons)]
    parent_set_id: str | None = Field(default=None, pattern=_HEX32)
    proposals: tuple[Proposal, ...]

    @property
    def incomplete(self) -> bool:
        return self.incomplete_reason is not None

    @model_validator(mode="after")
    def _consistent(self, info: ValidationInfo) -> ProposalSet:
        if len(self.proposals) > MAX_PROPOSALS:
            raise ValueError(f"a set holds at most {MAX_PROPOSALS} proposals")
        ids = [p.proposal_id for p in self.proposals]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate proposal_id")
        if self.images_done > self.images_total:
            raise ValueError("images_done exceeds images_total")
        if (self.images_done < self.images_total) != (self.incomplete_reason is not None):
            raise ValueError(
                "a run that did not reach every image is incomplete and says why; a complete run "
                "has no reason"
            )
        unsealed = info.context is not None and info.context.get(_UNSEALED) is True
        if not unsealed and self.set_id != _digest(
            self.model_dump(mode="json", exclude={"set_id"})
        ):
            raise ValueError("set_id is not the digest of the content: the set was changed")
        return self

    @classmethod
    def build(cls, *, proposals: Sequence[Proposal], **fields: object) -> ProposalSet:
        """A set with its ``set_id`` computed; ``proposals`` are put in canonical order."""
        body = {**fields, "proposals": tuple(sorted(proposals, key=_order)), "set_id": "0" * 32}
        draft = cls.model_validate(body, context={_UNSEALED: True})
        return draft.model_copy(
            update={"set_id": _digest(draft.model_dump(mode="json", exclude={"set_id"}))}
        )

    def derive(self, proposals: Sequence[Proposal]) -> ProposalSet:
        """The reviewed successor of this set: the same proposals with new statuses and boxes.

        Only ``status``, ``edited`` and the box may differ from the parent's proposal; the set
        names this one as ``parent_set_id``.

        Raises:
            ValueError: a proposal that is not in this set, a missing one, or a changed identity
                (image, class, score, prompt or kind: the same proposal must stay the same).
        """
        mine = {p.proposal_id: p for p in self.proposals}
        given = {p.proposal_id: p for p in proposals}
        for pid, new in given.items():
            old = mine.get(pid)
            if old is None:
                raise ValueError(f"proposal {pid} is not in the set")
            same = old.model_copy(
                update={
                    "status": new.status,
                    "edited": new.edited,
                    "bbox_xyxy_abs": new.bbox_xyxy_abs,
                }
            )
            if same != new:
                raise ValueError(f"proposal {pid} is not the same proposal as in the set")
        if len(given) != len(proposals) or set(given) != set(mine):
            raise ValueError(
                "a review covers every proposal of the set exactly once; one is missing"
            )
        fields = self.model_dump(exclude={"set_id", "proposals", "parent_set_id"})
        return ProposalSet.build(
            proposals=list(given.values()), parent_set_id=self.set_id, **fields
        )

    def accepted(self) -> tuple[Proposal, ...]:
        """The accepted proposals, in the canonical order."""
        return tuple(p for p in self.proposals if p.status == "accepted")


_UNSEALED = "unsealed"
"""Validation context key of :meth:`ProposalSet.build`, which validates before the id exists."""


def _digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:32]


_MAX_REPORTED = 5


def validate_against(ps: ProposalSet, ds: Dataset) -> None:
    """Check that ``ps`` belongs to ``ds``: the same version, and every proposal names an image
    and a class of the dataset and a box inside that image.

    Raises:
        ValidationError: listing up to five proposal ids (hex, safe to show), then a count.
    """
    if ps.dataset_version != ds.manifest.version:
        raise ValidationError(
            f"the proposal set was made for another dataset version ({ps.dataset_version}); this "
            f"one is {ds.manifest.version}: propose again on this version"
        )
    sizes = {i.image_id: (i.width, i.height) for i in ds.images}
    classes = {c.id for c in ds.categories}
    problems: list[str] = []
    for p in ps.proposals:
        size = sizes.get(p.image_id)
        if size is None:
            problems.append(f"proposal {p.proposal_id} names an unknown image")
        elif p.bbox_xyxy_abs[2] > size[0] or p.bbox_xyxy_abs[3] > size[1]:
            problems.append(f"proposal {p.proposal_id} has a box outside its image")
        if p.category_id not in classes:
            problems.append(f"proposal {p.proposal_id} names an unknown class")
    if problems:
        more = len(problems) - _MAX_REPORTED
        suffix = f"; and {more} more" if more > 0 else ""
        raise ValidationError(
            "the proposal set does not fit the dataset: "
            + "; ".join(problems[:_MAX_REPORTED])
            + suffix
        )
