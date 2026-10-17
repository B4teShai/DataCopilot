"""Review through CVAT: proposals leave as a separate layer, decisions come back as a new set.

``export_review`` writes the dataset's own boxes (labels unchanged, for context) and the proposals
as boxes whose label is the class name plus :data:`SUFFIX`, through the CVAT writer, so the
reviewer sees them as a layer of their own. Each proposal box carries three attributes:
``proposal_id``, ``score`` (text) and ``decision``, which the reviewer fills in with ``accepted`` or
``rejected`` (empty means undecided). A reviewer who moves the box before accepting it makes an
*edited* proposal. Nothing is ever decided by deleting a box: a box that is missing from the file
keeps the status it had, so a truncated file cannot reject anything.

``import_review`` reads the decisions back through the CVAT reader (bounded, no entities) and
returns a **new** set that names the reviewed one as its parent; the human layer of the file is
ignored (counted), so reviewing never edits a label. The file is untrusted, and each of these is an
error, not something to skip: a proposal id that is malformed, unknown or used twice, a proposal box
that lost its layer label or changed class, a proposal drawn on another image, an invalid decision,
a new box without an id in the layer, an accepted box edited outside its image, and a file that
decides nothing. Nothing is half applied.

Events: ``review_exported``, ``review_imported`` (counts only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from datacopilot import convert
from datacopilot.core.errors import ValidationError
from datacopilot.core.proposals import Proposal, ProposalSet, validate_against
from datacopilot.core.schema import (
    SCHEMA_VERSION,
    Annotation,
    Category,
    Dataset,
    DatasetManifest,
    validate_tables,
)
from datacopilot.core.version import models_version
from datacopilot.io import registry
from datacopilot.io.report import ConversionReport
from datacopilot.logging import get_logger

SUFFIX = "__proposed"
"""Appended to a class name to name its proposal layer in the annotation tool."""
EDIT_TOLERANCE_PX = 0.01
"""A box that moved by at most this many pixels in every coordinate was not edited (annotation
tools round coordinates)."""
_ID = re.compile(r"^[0-9a-f]{16}$")
_DECISIONS = {"": None, "pending": None, "accepted": "accepted", "rejected": "rejected"}


@dataclass(frozen=True)
class ReviewSummary:
    """What a review import found. The first four count the statuses of the new set; the others
    describe the file: proposals whose box was moved but that were not decided, and boxes of the
    human layer that were left alone."""

    accepted: int
    rejected: int
    pending: int
    edited: int
    edited_pending: int
    ignored_boxes: int


def export_review(ds: Dataset, ps: ProposalSet, out: Path) -> ConversionReport:
    """Write ``ds`` and the proposals of ``ps`` as a CVAT dataset in ``out`` (absent or empty).

    Raises:
        ValidationError: the set does not fit the dataset, has no proposals, or a class already
            ends in :data:`SUFFIX`.
        FormatError: the output exists and is not empty.
    """
    validate_against(ps, ds)
    if not ps.proposals:
        raise ValidationError("the proposal set has no proposals: nothing to review")
    if any(c.name.endswith(SUFFIX) for c in ds.categories):
        raise ValidationError(f"a class name ends in the reserved layer suffix {SUFFIX!r}")
    names = {c.id: c.name for c in ds.categories}
    next_id = max(names, default=-1) + 1
    layer_of: dict[int, int] = {}
    categories = list(ds.categories)
    for category_id in sorted({p.category_id for p in ps.proposals}):
        layer_of[category_id] = next_id
        categories.append(Category(id=next_id, name=names[category_id] + SUFFIX))
        next_id += 1
    layer = [
        Annotation(
            ann_id=f"proposal:{p.proposal_id}",
            image_id=p.image_id,
            category_id=layer_of[p.category_id],
            kind="bbox",
            bbox_xyxy_abs=p.bbox_xyxy_abs,
            provenance="model",
            attributes={
                "proposal_id": p.proposal_id,
                "score": f"{p.score:.3f}",
                "decision": "" if p.status == "pending" else p.status,
            },
        )
        for p in ps.proposals
    ]
    annotations = [*ds.annotations, *layer]
    validate_tables(ds.images, annotations, categories, task=ds.manifest.task)
    manifest = DatasetManifest(
        version=models_version(ds.images, annotations, categories, schema_version=SCHEMA_VERSION),
        schema_version=SCHEMA_VERSION,
        name=ds.manifest.name,
        task=ds.manifest.task,
        categories=tuple(sorted(categories, key=lambda c: c.id)),
        image_count=len(ds.images),
        annotation_count=len(annotations),
        created_at=datetime.now(timezone.utc),
    )
    report = convert.export(
        Dataset(manifest, ds.images, tuple(annotations), ds.root), "cvat_xml", out
    )
    get_logger(__name__).info(
        "review_exported", target="cvat", set_id=ps.set_id, proposal_count=len(ps.proposals)
    )
    return report


def import_review(
    ds: Dataset, ps: ProposalSet, directory: Path
) -> tuple[ProposalSet, ReviewSummary]:
    """Read the decisions in ``directory`` (a CVAT export of :func:`export_review`) into a new set.

    Raises:
        ValidationError: any of the refusals listed in the module docstring, or a set that does
            not fit the dataset.
        FormatError: the CVAT file is unreadable, malformed or hostile (from the reader).
    """
    validate_against(ps, ds)
    result = registry.get_reader("cvat_xml").read(Path(directory))
    label = {c.id: c.name for c in result.categories}
    own = {c.id: c.name for c in ds.categories}
    sizes = {i.image_id: (i.width, i.height) for i in ds.images}
    by_id = {p.proposal_id: p for p in ps.proposals}
    seen: set[str] = set()
    decided: dict[str, Proposal] = {}
    edited_pending = ignored = 0
    for ann in result.annotations:
        name = label[ann.category_id]
        if not name.endswith(SUFFIX):
            if "proposal_id" in ann.attributes:
                raise ValidationError(
                    "a box with a proposal_id must keep its label in the proposal layer"
                )
            ignored += 1
            continue
        pid = str(ann.attributes.get("proposal_id", "")).strip()
        if not _ID.fullmatch(pid):
            raise ValidationError("a box of the proposal layer has no valid proposal_id")
        if pid not in by_id:
            raise ValidationError(f"the decisions file names an unknown proposal {pid}")
        if pid in seen:
            raise ValidationError(f"proposal {pid} is decided twice in the decisions file")
        seen.add(pid)
        proposal = by_id[pid]
        if ann.image_id != proposal.image_id:
            raise ValidationError(f"proposal {pid} is drawn on another image than its own")
        if name[: -len(SUFFIX)] != own[proposal.category_id]:
            raise ValidationError(f"proposal {pid} changed its class, which is not supported")
        text = str(ann.attributes.get("decision", "")).strip().lower()
        if text not in _DECISIONS:
            raise ValidationError(
                f"proposal {pid} has an invalid decision: use accepted or rejected"
            )
        decision = _DECISIONS[text]
        box = ann.bbox_xyxy_abs
        if box is None:  # the reader never produces this for a box; refuse rather than assume
            raise ValidationError(f"proposal {pid} has no box")
        old = proposal.bbox_xyxy_abs
        moved = any(abs(a - b) > EDIT_TOLERANCE_PX for a, b in zip(box, old, strict=True))
        if decision == "accepted":
            width, height = sizes[proposal.image_id]
            if moved and (box[2] > width or box[3] > height):
                raise ValidationError(f"proposal {pid} was edited to a box outside its image")
            decided[pid] = proposal.model_copy(
                update={
                    "status": "accepted",
                    "edited": proposal.edited or moved,
                    "bbox_xyxy_abs": box if moved else proposal.bbox_xyxy_abs,
                }
            )
        elif decision == "rejected":
            decided[pid] = proposal.model_copy(update={"status": "rejected", "edited": False})
        elif moved:
            edited_pending += 1
    if not decided:
        raise ValidationError("the decisions file holds no decisions (accepted or rejected)")
    reviewed = ps.derive([decided.get(p.proposal_id, p) for p in ps.proposals])
    statuses = [p.status for p in reviewed.proposals]
    summary = ReviewSummary(
        accepted=statuses.count("accepted"),
        rejected=statuses.count("rejected"),
        pending=statuses.count("pending"),
        edited=sum(p.edited for p in reviewed.proposals),
        edited_pending=edited_pending,
        ignored_boxes=ignored,
    )
    get_logger(__name__).info(
        "review_imported",
        set_id=reviewed.set_id,
        parent_set_id=ps.set_id,
        accepted=summary.accepted,
        rejected=summary.rejected,
        pending=summary.pending,
        edited=summary.edited,
        ignored_boxes=ignored,
    )
    return reviewed, summary
