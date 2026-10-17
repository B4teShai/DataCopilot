"""``apply``: write the accepted proposals as a NEW dataset that records where it came from.

Nothing that exists is modified: the output is created with ``DatasetStore.create`` (which refuses
an existing path), its lineage is the parent's lineage plus the parent's version, and the parent's
files are never opened for writing. The output directory must lie outside the source dataset.

A set is applied only when all of this holds, and each failure writes nothing:

- it was made for exactly this dataset version, and every proposal fits the dataset (an image and a
  class that exist, a box inside the image);
- its proposer is installed and its license is on ``DATACOPILOT_ALLOWED_LICENSES``, and a model the
  registry knows is the pinned revision with the pinned files (a set that claims another checkpoint
  was not made by the model that was license-checked);
- it is complete, unless the caller accepts an incomplete run (``allow_incomplete``);
- at least one proposal was accepted. Pending and rejected proposals are never applied.

Each new row has ``provenance == "proposal_accepted"``, the model's score, and the ``proposal_id``
and ``set_id`` it came from in ``source``. Event: ``proposals_applied``.
"""

from __future__ import annotations

from pathlib import Path

from datacopilot.core.errors import ValidationError
from datacopilot.core.proposals import ProposalSet, validate_against
from datacopilot.core.schema import Annotation, Dataset, DatasetManifest
from datacopilot.logging import get_logger
from datacopilot.models import registry as models
from datacopilot.proposers.registry import check_proposer
from datacopilot.store import DatasetStore, StoreLimits


def _check_model(ps: ProposalSet, allowed_licenses: frozenset[str]) -> None:
    check_proposer(ps.proposer, allowed_licenses)  # installed, valid, license allowed
    if models.has_model(ps.proposer):
        card = models.get_model(ps.proposer, allowed_licenses)
        if (ps.model_id, ps.model_revision, ps.weights) != (
            card.model_id,
            card.revision,
            card.pinned_files(),
        ):
            raise ValidationError(
                "the proposal set was not made by the pinned revision and files of "
                f"model {card.model_id!r}; propose again"
            )


def _check_output(out: Path, source: Path | None) -> None:
    if source is None:
        return
    src, dst = source.resolve(), out.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValidationError("the output must be a new directory outside the source dataset")


def apply_proposals(
    ds: Dataset,
    ps: ProposalSet,
    out: Path,
    *,
    allowed_licenses: frozenset[str],
    allow_incomplete: bool = False,
    source: Path | None = None,
    limits: StoreLimits | None = None,
) -> DatasetManifest:
    """Write ``ds`` plus the accepted proposals of ``ps`` as the new dataset ``out``.

    Args:
        ds: the dataset ``ps`` was made for (it is not modified).
        out: a new ``.dcp`` path; ``source`` is the directory ``ds`` was read from, when known, so
            that an output inside it can be refused.

    Raises:
        ValidationError: the set does not fit the dataset, was made by another model than the
            pinned one, is incomplete (without ``allow_incomplete``), has nothing accepted, or the
            output lies inside the source.
        FormatError, LicenseError: the proposer is not installed or not allowed.
        StoreError: the output exists.
    """
    validate_against(ps, ds)
    _check_model(ps, allowed_licenses)
    if ps.incomplete and not allow_incomplete:
        raise ValidationError(
            f"the proposal set is incomplete ({ps.incomplete_reason}: {ps.images_done} of "
            f"{ps.images_total} images); applying it needs allow_incomplete"
        )
    accepted = ps.accepted()
    if not accepted:
        raise ValidationError("the proposal set has no accepted proposals: review it first")
    _check_output(out, source)
    rows = [
        Annotation(
            ann_id=f"proposal:{p.proposal_id}",
            image_id=p.image_id,
            category_id=p.category_id,
            kind="bbox",
            bbox_xyxy_abs=p.bbox_xyxy_abs,
            score=p.score,
            provenance="proposal_accepted",
            source={"proposal_id": p.proposal_id, "set_id": ps.set_id},
        )
        for p in accepted
    ]
    store = DatasetStore.create(
        out,
        ds.images,
        [*ds.annotations, *rows],
        ds.categories,
        name=ds.manifest.name,
        task=ds.manifest.task,
        limits=limits,
        lineage=(*ds.manifest.lineage, ds.manifest.version),
    )
    manifest = store.manifest()
    get_logger(__name__).info(
        "proposals_applied",
        parent_version=ds.manifest.version,
        dataset_version=manifest.version,
        applied_count=len(rows),
        set_id=ps.set_id,
    )
    return manifest
