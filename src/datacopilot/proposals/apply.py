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
- it came through ``review import`` (it names the set it was reviewed from, and when that set is
  available the two must agree on every proposal except status, edited flag and box) and is not
  from a test double unless the caller says so (``allow_test_proposer``);
- it is complete, unless the caller accepts an incomplete run (``allow_incomplete``);
- at least one proposal was accepted. Pending and rejected proposals are never applied.

A set's id is a digest: it proves the file was not changed, not who wrote it. Someone who can write
into the dataset directory can write a new, self-consistent set; the review marker and the parent
check make that a deliberate act, not an accident, and nothing here replaces trusting the people
who can write to the directory.

Each new row has ``provenance == "proposal_accepted"`` and, in ``source``, the ``proposal_id``,
``set_id``, ``model_id``, ``model_revision`` and whether a reviewer ``edited`` the box. The model's
score is kept only for an unedited box: an edited box is the reviewer's, not the model's. The
reviewed set (and its parent) are copied into the new dataset's ``proposals/``, so the provenance
survives the deletion of the source. Event: ``proposals_applied``.
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


def _check_model(ps: ProposalSet, allowed_licenses: frozenset[str], allow_test: bool) -> None:
    plugin = check_proposer(ps.proposer, allowed_licenses)  # installed, valid, license allowed
    if ps.test_only or getattr(plugin, "test_only", False):
        if not allow_test:
            raise ValidationError(
                f"the proposal set comes from the test double {ps.proposer!r}, whose boxes say "
                "nothing about the pictures; applying it needs allow_test_proposer"
            )
    first_party = models.has_model(ps.proposer) or models.has_model(ps.model_id)
    if first_party:
        # a registered model is only ever produced by its own proposer, at the pinned revision
        card = models.get_model(ps.proposer if models.has_model(ps.proposer) else ps.model_id,
                                allowed_licenses)  # fmt: skip
        if ps.proposer != card.model_id or (ps.model_id, ps.model_revision, ps.weights) != (
            card.model_id,
            card.revision,
            card.pinned_files(),
        ):
            raise ValidationError(
                "the proposal set was not made by the pinned revision and files of "
                f"model {card.model_id!r}; propose again"
            )


def _reviews(ps: ProposalSet, parent: ProposalSet) -> bool:
    """``ps`` is exactly what a review of ``parent`` can produce (only status, edited flag and box
    differ)."""
    if parent.set_id != ps.parent_set_id:
        return False
    try:
        return parent.derive(list(ps.proposals)).set_id == ps.set_id
    except ValueError:
        return False


def _check_output(out: Path, source: Path) -> None:
    src, dst = source.resolve(), out.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValidationError("the output must be a new directory outside the source dataset")


def apply_proposals(
    ds: Dataset,
    ps: ProposalSet,
    out: Path,
    *,
    allowed_licenses: frozenset[str],
    source: Path,
    allow_incomplete: bool = False,
    allow_test_proposer: bool = False,
    parent: ProposalSet | None = None,
    limits: StoreLimits | None = None,
) -> DatasetManifest:
    """Write ``ds`` plus the accepted proposals of ``ps`` as the new dataset ``out``.

    Args:
        ds: the dataset ``ps`` was made for (it is not modified).
        out: a new ``.dcp`` path; ``source`` is the directory ``ds`` was read from (required), so
            that an output inside it is refused.
        parent: the set ``ps`` was reviewed from, when available; it is checked and copied.

    Raises:
        ValidationError: the set does not fit the dataset, did not come through a review, comes
            from a test double (without ``allow_test_proposer``), was made by another model than
            the pinned one, is incomplete (without ``allow_incomplete``), has nothing accepted, or
            the output lies inside the source.
        FormatError, LicenseError: the proposer is not installed or not allowed.
        StoreError: the output exists.
    """
    validate_against(ps, ds)
    if ps.parent_set_id is None:
        raise ValidationError(
            "the proposal set has not been through `review import`: review it first (a fresh "
            "proposal run is never applied)"
        )
    if parent is not None and not _reviews(ps, parent):
        raise ValidationError("the proposal set is not a review of its parent set")
    _check_model(ps, allowed_licenses, allow_test_proposer)
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
            score=None if p.edited else p.score,
            provenance="proposal_accepted",
            source={
                "proposal_id": p.proposal_id,
                "set_id": ps.set_id,
                "model_id": ps.model_id,
                "model_revision": ps.model_revision,
                "edited": p.edited,
            },
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
    store.write_proposals(ps)  # the provenance travels with the dataset
    if parent is not None:
        store.write_proposals(parent)
    manifest = store.manifest()
    get_logger(__name__).info(
        "proposals_applied",
        parent_version=ds.manifest.version,
        dataset_version=manifest.version,
        applied_count=len(rows),
        set_id=ps.set_id,
    )
    return manifest
