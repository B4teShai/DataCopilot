"""The propose, review and apply commands as use cases over a ``dataset.dcp`` directory.

The command line and the Python API call these with the same validation, limits and license
guard. The proposer is resolved (and its license checked) before the dataset is read; proposal sets
are stored under ``proposals/`` and are looked up by their content digest.

Events: ``proposal_set_written``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from datacopilot.core.errors import StoreError, ValidationError
from datacopilot.core.proposals import ProposalSet
from datacopilot.core.schema import Dataset, DatasetManifest
from datacopilot.io.report import ConversionReport
from datacopilot.logging import get_logger
from datacopilot.proposals import apply as apply_module
from datacopilot.proposals import review, run
from datacopilot.proposers import registry
from datacopilot.settings import Settings
from datacopilot.store import DatasetStore, StoreLimits

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _limits(settings: Settings) -> StoreLimits:
    return StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )


def _open(dataset: Path, settings: Settings) -> tuple[DatasetStore, Dataset]:
    store = DatasetStore.open(dataset, limits=_limits(settings))
    ds = store.read()
    return store, Dataset(ds.manifest, ds.images, ds.annotations, Path(dataset).resolve().parent)


def propose_dataset(
    dataset: Path,
    proposer_name: str,
    *,
    settings: Settings,
    classes: Sequence[str] | None = None,
    min_score: float = run.DEFAULT_MIN_SCORE,
    class_min_score: Mapping[str, float] | None = None,
    time_budget_seconds: float = run.DEFAULT_TIME_BUDGET_SECONDS,
) -> tuple[ProposalSet, bool]:
    """Run the named proposer over ``dataset`` and store the set; returns it and whether the
    proposer is a test double (its boxes say nothing about the pictures).

    Raises:
        FormatError, LicenseError: the proposer is unusable or not allowed.
        ValueError: an unknown proposer, an invalid option, an unknown class, or no usable class
            name (usage errors).
        ValidationError, StoreError, ModelError: unreadable or non-detection dataset, model failure.
    """
    available = registry.proposer_names()
    if not _NAME.fullmatch(proposer_name) or proposer_name not in available:
        raise run.OptionError(f"unknown proposer; available: {', '.join(available) or 'none'}")
    proposer = registry.get_proposer(proposer_name, settings.allowed_licenses)
    store, ds = _open(dataset, settings)
    ps = run.propose(
        ds,
        proposer,
        proposer_name=proposer_name,
        root=Path(dataset).resolve().parent,
        classes=classes,
        min_score=min_score,
        class_min_score=class_min_score,
        time_budget_seconds=time_budget_seconds,
    )
    store.write_proposals(ps)
    get_logger(__name__).info(
        "proposal_set_written",
        set_id=ps.set_id,
        proposal_count=len(ps.proposals),
        images_done=ps.images_done,
        images_total=ps.images_total,
        incomplete_reason=ps.incomplete_reason,
    )
    return ps, bool(getattr(proposer, "test_only", False))


def export_review_set(
    dataset: Path, set_id: str, out: Path, *, settings: Settings
) -> ConversionReport:
    """Write the set's proposals as a CVAT layer next to the dataset's own boxes into ``out``."""
    store, ds = _open(dataset, settings)
    return review.export_review(ds, store.read_proposals(set_id), out)


def import_review_set(
    dataset: Path, set_id: str, directory: Path, *, settings: Settings
) -> tuple[ProposalSet, review.ReviewSummary]:
    """Read the decisions in ``directory`` and store the reviewed set (a new one, naming ``set_id``
    as its parent)."""
    store, ds = _open(dataset, settings)
    reviewed, summary = review.import_review(ds, store.read_proposals(set_id), directory)
    store.write_proposals(reviewed)
    return reviewed, summary


def apply_set(
    dataset: Path,
    set_id: str,
    out: Path,
    *,
    settings: Settings,
    allow_incomplete: bool = False,
    allow_test_proposer: bool = False,
) -> DatasetManifest:
    """Write the accepted proposals of the set as the new dataset ``out`` (see ``apply.py``)."""
    store, ds = _open(dataset, settings)
    ps = store.read_proposals(set_id)
    parent = None
    if ps.parent_set_id is not None:
        try:
            parent = store.read_proposals(ps.parent_set_id)
        except StoreError:
            raise ValidationError(
                "the set this review was made from is not in the dataset: apply needs both"
            ) from None
    return apply_module.apply_proposals(
        ds,
        ps,
        out,
        allowed_licenses=settings.allowed_licenses,
        allow_incomplete=allow_incomplete,
        allow_test_proposer=allow_test_proposer,
        source=Path(dataset),
        parent=parent,
        limits=_limits(settings),
    )
