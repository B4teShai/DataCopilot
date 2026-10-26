"""The ``split`` use case: re-split a ``dataset.dcp`` by group and write it as a NEW dataset.

Groups are the connected components of two relations over the images:

- *same key*: images whose ``--by-group`` key (``groupkey.py``; default ``group_id``) has the same
  normalised label; the new version's ``group_id`` is that label, so the leakage check's
  ``group_overlap`` verifies the result. An existing ``group_id`` is *always* a constraint as
  well, whatever the key: an image the key does not label keeps its ``group_id``, and one whose
  ``group_id`` the key replaces keeps the old value in ``source["group_id_before_split"]``;
- *near copies* (``near_duplicates``): every pair of images whose embeddings have a cosine
  similarity of at least the threshold, found exactly (``knn.pairs_at_least``, not a top-k
  search, so a crowd of copies cannot hide an edge). The pairs are taken down to
  ``threshold - SIM_EPSILON - BACKEND_TOLERANCE``: the leakage check compares with
  ``SIM_EPSILON`` of slack and may use FAISS, whose similarities differ by up to
  ``BACKEND_TOLERANCE``, so it cannot find a pair the split did not group.

The groups are assigned whole (``assign.py``) and the result is checked again before anything is
written: a group in two splits is an internal error, never a written dataset.

Fail closed: a near-duplicate split needs every image embedded. An image that could not be read or
embedded, an embedder error, a search ended by the time budget or more than ``MAX_PAIRS`` pairs
refuse the split (``ValidationError``) and nothing is written, because an image whose copies are
unknown could be the leaked one.

Append-only: the output is created with ``DatasetStore.create`` (it refuses an existing path) with
the parent's lineage plus the parent's version; the source directory is read, never written (a
near-duplicate split neither reads nor writes the embedding cache). A result equal to the current
version or to an earlier one in its lineage is refused. Relative image paths are rewritten for the
output's directory (``store.relocate_images``); an output whose directory does not hold the images
is refused, because no check would read them there. A test-only embedder (``hash``) is refused
unless the caller opts in. Event: ``dataset_split``.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.audit.embedding import DEFAULT_DTYPE, DEFAULT_EMBED_BATCH, PipelineProvider
from datacopilot.audit.usecase import check_threshold, embedder_label, resolve_embedder
from datacopilot.core.errors import StoreError, ValidationError
from datacopilot.core.schema import SCHEMA_VERSION, Dataset, Image, SplitReport
from datacopilot.core.version import models_version
from datacopilot.embed import knn
from datacopilot.embed.protocol import Embedder
from datacopilot.logging import current_request_id, get_logger
from datacopilot.settings import Settings
from datacopilot.splits import assign, groupkey
from datacopilot.store import DatasetStore, StoreLimits, ensure_outside, relocate_images

PREVIOUS_GROUP_ID = "group_id_before_split"
"""``Image.source`` key that keeps a ``group_id`` the ``--by-group`` key replaced."""
DEFAULT_THRESHOLD = 0.95
"""Cosine similarity at or above which two images are near copies (the leakage check's default)."""
MAX_PAIRS = 2_000_000
"""Near-copy pairs a split accepts; more means a threshold far too low for grouping, and the split
is refused rather than collapsing the dataset into one group."""


class _UnionFind:
    """Union-find over ``n`` rows with path halving; ``find`` returns the smallest-index root."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _near_copy_pairs(
    ds: Dataset,
    dataset_dir: Path,
    position: dict[str, int],
    uf: _UnionFind,
    *,
    settings: Settings,
    created: Embedder,
    threshold: float,
    embed_batch: int | None,
) -> tuple[int, str]:
    """Union the rows of every near-copy pair; returns the pair count and the embedder label.

    The embedding cache is neither read nor written (``use_cache=False``): the source dataset is
    never written by ``split``, so the vectors are computed for this run only.
    """
    provider = PipelineProvider(
        created,
        batch_size=embed_batch if embed_batch is not None else DEFAULT_EMBED_BATCH,
        dtype=DEFAULT_DTYPE,
        use_cache=False,
    )
    config = CheckConfig()
    ctx = CheckContext(
        run_id=current_request_id() or str(uuid.uuid4()),
        root=dataset_dir.parent,
        cache_dir=settings.cache_dir,
        workers=settings.workers,
        cache_key=settings.cache_key,
        in_ci=settings.ci,
        dataset_dir=None,
        embedder=embedder_label(created),
        embeddings=provider,
        config=config,
    )
    es = provider.get(ds, ctx)
    problems: list[str] = []
    if es.refused is not None:
        problems.append(es.refused)
    if es.error is not None:
        problems.append(f"embedder error {es.error}")
    if es.missing:
        problems.append(", ".join(f"{r}={len(v)}" for r, v in sorted(es.missing.items())))
    if problems:
        raise ValidationError(
            "a near-duplicate split needs every image embedded (an image whose copies are "
            f"unknown could be the leaked one): {'; '.join(problems)}"
        )
    rows: NDArray[np.int64] = np.array([position[i] for i in es.ids], dtype=np.int64)

    def take(a: NDArray[np.int64], b: NDArray[np.int64]) -> None:
        for i, j in zip(rows[a].tolist(), rows[b].tolist(), strict=True):
            uf.union(i, j)

    floor = threshold - knn.SIM_EPSILON - knn.BACKEND_TOLERANCE
    deadline = time.monotonic() + config.max_check_seconds
    result = knn.pairs_at_least(es.matrix, floor, take, max_pairs=MAX_PAIRS, deadline=deadline)
    if result.capped:
        raise ValidationError(
            f"more than {MAX_PAIRS} near-duplicate pairs at threshold {threshold}: raise "
            "--similarity-threshold or group by key"
        )
    if result.rows < len(es.ids):
        raise ValidationError(
            f"the near-duplicate search ended at the time budget ({result.rows} of "
            f"{len(es.ids)} images); nothing was written"
        )
    return result.pairs, embedder_label(created)


def _groups(
    images: Sequence[Image], ds: Dataset, uf: _UnionFind
) -> tuple[list[assign.Group], list[list[int]]]:
    members: dict[int, list[int]] = {}
    for row in range(len(images)):
        members.setdefault(uf.find(row), []).append(row)
    classes: dict[str, Counter[int]] = {}
    for ann in ds.annotations:
        classes.setdefault(ann.image_id, Counter())[ann.category_id] += 1
    groups: list[assign.Group] = []
    rows_of: list[list[int]] = []
    for rows in members.values():
        tally: Counter[int] = Counter()
        for row in rows:
            tally.update(classes.get(images[row].image_id, Counter()))
        groups.append(assign.Group.of([images[r].image_id for r in rows], tally))
        rows_of.append(rows)
    return groups, rows_of


def split_dataset(
    dataset: Path,
    out: Path,
    *,
    settings: Settings,
    ratios: Sequence[float],
    group_key: str = "group_id",
    near_duplicates: bool = False,
    embedder: str | None = None,
    similarity_threshold: float | None = None,
    embed_batch: int | None = None,
    stratify: bool = False,
    seed: int = 0,
    allow_test_embedder: bool = False,
) -> SplitReport:
    """Re-split ``dataset`` into the new dataset ``out`` (see the module docstring).

    Raises:
        ValueError: bad ratios, key, seed or threshold, near duplicates without an embedder, or a
            test-only embedder without ``allow_test_embedder`` (usage errors).
        ValidationError: output inside the source or in a directory that does not hold the images,
            a near-duplicate split that cannot see every image, too many pairs, or a split equal to
            the current version or to an earlier one in its lineage.
        StoreError: the output exists or the dataset is unreadable.
        LicenseError, FormatError: the embedder is not allowed or unusable.
    """
    checked = assign.check_ratios(list(ratios))
    key = groupkey.parse(group_key)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")
    if near_duplicates and embedder is None:
        raise ValueError("a near-duplicate split needs an embedder")
    if not near_duplicates and (embedder is not None or similarity_threshold is not None):
        raise ValueError("an embedder and a similarity threshold need near_duplicates")
    check_threshold(similarity_threshold)
    threshold = DEFAULT_THRESHOLD if similarity_threshold is None else similarity_threshold
    created: Embedder | None = None
    if embedder is not None:
        created = resolve_embedder(embedder, settings.allowed_licenses)
        if getattr(created, "test_only", False) and not allow_test_embedder:
            raise ValueError(
                f"{embedder!r} is test-only (it compares file bytes, not pictures) and is refused "
                "for a split; pass allow_test_embedder=True (--allow-test-embedder)"
            )
    ensure_outside(out, dataset)
    if os.path.lexists(out):  # checked before any embedding work; create() checks again
        raise StoreError(f"{out} already exists; refusing to overwrite")
    limits = StoreLimits(
        max_rows=settings.max_table_rows, max_uncompressed_bytes=settings.max_table_bytes
    )
    store = DatasetStore.open(dataset, limits=limits)
    read = store.read()
    ds = Dataset(read.manifest, read.images, read.annotations, Path(dataset).resolve().parent)
    images = sorted(ds.images, key=lambda i: i.image_id)
    assert ds.root is not None  # noqa: S101  # set above
    relocated = relocate_images(images, ds.root, Path(out).resolve().parent)  # refuses early
    position = {image.image_id: n for n, image in enumerate(images)}
    uf = _UnionFind(len(images))
    labels = [groupkey.key_of(image, key) for image in images]
    old_labels = [groupkey.label(image.group_id) for image in images]
    for column in (labels, old_labels):  # an existing group_id is always a constraint too
        first_with: dict[str, int] = {}
        for row, value in enumerate(column):
            if value is not None:
                uf.union(first_with.setdefault(value, row), row)
    key_groups = sum(1 for n in Counter(lb for lb in labels if lb).values() if n > 1)
    pairs, label = 0, None
    if near_duplicates:
        assert created is not None  # noqa: S101  # checked above
        pairs, label = _near_copy_pairs(
            ds,
            Path(dataset).resolve(),
            position,
            uf,
            settings=settings,
            created=created,
            threshold=threshold,
            embed_batch=embed_batch,
        )
    groups, rows_of = _groups(images, ds, uf)
    result = assign.assign(groups, checked, seed=seed, stratify=stratify)
    new_images: list[Image] = []
    split_of_row: dict[int, str] = {}
    for group, rows in zip(groups, rows_of, strict=True):
        for row in rows:
            split_of_row[row] = result.split_of[group.key]
    for row, image in enumerate(relocated):
        update: dict[str, object] = {"split": split_of_row[row]}
        if key.kind != "group_id" and labels[row] is not None:
            update["group_id"] = labels[row]
            if image.group_id is not None and old_labels[row] != labels[row]:
                update["source"] = {**image.source, PREVIOUS_GROUP_ID: image.group_id}
        new_images.append(image.model_copy(update=update))
    across = sum(1 for rows in rows_of if len({split_of_row[r] for r in rows}) > 1)
    if across:  # cannot happen: every group was assigned as one unit
        raise ValidationError(f"internal error: {across} groups ended in more than one split")
    lineage = (*ds.manifest.lineage, ds.manifest.version)
    version = models_version(
        new_images, ds.annotations, ds.categories, schema_version=SCHEMA_VERSION
    )
    if version == ds.manifest.version:
        raise ValidationError(
            "the split equals the current version (same splits and groups): nothing to write"
        )
    if version in ds.manifest.lineage:
        raise ValidationError(
            f"the split equals an earlier version in this dataset's lineage ({version}); use "
            "that version instead: nothing to write"
        )
    written = DatasetStore.create(
        out,
        new_images,
        ds.annotations,
        ds.categories,
        name=ds.manifest.name,
        task=ds.manifest.task,
        limits=limits,
        lineage=lineage,
    ).manifest()
    sizes = [len(rows) for rows in rows_of]
    stratified: Literal["category"] | None = "category" if stratify else None
    report = SplitReport(
        parent_version=ds.manifest.version,
        dataset_version=written.version,
        seed=seed,
        ratios=checked,
        stratify=stratified,
        group_key=key.text,
        near_duplicate_threshold=threshold if near_duplicates else None,
        embedder=label,
        image_count=len(images),
        group_count=len(groups),
        multi_image_groups=sum(1 for s in sizes if s > 1),
        largest_group=max(sizes, default=0),
        key_groups=key_groups,
        images_without_key=sum(1 for lb in labels if lb is None),
        near_duplicate_pairs=pairs,
        groups_across_splits=across,
        moved_images=sum(
            1 for old, new in zip(images, new_images, strict=True) if old.split != new.split
        ),  # fmt: skip
        splits=result.splits,
        classes=result.classes,
        max_class_deviation=result.max_class_deviation,
    )
    get_logger(__name__).info(
        "dataset_split",
        parent_version=report.parent_version,
        dataset_version=report.dataset_version,
        image_count=report.image_count,
        group_count=report.group_count,
        near_duplicate_pairs=pairs,
        moved_count=report.moved_images,
    )
    return report
