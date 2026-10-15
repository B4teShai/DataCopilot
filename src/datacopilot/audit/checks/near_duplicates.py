"""``near_duplicates``: images whose embeddings are nearly parallel.

Needs an embedder (``--embedder``). Every image's ``knn_k`` nearest neighbours (exact cosine, see
``embed/knn.py``) at or above ``near_duplicate_cosine`` form pairs; pairs are merged into clusters
(connected components), so a chain of near copies is one finding. A cluster is WARN: redundancy
inflates metrics but does not by itself break the evaluation. ``cross_split`` marks a cluster that
touches two splits; the ``leakage`` check reports those as CRITICAL, and they stay visible here so
running this check alone never hides them.

Rules (``payload["rule"]``): ``near_duplicate`` (``group_id`` = 12-hex digest of the member ids,
``count``, ``cross_split``, ``splits``, ``min_cosine``, ``max_cosine``, up to 20 ``members`` as
short id, split and base name); and the shared fail-closed rules of ``embedtools.embedding_set``:
``embedder_missing``, ``embedder_error``, ``images_not_embedded``, ``audit_incomplete``,
``embedding_cache_rejected``, ``embedding_cache_mismatch``. ``phash_not_applied`` (WARN) counts
the images the optional perceptual-hash prefilter could not hash; their pairs are kept.
``audit_incomplete`` also reports a neighbour search that ended at ``max_check_seconds``
(``checked`` rows of ``total``) or a dataset over ``MAX_KNN_ROWS`` images (``too_many_images``):
pairs found before the end are still reported.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from datacopilot.audit import embedding
from datacopilot.audit.checks import embedtools
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.core.errors import ModelError
from datacopilot.core.schema import Dataset, Finding, Image, PluginSpec
from datacopilot.embed import imaging, knn

NAME = "near_duplicates"


def similar_pairs(
    matrix: NDArray[np.float32], k: int, threshold: float, deadline: float
) -> tuple[dict[tuple[int, int], float], int]:
    """Pairs ``(i, j)``, ``i < j``, of rows with cosine >= ``threshold`` among each row's ``k``
    nearest neighbours, with their similarity, and the number of rows searched."""
    pairs: dict[tuple[int, int], float] = {}
    rows = 0
    for block in knn.iter_knn(matrix, matrix, k, same_set=True, deadline=deadline):
        rows += len(block.indices)
        hit_rows, hit_cols = np.nonzero(block.sims >= threshold)
        for r, c in zip(hit_rows.tolist(), hit_cols.tolist(), strict=True):
            i, j = block.start + r, int(block.indices[r, c])
            key = (i, j) if i < j else (j, i)
            sim = float(block.sims[r, c])
            if sim > pairs.get(key, -1.0):
                pairs[key] = sim
    return pairs, rows


def phash_prefilter(
    ctx: CheckContext,
    ids: tuple[str, ...],
    by_id: dict[str, Image],
    pairs: dict[tuple[int, int], float],
    max_distance: int,
    deadline: float,
) -> tuple[dict[tuple[int, int], float], int]:
    """Drop the pairs whose difference hashes differ in more than ``max_distance`` bits.

    A pair is dropped only when *both* images were hashed and disagree; an image that cannot be
    read or hashed (or Pillow missing, or the time budget over) keeps its pairs, so the filter can
    only remove findings it has evidence against. Returns the kept pairs and the number of images
    that could not be hashed.
    """
    hashes: dict[int, int] = {}
    unverified = 0
    for row in sorted({r for pair in pairs for r in pair}):
        if time.monotonic() >= deadline:
            unverified += 1
            continue
        data, _ = embedding.read_image_bytes(by_id[ids[row]], ctx)
        if data is None:
            unverified += 1
            continue
        try:
            hashes[row] = imaging.dhash(data, 0)
        except (ModelError, ImportError):
            unverified += 1
    kept = {
        (a, b): sim
        for (a, b), sim in pairs.items()
        if not (
            a in hashes and b in hashes and imaging.hamming(hashes[a], hashes[b]) > max_distance
        )
    }
    return kept, unverified


class NearDuplicatesCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        es, findings = embedtools.embedding_set(ds, ctx, NAME)
        if es is None:
            return findings
        n = len(es.ids)
        if n < 2:
            return findings
        if n > knn.MAX_KNN_ROWS:
            return [*findings, embedtools.incomplete_finding(ctx, NAME, "too_many_images", 0, n)]
        deadline = time.monotonic() + ctx.config.max_check_seconds
        pairs, rows = similar_pairs(
            es.matrix, ctx.config.knn_k, ctx.config.near_duplicate_cosine, deadline
        )
        if rows < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", rows, n))
        by_id = dict(embedtools.images_by_id(ds))
        max_distance = ctx.config.near_duplicate_phash_max_distance
        if max_distance is not None and pairs:
            pairs, unverified = phash_prefilter(ctx, es.ids, by_id, pairs, max_distance, deadline)
            if unverified:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        "WARN",
                        payload={
                            "rule": "phash_not_applied",
                            "count": unverified,
                            "max_distance": max_distance,
                        },
                    )
                )
        grouped = embedtools.components(pairs)
        component_of = {node: n for n, comp in enumerate(grouped) for node in comp}
        sims_of: list[list[float]] = [[] for _ in grouped]
        for (a, _b), sim in pairs.items():
            sims_of[component_of[a]].append(sim)
        for component, sims in zip(grouped, sims_of, strict=True):
            member_ids = [es.ids[i] for i in component]
            members = [by_id[i] for i in member_ids]
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "WARN",
                    image_id=member_ids[0],
                    payload={
                        "rule": "near_duplicate",
                        "group_id": embedtools.short_digest(member_ids),
                        "count": len(members),
                        "cross_split": embedtools.spans_splits(m.split for m in members),
                        "splits": embedtools.split_counts(members),
                        "min_cosine": round(min(sims), 4),
                        "max_cosine": round(max(sims), 4),
                        "members": embedtools.members_payload(members),
                    },
                )
            )
        return findings


plugin = NearDuplicatesCheck()
