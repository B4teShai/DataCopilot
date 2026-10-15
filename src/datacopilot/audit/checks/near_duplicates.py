"""``near_duplicates``: images whose embeddings are nearly parallel.

Needs an embedder (``--embedder``). Every image's ``knn_k`` nearest neighbours (exact cosine, see
``embed/knn.py``) at or above ``near_duplicate_cosine`` form pairs; pairs are merged into clusters
(connected components), so a chain of near copies is one finding. A cluster inside one split is
WARN: redundancy inflates metrics but does not by itself break the evaluation. A cluster that
touches two splits (``cross_split``) is CRITICAL, the same rule as exact duplicates, so running or
gating on this check alone does not hide cross-split copies; ``leakage`` keeps its own findings
for them. Pairs found are re-embedded fresh before the verdict when a cached vector is involved
(``embedtools.reverify``).

Rules (``payload["rule"]``): ``near_duplicate`` (``group_id`` = 12-hex digest of the member ids,
``count``, ``cross_split``, ``splits``, ``min_cosine``, ``max_cosine``, up to 20 ``members`` as
short id, split and base name); and the shared fail-closed rules of ``embedtools.embedding_set``:
``embedder_missing``, ``embedder_error``, ``images_not_embedded``, ``audit_incomplete``,
``embedding_cache_rejected``, ``embedding_cache_mismatch``,
``embedding_cache_content_changed``, ``embedding_cache_unverified``,
``embedding_pair_contradicted``. ``phash_not_applied`` (WARN) counts
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
) -> tuple[dict[tuple[int, int], float], int, bool]:
    """Similar pairs ``(i, j)``, ``i < j``, with their similarity, the rows searched and a cap flag.

    A pair is kept when its cosine is at least ``threshold`` less ``SIM_EPSILON`` less
    ``BORDERLINE_MARGIN`` (the borderline pairs are re-checked by ``embedtools.reverify``), among
    each row's ``k`` nearest neighbours. The flag is True when the pair cap
    (``embedtools.MAX_PAIRS``) stopped the search.
    """
    pairs: dict[tuple[int, int], float] = {}
    rows = 0
    for block in knn.iter_knn(matrix, matrix, k, same_set=True, deadline=deadline):
        rows += len(block.indices)
        hit_rows, hit_cols = np.nonzero(
            block.sims >= threshold - knn.SIM_EPSILON - embedtools.BORDERLINE_MARGIN
        )
        for r, c in zip(hit_rows.tolist(), hit_cols.tolist(), strict=True):
            if not embedtools.add_pair(
                pairs, block.start + r, int(block.indices[r, c]), float(block.sims[r, c])
            ):
                return pairs, rows, True
    return pairs, rows, False


def phash_prefilter(
    ctx: CheckContext,
    ids: tuple[str, ...],
    by_id: dict[str, Image],
    pairs: dict[tuple[int, int], float],
    max_distance: int,
    deadline: float,
) -> tuple[dict[tuple[int, int], float], int]:
    """Drop the pairs whose difference hashes differ in more than ``max_distance`` bits.

    Pairs across a split boundary are never dropped (``spans_splits``): a hash that disagrees with
    the model must not be able to hide leakage, so the prefilter only prunes redundancy inside a
    split. Enabling it therefore weakens ``near_duplicates`` as a gate for within-split pairs.

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
        if embedtools.spans_splits((by_id[ids[a]].split, by_id[ids[b]].split))
        or not (
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
        pairs, rows, capped = similar_pairs(
            es.matrix, ctx.config.knn_k, ctx.config.near_duplicate_cosine, deadline
        )
        if capped:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "pair_cap", rows, n))
        elif rows < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", rows, n))
        by_id = dict(embedtools.images_by_id(ds))
        pairs, contradicted = embedtools.reverify(
            ds, ctx, es, pairs, ctx.config.near_duplicate_cosine
        )
        if contradicted:
            findings.append(embedtools.contradiction_finding(ctx, NAME, contradicted))
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
            cross = embedtools.spans_splits(m.split for m in members)
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "CRITICAL" if cross else "WARN",
                    image_id=member_ids[0],
                    payload={
                        "rule": "near_duplicate",
                        "group_id": embedtools.short_digest(member_ids),
                        "count": len(members),
                        "cross_split": cross,
                        "splits": embedtools.split_counts(members),
                        "min_cosine": round(min(sims), 4),
                        "max_cosine": round(max(sims), 4),
                        "members": embedtools.members_payload(members),
                    },
                )
            )
        return findings


plugin = NearDuplicatesCheck()
