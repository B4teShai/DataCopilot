"""``leakage``: the same picture on both sides of a split boundary.

Two independent signals, both CRITICAL because they make validation and test metrics optimistic:

* ``cross_split_near_duplicate``: for each split, the ``knn_k`` nearest neighbours among the
  images of *other* splits (exact cosine, ``embed/knn.py``) at or above ``leakage_cosine`` form
  pairs; pairs are merged into clusters. Searching only across splits means a crowd of
  near-identical frames inside one split (a video) cannot push the one leaking frame out of the
  neighbour list.
  ``unassigned`` counts as a split of its own and is compared with the real splits, as the
  exact-duplicate check does; two ``unassigned`` images are not compared with each other.
* ``group_overlap``: one ``Image.group_id`` (session, device, video, SKU) present in more than one
  split (or in a real split and ``unassigned``). This needs no embedding and is reported even when
  the embedder fails.

Findings carry short ids, splits and base names only. Rules also include the shared fail-closed
rules of ``embedtools.embedding_set`` (``embedder_missing``, ``embedder_error``,
``images_not_embedded``, ``audit_incomplete``, ``embedding_cache_rejected``,
``embedding_cache_mismatch``): images that could not be compared are never silently clean.
``k`` bounds the neighbours examined, so ``count`` of a cluster can understate a very large one.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from datacopilot.audit.checks import embedtools
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.core.schema import Dataset, Finding, Image, PluginSpec
from datacopilot.embed import knn

NAME = "leakage"
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2, "unassigned": 3}


def _group_findings(ds: Dataset, ctx: CheckContext) -> list[Finding]:
    by_group: dict[str, list[Image]] = defaultdict(list)
    for image in ds.images:
        if image.group_id:
            by_group[image.group_id].append(image)
    findings: list[Finding] = []
    ungrouped = len(ds.images) - sum(len(v) for v in by_group.values())
    if by_group and ungrouped:
        # Some images carry a group and some do not: whoever clears group_id on the images that
        # matter would otherwise make group_overlap disappear. A dataset with no groups at all
        # cannot be told apart from one that does not use them, and gets no note.
        findings.append(
            make_finding(
                ctx.run_id,
                NAME,
                "INFO",
                payload={
                    "rule": "images_without_group",
                    "count": ungrouped,
                    "total": len(ds.images),
                },
            )
        )
    for group_id in sorted(by_group):
        members = sorted(by_group[group_id], key=lambda i: i.image_id)
        if not embedtools.spans_splits(m.split for m in members):
            continue
        findings.append(
            make_finding(
                ctx.run_id,
                NAME,
                "CRITICAL",
                image_id=members[0].image_id,
                payload={
                    "rule": "group_overlap",
                    "group": embedtools.safe_label(group_id),
                    "group_ref": embedtools.short_digest([group_id]),
                    "count": len(members),
                    "splits": embedtools.split_counts(members),
                    "members": embedtools.members_payload(members),
                },
            )
        )
    return findings


class LeakageCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        findings = _group_findings(ds, ctx)
        es, prologue = embedtools.embedding_set(ds, ctx, NAME)
        findings += prologue
        if es is None:
            return findings
        n = len(es.ids)
        if n < 2:
            return findings
        if n > knn.MAX_KNN_ROWS:
            return [*findings, embedtools.incomplete_finding(ctx, NAME, "too_many_images", 0, n)]
        by_id = embedtools.images_by_id(ds)
        labels = np.array([_SPLIT_CODE[by_id[i].split] for i in es.ids], dtype=np.int8)
        threshold, k = ctx.config.leakage_cosine, ctx.config.knn_k
        deadline = time.monotonic() + ctx.config.max_check_seconds
        pairs: dict[tuple[int, int], float] = {}
        searched = 0
        for code in sorted(set(labels.tolist())):
            queries = np.flatnonzero(labels == code)
            others = np.flatnonzero(labels != code)
            if len(others) == 0:
                searched += len(queries)
                continue
            for block in knn.iter_knn(es.matrix[queries], es.matrix[others], k, deadline=deadline):
                searched += len(block.indices)
                hit_rows, hit_cols = np.nonzero(block.sims >= threshold)
                for r, c in zip(hit_rows.tolist(), hit_cols.tolist(), strict=True):
                    i = int(queries[block.start + r])
                    j = int(others[int(block.indices[r, c])])
                    key = (i, j) if i < j else (j, i)
                    sim = float(block.sims[r, c])
                    if sim > pairs.get(key, -1.0):
                        pairs[key] = sim
        if searched < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", searched, n))
        grouped = embedtools.components(pairs)
        component_of = {node: idx for idx, comp in enumerate(grouped) for node in comp}
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
                    "CRITICAL",
                    image_id=member_ids[0],
                    payload={
                        "rule": "cross_split_near_duplicate",
                        "group_id": embedtools.short_digest(member_ids),
                        "count": len(members),
                        "pairs": len(sims),
                        "splits": embedtools.split_counts(members),
                        "min_cosine": round(min(sims), 4),
                        "max_cosine": round(max(sims), 4),
                        "members": embedtools.members_payload(members),
                    },
                )
            )
        return findings


plugin = LeakageCheck()
