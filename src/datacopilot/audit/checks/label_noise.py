"""``label_noise``: annotations whose class no neighbouring image shares.

Needs an embedder. It uses the image embeddings the other embedding checks already share (no new
model, no new dependency) and reports leads for a human; it never edits or proposes a change.

Rule (``suspect_class``). The base set is every embedded image that has at least one annotation
(an unannotated image says nothing about classes, so it is never a neighbour). For each base image
the ``suspect_k`` nearest other base images by exact cosine similarity are found
(``embed/knn.py``, pinned to the NumPy backend so the result does not depend on whether FAISS is
installed; chunked, so memory stays near one block). Neighbours below ``suspect_min_similarity``
(compared with ``knn.SIM_EPSILON`` of slack, so a floor of 1.0 keeps identical images) are dropped;
an image left with fewer than ``suspect_min_neighbours`` is not scored and its annotations are
counted (``annotations_not_scored``). A neighbour *carries* class ``c`` when any of its annotations
has that class; at most ``MAX_CARRIED_CLASSES`` distinct classes (the lowest ids) count per image,
so one image with a million classes cannot make every row expensive (the images cut are counted
in ``classes_truncated``). The ``support`` of an annotation of class ``c`` is the fraction of the
usable neighbours that carry ``c``; the annotation is suspect when
``support <= suspect_max_support`` (default 0.0: no neighbour carries it).

The embedding describes the whole image, not the box, so for a photo with many objects the signal
is weak: a suspect annotation means "images that look like this one do not contain this class",
which is also true of a genuinely rare object. Findings are INFO leads, never defects.

Ranking and caps: suspects rank by ``(support, -mean_similarity, ann_id)``. At most
``max_suspects_per_image`` per image, ``max_suspects_per_class`` per class and ``MAX_SUSPECTS`` in
all are kept (the best by that rank) and the rest are counted in ``suspects_truncated``; the rank
only decides which suspects survive, because the audit engine orders the findings it stores by
severity, image and annotation. Memory is bounded: only small tuples are kept while streaming, and
payloads are built for the survivors. The result is deterministic and independent of the order of
the rows.

Rules: ``suspect_class`` (``category_id``, ``category`` cut to 64 characters, ``support``,
``neighbours``, ``mean_similarity``, ``neighbour_classes`` as the five most common classes among
the neighbours with their image counts, ``file_name``), ``annotations_not_scored`` (WARN, CRITICAL
above ``max_unchecked_fraction`` of the boxes or with ``--require-local-images``, like images a
check could not examine), ``classes_truncated``, ``suspects_truncated``, ``skipped`` and the shared
fail-closed rules of ``embedtools.embedding_set``. ``skipped`` carries a ``reason``:
``embedder_missing``, ``no_embeddings`` (every image failed to embed), ``too_many_images`` (the
embedding run refused the dataset size), ``too_few_images`` (WARN) and, as there is nothing to
check, ``no_annotations`` and ``single_class`` (INFO). ``audit_incomplete`` also reports a
neighbour search that ended at ``max_check_seconds``.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict

import numpy as np
from pydantic import JsonValue

from datacopilot.audit.checks import embedtools
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.paths import display_name, unchecked_severity
from datacopilot.core.schema import Annotation, Dataset, Finding, PluginSpec, Severity
from datacopilot.embed import knn

NAME = "label_noise"
TOP_NEIGHBOUR_CLASSES = 5
"""Classes listed in the neighbour histogram of one finding."""
MAX_CARRIED_CLASSES = 256
"""Distinct classes of one image that count as carried (the lowest ids)."""
MAX_SUSPECTS = 50_000
"""Suspects kept in all; the rest are counted in ``suspects_truncated``."""

Candidate = tuple[float, float, str, int, str, int, tuple[tuple[int, int], ...]]
"""``(support, -mean_similarity, ann_id, category_id, image_id, neighbours, top classes)``: the
first three are the rank (``ann_id`` is unique, so the order is total); small tuples only, the
payload is built for the survivors."""

_SKIP_SEVERITY: dict[str, Severity] = {"no_annotations": "INFO", "single_class": "INFO"}


def _skipped(ctx: CheckContext, reason: str, **extra: JsonValue) -> Finding:
    return make_finding(
        ctx.run_id,
        NAME,
        _SKIP_SEVERITY.get(reason, "WARN"),
        payload={"rule": "skipped", "reason": reason, **extra},
    )


def _rank(candidate: Candidate) -> tuple[float, float, str]:
    return candidate[:3]


class LabelNoiseCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        es, findings = embedtools.embedding_set(ds, ctx, NAME)
        if es is None:
            return [*findings, _skipped(ctx, "embedder_missing")]
        cfg = ctx.config
        anns_of: dict[str, list[Annotation]] = defaultdict(list)
        for ann in ds.annotations:
            anns_of[ann.image_id].append(ann)
        if not es.ids:
            return [*findings, _skipped(ctx, es.refused or "no_embeddings")]
        rows = [r for r, image_id in enumerate(es.ids) if image_id in anns_of]
        if not rows:
            return [*findings, _skipped(ctx, "no_annotations")]
        n = len(rows)
        if n < cfg.suspect_min_images:
            return [
                *findings,
                _skipped(ctx, "too_few_images", images=n, min_images=cfg.suspect_min_images),
            ]
        image_ids = [es.ids[r] for r in rows]
        all_classes = [{a.category_id for a in anns_of[i]} for i in image_ids]
        if len(set().union(*all_classes)) < 2:
            return [*findings, _skipped(ctx, "single_class", images=n)]
        carried: list[frozenset[int]] = []
        cut_images = 0
        for classes in all_classes:
            if len(classes) > MAX_CARRIED_CLASSES:
                cut_images += 1
                classes = set(sorted(classes)[:MAX_CARRIED_CLASSES])
            carried.append(frozenset(classes))
        names = {c.id: c.name for c in ds.categories}
        files = {i.image_id: display_name(i.file_path) for i in ds.images}
        matrix = es.matrix[np.array(rows, dtype=np.int64)]
        deadline = time.monotonic() + cfg.max_check_seconds
        floor = cfg.suspect_min_similarity - knn.SIM_EPSILON
        per_class: dict[int, list[Candidate]] = defaultdict(list)
        held = 0
        dropped = 0
        not_scored = 0
        boxes = sum(len(anns_of[i]) for i in image_ids)
        done = 0
        stopped = False
        for block in knn.iter_knn(
            matrix,
            matrix,
            min(cfg.suspect_k, n - 1),
            same_set=True,
            backend="numpy",  # the result must not depend on whether FAISS is installed
            deadline=deadline,
        ):
            for offset in range(len(block.indices)):
                if time.monotonic() >= deadline:
                    stopped = True
                    break
                row = block.start + offset
                done += 1
                usable = [
                    (int(j), float(sim))
                    for j, sim in zip(block.indices[offset], block.sims[offset], strict=True)
                    if sim >= floor
                ]
                image_id = image_ids[row]
                if len(usable) < cfg.suspect_min_neighbours:
                    not_scored += len(anns_of[image_id])
                    continue
                histogram: Counter[int] = Counter()
                for j, _ in usable:
                    histogram.update(carried[j])
                mean_sim = sum(sim for _, sim in usable) / len(usable)
                suspects: list[Candidate] = [
                    (
                        histogram[a.category_id] / len(usable),
                        -mean_sim,
                        a.ann_id,
                        a.category_id,
                        image_id,
                        len(usable),
                        (),
                    )
                    for a in anns_of[image_id]
                    if histogram[a.category_id] / len(usable) <= cfg.suspect_max_support
                ]
                if not suspects:
                    continue
                best = heapq.nsmallest(cfg.max_suspects_per_image, suspects, key=_rank)
                dropped += len(suspects) - len(best)
                top = tuple(
                    heapq.nlargest(
                        TOP_NEIGHBOUR_CLASSES, histogram.items(), key=lambda kv: (kv[1], -kv[0])
                    )
                )
                for cand in best:
                    bucket = per_class[cand[3]]
                    bucket.append((*cand[:6], top))
                    held += 1
                    if len(bucket) > 2 * cfg.max_suspects_per_class:
                        bucket.sort(key=_rank)
                        cut = len(bucket) - cfg.max_suspects_per_class
                        del bucket[cfg.max_suspects_per_class :]
                        dropped += cut
                        held -= cut
                if held > 2 * MAX_SUSPECTS:
                    everyone = sorted((c for b in per_class.values() for c in b), key=_rank)
                    dropped += len(everyone) - MAX_SUSPECTS
                    per_class.clear()
                    for cand in everyone[:MAX_SUSPECTS]:
                        per_class[cand[3]].append(cand)
                    held = MAX_SUSPECTS
            if stopped:
                break
        kept: list[Candidate] = []
        for bucket in per_class.values():
            bucket.sort(key=_rank)
            dropped += max(0, len(bucket) - cfg.max_suspects_per_class)
            kept.extend(bucket[: cfg.max_suspects_per_class])
        kept.sort(key=_rank)
        if len(kept) > MAX_SUSPECTS:
            dropped += len(kept) - MAX_SUSPECTS
            del kept[MAX_SUSPECTS:]
        for support, neg_sim, ann_id, cid, image_id, neighbours, top in kept:
            label = embedtools.safe_label(names.get(cid, str(cid)))
            payload: dict[str, JsonValue] = {
                "rule": "suspect_class",
                "category_id": cid,
                "category": label,
                "support": round(support, 3),
                "neighbours": neighbours,
                "mean_similarity": round(-neg_sim, 4),
                "neighbour_classes": [
                    {
                        "category_id": c,
                        "category": embedtools.safe_label(names.get(c, str(c))),
                        "images": count,
                    }
                    for c, count in top
                ],
                "file_name": files.get(image_id, ""),
            }
            findings.append(
                make_finding(
                    ctx.run_id, NAME, "INFO", image_id=image_id, ann_id=ann_id, payload=payload
                )
            )
        if cut_images:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "INFO",
                    payload={
                        "rule": "classes_truncated",
                        "count": cut_images,
                        "max_classes": MAX_CARRIED_CLASSES,
                    },
                )
            )
        if dropped:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "INFO",
                    payload={"rule": "suspects_truncated", "count": dropped},
                )
            )
        if not_scored:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    unchecked_severity(
                        not_scored,
                        boxes,
                        max_fraction=cfg.max_unchecked_fraction,
                        require_local=ctx.require_local_images,
                    ),
                    payload={
                        "rule": "annotations_not_scored",
                        "count": not_scored,
                        "total": boxes,
                        "min_similarity": cfg.suspect_min_similarity,
                        "min_neighbours": cfg.suspect_min_neighbours,
                    },
                )
            )
        if done < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", done, n))
        return findings


plugin = LabelNoiseCheck()
