"""``label_noise``: annotations whose class no neighbouring image shares.

Needs an embedder. It uses the image embeddings the other embedding checks already share (no new
model, no new dependency) and reports leads for a human; it never edits or proposes a change.

Rule (``suspect_class``). The base set is every embedded image that has at least one annotation
(an unannotated image says nothing about classes, so it is never a neighbour). For each base image
the ``suspect_k`` nearest other base images by exact cosine similarity are found
(``embed/knn.py``: chunked, so memory stays near one block). Neighbours below
``suspect_min_similarity`` are dropped; an image left with fewer than ``suspect_min_neighbours``
is not scored and its annotations are counted (``annotations_not_scored``). A neighbour *carries*
class ``c`` when any of its annotations has that class. The ``support`` of an annotation of class
``c`` is the fraction of the usable neighbours that carry ``c``; the annotation is suspect when
``support <= suspect_max_support`` (default 0.0: no neighbour carries it).

The embedding describes the whole image, not the box, so for a photo with many objects the signal
is weak: a suspect annotation means "images that look like this one do not contain this class",
which is also true of a genuinely rare object. Findings are INFO leads, never defects.

Order and caps: suspects sort by ``(support, -mean_similarity, ann_id)``; at most
``max_suspects_per_image`` per image and ``max_suspects_per_class`` per class are reported and the
rest are counted in ``suspects_truncated``. The result is deterministic and independent of the
order of the rows.

Rules: ``suspect_class`` (``category_id``, ``category`` cut to 64 characters, ``support``,
``neighbours``, ``mean_similarity``, ``neighbour_classes`` as the five most common classes among
the neighbours with their image counts, ``file_name``), ``annotations_not_scored``,
``suspects_truncated``, ``skipped`` (``reason``: ``embedder_missing``, ``no_embeddings``,
``no_annotations``, ``too_few_images``, ``single_class``) and the shared fail-closed rules of
``embedtools.embedding_set``. A skipped check is a WARN, never an empty result;
``audit_incomplete`` also reports a neighbour search that ended at ``max_check_seconds`` or a
dataset over ``MAX_KNN_ROWS`` images.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import numpy as np
from pydantic import JsonValue

from datacopilot.audit.checks import embedtools
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.paths import display_name
from datacopilot.core.schema import Annotation, Dataset, Finding, PluginSpec
from datacopilot.embed import knn

NAME = "label_noise"
TOP_NEIGHBOUR_CLASSES = 5
"""Classes listed in the neighbour histogram of one finding."""

Candidate = tuple[float, float, str, int, str, dict[str, JsonValue]]
"""``(support, -mean_similarity, ann_id, category_id, image_id, payload)``: the first three are
the sort key (``ann_id`` is unique, so the order is total)."""


def _skipped(ctx: CheckContext, reason: str, **extra: JsonValue) -> Finding:
    return make_finding(
        ctx.run_id, NAME, "WARN", payload={"rule": "skipped", "reason": reason, **extra}
    )


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
            return [*findings, _skipped(ctx, "no_embeddings")]
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
        carried = [frozenset(a.category_id for a in anns_of[i]) for i in image_ids]
        if len(frozenset().union(*carried)) < 2:
            return [*findings, _skipped(ctx, "single_class", images=n)]
        if n > knn.MAX_KNN_ROWS:
            return [*findings, embedtools.incomplete_finding(ctx, NAME, "too_many_images", 0, n)]
        names = {c.id: c.name for c in ds.categories}
        files = {i.image_id: display_name(i.file_path) for i in ds.images}
        matrix = es.matrix[np.array(rows, dtype=np.int64)]
        deadline = time.monotonic() + cfg.max_check_seconds
        per_class: dict[int, list[Candidate]] = defaultdict(list)
        dropped = 0
        not_scored = 0
        done = 0
        for block in knn.iter_knn(
            matrix, matrix, min(cfg.suspect_k, n - 1), same_set=True, deadline=deadline
        ):
            for offset in range(len(block.indices)):
                row = block.start + offset
                done += 1
                usable = [
                    (int(j), float(s))
                    for j, s in zip(block.indices[offset], block.sims[offset], strict=True)
                    if s >= cfg.suspect_min_similarity
                ]
                image_id = image_ids[row]
                if len(usable) < cfg.suspect_min_neighbours:
                    not_scored += len(anns_of[image_id])
                    continue
                histogram: Counter[int] = Counter()
                for j, _ in usable:
                    histogram.update(carried[j])
                mean_sim = sum(s for _, s in usable) / len(usable)
                top = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
                top_classes: list[JsonValue] = [
                    {
                        "category_id": cid,
                        "category": embedtools.safe_label(names.get(cid, str(cid))),
                        "images": count,
                    }
                    for cid, count in top[:TOP_NEIGHBOUR_CLASSES]
                ]
                suspects: list[Candidate] = []
                for ann in anns_of[image_id]:
                    support = histogram[ann.category_id] / len(usable)
                    if support > cfg.suspect_max_support:
                        continue
                    payload: dict[str, JsonValue] = {
                        "rule": "suspect_class",
                        "category_id": ann.category_id,
                        "category": embedtools.safe_label(
                            names.get(ann.category_id, str(ann.category_id))
                        ),
                        "support": round(support, 3),
                        "neighbours": len(usable),
                        "mean_similarity": round(mean_sim, 4),
                        "neighbour_classes": top_classes,
                        "file_name": files.get(image_id, ""),
                    }
                    suspects.append(
                        (support, -mean_sim, ann.ann_id, ann.category_id, image_id, payload)
                    )
                suspects.sort(key=lambda c: c[:3])
                dropped += max(0, len(suspects) - cfg.max_suspects_per_image)
                for cand in suspects[: cfg.max_suspects_per_image]:
                    bucket = per_class[cand[3]]
                    bucket.append(cand)
                    if len(bucket) > 2 * cfg.max_suspects_per_class:
                        bucket.sort(key=lambda c: c[:3])
                        dropped += len(bucket) - cfg.max_suspects_per_class
                        del bucket[cfg.max_suspects_per_class :]
        kept: list[Candidate] = []
        for bucket in per_class.values():
            bucket.sort(key=lambda c: c[:3])
            dropped += max(0, len(bucket) - cfg.max_suspects_per_class)
            kept.extend(bucket[: cfg.max_suspects_per_class])
        kept.sort(key=lambda c: c[:3])
        for _, _, ann_id, _, image_id, payload in kept:
            findings.append(
                make_finding(
                    ctx.run_id, NAME, "INFO", image_id=image_id, ann_id=ann_id, payload=payload
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
                    "INFO",
                    payload={
                        "rule": "annotations_not_scored",
                        "count": not_scored,
                        "min_similarity": cfg.suspect_min_similarity,
                        "min_neighbours": cfg.suspect_min_neighbours,
                    },
                )
            )
        if done < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", done, n))
        return findings


plugin = LabelNoiseCheck()
