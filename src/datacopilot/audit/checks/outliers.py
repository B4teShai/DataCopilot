"""``outliers``: images that look unlike the rest of their class.

Needs an embedder. An image's class is the category of most of its boxes (ties: the lowest
category id); for classification data that is its label; an image without annotations has no class
and is counted, not scored. Within each class with at least ``outlier_min_class_images``
embedded images the score is the mean cosine distance (``1 - cosine``) to the ``outlier_k``
nearest images of the *same class*. An image is an outlier when its robust z-score
``(score - median) / scale`` exceeds ``outlier_z``, with ``scale = max(1.4826 * MAD, 0.02)``; the
floor keeps a class of near-identical images from flagging its tiny spread. At most
``max_outliers_per_class`` are reported per class, highest score first (ties: lower image id).
Findings are INFO: an outlier is a lead for review (label error, odd capture), not a defect.

Rules: ``outlier`` (``category_id``, ``category`` cut to 64 characters, ``score``, ``class_median``,
``z``), ``classes_skipped`` (classes too small to score), ``images_without_class``, and the shared
fail-closed rules of ``embedtools.embedding_set``; ``audit_incomplete`` also reports a neighbour
search that ended at ``max_check_seconds`` or a dataset over ``MAX_KNN_ROWS`` images.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import numpy as np

from datacopilot.audit.checks import embedtools
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.core.schema import Dataset, Finding, PluginSpec
from datacopilot.embed import knn

NAME = "outliers"
MIN_SCALE = 0.02
"""Floor of the score spread in cosine-distance units."""
MAD_TO_SIGMA = 1.4826


def _image_classes(ds: Dataset) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for ann in ds.annotations:
        counts[ann.image_id][ann.category_id] += 1
    return {
        image_id: min(per, key=lambda cid: (-per[cid], cid)) for image_id, per in counts.items()
    }


class OutliersCheck:
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
        cfg = ctx.config
        class_of = _image_classes(ds)
        names = {c.id: c.name for c in ds.categories}
        rows_by_class: dict[int, list[int]] = defaultdict(list)
        without_class = 0
        for row, image_id in enumerate(es.ids):
            cid = class_of.get(image_id)
            if cid is None:
                without_class += 1
            else:
                rows_by_class[cid].append(row)
        if without_class:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "INFO",
                    payload={"rule": "images_without_class", "count": without_class},
                )
            )
        deadline = time.monotonic() + cfg.max_check_seconds
        skipped = 0
        scored = 0
        total_rows = sum(len(r) for r in rows_by_class.values())
        for cid in sorted(rows_by_class):
            rows = np.array(rows_by_class[cid], dtype=np.int64)
            if len(rows) < cfg.outlier_min_class_images:
                skipped += 1
                scored += len(rows)
                continue
            sub = es.matrix[rows]
            k = min(cfg.outlier_k, len(rows) - 1)
            parts = list(knn.iter_knn(sub, sub, k, same_set=True, deadline=deadline))
            done = sum(len(b.indices) for b in parts)
            scored += done
            if done < len(rows):
                break
            sims = np.concatenate([b.sims for b in parts]).astype(np.float64)
            score = (1.0 - sims).mean(axis=1)
            median = float(np.median(score))
            scale = max(MAD_TO_SIGMA * float(np.median(np.abs(score - median))), MIN_SCALE)
            z = (score - median) / scale
            flagged = [i for i in range(len(rows)) if z[i] > cfg.outlier_z]
            flagged.sort(key=lambda i: (-score[i], es.ids[int(rows[i])]))
            for i in flagged[: cfg.max_outliers_per_class]:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        "INFO",
                        image_id=es.ids[int(rows[i])],
                        payload={
                            "rule": "outlier",
                            "category_id": cid,
                            "category": embedtools.safe_label(names.get(cid, str(cid))),
                            "score": round(float(score[i]), 4),
                            "class_median": round(median, 4),
                            "z": round(float(z[i]), 2),
                        },
                    )
                )
        if scored < total_rows:
            findings.append(
                embedtools.incomplete_finding(ctx, NAME, "time_budget", scored, total_rows)
            )
        if skipped:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "INFO",
                    payload={
                        "rule": "classes_skipped",
                        "count": skipped,
                        "min_images": cfg.outlier_min_class_images,
                    },
                )
            )
        return findings


plugin = OutliersCheck()
