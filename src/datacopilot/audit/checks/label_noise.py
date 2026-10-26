"""``label_noise``: annotations whose class their nearest look-alikes do not share.

Needs an embedder. It reports leads for a human; it never edits or proposes a change. Two rules
share one scoring core; every finding says which one produced it (``basis``).

**Crop rule (default, ``suspect_basis = "crop"``, detection datasets).** Each box large enough is
cropped and embedded (``audit/crops.py``: padding, size floor, per-image cap and crop size are
documented there and in the settings). For each crop the ``suspect_k`` nearest other crops by exact
cosine similarity are found; a neighbour *carries* the classes of the boxes it was cut from. A box
of class ``c`` is suspect when at most ``suspect_max_support`` of its usable neighbours carry
``c``. Boxes that were not cropped (too small, over the per-image cap), whose crop could not be
embedded, or that have fewer than ``suspect_min_neighbours`` neighbours at or above
``suspect_min_similarity`` are counted by reason in ``annotations_not_scored``.

**Image rule (``suspect_basis = "image"``, classification datasets, and the fallback).** The base
set is every embedded image that has at least one annotation; a neighbour image carries class
``c`` when any of its annotations has that class (at most ``MAX_CARRIED_CLASSES`` distinct
classes, the lowest ids, count per image; the images cut are counted in ``classes_truncated``).
The embedding describes the whole photo, so for a photo with many objects the signal is weak.
When the crop rule was asked for but crops cannot be cut (Pillow missing, a provider without crop
support, more boxes than the crop cap), the image rule runs instead and a WARN
``crop_rule_unavailable`` finding names the reason.

Common to both: neighbours below ``suspect_min_similarity`` (with ``knn.SIM_EPSILON`` of slack)
are dropped; the search is pinned to the NumPy backend and chunked, so the result does not depend
on whether FAISS is installed and memory stays near one block. Suspects rank by
``(support, -mean_similarity, ann_id)``; at most ``max_suspects_per_image``,
``max_suspects_per_class`` and ``MAX_SUSPECTS`` are kept and the rest counted in
``suspects_truncated``. The result is deterministic and independent of the row order.

Rules: ``suspect_class`` (``basis``, ``category_id``, ``category`` cut to 64 characters,
``support``, ``neighbours``, ``mean_similarity``, ``neighbour_classes`` as the five most common
classes among the neighbours with their ``images`` or ``boxes`` count, ``file_name``),
``annotations_not_scored`` (WARN, CRITICAL above ``max_unchecked_fraction`` of the boxes or with
``--require-local-images``; ``reasons`` for the crop rule), ``crop_rule_unavailable``,
``classes_truncated``, ``suspects_truncated``, ``skipped`` and the shared fail-closed rules of
``embedtools``. ``skipped`` carries a ``reason``: ``embedder_missing``, ``no_embeddings``,
``too_many_images``, ``too_few_images`` or ``too_few_boxes`` (WARN) and, as there is nothing to
check, ``no_annotations`` and ``single_class`` (INFO). ``audit_incomplete`` also reports a
neighbour search that ended at ``max_check_seconds``.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from datacopilot.audit.checks import embedtools
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.audit.crops import CropPlan, Crops
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


@dataclass
class _Scored:
    kept: list[Candidate] = field(default_factory=list)
    dropped: int = 0
    not_scored: int = 0
    done: int = 0


def _score(
    matrix: NDArray[np.float32],
    carried: Sequence[frozenset[int]],
    anns: Sequence[Sequence[Annotation]],
    image_ids: Sequence[str],
    cfg: CheckConfig,
) -> _Scored:
    """Neighbour support of every annotation of every row (an image or a crop) and the capped,
    ranked suspects. Rows that run out of time are not counted in ``done``."""
    n = len(carried)
    out = _Scored()
    deadline = time.monotonic() + cfg.max_check_seconds
    floor = cfg.suspect_min_similarity - knn.SIM_EPSILON
    per_class: dict[int, list[Candidate]] = defaultdict(list)
    held = 0
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
            out.done += 1
            usable = [
                (int(j), float(sim))
                for j, sim in zip(block.indices[offset], block.sims[offset], strict=True)
                if sim >= floor
            ]
            if len(usable) < cfg.suspect_min_neighbours:
                out.not_scored += len(anns[row])
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
                    image_ids[row],
                    len(usable),
                    (),
                )
                for a in anns[row]
                if histogram[a.category_id] / len(usable) <= cfg.suspect_max_support
            ]
            if not suspects:
                continue
            best = heapq.nsmallest(cfg.max_suspects_per_image, suspects, key=_rank)
            out.dropped += len(suspects) - len(best)
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
                    out.dropped += cut
                    held -= cut
            if held > 2 * MAX_SUSPECTS:
                everyone = sorted((c for b in per_class.values() for c in b), key=_rank)
                out.dropped += len(everyone) - MAX_SUSPECTS
                per_class.clear()
                for cand in everyone[:MAX_SUSPECTS]:
                    per_class[cand[3]].append(cand)
                held = MAX_SUSPECTS
        if stopped:
            break
    for bucket in per_class.values():
        bucket.sort(key=_rank)
        out.dropped += max(0, len(bucket) - cfg.max_suspects_per_class)
        out.kept.extend(bucket[: cfg.max_suspects_per_class])
    out.kept.sort(key=_rank)
    if len(out.kept) > MAX_SUSPECTS:
        out.dropped += len(out.kept) - MAX_SUSPECTS
        del out.kept[MAX_SUSPECTS:]
    return out


def _report(
    ds: Dataset, ctx: CheckContext, scored: _Scored, basis: str, unit: str
) -> list[Finding]:
    names = {c.id: c.name for c in ds.categories}
    files = {i.image_id: display_name(i.file_path) for i in ds.images}
    findings: list[Finding] = []
    for support, neg_sim, ann_id, cid, image_id, neighbours, top in scored.kept:
        payload: dict[str, JsonValue] = {
            "rule": "suspect_class",
            "basis": basis,
            "category_id": cid,
            "category": embedtools.safe_label(names.get(cid, str(cid))),
            "support": round(support, 3),
            "neighbours": neighbours,
            "mean_similarity": round(-neg_sim, 4),
            "neighbour_classes": [
                {
                    "category_id": c,
                    "category": embedtools.safe_label(names.get(c, str(c))),
                    unit: count,
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
    if scored.dropped:
        findings.append(
            make_finding(
                ctx.run_id,
                NAME,
                "INFO",
                payload={"rule": "suspects_truncated", "count": scored.dropped},
            )
        )
    return findings


def _not_scored(
    ctx: CheckContext, count: int, boxes: int, reasons: dict[str, int] | None
) -> Finding:
    cfg = ctx.config
    payload: dict[str, JsonValue] = {
        "rule": "annotations_not_scored",
        "count": count,
        "total": boxes,
        "min_similarity": cfg.suspect_min_similarity,
        "min_neighbours": cfg.suspect_min_neighbours,
    }
    if reasons is not None:
        payload["reasons"] = {r: n for r, n in sorted(reasons.items()) if n}
    return make_finding(
        ctx.run_id,
        NAME,
        unchecked_severity(
            count,
            boxes,
            max_fraction=cfg.max_unchecked_fraction,
            require_local=ctx.require_local_images,
        ),  # fmt: skip
        payload=payload,
    )


class LabelNoiseCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="2"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        cfg = ctx.config
        provider = ctx.embeddings
        fallback: str | None = None
        if cfg.suspect_basis == "crop" and provider is not None and ds.manifest.task == "detection":
            crops_of = getattr(provider, "crops", None)
            if crops_of is None:
                fallback = "provider_without_crops"
            else:
                plan = CropPlan(cfg.crop_min_side_px, cfg.crop_padding, cfg.max_crops_per_image)
                result: Crops = crops_of(ds, ctx, plan)
                if result.unavailable is None:
                    return self._crop_rule(ds, ctx, result)
                fallback = result.unavailable
        findings = self._image_rule(ds, ctx)
        if fallback is not None:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "WARN",
                    payload={"rule": "crop_rule_unavailable", "reason": fallback, "basis": "image"},
                )
            )
        return findings

    def _crop_rule(self, ds: Dataset, ctx: CheckContext, result: Crops) -> list[Finding]:
        cfg = ctx.config
        findings = embedtools.qualify(result.es, ctx, NAME, report_missing=False)
        es = result.es
        by_id = {a.ann_id: a for a in ds.annotations}
        anns_of_key: dict[str, list[Annotation]] = defaultdict(list)
        for ann_id, key in sorted(result.key_of.items()):
            anns_of_key[key].append(by_id[ann_id])
        reasons: Counter[str] = Counter(result.skipped)
        for reason, keys in es.missing.items():
            reasons[reason] += sum(len(anns_of_key[k]) for k in keys)
        if result.boxes == 0:
            return [*findings, _skipped(ctx, "no_annotations")]
        if not es.ids and es.missing:
            findings.append(_not_scored(ctx, result.boxes, result.boxes, dict(reasons)))
            return [*findings, _skipped(ctx, "no_embeddings")]
        n = len(es.ids)
        if n < cfg.suspect_min_images:  # like too_few_images: a skip, not a count of boxes
            return [
                *findings,
                _skipped(
                    ctx,
                    "too_few_boxes",
                    boxes=n,
                    min_boxes=cfg.suspect_min_images,
                    not_cropped=dict(sorted(result.skipped.items())),
                ),
            ]
        anns = [anns_of_key[k] for k in es.ids]
        carried = [frozenset(a.category_id for a in row) for row in anns]
        if len(frozenset().union(*carried)) < 2:
            return [*findings, _skipped(ctx, "single_class", boxes=n)]
        scored = _score(es.matrix, carried, anns, [result.image_of[k] for k in es.ids], cfg)
        findings += _report(ds, ctx, scored, "crop", "boxes")
        reasons["too_few_neighbours"] += scored.not_scored
        not_scored = sum(reasons.values())
        if not_scored:
            findings.append(_not_scored(ctx, not_scored, result.boxes, dict(reasons)))
        if scored.done < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", scored.done, n))
        return findings

    def _image_rule(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
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
        matrix = es.matrix[np.array(rows, dtype=np.int64)]
        anns = [anns_of[i] for i in image_ids]
        scored = _score(matrix, carried, anns, image_ids, cfg)
        findings += _report(ds, ctx, scored, "image", "images")
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
        if scored.not_scored:
            boxes = sum(len(a) for a in anns)
            findings.append(_not_scored(ctx, scored.not_scored, boxes, None))
        if scored.done < n:
            findings.append(embedtools.incomplete_finding(ctx, NAME, "time_budget", scored.done, n))
        return findings


plugin = LabelNoiseCheck()
