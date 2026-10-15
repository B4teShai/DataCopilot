"""``boxes``: geometric sanity of bounding boxes, vectorised with NumPy.

Per-annotation rules are computed on whole arrays; the pair rules (duplicates and conflicting
classes) run per image on blocks of at most ``PAIR_BLOCK`` columns, so memory is bounded by
``MAX_BOXES_PER_IMAGE_FOR_PAIRS x PAIR_BLOCK`` floats however many boxes a hostile file declares.

Rules (``payload["rule"]``), CRITICAL: ``invalid_box`` (a non-finite coordinate), ``out_of_bounds``
(outside the image), ``zero_area`` (empty or inverted); WARN: ``tiny_box`` (a side below
``tiny_box_side_px`` or an area below ``tiny_box_area_fraction`` of the image),
``extreme_aspect`` (``max(w/h, h/w)`` above ``max_box_aspect_ratio``), ``duplicate_box`` (same
image and class, IoU above ``duplicate_box_iou`` with an earlier box) and ``conflicting_class``
(the same, with a different class). "Earlier" means a lower ``ann_id`` within the image, and each
box is reported once per rule, against its earliest match of that kind (one box can be both a
duplicate and a conflict). ``pairwise_skipped`` (WARN: images with more than
``MAX_BOXES_PER_IMAGE_FOR_PAIRS`` valid boxes are not searched for pairs, so a gate must notice)
and ``boxes_truncated`` (a rule hit
``MAX_FINDINGS_PER_RULE``; carries the number dropped, with the severity of the truncated rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.paths import clip
from datacopilot.core.geometry import iou_xyxy
from datacopilot.core.schema import Dataset, Finding, PluginSpec, Severity

NAME = "boxes"
MAX_FINDINGS_PER_RULE = 10_000
"""Findings kept per rule; the rest are counted in one ``boxes_truncated`` finding."""
MAX_BOXES_PER_IMAGE_FOR_PAIRS = 20_000
"""Largest per-image box count for the pair search (20,000 x 256 x 8 bytes is 40 MB per block)."""
PAIR_BLOCK = 256
"""Columns of the IoU matrix computed at once."""

_SEVERITY: dict[str, Severity] = {
    "invalid_box": "CRITICAL",
    "out_of_bounds": "CRITICAL",
    "zero_area": "CRITICAL",
    "tiny_box": "WARN",
    "extreme_aspect": "WARN",
    "duplicate_box": "WARN",
    "conflicting_class": "WARN",
}


@dataclass
class _Hit:
    """One flagged annotation (row ``index`` into the arrays) with its payload."""

    index: int
    payload: dict[str, object]


def _r(value: float) -> float:
    return round(float(value), 4)


def _pairs(
    order: NDArray[np.int64],
    box: NDArray[np.float64],
    cat: NDArray[np.int64],
    ann_ids: list[str],
    threshold: float,
    hits: dict[str, list[_Hit]],
) -> None:
    """Report, for each box, its earliest same-image match above ``threshold`` (one block of
    columns at a time)."""
    n = len(order)
    b = box[order]
    cb = cat[order]
    for j0 in range(0, n, PAIR_BLOCK):
        j1 = min(n, j0 + PAIR_BLOCK)
        iou = iou_xyxy(b[:j1], b[j0:j1])
        earlier = np.arange(j1)[:, None] < np.arange(j0, j1)[None, :]
        over = earlier & (iou > threshold)
        same_class = cb[:j1, None] == cb[None, j0:j1]
        # Two independent searches: the earliest same-class match is a duplicate, the earliest
        # different-class match a conflict; one box can be both.
        for rule, match in (
            ("duplicate_box", over & same_class),
            ("conflicting_class", over & ~same_class),
        ):
            has = match.any(axis=0)
            first = match.argmax(axis=0)
            for col in np.flatnonzero(has):
                later_row, earlier_row = int(order[j0 + col]), int(order[first[col]])
                payload: dict[str, object] = {
                    "rule": rule,
                    "other_ann_id": clip(ann_ids[earlier_row], 256),
                    "iou": _r(iou[first[col], col]),
                    "category_id": int(cat[later_row]),
                }
                if rule == "conflicting_class":
                    payload["other_category_id"] = int(cat[earlier_row])
                hits[rule].append(_Hit(later_row, payload))


class BoxesCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        if ds.manifest.task != "detection":
            return []
        cfg = ctx.config
        size = {i.image_id: (i.width, i.height) for i in ds.images}
        rows = [
            a
            for a in ds.annotations
            if a.kind == "bbox" and a.bbox_xyxy_abs is not None and a.image_id in size
        ]
        if not rows:
            return []
        ann_ids = [a.ann_id for a in rows]
        image_ids = [a.image_id for a in rows]
        box = np.array([a.bbox_xyxy_abs for a in rows], dtype=np.float64)
        cat = np.array([a.category_id for a in rows], dtype=np.int64)
        dims = np.array([size[i] for i in image_ids], dtype=np.float64)
        width, height = dims[:, 0], dims[:, 1]
        hits: dict[str, list[_Hit]] = {rule: [] for rule in _SEVERITY}
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            finite = np.isfinite(box).all(axis=1)
            w, h = x2 - x1, y2 - y1
            zero = finite & ((w <= 0) | (h <= 0))
            oob = finite & ((x1 < 0) | (y1 < 0) | (x2 > width) | (y2 > height))
            ok = finite & ~zero
            tiny = ok & (
                (np.minimum(w, h) < cfg.tiny_box_side_px)
                | (w * h < cfg.tiny_box_area_fraction * width * height)
            )
            aspect = np.where(ok, np.maximum(w / h, h / w), 0.0)
            extreme = ok & (aspect > cfg.max_box_aspect_ratio)
        for i in np.flatnonzero(~finite):
            hits["invalid_box"].append(_Hit(int(i), {"rule": "invalid_box"}))
        for i in np.flatnonzero(oob):
            hits["out_of_bounds"].append(
                _Hit(
                    int(i),
                    {
                        "rule": "out_of_bounds",
                        "bbox": [_r(v) for v in box[i]],
                        "width": int(width[i]),
                        "height": int(height[i]),
                    },
                )
            )
        for i in np.flatnonzero(zero):
            hits["zero_area"].append(
                _Hit(int(i), {"rule": "zero_area", "bbox": [_r(v) for v in box[i]]})
            )
        for i in np.flatnonzero(tiny):
            hits["tiny_box"].append(
                _Hit(
                    int(i),
                    {
                        "rule": "tiny_box",
                        "width": _r(w[i]),
                        "height": _r(h[i]),
                        "area_fraction": round(float(w[i] * h[i] / (width[i] * height[i])), 6),
                    },
                )
            )
        for i in np.flatnonzero(extreme):
            hits["extreme_aspect"].append(
                _Hit(int(i), {"rule": "extreme_aspect", "aspect_ratio": _r(aspect[i])})
            )
        skipped_images = self._search_pairs(
            ok, image_ids, ann_ids, box, cat, cfg.duplicate_box_iou, hits
        )
        findings: list[Finding] = []
        for rule, items in hits.items():
            items.sort(key=lambda hit: (image_ids[hit.index], ann_ids[hit.index]))
            for hit in items[:MAX_FINDINGS_PER_RULE]:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        _SEVERITY[rule],
                        image_id=image_ids[hit.index],
                        ann_id=ann_ids[hit.index],
                        payload=hit.payload,  # type: ignore[arg-type]
                    )
                )
            if len(items) > MAX_FINDINGS_PER_RULE:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        _SEVERITY[rule],
                        payload={
                            "rule": "boxes_truncated",
                            "truncated_rule": rule,
                            "dropped": len(items) - MAX_FINDINGS_PER_RULE,
                        },
                    )
                )
        if skipped_images:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "WARN",
                    payload={
                        "rule": "pairwise_skipped",
                        "images": skipped_images,
                        "max_boxes": MAX_BOXES_PER_IMAGE_FOR_PAIRS,
                    },
                )
            )
        return findings

    @staticmethod
    def _search_pairs(
        ok: NDArray[np.bool_],
        image_ids: list[str],
        ann_ids: list[str],
        box: NDArray[np.float64],
        cat: NDArray[np.int64],
        threshold: float,
        hits: dict[str, list[_Hit]],
    ) -> int:
        """Run the pair search per image; return the number of images skipped for size."""
        by_image: dict[str, list[int]] = {}
        for i in np.flatnonzero(ok):
            by_image.setdefault(image_ids[int(i)], []).append(int(i))
        skipped = 0
        for members in by_image.values():
            if len(members) < 2:
                continue
            if len(members) > MAX_BOXES_PER_IMAGE_FOR_PAIRS:
                skipped += 1
                continue
            order = np.array(sorted(members, key=ann_ids.__getitem__), dtype=np.int64)
            _pairs(order, box, cat, ann_ids, threshold, hits)
        return skipped


plugin = BoxesCheck()
