"""``imbalance``: class balance per split.

For each split (train, val, test, unassigned) the per-class annotation counts give:

* the long-tail ratio, ``max count / min count`` over the classes that occur in the split;
* the effective number of classes, ``exp(H)`` where ``H`` is the Shannon entropy of the class
  distribution (``K`` for a perfectly balanced split of ``K`` classes, ``1`` when one class holds
  everything).

Rules (``payload["rule"]``): ``long_tail`` (WARN at ``imbalance_warn_ratio``, INFO at
``imbalance_info_ratio``, nothing below), ``class_missing_in_split`` (WARN: a train, val or test
split that has annotations lacks classes that occur in another split) and ``class_unused`` (INFO:
categories with no annotation anywhere). Class ids in payloads are capped at ``MAX_IDS_LISTED``;
``count`` is exact.
"""

from __future__ import annotations

import math
from collections import Counter

from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.paths import clip
from datacopilot.core.schema import Dataset, Finding, PluginSpec, Severity

NAME = "imbalance"
TOP_K = 5
"""Head and tail classes listed in a ``long_tail`` payload."""
MAX_IDS_LISTED = 100
"""Class ids listed in a ``class_missing_in_split`` or ``class_unused`` payload."""
_SPLITS = ("train", "val", "test", "unassigned")
_REAL_SPLITS = ("train", "val", "test")


def effective_classes(counts: list[int]) -> float:
    """``exp`` of the Shannon entropy of ``counts`` (zeros ignored); 0.0 when all are zero."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return math.exp(-sum(c / total * math.log(c / total) for c in counts if c > 0))


def split_class_counts(ds: Dataset) -> dict[str, Counter[int]]:
    """``split -> category_id -> annotations`` for every split that has at least one annotation."""
    split_of = {i.image_id: i.split for i in ds.images}
    out: dict[str, Counter[int]] = {}
    for ann in ds.annotations:
        out.setdefault(split_of[ann.image_id], Counter())[ann.category_id] += 1
    return out


class ImbalanceCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        names = {c.id: c.name for c in ds.categories}
        counts = split_class_counts(ds)
        cfg = ctx.config
        findings: list[Finding] = []
        present_anywhere: set[int] = set().union(*counts.values()) if counts else set()
        for split in _SPLITS:
            per_class = counts.get(split)
            if not per_class:
                continue
            values = list(per_class.values())
            ratio = max(values) / min(values)
            severity: Severity | None = None
            if ratio >= cfg.imbalance_warn_ratio:
                severity = "WARN"
            elif ratio >= cfg.imbalance_info_ratio:
                severity = "INFO"
            if severity is not None:
                ranked = sorted(per_class.items(), key=lambda kv: (-kv[1], kv[0]))
                head, tail = ranked[:TOP_K], sorted(ranked[-TOP_K:], key=lambda kv: (kv[1], kv[0]))
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        severity,
                        payload={
                            "rule": "long_tail",
                            "split": split,
                            "ratio": round(ratio, 4),
                            "effective_classes": round(effective_classes(values), 4),
                            "classes_present": len(per_class),
                            "total": sum(values),
                            "head": [
                                {"id": c, "name": clip(names.get(c, "")), "count": n}
                                for c, n in head
                            ],
                            "tail": [
                                {"id": c, "name": clip(names.get(c, "")), "count": n}
                                for c, n in tail
                            ],
                        },
                    )
                )
        for split in _REAL_SPLITS:
            per_class = counts.get(split)
            if not per_class:
                continue
            missing = sorted(present_anywhere - set(per_class))
            if missing:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        "WARN",
                        payload={
                            "rule": "class_missing_in_split",
                            "split": split,
                            "count": len(missing),
                            "class_ids": list(missing[:MAX_IDS_LISTED]),
                        },
                    )
                )
        unused = sorted(c.id for c in ds.categories if c.id not in present_anywhere)
        if unused:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "INFO",
                    payload={
                        "rule": "class_unused",
                        "count": len(unused),
                        "class_ids": list(unused[:MAX_IDS_LISTED]),
                    },
                )
            )
        return findings


plugin = ImbalanceCheck()
