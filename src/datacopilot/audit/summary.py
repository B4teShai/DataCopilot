"""Dataset summary stored with every run (feeds the report's histogram and split-by-class table)."""

from __future__ import annotations

from collections import Counter

from datacopilot.core.schema import MAX_SUMMARY_NAME_CHARS, Dataset, Summary, SummaryClass

MAX_SUMMARY_CLASSES = 2_000
"""Classes listed in a summary; a dataset with more keeps only its totals."""


def build_summary(ds: Dataset) -> Summary:
    """Totals, images per split and annotations per (split, class) of ``ds``."""
    split_of = {i.image_id: i.split for i in ds.images}
    split_images: Counter[str] = Counter(split_of.values())
    omitted = len(ds.categories) > MAX_SUMMARY_CLASSES
    counts: dict[str, Counter[int]] = {}
    if not omitted:
        for ann in ds.annotations:
            counts.setdefault(split_of[ann.image_id], Counter())[ann.category_id] += 1
    return Summary(
        task=ds.manifest.task,
        image_count=len(ds.images),
        annotation_count=len(ds.annotations),
        split_images=dict(split_images),
        classes=()
        if omitted
        else tuple(
            SummaryClass(id=c.id, name=c.name[:MAX_SUMMARY_NAME_CHARS])
            for c in sorted(ds.categories, key=lambda c: c.id)
        ),
        classes_omitted=omitted,
        class_counts={
            split: {str(cid): n for cid, n in sorted(per.items())} for split, per in counts.items()
        },
    )
