"""Human descriptions of the first-party findings, keyed by ``check/rule``."""

from __future__ import annotations

from datacopilot.report.text import clean_text

_DESCRIPTIONS: dict[str, str] = {
    "integrity/file_missing": "Image file is missing",
    "integrity/unsafe_path": "Image path is unsafe and was not opened",
    "integrity/not_a_regular_file": "Image path is not a regular file",
    "integrity/unreadable": "Image file cannot be read",
    "integrity/empty_file": "Image file is empty",
    "integrity/invalid_header": "Image header is invalid or truncated",
    "integrity/size_mismatch": "Declared size differs from the image header",
    "integrity/file_too_large": "Image file exceeds the size limit and was not read",
    "integrity/unsupported_format": "Image format is not supported by the header reader",
    "integrity/exif_orientation": "Image has a non-default EXIF orientation",
    "integrity/remote_not_checked": "Remote images were not checked",
    "integrity/external_not_checked": "Images outside the dataset root were not checked",
    "duplicates/exact_duplicate": "Images with identical file content",
    "duplicates/content_changed": "File content no longer matches the image id",
    "duplicates/hash_cache_mismatch": "Hash cache disagreed with the files and was discarded",
    "duplicates/images_not_hashed": "Some images could not be hashed",
    "imbalance/long_tail": "Class distribution is long-tailed",
    "imbalance/class_missing_in_split": "Classes missing from a split",
    "imbalance/class_unused": "Classes without any annotation",
    "boxes/invalid_box": "Box has a non-finite coordinate",
    "boxes/out_of_bounds": "Box lies outside its image",
    "boxes/zero_area": "Box has zero or negative area",
    "boxes/tiny_box": "Box is very small",
    "boxes/extreme_aspect": "Box has an extreme aspect ratio",
    "boxes/duplicate_box": "Duplicate box of the same class",
    "boxes/conflicting_class": "Same box labelled with different classes",
    "boxes/pairwise_skipped": "Pair search skipped for images with too many boxes",
    "boxes/boxes_truncated": "Findings of one rule were truncated",
    "near_duplicates/near_duplicate": "Images with near-identical embeddings",
    "near_duplicates/phash_not_applied": "Perceptual-hash prefilter could not hash some images",
    "leakage/cross_split_near_duplicate": "Near-identical images in different splits",
    "leakage/group_overlap": "A group appears in more than one split",
    "leakage/images_without_group": "Images without a group id are not covered by group overlap",
    "outliers/outlier": "Image is unlike the rest of its class",
    "outliers/classes_skipped": "Classes too small for the outlier check",
    "outliers/images_without_class": "Images without annotations were not scored",
}


_GENERIC: dict[str, str] = {
    "findings_truncated": "Findings of this check were truncated at the per-check cap",
    "audit_incomplete": "The check ended before examining every image (budget reached)",
    "embedder_missing": "No embedder was configured for a check that needs one",
    "embedder_error": "The embedder failed; images were not compared",
    "images_not_embedded": "Some images have no embedding and were not compared",
    "embedding_cache_rejected": "Part of the embedding cache failed verification and was ignored",
    "embedding_cache_content_changed": "Image files changed since their embeddings were cached",
    "embedding_cache_unverified": "Cached embeddings could not be verified and were not used",
    "embedding_cache_mismatch": "Cached embeddings disagreed with the images and were discarded",
}


def describe_rule(check: str, rule: str) -> str:
    """Description of a rule; unknown (plugin) rules get a bounded, cleaned fallback."""
    known = _DESCRIPTIONS.get(f"{check}/{rule}")
    if known is not None:
        return known
    generic = _GENERIC.get(rule)
    return generic if generic is not None else f"{check}: {clean_text(rule, 64)}"
