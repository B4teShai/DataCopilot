"""Inputs shared by every check: identity of the run, filesystem roots, workers and thresholds."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator

from datacopilot.audit.embedding import EmbeddingProvider

MAX_WORKERS = 256
"""Upper bound for ``workers``; a typo such as 100000 must not fork a hundred thousand processes."""


class CheckConfig(BaseModel, frozen=True, extra="forbid"):
    """Thresholds of the first-party checks. Defaults are conservative; all are overridable.

    Attributes:
        imbalance_info_ratio: long-tail ratio (largest class count over smallest) at which the
            imbalance check reports INFO.
        imbalance_warn_ratio: ratio at which it reports WARN; not below ``imbalance_info_ratio``.
        tiny_box_side_px: a box whose width or height is below this many pixels is "tiny".
        tiny_box_area_fraction: a box covering less than this fraction of its image is "tiny".
        max_box_aspect_ratio: ``max(w/h, h/w)`` above which a box is "extreme".
        duplicate_box_iou: same-class boxes of one image with IoU above this are duplicates.
        max_image_bytes: files larger than this are reported, never hashed or read.
        max_check_seconds: wall-time budget of one file-reading check; images not reached are
            reported in an ``audit_incomplete`` finding.
        max_hash_bytes: total bytes the duplicates check may hash in one run (same finding).
        max_unchecked_fraction: when more than this fraction of the images could not be checked
            (remote, outside the root, unhashable) the finding is CRITICAL, else WARN.
        max_embed_image_bytes: files larger than this are not read for embedding (reported).
        near_duplicate_cosine: image pairs whose embeddings have at least this cosine similarity
            are near-duplicates.
        leakage_cosine: cross-split pairs at or above this cosine similarity are leakage.
        knn_k: neighbours examined per image by the near-duplicate and leakage checks.
        near_duplicate_phash_max_distance: when set, a candidate pair is kept only if the 64-bit
            difference hashes of its two images differ in at most this many bits (a pair that cannot
            be hashed is kept).
        outlier_k: neighbours (within the class) whose mean distance is the outlier score.
        outlier_z: robust z-score (median and MAD) above which an image is an outlier.
        outlier_min_class_images: classes with fewer embedded images are skipped.
        max_outliers_per_class: outliers reported per class (highest score first).
        suspect_k: neighbouring images examined per annotation by the ``label_noise`` check.
        suspect_min_similarity: neighbours with a lower cosine similarity are not used.
        suspect_min_neighbours: an annotation with fewer usable neighbours is not scored (counted).
        suspect_max_support: an annotation is suspect when at most this fraction of its usable
            neighbours carry its class (0.0: none of them does).
        suspect_min_images: fewer annotated, embedded images than this skips the check (reported).
        max_suspects_per_image: suspect annotations reported per image (the rest are counted).
        max_suspects_per_class: suspect annotations reported per class (the rest are counted).
    """

    imbalance_info_ratio: float = Field(default=3.0, gt=1.0, le=1e9, allow_inf_nan=False)
    imbalance_warn_ratio: float = Field(default=10.0, gt=1.0, le=1e9, allow_inf_nan=False)
    tiny_box_side_px: float = Field(default=2.0, gt=0.0, le=1e6, allow_inf_nan=False)
    tiny_box_area_fraction: float = Field(default=1e-4, gt=0.0, lt=1.0, allow_inf_nan=False)
    max_box_aspect_ratio: float = Field(default=20.0, gt=1.0, le=1e9, allow_inf_nan=False)
    duplicate_box_iou: float = Field(default=0.9, gt=0.0, le=1.0, allow_inf_nan=False)
    max_image_bytes: int = Field(default=1024**3, ge=1)
    max_check_seconds: float = Field(default=1800.0, gt=0.0, le=1e7, allow_inf_nan=False)
    max_hash_bytes: int = Field(default=200 * 1024**3, ge=1)
    max_unchecked_fraction: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    max_embed_image_bytes: int = Field(default=32 * 1024**2, ge=1)
    near_duplicate_cosine: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    leakage_cosine: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    knn_k: int = Field(default=10, ge=1, le=100)
    near_duplicate_phash_max_distance: int | None = Field(default=None, ge=0, le=64)
    outlier_k: int = Field(default=5, ge=1, le=100)
    outlier_z: float = Field(default=3.5, gt=0.0, le=100.0, allow_inf_nan=False)
    outlier_min_class_images: int = Field(default=8, ge=3, le=1_000_000)
    max_outliers_per_class: int = Field(default=20, ge=1, le=10_000)
    suspect_k: int = Field(default=10, ge=1, le=100)
    suspect_min_similarity: float = Field(default=0.3, gt=0.0, le=1.0, allow_inf_nan=False)
    suspect_min_neighbours: int = Field(default=3, ge=1, le=100)
    suspect_max_support: float = Field(default=0.0, ge=0.0, lt=1.0, allow_inf_nan=False)
    suspect_min_images: int = Field(default=30, ge=5, le=1_000_000)
    max_suspects_per_image: int = Field(default=3, ge=1, le=1_000)
    max_suspects_per_class: int = Field(default=20, ge=1, le=10_000)

    @model_validator(mode="after")
    def _ordered(self) -> CheckConfig:
        if self.imbalance_warn_ratio < self.imbalance_info_ratio:
            raise ValueError("imbalance_warn_ratio must not be below imbalance_info_ratio")
        if self.suspect_min_neighbours > self.suspect_k:
            raise ValueError("suspect_min_neighbours must not exceed suspect_k")
        return self


class CheckContext(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """What a check may use besides the dataset.

    Attributes:
        run_id: owner of every finding the check returns (the CLI ``request_id``).
        root: directory that relative ``Image.file_path`` values resolve against (the parent of
            the ``.dcp`` directory).
        cache_dir: where resumable caches live.
        workers: thread count for the file-reading checks.
        embedder: embedder registry id, if the run uses one (none in the CPU checks).
        embeddings: the provider of image embeddings for the embedding checks (computes once per
            audit and caches in memory); None when no embedder was requested.
        show_group_labels: include the (cleaned, cut) group label in ``group_overlap`` findings;
            by default only the hashed ``group_ref`` is shown, because group ids come from file
            paths and can carry names.
        dataset_dir: the ``dataset.dcp`` directory, where the embedding cache lives; None means no
            persistent embedding cache.
        include_model_annotations: audit rows with ``provenance == "model"`` too; by default they
            are left out of every check (model output is not ground truth) and counted in the
            summary.
        require_local_images: any image that a check could not examine is CRITICAL.
        cache_key: out-of-band MAC secret of the hash cache (``DATACOPILOT_CACHE_KEY``).
        incomplete_is_critical: severity of ``audit_incomplete`` (the runner sets WARN when no gate
            is applied).
        in_ci: a CI environment; the hash cache then needs ``cache_key``.
        use_hash_cache: False neither reads nor writes the hash cache (use in CI gates).
        allow_external_images: when False (default) images whose real path lies outside ``root``
            are neither opened nor hashed; they are counted in an INFO finding. Remote URIs are
            never fetched.
        config: thresholds.
    """

    run_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    root: Path
    cache_dir: Path
    workers: int = Field(default=1, ge=1, le=MAX_WORKERS)
    embedder: str | None = None
    embeddings: EmbeddingProvider | None = Field(default=None, exclude=True)
    dataset_dir: Path | None = None
    show_group_labels: bool = False
    require_local_images: bool = False
    include_model_annotations: bool = False
    cache_key: SecretStr | None = None
    in_ci: bool = False
    incomplete_is_critical: bool = True
    use_hash_cache: bool = True
    allow_external_images: bool = False
    config: CheckConfig = Field(default_factory=CheckConfig)
