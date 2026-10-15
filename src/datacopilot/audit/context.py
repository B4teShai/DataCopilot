"""Inputs shared by every check: identity of the run, filesystem roots, workers and thresholds."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator

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

    @model_validator(mode="after")
    def _ordered(self) -> CheckConfig:
        if self.imbalance_warn_ratio < self.imbalance_info_ratio:
            raise ValueError("imbalance_warn_ratio must not be below imbalance_info_ratio")
        return self


class CheckContext(BaseModel, frozen=True):
    """What a check may use besides the dataset.

    Attributes:
        run_id: owner of every finding the check returns (the CLI ``request_id``).
        root: directory that relative ``Image.file_path`` values resolve against (the parent of
            the ``.dcp`` directory).
        cache_dir: where resumable caches live.
        workers: thread count for the file-reading checks.
        embedder: embedder registry id, if the run uses one (none in the CPU checks).
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
    require_local_images: bool = False
    cache_key: SecretStr | None = None
    in_ci: bool = False
    incomplete_is_critical: bool = True
    use_hash_cache: bool = True
    allow_external_images: bool = False
    config: CheckConfig = Field(default_factory=CheckConfig)
