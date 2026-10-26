"""Box-crop embeddings for ``label_noise``: one vector per box instead of one per photo.

**Which boxes.** Every ``bbox`` annotation whose shorter side is at least ``min_side`` pixels is
cropped; per image at most ``per_image`` boxes are, the largest first (ties by ``ann_id``), and the
rest are counted (``too_small``, ``over_image_cap``). The crop is the box grown by ``padding`` of
its width and height on each side (some context helps a model recognise the object), clamped to
the image and rounded outwards to whole pixels. A crop whose long side exceeds ``MAX_CROP_SIDE`` is
downscaled to it (bicubic) before it is encoded, so one crop is a small, bounded PNG whatever the
image size; the embedder then preprocesses it like any image.

**Keys and cache.** A crop's key is the first 16 hex digits of
``sha256(image_id, crop rectangle, geometry tag)``; two boxes with the same rectangle share one
crop. Vectors are stored in the existing authenticated embedding cache under a *variant* named by
the geometry (``CropPlan.variant``), so a cache made with another padding, size floor, cap or crop
size is a different cache and never read as this one. The trust rules are those of the image
pipeline (``audit/embedding.py``): every cached crop carries the sha256 of its source image and
the image file is hashed again before the vector is used (a changed file discards the cache), and
a deterministic sample of cached crops is cut and embedded again and must agree (``SPOT_*``).

**Reading.** Images are read through ``read_verified`` (contained path, no symlinks, size cap,
bytes must hash to the image id) and decoded once per image with the bounded decoder
(``embed/imaging.py``); the decoded size must equal the row's ``width`` and ``height``, otherwise
the box coordinates do not describe these pixels and the image's crops are not embedded
(``size_mismatch``). Every planned crop ends with a vector or a named reason, as in the image
pipeline; nothing is dropped silently.

Pillow is needed to cut crops; without it the result says ``unavailable = "pillow_missing"`` and
the check falls back to the whole-image rule and says so.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from datacopilot.audit import pool
from datacopilot.audit.embedding import (
    MAX_BATCH_BYTES,
    MAX_READ_WINDOW,
    SPOT_MIN_COSINE,
    SPOT_MIN_VERIFIED_FRACTION,
    EmbeddingSet,
    hash_image,
    pick_sample,
    read_verified,
)
from datacopilot.audit.paths import safe_text
from datacopilot.core.errors import DataCopilotError, ModelError
from datacopilot.core.schema import Dataset, Image
from datacopilot.embed import cache as embed_cache
from datacopilot.embed import imaging
from datacopilot.embed.protocol import Embedder, ImageDecodeError
from datacopilot.logging import get_logger

if TYPE_CHECKING:
    from datacopilot.audit.context import CheckContext

CROP_FORMAT_VERSION = 1
"""Bumped when cutting or encoding crops changes their pixels (it is part of the cache variant)."""
MAX_CROP_SIDE = 512
"""Longest side of an encoded crop; larger crops are downscaled (models take 224 to 518 pixels)."""
MAX_CROPS = 500_000
"""Crops one run may plan (the neighbour search's row cap); more refuses the crop rule."""
_DECODE_REASONS = frozenset({"decode_failed", "too_many_pixels", "multi_frame"})

Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class CropPlan:
    """Crop geometry: shorter-side floor in pixels, padding fraction, boxes per image."""

    min_side: int
    padding: float
    per_image: int

    @property
    def variant(self) -> str:
        """Cache variant name; every parameter that changes the pixels is in it."""
        return (
            f"crop-v{CROP_FORMAT_VERSION}-p{round(self.padding * 1000)}-m{self.min_side}"
            f"-c{self.per_image}-s{MAX_CROP_SIDE}"
        )


@dataclass(frozen=True)
class _Job:
    image: Image
    crops: tuple[tuple[str, Rect], ...]  # (key, rectangle), unique keys


@dataclass(frozen=True)
class Crops:
    """Crop embeddings of one dataset and the account of every box.

    Attributes:
        es: the vectors in the image pipeline's form; ``ids`` are crop keys, ``total`` the planned
            crops and ``missing`` the crops without a vector by reason.
        key_of: ``ann_id -> crop key`` for every planned box.
        image_of: ``crop key -> image_id``.
        skipped: boxes that were not planned, by reason (``too_small``, ``over_image_cap``).
        boxes: ``bbox`` annotations considered.
        unavailable: why no crop could be cut at all (``pillow_missing``, ``too_many_boxes``);
            the caller then uses the whole-image rule and reports it.
    """

    es: EmbeddingSet
    key_of: dict[str, str]
    image_of: dict[str, str]
    skipped: dict[str, int]
    boxes: int
    unavailable: str | None = None


def crop_rect(
    box: tuple[float, float, float, float], width: int, height: int, padding: float
) -> Rect:
    """The padded box clamped to the image and rounded outwards to whole pixels."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, math.floor(x1 - px))
    top = max(0, math.floor(y1 - py))
    right = min(width, math.ceil(x2 + px))
    bottom = min(height, math.ceil(y2 + py))
    return left, top, max(right, left + 1), max(bottom, top + 1)


def _key(image_id: str, rect: Rect, variant: str) -> str:
    text = f"{image_id}\n{rect[0]},{rect[1]},{rect[2]},{rect[3]}\n{variant}"
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


def plan_crops(
    ds: Dataset, plan: CropPlan
) -> tuple[list[_Job], dict[str, str], dict[str, str], dict[str, int], int]:
    """The crops to embed, per image (sorted by image id), and the boxes left out by reason."""
    by_image: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for ann in ds.annotations:
        if ann.bbox_xyxy_abs is not None:
            by_image[ann.image_id].append((ann.ann_id, ann.bbox_xyxy_abs))
    skipped: dict[str, int] = defaultdict(int)
    key_of: dict[str, str] = {}
    image_of: dict[str, str] = {}
    jobs: list[_Job] = []
    boxes = 0
    for image in sorted(ds.images, key=lambda i: i.image_id):
        found = by_image.get(image.image_id, [])
        boxes += len(found)
        large = []
        for ann_id, box in found:
            if min(box[2] - box[0], box[3] - box[1]) < plan.min_side:
                skipped["too_small"] += 1
            else:
                large.append((-(box[2] - box[0]) * (box[3] - box[1]), ann_id, box))
        large.sort()
        skipped["over_image_cap"] += max(0, len(large) - plan.per_image)
        crops: dict[str, Rect] = {}
        for _, ann_id, box in large[: plan.per_image]:
            rect = crop_rect(box, image.width, image.height, plan.padding)
            key = _key(image.image_id, rect, plan.variant)
            crops[key] = rect
            key_of[ann_id] = key
            image_of[key] = image.image_id
        if crops:
            jobs.append(_Job(image, tuple(sorted(crops.items()))))
    return jobs, key_of, image_of, {k: v for k, v in skipped.items() if v}, boxes


def _encode(rgb: NDArray[np.uint8], rect: Rect) -> bytes:
    from PIL import Image as PILImage

    left, top, right, bottom = rect
    crop = PILImage.fromarray(np.ascontiguousarray(rgb[top:bottom, left:right]), "RGB")
    long_side = max(crop.size)
    if long_side > MAX_CROP_SIDE:
        scale = MAX_CROP_SIDE / long_side
        size = (max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale)))
        crop = crop.resize(size, PILImage.Resampling.BICUBIC)
    out = io.BytesIO()
    crop.save(out, format="PNG", compress_level=1)
    return out.getvalue()


class _State:
    def __init__(self) -> None:
        self.error: str | None = None
        self.detail: str | None = None


Result = tuple[str, NDArray[np.float32] | None, str | None, str | None]
"""``(crop key, vector, reason, source image sha256)``."""


class _Embedder:
    """Cuts and embeds the crops of a list of jobs; see the module docstring."""

    def __init__(
        self, embedder: Embedder, batch: int, dtype: embed_cache.Dtype, ctx: CheckContext
    ) -> None:
        self.embedder, self.batch, self.dtype, self.ctx = embedder, batch, dtype, ctx

    def stream(self, jobs: list[_Job], deadline: float, state: _State) -> Iterator[Result]:
        ctx = self.ctx
        reads = pool.bounded_map(
            lambda job: (job, *read_verified(job.image, ctx)),
            jobs,
            workers=ctx.workers,
            window=max(1, min(ctx.workers, MAX_READ_WINDOW)),
            deadline=deadline,
        )
        pending: list[tuple[str, bytes, str]] = []
        consumed = 0
        try:
            for job, data, digest, reason in reads:
                consumed += 1
                if state.error is not None:
                    yield from ((key, None, "embedder_error", None) for key, _ in job.crops)
                    continue
                if data is None or digest is None:
                    yield from ((key, None, reason or "unreadable", None) for key, _ in job.crops)
                    continue
                cut, why = self._cut(job, data)
                if why is not None:
                    yield from ((key, None, why, None) for key, _ in job.crops)
                    continue
                pending.extend((key, png, digest) for key, png in cut)
                while state.error is None and (
                    len(pending) >= self.batch
                    or sum(len(png) for _, png, _ in pending) >= MAX_BATCH_BYTES
                ):
                    head, pending = self._head(pending)
                    yield from self._flush(head, deadline, state)
            while pending and state.error is None:
                head, pending = self._head(pending)
                yield from self._flush(head, deadline, state)
        finally:
            reads.close()
        yield from ((key, None, "embedder_error", None) for key, _, _ in pending)
        rest = "embedder_error" if state.error else "time_budget"
        for job in jobs[consumed:]:
            yield from ((key, None, rest, None) for key, _ in job.crops)

    def _head(
        self, pending: list[tuple[str, bytes, str]]
    ) -> tuple[list[tuple[str, bytes, str]], list[tuple[str, bytes, str]]]:
        """At most ``batch`` crops and ``MAX_BATCH_BYTES`` encoded bytes (at least one crop)."""
        size = 0
        for n, (_, png, _) in enumerate(pending[: self.batch]):
            size += len(png)
            if n and size > MAX_BATCH_BYTES:
                return pending[:n], pending[n:]
        return pending[: self.batch], pending[self.batch :]

    def _cut(self, job: _Job, data: bytes) -> tuple[list[tuple[str, bytes]], str | None]:
        try:
            rgb = imaging.decode_rgb(data, 0)
        except ImageDecodeError as exc:
            return [], exc.reason if exc.reason in _DECODE_REASONS else "decode_failed"
        height, width = rgb.shape[:2]
        if (width, height) != (job.image.width, job.image.height):
            return [], "size_mismatch"
        return [(key, _encode(rgb, rect)) for key, rect in job.crops], None

    def _flush(
        self, batch: list[tuple[str, bytes, str]], deadline: float, state: _State
    ) -> Iterator[Result]:
        dims = self.embedder.dims
        retried = False
        while batch:
            if time.monotonic() >= deadline:
                yield from ((key, None, "time_budget", None) for key, _, _ in batch)
                return
            try:
                raw = self.embedder.embed([png for _, png, _ in batch])
            except ImageDecodeError as exc:
                bad = {i: r for i, r in exc.failures if 0 <= i < len(batch)}
                if not bad or retried:
                    yield from ((key, None, "decode_failed", None) for key, _, _ in batch)
                    return
                for i in sorted(bad):
                    reason = bad[i] if bad[i] in _DECODE_REASONS else "decode_failed"
                    yield batch[i][0], None, reason, None
                batch = [item for i, item in enumerate(batch) if i not in bad]
                retried = True
                continue
            except Exception as exc:  # noqa: BLE001  # recorded by type only; text may hold secrets
                state.error = type(exc).__name__
                if isinstance(exc, DataCopilotError):
                    state.detail = safe_text(str(exc), 300)
                yield from ((key, None, "embedder_error", None) for key, _, _ in batch)
                return
            out = np.asarray(raw)
            if (
                out.shape != (len(batch), dims)
                or not np.issubdtype(out.dtype, np.floating)
                or not np.isfinite(out).all()
            ):
                state.error = "InvalidEmbedderOutput"
                yield from ((key, None, "embedder_error", None) for key, _, _ in batch)
                return
            good = np.linalg.norm(out.astype(np.float32), axis=1) > 0
            canonical = (
                embed_cache.canonical_vectors(out[good].astype(np.float32), self.dtype)
                if good.any()
                else np.zeros((0, dims), dtype=np.float32)
            )
            row = 0
            for n, (key, _, digest) in enumerate(batch):
                if good[n]:
                    yield key, canonical[row], None, digest
                    row += 1
                else:
                    yield key, None, "zero_vector", None
            return


def _only(jobs: list[_Job], keys: set[str]) -> list[_Job]:
    out = []
    for job in jobs:
        kept = tuple((k, r) for k, r in job.crops if k in keys)
        if kept:
            out.append(_Job(job.image, kept))
    return out


def compute(
    embedder: Embedder,
    ds: Dataset,
    ctx: CheckContext,
    plan: CropPlan,
    *,
    identity: embed_cache.CacheIdentity,
    secret: bytes | None,
    disabled_reason: str | None,
    batch: int,
    dtype: embed_cache.Dtype,
) -> Crops:
    """Embed the crops of ``ds`` (see the module docstring); never raises for an image or a model
    failure, which end up in ``Crops.es``."""
    started = time.perf_counter()
    log = get_logger(__name__)
    dims = embedder.dims
    test_only = bool(getattr(embedder, "test_only", False))

    def empty(total: int = 0) -> EmbeddingSet:
        return EmbeddingSet(
            ids=(),
            matrix=np.zeros((0, dims), dtype=np.float32),
            total=total,
            model_id=embedder.model_id,
            revision=embedder.revision,
            dims=dims,
            test_only=test_only,
        )

    try:
        imaging._pil()  # noqa: SLF001  # the one place that knows how Pillow is imported
    except ModelError:
        return Crops(empty(), {}, {}, {}, 0, unavailable="pillow_missing")
    jobs, key_of, image_of, skipped, boxes = plan_crops(ds, plan)
    if len(image_of) > MAX_CROPS:
        return Crops(empty(len(image_of)), {}, {}, skipped, boxes, unavailable="too_many_boxes")
    deadline = time.monotonic() + ctx.config.max_check_seconds
    keys = sorted(image_of)
    variant_identity = dataclasses.replace(identity, variant=plan.variant)
    cache = (
        embed_cache.EmbeddingCache.open(
            (ctx.dataset_dir or Path()) / "embeddings",
            variant_identity,
            secret,
            max_rows=2 * len(keys) + embed_cache.WRITE_SHARD_ROWS,
        )
        if secret is not None
        else embed_cache.EmbeddingCache.disabled(disabled_reason or "no_key")
    )
    loaded = cache.load(wanted=frozenset(keys))
    vectors: dict[str, NDArray[np.float32]] = {}
    stored: dict[str, str] = {}
    for row, key in enumerate(loaded.ids):
        if key in image_of:
            vectors[key], stored[key] = loaded.matrix[row], loaded.digests[row]
    worker = _Embedder(embedder, batch, dtype, ctx)
    state = _State()
    changed = sampled = verified = mismatched = 0
    unverified: str | None = None
    if vectors and cache.enabled:
        images = {i.image_id: i for i in ds.images}
        cached_images = sorted({image_of[k] for k in vectors})
        digests = pool.bounded_map(
            lambda image_id: hash_image(images[image_id], ctx),
            cached_images,
            workers=ctx.workers,
            window=max(1, min(ctx.workers, MAX_READ_WINDOW)),
            deadline=deadline,
        )
        now: dict[str, str | None] = {}
        for image_id, (digest, _reason) in zip(cached_images, digests, strict=False):
            now[image_id] = digest
        for key in list(vectors):
            digest = now.get(image_of[key])
            if digest != stored[key]:
                del vectors[key]
                changed += digest is not None
        if changed:
            log.warning("embedding_cache_content_changed", count=changed)
            vectors.clear()
            cache.reset()
    if vectors and cache.enabled:
        sample = set(pick_sample(sorted(vectors), ctx.run_id))
        sampled = len(sample)
        for key, vector, reason, _digest in worker.stream(_only(jobs, sample), deadline, state):
            if vector is None:
                if reason == "content_changed":
                    verified += 1
                    mismatched += 1
                continue
            verified += 1
            if float(np.dot(vector, vectors[key])) < SPOT_MIN_COSINE:
                mismatched += 1
        if verified < math.ceil(sampled * SPOT_MIN_VERIFIED_FRACTION):
            unverified = "too_few_verifiable"
        if mismatched or unverified:
            log.warning("embedding_cache_discarded", sampled=sampled, mismatched=mismatched)
            vectors.clear()
            cache.reset()
    cached_ids = frozenset(vectors)
    missing: dict[str, list[str]] = defaultdict(list)
    todo = set(keys) - set(vectors)
    pending: list[tuple[str, NDArray[np.float32], str]] = []
    unwritable = False

    def flush() -> None:
        nonlocal unwritable
        if pending and not unwritable and cache.enabled:
            try:
                cache.append([k for k, _, _ in pending], np.stack([v for _, v, _ in pending]),
                             [d for _, _, d in pending])  # fmt: skip
            except (embed_cache.CacheFullError, OSError):
                unwritable = True
        pending.clear()

    if state.error is not None:
        missing["embedder_error"].extend(sorted(todo))
    else:
        for key, vector, reason, digest in worker.stream(_only(jobs, todo), deadline, state):
            if vector is None or digest is None:
                missing[reason or "unreadable"].append(key)
                continue
            vectors[key] = vector
            pending.append((key, vector, digest))
            if len(pending) >= embed_cache.WRITE_SHARD_ROWS:
                flush()
        flush()
    ids = tuple(k for k in keys if k in vectors)
    matrix = (
        np.stack([vectors[k] for k in ids]).astype(np.float32)
        if ids
        else np.zeros((0, dims), dtype=np.float32)
    )
    frozen = {reason: tuple(sorted(found)) for reason, found in sorted(missing.items())}
    es = EmbeddingSet(
        ids=ids,
        matrix=matrix,
        total=len(keys),
        model_id=embedder.model_id,
        revision=embedder.revision,
        dims=dims,
        missing=frozen,
        incomplete=("time_budget", len(ids), len(keys)) if "time_budget" in frozen else None,
        error=state.error,
        error_detail=state.detail,
        cache_state=loaded.state,
        cache_rejected=dict(loaded.rejected),
        cache_hits=len(cached_ids),
        spot_checked=verified,
        spot_mismatched=mismatched,
        cached_ids=cached_ids,
        test_only=test_only,
        cache_content_changed=changed,
        cache_unverified=unverified,
        cache_unwritable=unwritable,
        cache_disabled_reason=disabled_reason,
    )
    log.info(
        "crop_embeddings_computed",
        model_id=embedder.model_id,
        model_revision=embedder.revision,
        crop_count=len(keys),
        embedded_count=len(ids),
        cache_hits=len(cached_ids),
        missing_count=es.missing_count,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return Crops(es, key_of, image_of, skipped, boxes)
