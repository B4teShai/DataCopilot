"""Embedding pipeline for the audit: cache, reading, batching and the fail-closed bookkeeping.

``PipelineProvider.get`` returns an :class:`EmbeddingSet` for a dataset exactly once per audit
(the near-duplicate, leakage and outlier checks share it). Every image ends in one of two states:
it has a vector in ``ids``/``matrix``, or it appears in ``missing`` under a named reason. Nothing is
dropped silently; the checks turn ``missing``, ``error`` and ``incomplete`` into gating findings.

Order of work: open the cache (authenticated, see ``embed/cache.py``), take every verified vector,
re-embed a deterministic sample of them and compare (a mismatch discards the cache and recomputes
everything: the defence against a forged cache by someone who holds the key), then read, verify and
embed the remaining images in batches, writing a shard per ``WRITE_SHARD_ROWS`` vectors so an
interrupted run resumes. Image bytes are read through the same containment as the other checks
(``audit/paths.py``, ``O_NOFOLLOW``, regular files, size cap) and must hash to the image id: a file
changed since ingest is never embedded under its old id. The embedder only ever sees bytes.

The trust policy mirrors the hash cache: with ``DATACOPILOT_CACHE_KEY`` the key is used; without
it a private secret beside ``DATACOPILOT_CACHE_DIR`` is used locally, and in CI (``CI`` set) the
cache is neither read nor written.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from datacopilot.audit import pool
from datacopilot.audit.hashcache import resolve_secret
from datacopilot.audit.paths import resolve_image_path
from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Dataset, Image
from datacopilot.embed import cache as embed_cache
from datacopilot.embed.protocol import Embedder, ImageDecodeError
from datacopilot.io import files
from datacopilot.logging import get_logger

if TYPE_CHECKING:
    from datacopilot.audit.context import CheckContext

__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_EMBED_BATCH",
    "MAX_BATCH_BYTES",
    "MAX_EMBED_BATCH",
    "EmbeddingProvider",
    "EmbeddingSet",
    "PipelineProvider",
    "pick_sample",
    "read_image_bytes",
    "sample_size",
]

DEFAULT_EMBED_BATCH = 32
MAX_EMBED_BATCH = 1024
DEFAULT_DTYPE: embed_cache.Dtype = "float16"
MAX_BATCH_BYTES = 256 * 1024**2
"""Encoded bytes one batch may hold; a batch is cut earlier when this would be exceeded."""
MAX_READ_WINDOW = 4
"""Images read ahead of the embedder, so memory is bounded by about this many files."""
SPOT_FRACTION = 0.01
SPOT_MIN = 4
SPOT_MAX = 64
"""Cache hits re-embedded per run: 1 percent, at least 4, at most 64 (all if fewer)."""
SPOT_MIN_COSINE = 0.99
"""A re-embedded vector must be at least this close to the cached one (float16 rounding, device
differences and a changed batch composition stay far above it)."""
_DECODE_REASONS = frozenset({"decode_failed", "too_many_pixels"})


@dataclass(frozen=True)
class EmbeddingSet:
    """The embeddings of one dataset and a complete account of the images that have none.

    Attributes:
        ids: image ids that have a vector, sorted.
        matrix: canonical ``float32`` vectors, row ``i`` belongs to ``ids[i]``.
        total: images in the dataset.
        missing: ``reason -> sorted image ids`` without a vector (``remote``, ``outside_root``,
            ``unsafe_path``, ``file_missing``, ``not_a_regular_file``, ``unreadable``,
            ``file_too_large``, ``content_changed``, ``decode_failed``, ``too_many_pixels``,
            ``zero_vector``, ``embedder_error``, ``time_budget``).
        incomplete: ``("time_budget", embedded, total)`` when the time budget ended the run.
        error: type name of the exception that stopped the embedder, or ``InvalidEmbedderOutput``.
        cache_state: ``absent``, ``ok``, ``stale``, ``rejected`` or ``disabled``.
        cache_rejected: rejected cache parts by reason (see ``embed/cache.py``).
        cache_hits: vectors taken from the cache and kept.
        spot_checked: cache hits that were re-embedded and compared.
        spot_mismatched: of those, how many disagreed (the cache was then discarded).
        cache_unwritable: the cache could not be written (read-only dataset, cache full).
        cache_disabled_reason: why the cache was not used, when it was not.
    """

    ids: tuple[str, ...]
    matrix: NDArray[np.float32]
    total: int
    model_id: str
    revision: str
    dims: int
    missing: dict[str, tuple[str, ...]] = field(default_factory=dict)
    incomplete: tuple[str, int, int] | None = None
    error: str | None = None
    cache_state: str = "disabled"
    cache_rejected: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    spot_checked: int = 0
    spot_mismatched: int = 0
    cache_unwritable: bool = False
    cache_disabled_reason: str | None = None

    @property
    def missing_count(self) -> int:
        return sum(len(v) for v in self.missing.values())


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What a check needs: the embeddings of the audited dataset, computed at most once."""

    def get(self, ds: Dataset, ctx: CheckContext) -> EmbeddingSet: ...


def sample_size(hits: int) -> int:
    """Number of cache hits to re-verify when there are ``hits`` of them."""
    return min(hits, SPOT_MAX, max(SPOT_MIN, math.ceil(hits * SPOT_FRACTION)))


def pick_sample(keys: list[str], run_id: str) -> list[str]:
    """Deterministic-random sample of ``keys``: ordered by ``sha256(run_id:key)``, so it is
    reproducible for a run and not predictable by someone who writes the cache beforehand."""
    ranked = sorted(keys, key=lambda k: hashlib.sha256(f"{run_id}:{k}".encode()).digest())
    return ranked[: sample_size(len(keys))]


def read_image_bytes(image: Image, ctx: CheckContext) -> tuple[bytes | None, str | None]:
    """The bytes of one image, or the reason they cannot be used. Never raises for file problems.

    The path is classified and contained like every other read of the audit (remote, unsafe and
    outside-root paths are not opened), the file is opened without following symlinks, must be a
    regular file no larger than ``max_embed_image_bytes`` and must hash to ``image.image_id``.
    """
    resolved = resolve_image_path(
        image.file_path, ctx.root, allow_external=ctx.allow_external_images
    )
    if resolved.kind != "local" or resolved.path is None:
        reason = {"remote": "remote", "external": "outside_root"}.get(resolved.kind, "unsafe_path")
        return None, reason
    limit = ctx.config.max_embed_image_bytes
    try:
        with files.open_regular(resolved.path, nofollow=True) as handle:
            if os.fstat(handle.fileno()).st_size > limit:
                return None, "file_too_large"
            data = handle.read(limit + 1)
    except FormatError as exc:
        kind = "not_a_regular_file" if "not a regular file" in str(exc) else "file_missing"
        return None, kind
    except OSError:
        return None, "unreadable"
    if len(data) > limit:
        return None, "file_too_large"
    if hashlib.sha256(data).hexdigest()[:16] != image.image_id:
        return None, "content_changed"
    return data, None


class _State:
    """Mutable outcome of one streamed run."""

    def __init__(self) -> None:
        self.error: str | None = None


class PipelineProvider:
    """Computes (and caches) the embeddings of a dataset with one :class:`Embedder`.

    Args:
        embedder: a created embedder (the license guard has already passed).
        batch_size: images per ``embed`` call, 1 to ``MAX_EMBED_BATCH``.
        dtype: storage dtype of the cache; fresh vectors are rounded identically.
        use_cache: False never reads or writes the cache.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        batch_size: int = DEFAULT_EMBED_BATCH,
        dtype: embed_cache.Dtype = DEFAULT_DTYPE,
        use_cache: bool = True,
    ) -> None:
        if not 1 <= batch_size <= MAX_EMBED_BATCH:
            raise ValueError(f"batch size must be between 1 and {MAX_EMBED_BATCH}")
        if dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")
        if not 1 <= embedder.dims <= embed_cache.MAX_DIMS:
            raise ValueError(f"embedder dims must be between 1 and {embed_cache.MAX_DIMS}")
        self._embedder = embedder
        self._batch = batch_size
        self._dtype: embed_cache.Dtype = dtype
        self._use_cache = use_cache
        self._memo: tuple[str, EmbeddingSet] | None = None

    # ------------------------------------------------------------------ cache trust
    def _secret(self, ctx: CheckContext) -> tuple[bytes | None, str | None]:
        if not self._use_cache:
            return None, "disabled_by_option"
        if ctx.dataset_dir is None:
            return None, "no_dataset_directory"
        if ctx.cache_key is not None:
            return ctx.cache_key.get_secret_value().encode("utf-8"), None
        try:
            return resolve_secret(ctx.cache_dir, None, ctx.in_ci)
        except OSError:
            return None, "cache_directory_unavailable"

    def preflight(self, ctx: CheckContext) -> str | None:
        """Why the persistent cache cannot be used with this context, or None when it can."""
        return self._secret(ctx)[1]

    def _identity(self, ds: Dataset) -> embed_cache.CacheIdentity:
        weights = getattr(self._embedder, "weights", None) or {}
        return embed_cache.CacheIdentity(
            model_id=self._embedder.model_id,
            revision=self._embedder.revision,
            dims=self._embedder.dims,
            dtype=self._dtype,
            dataset_version=ds.manifest.version,
            weights=dict(weights),
        )

    # ------------------------------------------------------------------ the run
    def get(self, ds: Dataset, ctx: CheckContext) -> EmbeddingSet:
        """Embeddings of ``ds``; later calls for the same dataset version return the same object.

        Exceptions other than ``Exception`` (``KeyboardInterrupt``, ``SystemExit``) propagate; an
        embedder failure is recorded in ``EmbeddingSet.error`` instead of being raised.
        """
        if self._memo is not None and self._memo[0] == ds.manifest.version:
            return self._memo[1]
        result = self._compute(ds, ctx)
        self._memo = (ds.manifest.version, result)
        return result

    def _compute(self, ds: Dataset, ctx: CheckContext) -> EmbeddingSet:
        started = time.perf_counter()
        log = get_logger(__name__)
        deadline = time.monotonic() + ctx.config.max_check_seconds
        dims = self._embedder.dims
        ids_sorted = sorted(image.image_id for image in ds.images)
        by_id = {image.image_id: image for image in ds.images}
        position = {image_id: n for n, image_id in enumerate(ids_sorted)}
        matrix = np.zeros((len(ids_sorted), dims), dtype=np.float32)
        have = np.zeros(len(ids_sorted), dtype=bool)
        missing: dict[str, list[str]] = defaultdict(list)
        state = _State()

        secret, disabled_reason = self._secret(ctx)
        cache = (
            embed_cache.EmbeddingCache.open(
                (ctx.dataset_dir or Path()) / "embeddings", self._identity(ds), secret
            )
            if secret is not None
            else embed_cache.EmbeddingCache.disabled(disabled_reason or "no_key")
        )
        loaded = cache.load()
        hits = 0
        for row, image_id in enumerate(loaded.ids):
            slot = position.get(image_id)
            if slot is not None:
                matrix[slot], have[slot] = loaded.matrix[row], True
                hits += 1
        rejected = dict(loaded.rejected)
        if disabled_reason is not None and disabled_reason not in (
            "disabled_by_option",
            "no_dataset_directory",
        ):
            log.warning("embedding_cache_disabled", reason=disabled_reason)
        if rejected:
            log.warning("embedding_cache_rejected", state=loaded.state, reasons=rejected)

        sampled = mismatched = 0
        if hits and cache.enabled:
            sampled, mismatched = self._spot_check(
                ctx, by_id, ids_sorted, matrix, have, position, state, deadline
            )
            if mismatched:
                log.warning("embedding_cache_discarded", sampled=sampled, mismatched=mismatched)
                cache.reset()
                have[:] = False
                matrix[:] = 0.0
                hits = 0

        unwritable = self._embed_missing(
            ctx, cache, by_id, ids_sorted, matrix, have, missing, state, deadline
        )
        slots = np.flatnonzero(have)
        ids = tuple(ids_sorted[int(s)] for s in slots)
        frozen = {reason: tuple(sorted(found)) for reason, found in missing.items()}
        incomplete = ("time_budget", len(ids), len(ids_sorted)) if "time_budget" in frozen else None
        result = EmbeddingSet(
            ids=ids,
            matrix=matrix[slots],
            total=len(ids_sorted),
            model_id=self._embedder.model_id,
            revision=self._embedder.revision,
            dims=dims,
            missing=frozen,
            incomplete=incomplete,
            error=state.error,
            cache_state=loaded.state,
            cache_rejected=rejected,
            cache_hits=hits,
            spot_checked=sampled,
            spot_mismatched=mismatched,
            cache_unwritable=unwritable,
            cache_disabled_reason=disabled_reason,
        )
        log.info(
            "embeddings_computed",
            model_id=result.model_id,
            model_revision=result.revision,
            image_count=result.total,
            embedded_count=len(ids),
            cache_hits=hits,
            missing_count=result.missing_count,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return result

    def _spot_check(
        self,
        ctx: CheckContext,
        by_id: dict[str, Image],
        ids_sorted: list[str],
        matrix: NDArray[np.float32],
        have: NDArray[np.bool_],
        position: dict[str, int],
        state: _State,
        deadline: float,
    ) -> tuple[int, int]:
        """Re-embed a sample of the cached vectors; returns ``(compared, disagreeing)``."""
        cached = [ids_sorted[int(s)] for s in np.flatnonzero(have)]
        sample = [by_id[i] for i in pick_sample(cached, ctx.run_id)]
        compared = disagreeing = 0
        for image_id, vector, _reason in self._stream(sample, ctx, deadline, state):
            if vector is None:
                continue  # unreadable now (for example no images on a CI runner): cannot compare
            compared += 1
            if float(np.dot(vector, matrix[position[image_id]])) < SPOT_MIN_COSINE:
                disagreeing += 1
        return compared, disagreeing

    def _embed_missing(
        self,
        ctx: CheckContext,
        cache: embed_cache.EmbeddingCache,
        by_id: dict[str, Image],
        ids_sorted: list[str],
        matrix: NDArray[np.float32],
        have: NDArray[np.bool_],
        missing: dict[str, list[str]],
        state: _State,
        deadline: float,
    ) -> bool:
        todo = [by_id[ids_sorted[int(s)]] for s in np.flatnonzero(~have)]
        position = {image_id: n for n, image_id in enumerate(ids_sorted)}
        if state.error is not None:  # the spot check already met a failing embedder
            for image in todo:
                missing["embedder_error"].append(image.image_id)
            return False
        pending_ids: list[str] = []
        pending_vecs: list[NDArray[np.float32]] = []
        unwritable = False

        def flush() -> None:
            nonlocal unwritable
            if not pending_ids or unwritable or not cache.enabled:
                pending_ids.clear()
                pending_vecs.clear()
                return
            try:
                cache.append(pending_ids, np.stack(pending_vecs))
            except (embed_cache.CacheFullError, OSError):
                unwritable = True
            pending_ids.clear()
            pending_vecs.clear()

        for image_id, vector, reason in self._stream(todo, ctx, deadline, state):
            if vector is None:
                missing[reason or "unreadable"].append(image_id)
                continue
            slot = position[image_id]
            matrix[slot], have[slot] = vector, True
            pending_ids.append(image_id)
            pending_vecs.append(vector)
            if len(pending_ids) >= embed_cache.WRITE_SHARD_ROWS:
                flush()
        flush()
        return unwritable

    # ------------------------------------------------------------------ reading and embedding
    def _read(self, image: Image, ctx: CheckContext) -> tuple[Image, bytes | None, str | None]:
        data, reason = read_image_bytes(image, ctx)
        return image, data, reason

    def _stream(
        self, images: list[Image], ctx: CheckContext, deadline: float, state: _State
    ) -> Iterator[tuple[str, NDArray[np.float32] | None, str | None]]:
        """Yield ``(image_id, vector or None, reason or None)`` for every image, in order."""
        window = max(1, min(ctx.workers, MAX_READ_WINDOW))
        reads = pool.bounded_map(
            lambda image: self._read(image, ctx),
            images,
            workers=ctx.workers,
            window=window,
            deadline=deadline,
        )
        batch: list[tuple[Image, bytes]] = []
        batch_bytes = 0
        consumed = 0
        try:
            for image, data, reason in reads:
                consumed += 1
                if data is None:
                    yield image.image_id, None, reason
                    continue
                if batch and (
                    len(batch) >= self._batch or batch_bytes + len(data) > MAX_BATCH_BYTES
                ):
                    yield from self._flush(batch, deadline, state)
                    batch, batch_bytes = [], 0
                    if state.error is not None:  # stop reading; the rest cannot be embedded
                        yield image.image_id, None, "embedder_error"
                        break
                batch.append((image, data))
                batch_bytes += len(data)
            else:
                yield from self._flush(batch, deadline, state)
        finally:
            reads.close()
        for image in images[consumed:]:
            yield image.image_id, None, "embedder_error" if state.error else "time_budget"

    def _flush(
        self, batch: list[tuple[Image, bytes]], deadline: float, state: _State
    ) -> Iterator[tuple[str, NDArray[np.float32] | None, str | None]]:
        dims = self._embedder.dims
        batch = list(batch)
        while batch:
            if time.monotonic() >= deadline:
                for image, _ in batch:
                    yield image.image_id, None, "time_budget"
                return
            try:
                raw = self._embedder.embed([data for _, data in batch])
            except ImageDecodeError as exc:
                if not 0 <= exc.index < len(batch):
                    state.error = "InvalidEmbedderOutput"
                    for image, _ in batch:
                        yield image.image_id, None, "embedder_error"
                    return
                image, _ = batch.pop(exc.index)
                reason = exc.reason if exc.reason in _DECODE_REASONS else "decode_failed"
                yield image.image_id, None, reason
                continue
            except Exception as exc:  # noqa: BLE001  # recorded by type only; text may hold secrets
                state.error = type(exc).__name__
                for image, _ in batch:
                    yield image.image_id, None, "embedder_error"
                return
            out = np.asarray(raw)
            if (
                out.shape != (len(batch), dims)
                or not np.issubdtype(out.dtype, np.floating)
                or not np.isfinite(out).all()
            ):
                state.error = "InvalidEmbedderOutput"
                for image, _ in batch:
                    yield image.image_id, None, "embedder_error"
                return
            norms = np.linalg.norm(out.astype(np.float32), axis=1)
            good = norms > 0
            canonical = (
                embed_cache.canonical_vectors(out[good].astype(np.float32), self._dtype)
                if good.any()
                else np.zeros((0, dims), dtype=np.float32)
            )
            row = 0
            for n, (image, _) in enumerate(batch):
                if good[n]:
                    yield image.image_id, canonical[row], None
                    row += 1
                else:
                    yield image.image_id, None, "zero_vector"
            return
