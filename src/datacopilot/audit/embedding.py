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
import re
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
from datacopilot.audit.paths import resolve_image_path, safe_text
from datacopilot.core.errors import DataCopilotError, FormatError
from datacopilot.core.schema import Dataset, Image
from datacopilot.embed import cache as embed_cache
from datacopilot.embed import knn
from datacopilot.embed.protocol import Embedder, ImageDecodeError
from datacopilot.io import files
from datacopilot.logging import get_logger

if TYPE_CHECKING:
    from datacopilot.audit.context import CheckContext
    from datacopilot.audit.crops import CropPlan, Crops

__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_EMBED_BATCH",
    "MAX_BATCH_BYTES",
    "MAX_EMBED_BATCH",
    "EmbeddingProvider",
    "EmbeddingSet",
    "PipelineProvider",
    "pick_sample",
    "hash_image",
    "read_image_bytes",
    "read_verified",
    "short_id",
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
SPOT_MIN = 10
SPOT_MAX = 64
"""Cache hits re-embedded per run: 1 percent, at least 10, at most 64 (all if fewer)."""
SPOT_MIN_VERIFIED_FRACTION = 0.5
"""At least this fraction of the sampled hits must be verifiable (image readable, embedder
working). Otherwise the cached vectors are not trusted: they are dropped and recomputed, and an
image that cannot be read is then reported as not embedded, which gates."""
SPOT_MIN_COSINE = 0.999
"""A re-embedded vector must be at least this close to the cached one. Both are canonical vectors
(``embed/cache.py``): a fresh vector is rounded to the storage dtype exactly like the stored one, so
the only difference is the backend's own nondeterminism (kernel choice, batch composition, CPU
versus GPU), which is about 1e-5 in cosine for the supported models; float16 rounding of a unit
vector moves each component by under 1e-3 but is identical on both sides and does not count. A
forged vector is far from this bound: the cosine of two unrelated images is below 0.9. One
disagreeing sample discards the whole cache for the run."""
Result = tuple[str, NDArray[np.float32] | None, str | None, str | None]
"""``(image_id, vector, reason, full sha256)``: a vector and a digest, or a reason."""
_PLAIN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DECODE_REASONS = frozenset({"decode_failed", "too_many_pixels", "multi_frame"})


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
            ``multi_frame``,
            ``zero_vector``, ``embedder_error``, ``time_budget``).
        incomplete: ``("time_budget", embedded, total)`` when the time budget ended the run.
        error: type name of the exception that stopped the embedder, or ``InvalidEmbedderOutput``.
        error_detail: the message of that exception when it is a ``DataCopilotError`` (messages of
            this package are written to be shown, for example the missing extra), cleaned and cut
            to 300 characters; None for any other exception, whose text may hold anything.
        cache_state: ``absent``, ``ok``, ``stale``, ``rejected`` or ``disabled``.
        cache_rejected: rejected cache parts by reason (see ``embed/cache.py``).
        cache_hits: vectors taken from the cache and kept.
        spot_checked: cache hits that were re-embedded and compared.
        spot_mismatched: of those, how many disagreed (the cache was then discarded).
        cached_ids: images whose vector came from the (verified) cache rather than from the model
            in this run; the checks re-embed the members of the pairs they report or nearly report.
        refused: ``too_many_images`` when the dataset is over the neighbour-search cap; nothing was
            read, embedded or cached.
        test_only: the embedder declares itself a test double (``test_only = True``); its verdicts
            say nothing about pictures.
        cache_content_changed: cached hits whose image file no longer hashes to the digest stored
            with the vector (the whole cache was then discarded and recomputed).
        cache_unverified: reason code (``too_few_verifiable``) when too few sampled hits could be
            verified and the cached vectors were therefore not trusted.
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
    error_detail: str | None = None
    cache_state: str = "disabled"
    cache_rejected: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    spot_checked: int = 0
    spot_mismatched: int = 0
    cached_ids: frozenset[str] = frozenset()
    refused: str | None = None
    test_only: bool = False
    cache_content_changed: int = 0
    cache_unverified: str | None = None
    cache_unwritable: bool = False
    cache_disabled_reason: str | None = None

    @property
    def missing_count(self) -> int:
        return sum(len(v) for v in self.missing.values())


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What a check needs: the embeddings of the audited dataset, computed at most once.

    A provider may also offer ``fresh_vectors(ds, ctx, image_ids)``, which embeds the named
    images again without the cache (see ``PipelineProvider``); checks use it when it exists.
    """

    def get(self, ds: Dataset, ctx: CheckContext) -> EmbeddingSet: ...


def sample_size(hits: int) -> int:
    """Number of cache hits to re-verify when there are ``hits`` of them."""
    return min(hits, SPOT_MAX, max(SPOT_MIN, math.ceil(hits * SPOT_FRACTION)))


def pick_sample(keys: list[str], run_id: str) -> list[str]:
    """Deterministic-random sample of ``keys``: ordered by ``sha256(run_id:key)``, so it is
    reproducible for a run and not predictable by someone who writes the cache beforehand."""
    ranked = sorted(keys, key=lambda k: hashlib.sha256(f"{run_id}:{k}".encode()).digest())
    return ranked[: sample_size(len(keys))]


def short_id(digest: str) -> str:
    """The 16-hex ``image_id`` a full sha256 digest corresponds to.

    ``image_id`` is a 64-bit label: it is cheap to compare but collisions can be constructed, so it
    is only a first filter. The full digest is what binds a cached vector to a file.
    """
    return digest[:16]


def _resolve(image: Image, ctx: CheckContext) -> tuple[Path | None, str | None]:
    """The real path to open, or the reason the image must not be opened."""
    resolved = resolve_image_path(
        image.file_path, ctx.root, allow_external=ctx.allow_external_images
    )
    if resolved.kind != "local" or resolved.path is None:
        reason = {"remote": "remote", "external": "outside_root"}.get(resolved.kind, "unsafe_path")
        return None, reason
    return resolved.path, None


def _open_reason(exc: Exception) -> str:
    if isinstance(exc, FormatError):
        return "not_a_regular_file" if "not a regular file" in str(exc) else "file_missing"
    return "unreadable"


def read_verified(image: Image, ctx: CheckContext) -> tuple[bytes | None, str | None, str | None]:
    """``(bytes, full sha256, None)`` of one image, or ``(None, None, reason)``. Never raises for
    file problems.

    The path is classified and contained like every other read of the audit (remote, unsafe and
    outside-root paths are not opened), the file is opened without following symlinks, must be a
    regular file no larger than ``max_embed_image_bytes`` and must hash to ``image.image_id``.
    """
    path, reason = _resolve(image, ctx)
    if path is None:
        return None, None, reason
    limit = ctx.config.max_embed_image_bytes
    try:
        with files.open_regular(path, nofollow=True) as handle:
            if os.fstat(handle.fileno()).st_size > limit:
                return None, None, "file_too_large"
            data = handle.read(limit + 1)
    except (FormatError, OSError) as exc:
        return None, None, _open_reason(exc)
    if len(data) > limit:
        return None, None, "file_too_large"
    digest = hashlib.sha256(data).hexdigest()
    if short_id(digest) != image.image_id:
        return None, None, "content_changed"
    return data, digest, None


def read_image_bytes(image: Image, ctx: CheckContext) -> tuple[bytes | None, str | None]:
    """The bytes of one image, or the reason they cannot be used (see :func:`read_verified`)."""
    data, _digest, reason = read_verified(image, ctx)
    return data, reason


def hash_image(image: Image, ctx: CheckContext) -> tuple[str | None, str | None]:
    """``(full sha256, None)`` of one image streamed in 1 MiB chunks, or ``(None, reason)``.

    Same containment and size cap as :func:`read_verified`; no image is held in memory. The digest
    is returned whether or not it matches ``image_id`` (the caller compares it with the cache).
    """
    path, reason = _resolve(image, ctx)
    if path is None:
        return None, reason
    limit = ctx.config.max_embed_image_bytes
    digest = hashlib.sha256()
    total = 0
    try:
        with files.open_regular(path, nofollow=True) as handle:
            if os.fstat(handle.fileno()).st_size > limit:
                return None, "file_too_large"
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    return None, "file_too_large"
                digest.update(chunk)
    except (FormatError, OSError) as exc:
        return None, _open_reason(exc)
    return digest.hexdigest(), None


class _State:
    """Mutable outcome of one streamed run."""

    def __init__(self) -> None:
        self.error: str | None = None
        self.detail: str | None = None


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
        if not _PLAIN_NAME.fullmatch(str(embedder.model_id)) or not _PLAIN_NAME.fullmatch(
            str(embedder.revision)
        ):
            raise ValueError("embedder model_id and revision must be plain names")
        self._embedder = embedder
        self._batch = batch_size
        self._dtype: embed_cache.Dtype = dtype
        self._use_cache = use_cache
        self._memo: tuple[str, EmbeddingSet] | None = None
        self._crop_memo: tuple[tuple[str, str], Crops] | None = None

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
        if len(ds.images) > knn.MAX_KNN_ROWS:  # decided before the cache or the model is touched
            return EmbeddingSet(
                ids=(),
                matrix=np.zeros((0, dims), dtype=np.float32),
                total=len(ds.images),
                model_id=self._embedder.model_id,
                revision=self._embedder.revision,
                dims=dims,
                refused="too_many_images",
                test_only=bool(getattr(self._embedder, "test_only", False)),
            )
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
                (ctx.dataset_dir or Path()) / "embeddings",
                self._identity(ds),
                secret,
                max_rows=2 * len(ds.images) + embed_cache.WRITE_SHARD_ROWS,
            )
            if secret is not None
            else embed_cache.EmbeddingCache.disabled(disabled_reason or "no_key")
        )
        loaded = cache.load(wanted=frozenset(ids_sorted))
        hits = 0
        stored: dict[int, str] = {}
        for row, image_id in enumerate(loaded.ids):
            slot = position.get(image_id)
            if slot is not None:
                matrix[slot], have[slot] = loaded.matrix[row], True
                stored[slot] = loaded.digests[row]
                hits += 1
        rejected = dict(loaded.rejected)
        if disabled_reason is not None and disabled_reason not in (
            "disabled_by_option",
            "no_dataset_directory",
        ):
            log.warning("embedding_cache_disabled", reason=disabled_reason)
        if rejected:
            log.warning("embedding_cache_rejected", state=loaded.state, reasons=rejected)

        changed = 0
        if hits and cache.enabled:
            changed = self._verify_files(ctx, by_id, ids_sorted, have, stored, deadline)
            hits = int(have.sum())
            if changed:
                log.warning("embedding_cache_content_changed", count=changed)
                cache.reset()
                have[:] = False
                matrix[:] = 0.0
                hits = 0
        sampled = mismatched = 0
        unverified: str | None = None
        if hits and cache.enabled:
            wanted, sampled, mismatched = self._spot_check(
                ctx, by_id, ids_sorted, matrix, have, position, state, deadline
            )
            if sampled < math.ceil(wanted * SPOT_MIN_VERIFIED_FRACTION):
                unverified = "too_few_verifiable"
                log.warning(
                    "embedding_cache_unverified",
                    reason=unverified,
                    sampled=wanted,
                    verified=sampled,
                )
            if mismatched:
                log.warning("embedding_cache_discarded", sampled=sampled, mismatched=mismatched)
            if mismatched or unverified:
                cache.reset()
                have[:] = False
                matrix[:] = 0.0
                hits = 0

        cached_ids = frozenset(ids_sorted[int(s)] for s in np.flatnonzero(have))
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
            error_detail=state.detail,
            cache_state=loaded.state,
            cache_rejected=rejected,
            cache_hits=hits,
            spot_checked=sampled,
            spot_mismatched=mismatched,
            cached_ids=cached_ids,
            test_only=bool(getattr(self._embedder, "test_only", False)),
            cache_content_changed=changed,
            cache_unverified=unverified,
            cache_unwritable=unwritable,
            cache_disabled_reason=disabled_reason,
        )
        if state.error is not None:
            log.warning("embedder_failed", error_type=state.error, detail=state.detail)
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

    def crops(self, ds: Dataset, ctx: CheckContext, plan: CropPlan) -> Crops:
        """Box-crop embeddings of ``ds`` (``audit/crops.py``), computed once per dataset version
        and crop geometry, cached beside the image embeddings under the geometry's variant."""
        from datacopilot.audit import crops as crop_pipeline

        memo_key = (ds.manifest.version, plan.variant)
        if self._crop_memo is not None and self._crop_memo[0] == memo_key:
            return self._crop_memo[1]
        secret, disabled_reason = self._secret(ctx)
        result = crop_pipeline.compute(
            self._embedder,
            ds,
            ctx,
            plan,
            identity=self._identity(ds),
            secret=secret,
            disabled_reason=disabled_reason,
            batch=self._batch,
            dtype=self._dtype,
        )
        self._crop_memo = (memo_key, result)
        return result

    def fresh_vectors(
        self, ds: Dataset, ctx: CheckContext, image_ids: list[str]
    ) -> dict[str, NDArray[np.float32]]:
        """Canonical vectors of the named images computed now, bypassing the cache.

        Used to re-check the pairs a check reports or nearly reports. Images that cannot be read
        or embedded are left out (the caller keeps the cached decision for those); an embedder
        failure returns what was computed so far. Bounded by the caller (a few hundred images).
        """
        by_id = {image.image_id: image for image in ds.images}
        images = [by_id[i] for i in image_ids if i in by_id]
        deadline = time.monotonic() + ctx.config.max_check_seconds
        out: dict[str, NDArray[np.float32]] = {}
        for image_id, vector, _reason, _digest in self._stream(images, ctx, deadline, _State()):
            if vector is not None:
                out[image_id] = vector
        return out

    def _verify_files(
        self,
        ctx: CheckContext,
        by_id: dict[str, Image],
        ids_sorted: list[str],
        have: NDArray[np.bool_],
        stored: dict[int, str],
        deadline: float,
    ) -> int:
        """Hash the file of every cache hit and drop the vectors it no longer matches.

        The cache is bound to the dataset rows, which do not see the image files, so this is what
        ties a cached vector to the bytes it was computed from. A hit whose file hashes to another
        digest is counted (the caller then discards the whole cache); a hit whose file cannot be
        read, is too large, is remote or outside the root, or was not reached before the deadline
        is simply dropped, so the main pass classifies it and it becomes an unchecked image rather
        than a trusted vector. Only I/O is spent here, no model.
        """
        slots = [int(s) for s in np.flatnonzero(have)]
        images = [by_id[ids_sorted[s]] for s in slots]
        window = max(1, min(ctx.workers, MAX_READ_WINDOW))
        results = pool.bounded_map(
            lambda image: hash_image(image, ctx),
            images,
            workers=ctx.workers,
            window=window,
            deadline=deadline,
        )
        reached = 0
        changed = 0
        for slot, (digest, _reason) in zip(slots, results, strict=False):
            reached += 1
            if digest == stored[slot]:
                continue
            have[slot] = False
            if digest is not None:
                changed += 1
        for slot in slots[reached:]:
            have[slot] = False
        return changed

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
    ) -> tuple[int, int, int]:
        """Re-embed a sample of the cached vectors.

        Returns ``(sample size, verified, disagreeing)``; a hit that cannot be re-read or
        re-embedded counts in the sample but not as verified.
        """
        cached = [ids_sorted[int(s)] for s in np.flatnonzero(have)]
        sample = [by_id[i] for i in pick_sample(cached, ctx.run_id)]
        compared = disagreeing = 0
        for image_id, vector, reason, _digest in self._stream(sample, ctx, deadline, state):
            if vector is None:
                if reason == "content_changed":  # a hit whose file changed: never "unverifiable"
                    compared += 1
                    disagreeing += 1
                continue  # unreadable now (for example no images on a CI runner): cannot compare
            compared += 1
            if float(np.dot(vector, matrix[position[image_id]])) < SPOT_MIN_COSINE:
                disagreeing += 1
        return len(sample), compared, disagreeing

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
        pending_digests: list[str] = []
        unwritable = False

        def flush() -> None:
            nonlocal unwritable
            if not pending_ids or unwritable or not cache.enabled:
                pending_ids.clear()
                pending_vecs.clear()
                pending_digests.clear()
                return
            try:
                cache.append(pending_ids, np.stack(pending_vecs), pending_digests)
            except (embed_cache.CacheFullError, OSError):
                unwritable = True
            pending_ids.clear()
            pending_vecs.clear()
            pending_digests.clear()

        for image_id, vector, reason, digest in self._stream(todo, ctx, deadline, state):
            if vector is None or digest is None:
                missing[reason or "unreadable"].append(image_id)
                continue
            slot = position[image_id]
            matrix[slot], have[slot] = vector, True
            pending_ids.append(image_id)
            pending_vecs.append(vector)
            pending_digests.append(digest)
            if len(pending_ids) >= embed_cache.WRITE_SHARD_ROWS:
                flush()
        flush()
        return unwritable

    # ------------------------------------------------------------------ reading and embedding
    def _read(
        self, image: Image, ctx: CheckContext
    ) -> tuple[Image, bytes | None, str | None, str | None]:
        data, digest, reason = read_verified(image, ctx)
        return image, data, digest, reason

    def _stream(
        self, images: list[Image], ctx: CheckContext, deadline: float, state: _State
    ) -> Iterator[Result]:
        """Yield ``(image_id, vector, reason, digest)`` per image, in order (see ``Result``)."""
        window = max(1, min(ctx.workers, MAX_READ_WINDOW))
        reads = pool.bounded_map(
            lambda image: self._read(image, ctx),
            images,
            workers=ctx.workers,
            window=window,
            deadline=deadline,
        )
        batch: list[tuple[Image, bytes, str]] = []
        batch_bytes = 0
        consumed = 0
        try:
            for image, data, digest, reason in reads:
                consumed += 1
                if data is None or digest is None:
                    yield image.image_id, None, reason or "unreadable", None
                    continue
                if batch and (
                    len(batch) >= self._batch or batch_bytes + len(data) > MAX_BATCH_BYTES
                ):
                    yield from self._flush(batch, deadline, state)
                    batch, batch_bytes = [], 0
                    if state.error is not None:  # stop reading; the rest cannot be embedded
                        yield image.image_id, None, "embedder_error", None
                        break
                batch.append((image, data, digest))
                batch_bytes += len(data)
            else:
                yield from self._flush(batch, deadline, state)
        finally:
            reads.close()
        for image in images[consumed:]:
            yield image.image_id, None, "embedder_error" if state.error else "time_budget", None

    def _flush(
        self, batch: list[tuple[Image, bytes, str]], deadline: float, state: _State
    ) -> Iterator[Result]:
        dims = self._embedder.dims
        batch = list(batch)
        while batch:
            if time.monotonic() >= deadline:
                for image, _, _ in batch:
                    yield image.image_id, None, "time_budget", None
                return
            try:
                raw = self._embedder.embed([data for _, data, _ in batch])
            except ImageDecodeError as exc:
                bad = {i: r for i, r in exc.failures if 0 <= i < len(batch)}
                if not bad:
                    state.error = "InvalidEmbedderOutput"
                    for image, _, _ in batch:
                        yield image.image_id, None, "embedder_error", None
                    return
                for i in sorted(bad):  # every reported failure is dropped; one retry for the rest
                    reason = bad[i] if bad[i] in _DECODE_REASONS else "decode_failed"
                    yield batch[i][0].image_id, None, reason, None
                batch = [item for i, item in enumerate(batch) if i not in bad]
                continue
            except Exception as exc:  # noqa: BLE001  # recorded by type only; text may hold secrets
                state.error = type(exc).__name__
                if isinstance(exc, DataCopilotError):
                    state.detail = safe_text(str(exc), 300)
                for image, _, _ in batch:
                    yield image.image_id, None, "embedder_error", None
                return
            out = np.asarray(raw)
            if (
                out.shape != (len(batch), dims)
                or not np.issubdtype(out.dtype, np.floating)
                or not np.isfinite(out).all()
            ):
                state.error = "InvalidEmbedderOutput"
                for image, _, _ in batch:
                    yield image.image_id, None, "embedder_error", None
                return
            norms = np.linalg.norm(out.astype(np.float32), axis=1)
            good = norms > 0
            canonical = (
                embed_cache.canonical_vectors(out[good].astype(np.float32), self._dtype)
                if good.any()
                else np.zeros((0, dims), dtype=np.float32)
            )
            row = 0
            for n, (image, _, digest) in enumerate(batch):
                if good[n]:
                    yield image.image_id, canonical[row], None, digest
                    row += 1
                else:
                    yield image.image_id, None, "zero_vector", None
            return
