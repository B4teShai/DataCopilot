"""``duplicates``: images whose file bytes are identical.

Every local image is hashed with a streamed sha256 (1 MiB chunks, so memory does not depend on the
image size) on ``ctx.workers`` threads (``hashlib`` releases the GIL on large chunks). Files are
opened without following symlinks, only after their real path has been confirmed to lie inside
the dataset root (see ``audit/paths.py``), and only if they are regular files. Results are cached
(authenticated, see ``audit/hashcache.py``) by (path, size, mtime); every run re-hashes a
deterministic-random sample of the cache hits and, on any mismatch, discards the cache and
re-hashes everything. One finding per group of identical files: CRITICAL when the group spans two
or more of train/val/test (leakage by content), WARN otherwise. ``unassigned`` is not a split.

Findings carry the 12-hex prefix of the shared digest (``group_id``), never the full digest, and
base names only, never directories.

Rules (``payload["rule"]``): ``exact_duplicate`` (WARN or CRITICAL), ``content_changed`` (CRITICAL:
the file no longer hashes to ``Image.image_id``, so ``manifest.version`` no longer describes the
images), ``hash_cache_mismatch`` (WARN: the spot check found a cached digest that is wrong) and
``audit_incomplete`` (CRITICAL, WARN without a gate: ``max_hash_bytes`` or
``max_check_seconds`` ended the run before every file was hashed) and ``images_not_hashed``
(WARN, or CRITICAL above ``max_unchecked_fraction`` or with ``require_local_images``: counts, by
reason, of images that were skipped).
A group that mixes ``unassigned`` with a real split is also leakage (CRITICAL).
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from datacopilot import __version__
from datacopilot.audit import pool
from datacopilot.audit.context import CheckContext
from datacopilot.audit.findings import make_finding
from datacopilot.audit.hashcache import MAX_CACHE_BYTES, HashCache, cache_key
from datacopilot.audit.paths import display_name, resolve_image_path, unchecked_severity
from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Dataset, Finding, Image, PluginSpec
from datacopilot.io import files
from datacopilot.logging import get_logger

__all__ = [
    "CACHE_FILE",
    "CHUNK_BYTES",
    "MAX_CACHE_BYTES",
    "MAX_MEMBERS_LISTED",
    "SAMPLE_MAX",
    "SAMPLE_MIN",
    "binding_for",
    "plugin",
]

NAME = "duplicates"
CACHE_FILE = "sha256-v2.cache"
CHUNK_BYTES = 1024 * 1024
"""Read size while hashing; peak memory per worker is one chunk."""
MAX_MEMBERS_LISTED = 20
"""Images named in one duplicate finding; ``count`` always carries the true group size."""
SAMPLE_FRACTION = 0.01
SAMPLE_MIN = 10
SAMPLE_MAX = 1000
"""Cache hits re-hashed per run: 1 percent, at least 10, at most 1,000 (all if fewer)."""
GROUP_ID_HEX = 12
_REAL_SPLITS = ("train", "val", "test")


def binding_for(ds: Dataset) -> str:
    """Key material that ties cache entries to this tool version and this dataset version."""
    return f"{__version__}|{ds.manifest.version}"


def sample_size(hits: int) -> int:
    """Number of cache hits to re-verify when there are ``hits`` of them."""
    return min(hits, SAMPLE_MAX, max(SAMPLE_MIN, math.ceil(hits * SAMPLE_FRACTION)))


def pick_sample(keys: list[str], run_id: str) -> list[str]:
    """Deterministic-random sample of ``keys``: ordered by ``sha256(run_id:key)``, so it is
    reproducible for a run and not predictable by someone who writes the cache beforehand."""
    ranked = sorted(keys, key=lambda k: hashlib.sha256(f"{run_id}:{k}".encode()).digest())
    return ranked[: sample_size(len(keys))]


def hash_file(
    path: Path, max_bytes: int, *, nofollow: bool = False, expected_key: str | None = None
) -> tuple[str | None, str]:
    """sha256 hex of ``path`` and a status: ``ok``, ``file_too_large``, ``not_a_regular_file``,
    ``file_missing``, ``unreadable`` or ``changed_during_audit``.

    The file is read in ``CHUNK_BYTES`` chunks and abandoned (``file_too_large``) as soon as more
    than ``max_bytes`` have been read, so a file that grows while it is hashed is bounded. With
    ``expected_key`` the descriptor's own ``fstat`` identity must produce that cache key, so the
    digest is never attributed to a file other than the one that was keyed.
    """
    digest = hashlib.sha256()
    total = 0
    try:
        with files.open_regular(path, nofollow=nofollow) as handle:
            if (
                expected_key is not None
                and cache_key(path, os.fstat(handle.fileno())) != expected_key
            ):
                return None, "changed_during_audit"
            while chunk := handle.read(CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    return None, "file_too_large"
                digest.update(chunk)
    except FormatError as exc:
        return None, "not_a_regular_file" if "not a regular file" in str(exc) else "file_missing"
    except OSError:
        return None, "unreadable"
    return digest.hexdigest(), "ok"


def _open_cache(ctx: CheckContext, ds: Dataset) -> HashCache:
    if not ctx.use_hash_cache:
        return HashCache.disabled()
    log = get_logger(__name__)
    try:
        key = ctx.cache_key.get_secret_value().encode("utf-8") if ctx.cache_key else None
        cache = HashCache.open(
            ctx.cache_dir, CACHE_FILE, binding=binding_for(ds), key=key, in_ci=ctx.in_ci
        )
    except OSError as exc:
        log.warning("audit_cache_unavailable", reason=type(exc).__name__)
        return HashCache.disabled()
    if cache.untrusted_reason is not None:
        log.warning("audit_cache_untrusted", reason=cache.untrusted_reason)
    return cache


class _Target:
    """One local image to hash: where, and the cache key of its current file version."""

    __slots__ = ("image", "key", "path", "size")

    def __init__(self, image: Image, path: Path, key: str, size: int) -> None:
        self.image, self.path, self.key, self.size = image, path, key, size


def _hash_all(
    ds: Dataset, ctx: CheckContext
) -> tuple[dict[str, str], Counter[str], int, int, tuple[str, int, int] | None]:
    """``image_id -> sha256`` for hashable images, skips by reason, the spot-check result
    ``(sampled, mismatched)`` and, if a budget ended the run early, ``(reason, checked, total)``."""
    skipped: Counter[str] = Counter()
    cache = _open_cache(ctx, ds)
    hashes: dict[str, str] = {}
    todo: list[_Target] = []
    hits: dict[str, _Target] = {}
    for image in ds.images:
        resolved = resolve_image_path(
            image.file_path, ctx.root, allow_external=ctx.allow_external_images
        )
        if resolved.kind != "local" or resolved.path is None:
            skipped[
                {"remote": "remote", "external": "outside_root"}.get(resolved.kind, "unsafe_path")
            ] += 1
            continue
        try:
            with files.open_regular(resolved.path, nofollow=True) as handle:
                info = os.fstat(handle.fileno())
        except FormatError as exc:
            skipped[
                "not_a_regular_file" if "not a regular file" in str(exc) else "file_missing"
            ] += 1
            continue
        except OSError:
            skipped["unreadable"] += 1
            continue
        if info.st_size > ctx.config.max_image_bytes:
            skipped["file_too_large"] += 1
            continue
        key = cache_key(resolved.path, info)
        target = _Target(image, resolved.path, key, info.st_size)
        cached = cache.get(key)
        if cached is not None:
            hashes[image.image_id] = cached
            hits[image.image_id] = target
        else:
            todo.append(target)
    sampled = mismatched = 0
    incomplete: tuple[str, int, int] | None = None
    try:
        max_bytes = ctx.config.max_image_bytes
        by_key = {t.key: t for t in hits.values()}
        for key in pick_sample(list(by_key), ctx.run_id):
            digest, status = hash_file(by_key[key].path, max_bytes, nofollow=True, expected_key=key)
            sampled += 1
            if status == "ok" and digest != cache.get(key):
                mismatched += 1
        if mismatched:
            get_logger(__name__).warning(
                "audit_cache_discarded", sampled=sampled, mismatched=mismatched
            )
            cache.discard()
            for image_id, target in hits.items():
                del hashes[image_id]
                todo.append(target)
        total_todo = len(todo)
        budget = ctx.config.max_hash_bytes
        used = 0
        for n, target in enumerate(todo):
            used += target.size
            if used > budget:
                skipped["byte_budget"] += total_todo - n
                incomplete = ("byte_budget", n, total_todo)
                todo = todo[:n]
                break
        deadline = time.monotonic() + ctx.config.max_check_seconds
        results = pool.bounded_map(
            lambda t: hash_file(t.path, max_bytes, nofollow=True, expected_key=t.key),
            todo,
            workers=ctx.workers,
            deadline=deadline,
        )
        reached = _collect(todo, results, cache, hashes, skipped)
        if reached < len(todo):
            skipped["time_budget"] += len(todo) - reached
            incomplete = incomplete or ("time_budget", reached, total_todo)
    finally:
        cache.flush()
    return hashes, skipped, sampled, mismatched, incomplete


def _collect(
    todo: list[_Target],
    results: Iterable[tuple[str | None, str]],
    cache: HashCache,
    hashes: dict[str, str],
    skipped: Counter[str],
) -> int:
    """Fold a prefix of the results into ``hashes`` / ``skipped``; return how many were consumed."""
    reached = 0
    for target, (digest, status) in zip(todo, results, strict=False):
        reached += 1
        if digest is None:
            skipped[status] += 1
        else:
            hashes[target.image.image_id] = digest
            cache.put(target.key, digest)
    return reached


def _group_finding(ctx: CheckContext, digest: str, members: list[Image]) -> Finding:
    members = sorted(members, key=lambda i: i.image_id)
    splits = Counter(i.split for i in members if i.split in _REAL_SPLITS)
    cross = len(splits) > 1 or (bool(splits) and any(i.split == "unassigned" for i in members))
    return make_finding(
        ctx.run_id,
        NAME,
        "CRITICAL" if cross else "WARN",
        image_id=members[0].image_id,
        payload={
            "rule": "exact_duplicate",
            "group_id": digest[:GROUP_ID_HEX],
            "count": len(members),
            "cross_split": cross,
            "splits": dict(sorted(Counter(i.split for i in members).items())),
            "members": [
                {"image_id": i.image_id, "split": i.split, "file_name": display_name(i.file_path)}
                for i in members[:MAX_MEMBERS_LISTED]
            ],
        },
    )


class DuplicatesCheck:
    spec = PluginSpec(
        name=NAME, kind="check", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        hashes, skipped, sampled, mismatched, incomplete = _hash_all(ds, ctx)
        findings: list[Finding] = []
        by_hash: dict[str, list[Image]] = defaultdict(list)
        for image in ds.images:
            digest = hashes.get(image.image_id)
            if digest is None:
                continue
            by_hash[digest].append(image)
            if digest[:16] != image.image_id:
                findings.append(
                    make_finding(
                        ctx.run_id,
                        NAME,
                        "CRITICAL",
                        image_id=image.image_id,
                        payload={
                            "rule": "content_changed",
                            "file_name": display_name(image.file_path),
                            "expected": image.image_id,
                            "actual": digest[:GROUP_ID_HEX],
                        },
                    )
                )
        findings += [_group_finding(ctx, d, m) for d, m in by_hash.items() if len(m) > 1]
        if incomplete is not None:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "CRITICAL" if ctx.incomplete_is_critical else "WARN",
                    payload={
                        "rule": "audit_incomplete",
                        "reason": incomplete[0],
                        "checked": incomplete[1],
                        "total": incomplete[2],
                    },
                )
            )
        if mismatched:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    "WARN",
                    payload={
                        "rule": "hash_cache_mismatch",
                        "sampled": sampled,
                        "mismatched": mismatched,
                    },
                )
            )
        if skipped:
            findings.append(
                make_finding(
                    ctx.run_id,
                    NAME,
                    unchecked_severity(
                        sum(skipped.values()),
                        len(ds.images),
                        max_fraction=ctx.config.max_unchecked_fraction,
                        require_local=ctx.require_local_images,
                    ),
                    payload={
                        "rule": "images_not_hashed",
                        "count": sum(skipped.values()),
                        "total": len(ds.images),
                        "reasons": dict(sorted(skipped.items())),
                    },
                )
            )
        return findings


plugin = DuplicatesCheck()
