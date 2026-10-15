"""Embedding cache: bounded, digest-verified, authenticated shards inside ``dataset.dcp``.

Layout (one directory per model revision)::

    dataset.dcp/embeddings/<model_id>@<revision>/
        manifest.json                  # identity, dataset version, weights, shard digests, MAC
        shard-<32 hex>.parquet         # image_id, vector list<float16|float32>, content_sha256

**Why this is defended like an input.** The directory travels with the dataset, so a pull request
can change it. A forged cache that maps a test image to a train image's vector would hide
leakage and make the gate pass. The layers, in the order they are applied on read:

1. *Containment.* The ``embeddings`` directory and every file are opened without following
   symlinks and must be regular files; names come from a fixed pattern, never from the manifest
   text as a path.
2. *Bounds from metadata.* Manifest, shard size, declared rows, dimensions, shard count and the
   decoded size (from the Parquet footer) are checked before any row is decoded.
3. *Digest per shard.* The sha256 of the shard bytes must equal the manifest entry; the shard is
   parsed from the very bytes that were hashed (no re-open, no time-of-check gap).
4. *Authentication.* The manifest carries an HMAC over its canonical JSON (which contains every
   shard digest), keyed by a secret (with a domain label, so it is not the hash cache's key) and
   bound to model, revision, dimensions, dtype, weight hashes, preprocessing version and the
   dataset version. Without the key an attacker cannot make a forged shard verify. Digests alone
   only detect damage; the MAC is what defends against forgery. When no key is available the
   cache is *disabled* (see ``audit/embedding.py``) rather than trusted.
5. *Content.* Ids are 16-hex and unique, vectors have exactly ``dims`` finite values and unit
   norm, the declared metadata matches the manifest, and an id may occur in only one shard.
6. *Binding to the image bytes.* Every row carries the full sha256 of the bytes its vector was
   computed from (covered by the shard digest and so by the MAC). The cache is bound to the dataset
   *rows* through the manifest version, which does not see the image files; the audit therefore
   hashes the file of every cache hit and recomputes any vector whose file changed. ``image_id`` is
   only a 64-bit label and is never the thing compared.
7. *One manifest, one parse.* The manifest is parsed once, with duplicate keys refused, and the
   same object feeds the MAC and the model.

A shard that fails any check is ignored and counted by reason; a manifest that fails makes the
whole cache untrusted. Nothing is repaired in place: the next write replaces it. Writes are
atomic (temporary file, fsync, rename, directory fsync); the manifest is renamed last, so a crash
leaves either the previous cache or the previous cache plus an unlisted shard that readers ignore
and the next write removes. The cache is **version-bound**: a manifest written for another
dataset version is "stale" and ignored (embeddings are recomputed), never reused.

**Rounding.** Vectors are L2-normalised, rounded to the storage dtype (IEEE-754 round to nearest
even; ``float16`` keeps 11 significant bits, absolute error below 1e-3 per component for unit
vectors) and converted back to ``float32``; they are *not* normalised again, so the norm of a
stored vector is within 1e-3 of one (float16) and the kNN index re-normalises its working copy.
:func:`canonical_vectors` is that function and the pipeline applies it to fresh vectors too, so a
computed vector and a cached one are bit-identical and a check's verdict cannot depend on whether
the cache was warm.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import stat
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationError, field_validator

from datacopilot.core.errors import FormatError
from datacopilot.io import files
from datacopilot.models.registry import MAX_DIMS

__all__ = [
    "MAX_CACHE_ROWS",
    "MAX_DIMS",
    "MAX_MANIFEST_BYTES",
    "MAX_SHARDS",
    "MAX_SHARD_BYTES",
    "MAX_SHARD_DECODED_BYTES",
    "MAX_SHARD_ROWS",
    "CacheFullError",
    "CacheIdentity",
    "EmbeddingCache",
    "LoadResult",
    "canonical_vectors",
    "encode_shard",
    "mac_for",
]

Dtype = Literal["float16", "float32"]
State = Literal["absent", "ok", "stale", "rejected", "disabled"]

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
"""Largest ``manifest.json`` read (2,048 shard entries need about 400 KiB)."""
MAX_SHARD_BYTES = 128 * 1024 * 1024
"""Largest shard file read into memory."""
MAX_SHARD_ROWS = 16_384
"""Most rows one shard may declare or contain."""
MAX_SHARD_DECODED_BYTES = 256 * 1024 * 1024
"""Most uncompressed bytes a shard may declare in its footer (vectors plus ids)."""
MAX_SHARDS = 2_048
"""Most shards one manifest may list; a cache that needs more is full."""
MAX_CACHE_ROWS = 2_000_000
"""Most embeddings one cache may hold (50,000 x 768 float16 is 77 MB; this is 3 GB of float16)."""
WRITE_SHARD_ROWS = 4_096
"""Rows the pipeline puts in one shard (a resumable unit of work)."""
FORMAT_VERSION = 2
"""2: shards carry the full sha256 of each image's bytes; the MAC key has a domain label and the
identity names the preprocessing version."""
MAC_LABEL = b"datacopilot/embedding-cache/v2"
"""Domain separation of the MAC key from every other use of the cache secret (the hash cache)."""
PREPROCESS_VERSION = 1
"""Bumped when image decoding or preprocessing changes the vectors a model produces."""
_ID = re.compile(r"^[0-9a-f]{16}$")
_NAME_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHARD_NAME = re.compile(r"^shard-[0-9a-f]{32}\.parquet$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[0-9a-f]{32}$")
_NORM_TOLERANCE = 1e-2
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_META = {
    "model_id": b"dcp_model_id",
    "revision": b"dcp_revision",
    "dims": b"dcp_dims",
    "dtype": b"dcp_dtype",
    "dataset_version": b"dcp_dataset_version",
}


class CacheFullError(Exception):
    """``MAX_CACHE_ROWS`` or ``MAX_SHARDS`` was reached; the caller continues in memory."""


@dataclass(frozen=True)
class CacheIdentity:
    """What a cache is *for*; any difference makes an existing cache unusable.

    Attributes:
        model_id: embedder name.
        revision: model revision (pinned commit for hub models).
        dims: vector width.
        dtype: storage dtype.
        dataset_version: ``manifest.version`` the vectors were computed for.
        weights: ``file -> sha256`` of the weights and configuration, recorded for provenance;
            ``{}`` for a model without weights.
        preprocess: version of the decoding and preprocessing code (``PREPROCESS_VERSION``).
    """

    model_id: str
    revision: str
    dims: int
    dtype: Dtype
    dataset_version: str
    weights: dict[str, str] = field(default_factory=dict)
    preprocess: int = PREPROCESS_VERSION

    def __post_init__(self) -> None:
        if not _NAME_PART.fullmatch(self.model_id) or not _NAME_PART.fullmatch(self.revision):
            raise ValueError("model_id and revision must be plain names")
        if not 1 <= self.dims <= MAX_DIMS:
            raise ValueError(f"dims must be between 1 and {MAX_DIMS}")
        if self.dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")
        if not _VERSION.fullmatch(self.dataset_version):
            raise ValueError("dataset_version must be 32 hex digits")
        if len(self.weights) > 16 or not all(
            _NAME_PART.fullmatch(k) and _SHA256.fullmatch(v) for k, v in self.weights.items()
        ):
            raise ValueError("weights must map plain file names to sha256 digests")
        if not 1 <= self.preprocess <= 10_000:
            raise ValueError("preprocess must be a small positive version number")

    @property
    def directory_name(self) -> str:
        return f"{self.model_id}@{self.revision}"

    def binding(self) -> bytes:
        material = json.dumps(
            [
                FORMAT_VERSION,
                self.model_id,
                self.revision,
                self.dims,
                self.dtype,
                self.dataset_version,
                sorted(self.weights.items()),
                self.preprocess,
            ],
            separators=(",", ":"),
        )
        return material.encode("utf-8")


@dataclass(frozen=True)
class LoadResult:
    """Verified contents of a cache.

    Attributes:
        ids: image ids, sorted.
        digests: full sha256 hex of the bytes each vector came from, in the order of ``ids``.
        matrix: canonical ``float32`` vectors, one row per id, in the order of ``ids``.
        rejected: count of rejected parts by reason (a shard counts once; a manifest-level
            failure counts once under its reason). Empty when nothing was refused.
        state: ``absent`` (no cache), ``ok``, ``stale`` (other dataset version or identity:
            ignored, not an attack), ``rejected`` (manifest failed verification) or ``disabled``.
    """

    ids: tuple[str, ...]
    digests: tuple[str, ...]
    matrix: NDArray[np.float32]
    rejected: dict[str, int]
    state: State


class _ShardEntry(BaseModel, frozen=True, extra="forbid"):
    name: str
    rows: int = Field(ge=1)
    bytes: int = Field(ge=1)
    sha256: str

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _SHARD_NAME.fullmatch(value):
            raise ValueError("invalid shard name")
        return value

    @field_validator("sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("invalid digest")
        return value

    @field_validator("rows")
    @classmethod
    def _rows(cls, value: int) -> int:
        if value > MAX_SHARD_ROWS:
            raise ValueError("too many rows")
        return value

    @field_validator("bytes")
    @classmethod
    def _bytes(cls, value: int) -> int:
        if value > MAX_SHARD_BYTES:
            raise ValueError("shard too large")
        return value


class _Manifest(BaseModel, frozen=True, extra="forbid"):
    format: int
    model_id: str
    revision: str
    dims: int
    dtype: Dtype
    dataset_version: str
    preprocess: int
    weights: dict[str, str] = Field(max_length=16)
    shards: tuple[_ShardEntry, ...] = Field(max_length=2_048)
    mac: str | None = None

    @field_validator("mac")
    @classmethod
    def _mac(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("invalid mac")
        return value


def canonical_vectors(vectors: NDArray[np.float32], dtype: Dtype) -> NDArray[np.float32]:
    """L2-normalise, then round to ``dtype`` and widen back to ``float32``.

    The result is exactly what a cache stores and later returns, so fresh and cached vectors are
    bit-identical. Rounding is IEEE-754 round to nearest even; for ``float16`` each component of a
    unit vector moves by less than 1e-3 and the norm stays within 1e-3 of one (the vector is not
    normalised again after rounding, because that would make the function non-idempotent). It is
    applied once, to fresh vectors; stored vectors are used as they are.

    Raises:
        ValueError: not a 2-D array, a non-finite value, or a zero vector.
    """
    arr = np.asarray(vectors)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError("vectors must be a 2-D array")
    work = arr.astype(np.float32, copy=False)
    if not np.isfinite(work).all():
        raise ValueError("vectors contain NaN or infinity")
    norms = np.linalg.norm(work, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("vectors contain a zero vector")
    unit = (work / norms).astype(np.float32)
    rounded: NDArray[np.float32] = unit.astype(np.dtype(dtype)).astype(np.float32)
    if (np.linalg.norm(rounded, axis=1) <= 0).any():
        raise ValueError("vectors collapse to zero at this precision")
    return rounded


def _require_canonical(vectors: NDArray[np.float32], dtype: Dtype) -> None:
    arr = np.asarray(vectors)
    if arr.ndim != 2 or arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise ValueError("vectors must be a finite 2-D float32 array")
    if not np.array_equal(arr.astype(np.dtype(dtype)).astype(np.float32), arr):
        raise ValueError("vectors must be canonical (see canonical_vectors)")
    if not np.all(np.abs(np.linalg.norm(arr, axis=1) - 1.0) <= _NORM_TOLERANCE):
        raise ValueError("vectors must have unit norm")


def _mac_key(secret: bytes, identity: CacheIdentity) -> bytes:
    return hashlib.sha256(MAC_LABEL + b"\x00" + secret + b"\x00" + identity.binding()).digest()


def _canonical_json(doc: dict[str, object]) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def mac_for(doc: dict[str, object], secret: bytes, identity: CacheIdentity) -> str:
    """HMAC-SHA256 hex of ``doc`` (without a ``mac`` key) under ``secret`` bound to ``identity``."""
    body = {k: v for k, v in doc.items() if k != "mac"}
    return hmac.new(_mac_key(secret, identity), _canonical_json(body), hashlib.sha256).hexdigest()


def _vector_type(dtype: Dtype) -> pa.DataType:
    return pa.float16() if dtype == "float16" else pa.float32()


def encode_shard(
    identity: CacheIdentity,
    image_ids: list[str],
    vectors: NDArray[np.float32],
    digests: list[str],
) -> bytes:
    """Parquet bytes of one shard: ids, canonical vectors in the storage dtype and the full sha256
    of the image bytes each vector was computed from.

    ``vectors`` must already be canonical (:func:`canonical_vectors`): storing them must not
    change a single bit, so what the caller holds in memory is what a later run reads back.

    The shard's schema metadata repeats the identity so a shard moved into another cache is
    refused even if its digest were listed.
    """
    _require_canonical(vectors, identity.dtype)
    n, dims = vectors.shape
    flat = pa.array(vectors.reshape(-1).astype(np.dtype(identity.dtype)))
    offsets = pa.array(np.arange(0, (n + 1) * dims, dims, dtype=np.int32))
    table = pa.table(
        {
            "image_id": pa.array(image_ids, type=pa.string()),
            "vector": pa.ListArray.from_arrays(offsets, flat),
            "content_sha256": pa.array(digests, type=pa.string()),
        }
    )
    meta = {
        _META["model_id"]: identity.model_id.encode(),
        _META["revision"]: identity.revision.encode(),
        _META["dims"]: str(identity.dims).encode(),
        _META["dtype"]: identity.dtype.encode(),
        _META["dataset_version"]: identity.dataset_version.encode(),
    }
    buffer = io.BytesIO()
    pq.write_table(
        table.replace_schema_metadata(meta), buffer, compression="zstd", row_group_size=n
    )
    return buffer.getvalue()


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``object_pairs_hook`` that refuses a repeated key.

    The manifest is parsed once and the same object feeds the MAC and the model, so a document in
    which the two parsers could disagree about a duplicated key does not exist.
    """
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


class _Reject(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _read_file(path: Path, limit: int, *, missing: str, unreadable: str, oversized: str) -> bytes:
    """Bytes of a regular file opened without following symlinks, at most ``limit``."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        raise _Reject(missing) from None
    except OSError:
        raise _Reject(unreadable) from None
    try:
        with files.open_regular(path, nofollow=True) as handle:
            if os.fstat(handle.fileno()).st_size > limit:
                raise _Reject(oversized)
            data = handle.read(limit + 1)
    except FormatError:
        raise _Reject(unreadable) from None
    except OSError:
        raise _Reject(unreadable) from None
    if len(data) > limit:
        raise _Reject(oversized)
    return data


def _decode_shard(
    data: bytes, entry: _ShardEntry, identity: CacheIdentity
) -> tuple[list[str], list[str], NDArray[np.float32]]:
    """Validate one shard that already matched its digest; ``_Reject`` names what was wrong."""
    try:
        pf = pq.ParquetFile(io.BytesIO(data))
        meta = pf.metadata
        if meta.num_rows > MAX_SHARD_ROWS:
            raise _Reject("too_many_rows")
        if meta.num_rows != entry.rows:
            raise _Reject("bad_shape")
        decoded = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        if (
            decoded > MAX_SHARD_DECODED_BYTES
            or meta.num_rows * identity.dims * 4 > MAX_SHARD_DECODED_BYTES
        ):
            raise _Reject("too_many_rows")
        schema = pf.schema_arrow
        declared = schema.metadata or {}
        expected = {
            _META["model_id"]: identity.model_id,
            _META["revision"]: identity.revision,
            _META["dims"]: str(identity.dims),
            _META["dtype"]: identity.dtype,
            _META["dataset_version"]: identity.dataset_version,
        }
        if any(declared.get(k) != v.encode() for k, v in expected.items()):
            raise _Reject("bad_shape")
        if schema.names != ["image_id", "vector", "content_sha256"]:
            raise _Reject("bad_shape")
        vtype = schema.field("vector").type
        if (
            schema.field("image_id").type != pa.string()
            or schema.field("content_sha256").type != pa.string()
            or not pa.types.is_list(vtype)
            or vtype.value_type != _vector_type(identity.dtype)
        ):
            raise _Reject("bad_shape")
        table = pf.read()
    except _Reject:
        raise
    except (pa.ArrowException, OSError, ValueError, TypeError):
        raise _Reject("bad_parquet") from None
    vec = table.column("vector").combine_chunks()
    if vec.null_count or not len(vec) == entry.rows:
        raise _Reject("bad_shape")
    lengths = pc.list_value_length(vec)
    if not pc.all(pc.equal(lengths, identity.dims)).as_py():
        raise _Reject("bad_shape")
    flat = vec.flatten()
    if flat.null_count:
        raise _Reject("bad_shape")
    matrix = (
        flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(entry.rows, identity.dims)
    )
    id_col = table.column("image_id")
    if id_col.null_count:
        raise _Reject("bad_values")
    image_ids = id_col.to_pylist()
    if not all(isinstance(i, str) and _ID.fullmatch(i) for i in image_ids) or len(
        set(image_ids)
    ) != len(image_ids):
        raise _Reject("bad_values")
    digest_col = table.column("content_sha256")
    if digest_col.null_count:
        raise _Reject("bad_values")
    digests = digest_col.to_pylist()
    if not all(isinstance(d, str) and _SHA256.fullmatch(d) for d in digests):
        raise _Reject("bad_values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.all(np.abs(norms - 1.0) <= _NORM_TOLERANCE):
        raise _Reject("bad_values")
    return image_ids, digests, matrix


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    staged = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            staged.unlink()
        except OSError:
            pass
        raise


def _is_real_directory(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode)


class EmbeddingCache:
    """Reads and appends verified shards for one :class:`CacheIdentity`.

    Use :meth:`open` with a secret, or :meth:`disabled`. ``load`` verifies everything and returns
    what can be trusted; ``append`` writes one more shard (raising ``OSError`` when the dataset
    directory is read-only and :class:`CacheFullError` at the caps) and is a no-op when disabled.
    """

    def __init__(
        self,
        root: Path | None,
        identity: CacheIdentity | None,
        secret: bytes | None,
        disabled_reason: str | None = None,
    ) -> None:
        self._root = root
        self._identity = identity
        self._secret = secret
        self.disabled_reason = disabled_reason
        self._shards: list[_ShardEntry] = []
        self._ids: set[str] = set()
        self._loaded = False

    @classmethod
    def open(cls, root: Path, identity: CacheIdentity, secret: bytes | None) -> EmbeddingCache:
        """Cache of ``identity`` below ``root`` (``<dataset>/embeddings``); nothing is touched yet.

        Without a ``secret`` the cache is disabled: unauthenticated vectors are never trusted.
        """
        if secret is None:
            return cls.disabled("no_key")
        return cls(root, identity, secret)

    @classmethod
    def disabled(cls, reason: str) -> EmbeddingCache:
        return cls(None, None, None, reason)

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def _directory(self) -> Path:
        assert self._root is not None and self._identity is not None  # noqa: S101
        return self._root / self._identity.directory_name

    def _empty(self, state: State, rejected: dict[str, int] | None = None) -> LoadResult:
        dims = self._identity.dims if self._identity else 1
        return LoadResult((), (), np.zeros((0, dims), dtype=np.float32), rejected or {}, state)

    def load(self) -> LoadResult:
        """Verify the cache and return every vector that passed (see the module docstring)."""
        self._shards, self._ids, self._loaded = [], set(), True
        if self._root is None or self._identity is None or self._secret is None:
            return self._empty("disabled")
        identity = self._identity
        if self._root.is_symlink() or (
            os.path.lexists(self._root) and not _is_real_directory(self._root)
        ):
            return self._empty("rejected", {"not_a_directory": 1})
        directory = self._directory()
        if not os.path.lexists(directory):
            return self._empty("absent")
        if not _is_real_directory(directory):
            return self._empty("rejected", {"not_a_directory": 1})
        try:
            raw = _read_file(
                directory / "manifest.json",
                MAX_MANIFEST_BYTES,
                missing="absent",
                unreadable="manifest_unreadable",
                oversized="manifest_unreadable",
            )
        except _Reject as exc:
            if exc.reason == "absent":
                return self._empty("absent")
            return self._empty("rejected", {exc.reason: 1})
        try:
            doc = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
            manifest = _Manifest.model_validate(doc)
        except (ValidationError, ValueError, RecursionError, TypeError):
            return self._empty("rejected", {"manifest_invalid": 1})
        if (
            manifest.format != FORMAT_VERSION
            or manifest.model_id != identity.model_id
            or manifest.revision != identity.revision
            or manifest.dims != identity.dims
            or manifest.dtype != identity.dtype
            or manifest.dataset_version != identity.dataset_version
            or manifest.weights != identity.weights
            or manifest.preprocess != identity.preprocess
        ):
            return self._empty("stale")
        expected_mac = mac_for(doc, self._secret, identity)
        if manifest.mac is None or not hmac.compare_digest(manifest.mac, expected_mac):
            return self._empty("rejected", {"bad_mac": 1})
        return self._load_shards(directory, manifest)

    def _load_shards(self, directory: Path, manifest: _Manifest) -> LoadResult:
        assert self._identity is not None  # noqa: S101
        identity = self._identity
        rejected: Counter[str] = Counter()
        all_ids: list[str] = []
        all_digests: list[str] = []
        blocks: list[NDArray[np.float32]] = []
        seen: set[str] = set()
        total = 0
        for entry in manifest.shards:
            total += entry.rows
            if total > MAX_CACHE_ROWS:
                rejected["too_many_rows"] += 1
                break
            try:
                data = _read_file(
                    directory / entry.name,
                    MAX_SHARD_BYTES,
                    missing="shard_missing",
                    unreadable="shard_unreadable",
                    oversized="bad_size",
                )
                if len(data) != entry.bytes:
                    raise _Reject("bad_size")
                if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), entry.sha256):
                    raise _Reject("bad_digest")
                shard_ids, shard_digests, matrix = _decode_shard(data, entry, identity)
                if not seen.isdisjoint(shard_ids):
                    raise _Reject("duplicate_ids")
            except _Reject as exc:
                rejected[exc.reason] += 1
                continue
            seen.update(shard_ids)
            all_ids.extend(shard_ids)
            all_digests.extend(shard_digests)
            blocks.append(matrix)
            self._shards.append(entry)
        self._ids = seen
        if not blocks:
            return self._empty("ok", dict(rejected))
        matrix = np.concatenate(blocks, axis=0)
        order = sorted(range(len(all_ids)), key=all_ids.__getitem__)
        return LoadResult(
            tuple(all_ids[i] for i in order),
            tuple(all_digests[i] for i in order),
            matrix[order],
            dict(rejected),
            "ok",
        )

    def reset(self) -> None:
        """Forget every shard: the next ``append`` publishes a fresh manifest and the old shard
        files, now unlisted, are removed. Used when the cache is found to disagree with the
        images (the spot check); nothing on disk changes until then."""
        self._shards, self._ids, self._loaded = [], set(), True

    def append(
        self, image_ids: list[str], vectors: NDArray[np.float32], digests: list[str]
    ) -> None:
        """Write ``vectors`` as one new shard and publish it in the manifest.

        ``digests`` are the full sha256 hex digests of the image bytes the vectors were computed
        from; a later run recomputes any vector whose file no longer hashes to its digest.

        Raises:
            ValueError: ids not unique 16-hex strings, already cached, or a bad shape or value.
            CacheFullError: ``MAX_CACHE_ROWS`` or ``MAX_SHARDS`` would be exceeded.
            OSError: the dataset directory is not writable, or ``embeddings`` is a symlink.
        """
        if self._root is None or self._identity is None or self._secret is None:
            return
        identity = self._identity
        _require_canonical(vectors, identity.dtype)
        if not self._loaded:
            self.load()
        n = len(image_ids)
        if n == 0 or n > MAX_SHARD_ROWS:
            raise ValueError(f"a shard holds between 1 and {MAX_SHARD_ROWS} rows")
        if vectors.shape != (n, identity.dims):
            raise ValueError("vectors must have shape (len(image_ids), dims)")
        if len(set(image_ids)) != n or not all(_ID.fullmatch(i) for i in image_ids):
            raise ValueError("image ids must be unique 16-hex strings")
        if len(digests) != n or not all(_SHA256.fullmatch(d) for d in digests):
            raise ValueError("digests must be one sha256 hex digest per image")
        if not self._ids.isdisjoint(image_ids):
            raise ValueError("image id is already cached")
        if len(self._ids) + n > MAX_CACHE_ROWS or len(self._shards) >= MAX_SHARDS:
            raise CacheFullError
        data = encode_shard(identity, image_ids, vectors, digests)
        if len(data) > MAX_SHARD_BYTES:
            raise ValueError("shard would exceed the size limit")
        if self._root.is_symlink():
            raise OSError("embeddings must be a directory inside the dataset, not a symlink")
        self._root.mkdir(exist_ok=True)
        directory = self._directory()
        if directory.is_symlink():
            raise OSError("cache directory is a symlink")
        directory.mkdir(exist_ok=True)
        entry = _ShardEntry(
            name=f"shard-{uuid.uuid4().hex}.parquet",
            rows=n,
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        _write_atomic(directory / entry.name, data)
        shards = [*self._shards, entry]
        doc: dict[str, object] = {
            "format": FORMAT_VERSION,
            "model_id": identity.model_id,
            "revision": identity.revision,
            "dims": identity.dims,
            "dtype": identity.dtype,
            "dataset_version": identity.dataset_version,
            "preprocess": identity.preprocess,
            "weights": dict(sorted(identity.weights.items())),
            "shards": [s.model_dump() for s in shards],
        }
        doc["mac"] = mac_for(doc, self._secret, identity)
        _write_atomic(directory / "manifest.json", json.dumps(doc, sort_keys=True).encode("ascii"))
        self._shards = shards
        self._ids.update(image_ids)
        self._collect_garbage(directory, {s.name for s in shards})

    @staticmethod
    def _collect_garbage(directory: Path, listed: set[str]) -> None:
        """Remove unlisted shards and abandoned temporary files (only our own name patterns)."""
        try:
            with os.scandir(directory) as it:
                names = [e.name for e in it if e.is_file(follow_symlinks=False)]
        except OSError:
            return
        for name in names:
            ours = (_SHARD_NAME.fullmatch(name) and name not in listed) or (
                name.startswith("shard-") and ".tmp-" in name
            )
            if ours:
                try:
                    (directory / name).unlink()
                except OSError:
                    pass
