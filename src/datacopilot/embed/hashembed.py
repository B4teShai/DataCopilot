"""``hash``: a dependency-free test embedder (SimHash of byte shingles).

It exists so the whole pipeline (cache, kNN, checks, CLI) can be tested and benchmarked without
model weights, a network or Pillow. It is *not* a visual embedding: it compares file bytes.
Two files that share most of their bytes (the same image with a different trailer or metadata)
get nearly parallel vectors; unrelated files get nearly orthogonal ones.

Method: the first ``MAX_HASH_BYTES`` bytes are cut into overlapping 8-byte shingles, duplicates
removed; each shingle votes +1 or -1 on each of ``DIMS`` dimensions according to a bit of a
splitmix64 hash of the shingle and a per-dimension seed; the votes are summed and normalised.
Everything is integer arithmetic on ``uint64``, so the result is identical on every platform.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from datacopilot.core.schema import PluginSpec
from datacopilot.embed.protocol import Embedder, ImageDecodeError
from datacopilot.models.registry import require_license

DIMS = 128
MAX_HASH_BYTES = 256 * 1024
"""Bytes of one file that are considered (bounds time and memory per image)."""
SHINGLE = 8
_CHUNK = 8192
_U64 = np.uint64


def _splitmix(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    z = x + _U64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    mixed: NDArray[np.uint64] = z ^ (z >> _U64(31))
    return mixed


_SEEDS = _splitmix(np.arange(1, DIMS + 1, dtype=np.uint64))


def _shingles(data: bytes) -> NDArray[np.uint64]:
    raw = np.frombuffer(data[:MAX_HASH_BYTES].ljust(SHINGLE, b"\x00"), dtype=np.uint8)
    windows = np.lib.stride_tricks.sliding_window_view(raw, SHINGLE)
    packed: NDArray[np.uint64] = np.zeros(windows.shape[0], dtype=np.uint64)
    for k in range(SHINGLE):
        packed = (packed << _U64(8)) | windows[:, k].astype(np.uint64)
    unique: NDArray[np.uint64] = np.unique(packed)
    return unique


class HashEmbedder:
    model_id = "hash"
    revision = "1"
    dims = DIMS
    weights: dict[str, str] = {}  # noqa: RUF012  # nothing to download, nothing to verify

    def embed(self, images: Sequence[bytes]) -> NDArray[np.float32]:
        out = np.zeros((len(images), DIMS), dtype=np.float32)
        for row, data in enumerate(images):
            if not data:
                raise ImageDecodeError(row, "decode_failed")
            shingles = _shingles(data)
            votes = np.zeros(DIMS, dtype=np.int64)
            for start in range(0, len(shingles), _CHUNK):
                block = shingles[start : start + _CHUNK]
                bits = _splitmix(block[:, None] ^ _SEEDS[None, :]) >> _U64(63)
                votes += (bits.astype(np.int64) * 2 - 1).sum(axis=0)
            vector = votes.astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ImageDecodeError(row, "decode_failed")
            out[row] = vector / norm
        return out


class HashPlugin:
    spec = PluginSpec(
        name="hash",
        kind="embedder",
        license="Apache-2.0",
        extra=None,
        gpu_memory_gb=None,
        version="1",
    )

    def create(self, allowed_licenses: frozenset[str]) -> Embedder:
        require_license("embedder 'hash'", self.spec.license, allowed_licenses)
        return HashEmbedder()


plugin = HashPlugin()
