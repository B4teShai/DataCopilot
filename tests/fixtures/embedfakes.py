"""Fake embedders and embedding sets for the embedding checks: no weights, no network.

``FakeEmbedder`` maps the *bytes* of an image to a vector, so a test plants near-duplicates by
registering the same vector (plus a little noise) for two different files. Unknown bytes get a
deterministic pseudo-random unit vector derived from their sha256, so unrelated images are far
apart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from datacopilot.embed.protocol import ImageDecodeError

DIMS = 16


def random_unit(seed_bytes: bytes, dims: int = DIMS) -> NDArray[np.float32]:
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dims).astype(np.float32)
    return v / np.linalg.norm(v)  # type: ignore[no-any-return]


def near(vector: NDArray[np.float32], noise: float, seed: int = 0) -> NDArray[np.float32]:
    """A unit vector at roughly cosine ``1 - noise**2 / 2`` from ``vector``."""
    jitter = np.random.default_rng(seed).standard_normal(len(vector)).astype(np.float32)
    out = vector + noise * jitter / np.linalg.norm(jitter)
    return (out / np.linalg.norm(out)).astype(np.float32)  # type: ignore[no-any-return]


class FakeEmbedder:
    """Embedder satisfying the protocol, with injectable behaviour."""

    def __init__(self, dims: int = DIMS, revision: str = "1") -> None:
        self.model_id, self.revision, self.dims = "fake", revision, dims
        self.weights: dict[str, str] = {}
        self.vectors: dict[bytes, NDArray[np.float32]] = {}
        self.calls: list[int] = []  # batch sizes
        self.seen: list[bytes] = []
        self.decode_errors: dict[bytes, str] = {}
        self.fail_after: int | None = None  # raise from the n-th batch on
        self.error: BaseException | None = None
        self.mangle: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None

    def plant(self, data: bytes, vector: NDArray[np.float32]) -> None:
        self.vectors[data] = vector

    def embed(self, images: Sequence[bytes]) -> NDArray[np.float32]:
        self.calls.append(len(images))
        self.seen.extend(images)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise self.error or RuntimeError("boom")
        out = np.zeros((len(images), self.dims), dtype=np.float32)
        for row, data in enumerate(images):
            if data in self.decode_errors:
                raise ImageDecodeError(row, self.decode_errors[data])
            out[row] = self.vectors.get(data, random_unit(data, self.dims))
        return self.mangle(out) if self.mangle else out
