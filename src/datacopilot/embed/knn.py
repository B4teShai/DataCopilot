"""Exact cosine k-nearest-neighbours with bounded memory and a deterministic order.

Vectors must be finite ``float32`` rows of unit length, so the inner product is the cosine
similarity. Two backends compute the same thing:

- **NumPy** (default, no extra dependency): query rows are processed in blocks; one block is a
  ``(rows, N)`` similarity matrix whose size is chosen from ``max_block_bytes``, so peak memory is
  about one block plus the vectors, never ``N x N``. Per query row the ``k`` best columns are chosen
  exactly: the ``k``-th best similarity is found with a partition, every column at or above it is a
  candidate, and the candidates are ordered by ``(similarity descending, index ascending)``.
  Ties therefore resolve to the lower index, and the result does not depend on the block size
  except through the platform's matrix-multiply rounding (about 1e-7 in similarity).
- **FAISS** ``IndexFlatIP`` (exact, from the ``[embed]`` extra), used by ``auto`` only above
  ``FAISS_MIN_ROWS`` base rows. Its neighbours equal the NumPy ones within ``BACKEND_TOLERANCE`` in
  similarity; at an exact tie on the ``k``-th place it may pick a different column, which is why
  rows are re-sorted by the same ``(similarity, index)`` order. No approximate index is offered:
  a missed neighbour would be a missed leak.

Caps: at most ``MAX_KNN_ROWS`` query rows and base rows and ``MAX_K`` neighbours per row, so a
hostile or mistaken input fails with a ``ValueError`` before any matrix is allocated. Missing
neighbours (``k`` larger than the base) are not padded: the result is narrower.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from datacopilot.core.errors import ModelError

MAX_KNN_ROWS = 500_000
"""Largest query set or base set. The exact search costs O(N^2 d), about 10 s per 50,000 x 768
and hours at this cap; callers also pass a deadline."""
MAX_K = 100
"""Most neighbours per row."""
DEFAULT_BLOCK_BYTES = 256 * 1024**2
"""Budget of one similarity block (float32 matrix plus its partition copy)."""
BYTES_PER_CELL = 8
FAISS_MIN_ROWS = 20_000
"""Base size from which ``backend="auto"`` prefers FAISS when it is installed."""
BACKEND_TOLERANCE = 1e-4
"""Largest similarity difference between the NumPy and FAISS backends."""
_NORM_TOLERANCE = 1e-2

Backend = Literal["auto", "numpy", "faiss"]


@dataclass(frozen=True)
class KnnBlock:
    """Neighbours of ``len(indices)`` consecutive query rows starting at ``start``.

    ``indices`` are base-row positions (int64) and ``sims`` the cosine similarities (float32),
    both ``(rows, k)`` and ordered by similarity descending, then index ascending.
    """

    start: int
    indices: NDArray[np.int64]
    sims: NDArray[np.float32]


def _import_faiss() -> Any:
    """Seam for tests: the ``faiss`` module, or None when it is not installed."""
    try:
        return importlib.import_module("faiss")
    except ImportError:
        return None


def block_rows(base_rows: int, max_block_bytes: int = DEFAULT_BLOCK_BYTES) -> int:
    """Query rows per block for a base of ``base_rows`` (at least one)."""
    return max(1, max_block_bytes // (BYTES_PER_CELL * max(base_rows, 1)))


def _validate(
    queries: NDArray[np.float32], base: NDArray[np.float32], k: int, same_set: bool
) -> int:
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}")
    for name, arr in (("queries", queries), ("base", base)):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array")
        if arr.dtype != np.float32:
            raise ValueError(f"{name} must be float32")
        if len(arr) > MAX_KNN_ROWS:
            raise ValueError(f"{name} has {len(arr)} rows; at most {MAX_KNN_ROWS} are supported")
    if queries.shape[1] != base.shape[1]:
        raise ValueError("queries and base differ in width")
    if same_set and queries is not base:
        raise ValueError("same_set requires queries and base to be the same array")
    for arr in (queries, base) if queries is not base else (base,):
        if len(arr) and not np.isfinite(arr).all():
            raise ValueError("vectors must be finite")
        if len(arr) and not np.all(np.abs(np.linalg.norm(arr, axis=1) - 1.0) <= _NORM_TOLERANCE):
            raise ValueError("vectors must have unit norm")
    return min(k, len(base) - (1 if same_set else 0))


def _order(
    sims: NDArray[np.float32], cols: NDArray[np.int64], k: int
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """The ``k`` best of ``cols`` by (similarity descending, index ascending)."""
    order = np.lexsort((cols, -sims))[:k]
    return cols[order], sims[order]


def _numpy_block(
    queries: NDArray[np.float32],
    base: NDArray[np.float32],
    start: int,
    stop: int,
    k: int,
    same_set: bool,
) -> KnnBlock:
    with np.errstate(all="ignore"):  # inputs are finite and unit length; some BLAS builds raise
        sims = queries[start:stop] @ base.T  # spurious floating-point flags on valid input
    rows = stop - start
    if same_set:
        sims[np.arange(rows), np.arange(start, stop)] = -np.inf
    width = sims.shape[1]
    kth = np.partition(sims, width - k, axis=1)[:, width - k]
    indices = np.empty((rows, k), dtype=np.int64)
    values = np.empty((rows, k), dtype=np.float32)
    for row in range(rows):
        line = sims[row]
        cols = np.flatnonzero(line >= kth[row]).astype(np.int64)
        indices[row], values[row] = _order(line[cols], cols, k)
    return KnnBlock(start, indices, values)


def _faiss_blocks(
    faiss: Any,
    queries: NDArray[np.float32],
    base: NDArray[np.float32],
    k: int,
    same_set: bool,
    rows_per_block: int,
    deadline: float | None,
) -> Iterator[KnnBlock]:
    try:
        index = faiss.IndexFlatIP(base.shape[1])
        index.add(base)
    except Exception as exc:  # noqa: BLE001  # never echo the library's text
        raise ModelError(f"faiss failed to build the index ({type(exc).__name__})") from None
    extra = 1 if same_set else 0
    for start in range(0, len(queries), rows_per_block):
        if deadline is not None and time.monotonic() >= deadline:
            return
        stop = min(start + rows_per_block, len(queries))
        try:
            sims, ids = index.search(queries[start:stop], k + extra)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"faiss search failed ({type(exc).__name__})") from None
        indices = np.empty((stop - start, k), dtype=np.int64)
        values = np.empty((stop - start, k), dtype=np.float32)
        for row in range(stop - start):
            keep = ids[row] >= 0
            if same_set:
                keep &= ids[row] != start + row
            cols = ids[row][keep].astype(np.int64)
            indices[row], values[row] = _order(
                np.asarray(sims[row][keep], dtype=np.float32), cols, k
            )
        yield KnnBlock(start, indices, values)


def iter_knn(
    queries: NDArray[np.float32],
    base: NDArray[np.float32],
    k: int,
    *,
    same_set: bool = False,
    backend: Backend = "auto",
    max_block_bytes: int = DEFAULT_BLOCK_BYTES,
    deadline: float | None = None,
) -> Iterator[KnnBlock]:
    """Yield the neighbours of ``queries`` among ``base`` block by block.

    ``same_set`` means ``queries is base`` and each row's own column is excluded. ``deadline`` is a
    ``time.monotonic()`` value checked before every block; the iterator then stops, and the caller
    detects the shortfall by counting the rows it received.

    Raises:
        ValueError: ``k`` or a size outside the caps, wrong shape or dtype, non-finite or
            non-unit vectors, an unknown backend.
        ModelError: FAISS was requested and is missing, or it failed.
    """
    if backend not in ("auto", "numpy", "faiss"):
        raise ValueError("backend must be auto, numpy or faiss")
    effective = _validate(queries, base, k, same_set)
    if effective <= 0 or len(queries) == 0:
        for start in range(0, len(queries), block_rows(max(len(base), 1), max_block_bytes)):
            stop = min(start + block_rows(max(len(base), 1), max_block_bytes), len(queries))
            if deadline is not None and time.monotonic() >= deadline:
                return
            yield KnnBlock(
                start,
                np.zeros((stop - start, 0), dtype=np.int64),
                np.zeros((stop - start, 0), dtype=np.float32),
            )
        return
    faiss = None
    if backend == "faiss" or (backend == "auto" and len(base) >= FAISS_MIN_ROWS):
        faiss = _import_faiss()
        if faiss is None and backend == "faiss":
            raise ModelError('faiss is not installed: pip install "datacopilot[embed]"')
    step = block_rows(len(base), max_block_bytes)
    if faiss is not None:
        yield from _faiss_blocks(faiss, queries, base, effective, same_set, step, deadline)
        return
    for start in range(0, len(queries), step):
        if deadline is not None and time.monotonic() >= deadline:
            return
        yield _numpy_block(
            queries, base, start, min(start + step, len(queries)), effective, same_set
        )


def knn(
    queries: NDArray[np.float32],
    base: NDArray[np.float32],
    k: int,
    *,
    same_set: bool = False,
    backend: Backend = "auto",
    max_block_bytes: int = DEFAULT_BLOCK_BYTES,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """``(indices, sims)`` of the ``k`` nearest base rows of every query row, in query order."""
    blocks = list(
        iter_knn(
            queries, base, k, same_set=same_set, backend=backend, max_block_bytes=max_block_bytes
        )
    )
    if not blocks:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return (
        np.concatenate([b.indices for b in blocks]),
        np.concatenate([b.sims for b in blocks]),
    )
