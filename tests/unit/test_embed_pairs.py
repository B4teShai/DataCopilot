"""Exact "every pair at or above a threshold" search: complete, bounded, deterministic."""

from __future__ import annotations

import time

import numpy as np
import pytest

from datacopilot.embed import knn


def unit(n: int, d: int = 8, seed: int = 0) -> np.ndarray:  # type: ignore[type-arg]
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # type: ignore[no-any-return]


def collect(vectors: np.ndarray, threshold: float, **kw: object):  # type: ignore[no-untyped-def,type-arg]
    found: list[tuple[int, int]] = []

    def take(rows: np.ndarray, cols: np.ndarray) -> None:  # type: ignore[type-arg]
        found.extend(zip(rows.tolist(), cols.tolist(), strict=True))

    result = knn.pairs_at_least(vectors, threshold, take, **kw)  # type: ignore[arg-type]
    return sorted(found), result


def brute(vectors: np.ndarray, threshold: float) -> list[tuple[int, int]]:  # type: ignore[type-arg]
    unit_rows = knn.unit_rows(vectors)
    with np.errstate(all="ignore"):  # spurious BLAS flags on valid input, as in knn.py
        sims = unit_rows @ unit_rows.T
    n = len(vectors)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if sims[i, j] >= threshold]


@pytest.mark.parametrize(("n", "threshold"), [(1, 0.5), (2, -1.0), (40, 0.3), (130, 0.6)])
def test_every_pair_is_found_once_with_the_lower_index_first(n: int, threshold: float) -> None:
    v = unit(n, seed=n)
    pairs, result = collect(v, threshold, max_pairs=100_000)
    assert pairs == brute(v, threshold)
    assert result.rows == n and result.pairs == len(pairs) and result.capped is False
    assert all(i < j for i, j in pairs)


def test_a_crowd_of_identical_rows_is_complete_unlike_a_top_k_search() -> None:
    v = np.repeat(unit(1), 150, axis=0)  # 150 copies: 11,175 pairs, far more than k per row
    pairs, result = collect(v, 0.99, max_pairs=20_000)
    assert len(pairs) == 150 * 149 // 2 == result.pairs


def test_the_block_size_does_not_change_the_result() -> None:
    v = unit(97, seed=3)
    whole, _ = collect(v, 0.2, max_pairs=10_000)
    small, result = collect(v, 0.2, max_pairs=10_000, max_block_bytes=8 * 97 * 5)
    assert small == whole and result.rows == 97


def test_the_pair_cap_stops_before_the_block_that_would_cross_it() -> None:
    v = np.repeat(unit(1), 60, axis=0)
    pairs, result = collect(v, 0.5, max_pairs=100, max_block_bytes=8 * 60 * 4)
    assert result.capped is True and len(pairs) <= 100 and result.pairs == len(pairs)
    assert result.rows < 60


def test_the_deadline_is_checked_before_each_block() -> None:
    v = unit(50)
    pairs, result = collect(v, 0.0, max_pairs=10_000, deadline=time.monotonic() - 1)
    assert pairs == [] and result.rows == 0 and result.capped is False


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 4), dtype=np.float64),
        np.full((3, 4), np.nan, dtype=np.float32),
        np.ones((3, 4), dtype=np.float32) * 3,
        np.ones(4, dtype=np.float32),
    ],
)
def test_invalid_vectors_are_refused(bad: np.ndarray) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        collect(bad, 0.5, max_pairs=10)


def test_size_and_cap_limits_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knn, "MAX_KNN_ROWS", 3)
    with pytest.raises(ValueError, match="rows"):
        collect(unit(4), 0.5, max_pairs=10)
    with pytest.raises(ValueError, match="max_pairs"):
        collect(unit(2), 0.5, max_pairs=0)
