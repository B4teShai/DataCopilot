"""Exact blocked cosine kNN: reference equality, tie order, bounded memory, caps, FAISS adapter."""

from __future__ import annotations

import time
import tracemalloc
import types
from typing import Any

import numpy as np
import pytest

from datacopilot.core.errors import ModelError
from datacopilot.embed import knn


def unit(n: int, d: int = 16, seed: int = 0) -> np.ndarray:  # type: ignore[type-arg]
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def reference(q: np.ndarray, b: np.ndarray, k: int, same: bool):  # type: ignore[no-untyped-def,type-arg]
    """Brute force with the documented order: similarity descending, then index ascending."""
    with np.errstate(all="ignore"):
        sims = q.astype(np.float64) @ b.astype(np.float64).T
    out_i, out_s = [], []
    for row in range(len(q)):
        order = sorted(
            (j for j in range(len(b)) if not (same and j == row)), key=lambda j: (-sims[row, j], j)
        )[:k]
        out_i.append(order)
        out_s.append([sims[row, j] for j in order])
    return np.array(out_i), np.array(out_s)


@pytest.mark.parametrize(("n", "k"), [(1, 1), (2, 1), (30, 5), (64, 63), (200, 10), (257, 7)])
def test_matches_a_brute_force_reference(n: int, k: int) -> None:
    v = unit(n, seed=n)
    idx, sims = knn.knn(v, v, k, same_set=True)
    kk = min(k, n - 1)
    assert idx.shape == (n, kk) and sims.shape == (n, kk)
    if kk:
        ref_i, ref_s = reference(v, v, kk, True)
        np.testing.assert_array_equal(idx, ref_i)
        np.testing.assert_allclose(sims, ref_s, atol=1e-5)
    assert idx.dtype == np.int64 and sims.dtype == np.float32


def test_a_query_set_against_a_different_base() -> None:
    q, b = unit(20, seed=1), unit(50, seed=2)
    idx, sims = knn.knn(q, b, 4)
    ref_i, ref_s = reference(q, b, 4, False)
    np.testing.assert_array_equal(idx, ref_i)
    np.testing.assert_allclose(sims, ref_s, atol=1e-5)


def test_self_is_never_a_neighbour_even_for_duplicate_vectors() -> None:
    v = np.repeat(unit(1, seed=3), 6, axis=0)  # six identical vectors
    idx, sims = knn.knn(v, v, 3, same_set=True)
    for row in range(6):
        assert row not in idx[row]
    np.testing.assert_allclose(sims, 1.0, atol=1e-6)


def test_ties_are_broken_by_the_lower_index() -> None:
    basis = np.zeros((10, 4), dtype=np.float32)
    basis[:, 0] = 1.0  # identical unit vectors: every similarity is exactly 1.0
    idx, _ = knn.knn(basis, basis, 3, same_set=True)
    np.testing.assert_array_equal(idx[0], [1, 2, 3])
    np.testing.assert_array_equal(idx[5], [0, 1, 2])  # the nearest *other* indices, lowest first
    np.testing.assert_array_equal(idx[9], [0, 1, 2])


def test_ties_at_the_boundary_keep_the_lowest_indices_deterministically() -> None:
    v = np.zeros((7, 3), dtype=np.float32)
    v[0] = (1, 0, 0)
    v[1:4] = (0, 1, 0)  # three tied neighbours of equal similarity 0 to the query
    v[4:] = (0, 0, 1)  # more at similarity 0
    q = v[:1]
    idx, _ = knn.knn(q, v, 2)  # best is index 0 itself (sim 1), then a tie at 0 among 1..6
    np.testing.assert_array_equal(idx[0], [0, 1])


def test_results_do_not_depend_on_the_block_size() -> None:
    v = unit(300, d=24, seed=5)
    full = knn.knn(v, v, 8, same_set=True)
    tiny = knn.knn(v, v, 8, same_set=True, max_block_bytes=300 * 8 * 3)  # about three rows
    np.testing.assert_array_equal(full[0], tiny[0])
    np.testing.assert_allclose(full[1], tiny[1], atol=1e-5)


def test_the_block_is_sized_from_the_memory_budget() -> None:
    assert knn.block_rows(base_rows=50_000, max_block_bytes=256 * 1024**2) == 671
    assert knn.block_rows(base_rows=10, max_block_bytes=1) == 1  # never zero


def test_peak_memory_stays_near_one_block_not_the_full_matrix() -> None:
    n = 3000
    v = unit(n, d=8, seed=6)
    budget = 2 * 1024**2  # one block is 2 MiB; the full similarity matrix would be 36 MB
    tracemalloc.start()
    knn.knn(v, v, 5, same_set=True, max_block_bytes=budget)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 6 * budget < n * n * 4


def test_iter_knn_yields_ordered_blocks_that_cover_every_query() -> None:
    v = unit(100, seed=7)
    blocks = list(knn.iter_knn(v, v, 3, same_set=True, max_block_bytes=100 * 8 * 10))
    assert [b.start for b in blocks] == list(range(0, 100, 10))
    assert sum(len(b.indices) for b in blocks) == 100


def test_a_passed_deadline_yields_nothing_and_a_distant_one_changes_nothing() -> None:
    v = unit(100, seed=8)
    assert list(knn.iter_knn(v, v, 3, same_set=True, deadline=time.monotonic() - 1)) == []
    far = time.monotonic() + 3600
    blocks = list(knn.iter_knn(v, v, 3, same_set=True, max_block_bytes=100 * 8 * 10, deadline=far))
    assert [b.start for b in blocks] == list(range(0, 100, 10))
    expected = knn.knn(v, v, 3, same_set=True)
    np.testing.assert_array_equal(np.concatenate([b.indices for b in blocks]), expected[0])


def test_deadline_is_checked_between_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    v = unit(100, seed=9)
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 10.0])  # the third check is past the deadline
    monkeypatch.setattr(knn.time, "monotonic", lambda: next(clock, 10.0))
    blocks = list(knn.iter_knn(v, v, 3, same_set=True, max_block_bytes=100 * 8 * 10, deadline=5.0))
    assert 0 < len(blocks) < 10


def test_empty_and_degenerate_inputs() -> None:
    empty = np.zeros((0, 4), dtype=np.float32)
    idx, sims = knn.knn(empty, empty, 3, same_set=True)
    assert idx.shape == (0, 0) and sims.shape == (0, 0)
    one = unit(1, d=4)
    idx, sims = knn.knn(one, one, 3, same_set=True)
    assert idx.shape == (1, 0)
    idx, sims = knn.knn(one, empty, 3)
    assert idx.shape == (1, 0)


# ------------------------------------------------------------ caps and validation
def test_caps_on_rows_and_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knn, "MAX_KNN_ROWS", 10)
    v = unit(11)
    with pytest.raises(ValueError, match="at most 10"):
        knn.knn(v, v, 1, same_set=True)
    with pytest.raises(ValueError, match="at most 10"):
        knn.knn(unit(2), v, 1)
    with pytest.raises(ValueError, match="k must be"):
        knn.knn(unit(5), unit(5), knn.MAX_K + 1, same_set=True)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k must be"):
            knn.knn(unit(5), unit(5), bad_k, same_set=True)


def test_inputs_are_validated() -> None:
    good = unit(5)
    nan = good.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        knn.knn(nan, good, 1)
    with pytest.raises(ValueError, match="unit"):
        knn.knn(good * 2, good, 1)
    with pytest.raises(ValueError, match="2-D"):
        knn.knn(good[0], good, 1)
    with pytest.raises(ValueError, match="width"):
        knn.knn(unit(3, d=4), good, 1)
    with pytest.raises(ValueError, match="float32"):
        knn.knn(good.astype(np.float64), good, 1)
    with pytest.raises(ValueError, match="same_set"):
        knn.knn(good, unit(5, seed=1), 1, same_set=True)


def test_neighbour_similarities_are_sorted_descending() -> None:
    v = unit(80, seed=11)
    _, sims = knn.knn(v, v, 10, same_set=True)
    assert (np.diff(sims, axis=1) <= 1e-7).all()


# ------------------------------------------------------------ FAISS adapter
class FakeIndexFlatIP:
    """Exact inner-product index in NumPy, with FAISS's own (arbitrary) tie behaviour."""

    instances: list[FakeIndexFlatIP] = []

    def __init__(self, d: int) -> None:
        self.d, self.data = d, np.zeros((0, d), dtype=np.float32)
        FakeIndexFlatIP.instances.append(self)

    def add(self, x: np.ndarray) -> None:  # type: ignore[type-arg]
        self.data = np.concatenate([self.data, x])

    def search(self, x: np.ndarray, k: int):  # type: ignore[no-untyped-def,type-arg]
        with np.errstate(all="ignore"):
            sims = x @ self.data.T
        # like the real library, break exact ties arbitrarily: here, the *higher* column first
        flipped = np.argsort(-sims[:, ::-1], axis=1, kind="stable")[:, :k]
        order = (sims.shape[1] - 1) - flipped
        return np.take_along_axis(sims, order, axis=1), order.astype(np.int64)


@pytest.fixture
def fake_faiss(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    FakeIndexFlatIP.instances = []
    module = types.ModuleType("faiss")
    module.IndexFlatIP = FakeIndexFlatIP  # type: ignore[attr-defined]
    monkeypatch.setattr(knn, "_import_faiss", lambda: module)
    return module


def test_faiss_results_equal_the_numpy_results_within_the_stated_tolerance(
    fake_faiss: types.ModuleType,
) -> None:
    q, b = unit(60, d=32, seed=12), unit(200, d=32, seed=13)
    exact = knn.knn(q, b, 6, backend="numpy")
    fast = knn.knn(q, b, 6, backend="faiss")
    assert FakeIndexFlatIP.instances
    np.testing.assert_array_equal(exact[0], fast[0])
    np.testing.assert_allclose(exact[1], fast[1], atol=knn.BACKEND_TOLERANCE)
    assert fast[0].dtype == np.int64 and fast[1].dtype == np.float32


def test_faiss_self_exclusion_and_the_documented_tie_caveat(fake_faiss: types.ModuleType) -> None:
    """The fake breaks ties the opposite way to NumPy. Rows are re-sorted by (similarity, index),
    so within the returned set the order is the documented one; *which* tied columns make it into
    the k-th place may differ from the NumPy backend, and the similarities do not."""
    basis = np.zeros((6, 4), dtype=np.float32)
    basis[:, 0] = 1.0  # every similarity is exactly 1.0: all columns tie
    idx, sims = knn.knn(basis, basis, 2, same_set=True, backend="faiss")
    exact_idx, exact_sims = knn.knn(basis, basis, 2, same_set=True, backend="numpy")
    for row in range(6):
        assert row not in idx[row]
        assert list(idx[row]) == sorted(idx[row])  # re-sorted: lower index first among ties
    np.testing.assert_allclose(sims, exact_sims, atol=knn.BACKEND_TOLERANCE)
    assert not np.array_equal(idx, exact_idx)  # the caveat is real: the tied choice differs


def test_auto_uses_faiss_only_for_large_inputs_when_it_is_installed(
    fake_faiss: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = unit(40)
    knn.knn(v, v, 3, same_set=True)  # below the threshold: exact NumPy
    assert FakeIndexFlatIP.instances == []
    monkeypatch.setattr(knn, "FAISS_MIN_ROWS", 10)
    knn.knn(v, v, 3, same_set=True)
    assert FakeIndexFlatIP.instances


def test_auto_without_faiss_uses_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knn, "_import_faiss", lambda: None)
    monkeypatch.setattr(knn, "FAISS_MIN_ROWS", 1)
    v = unit(30)
    assert knn.knn(v, v, 3, same_set=True)[0].shape == (30, 3)


def test_requesting_faiss_when_it_is_missing_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(knn, "_import_faiss", lambda: None)
    v = unit(5)
    with pytest.raises(ModelError, match=r"datacopilot\[embed\]"):
        knn.knn(v, v, 1, same_set=True, backend="faiss")


def test_a_faiss_failure_is_a_model_error_without_its_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def __init__(self, d: int) -> None:
            raise RuntimeError("secret detail")

    module: Any = types.ModuleType("faiss")
    module.IndexFlatIP = Broken
    monkeypatch.setattr(knn, "_import_faiss", lambda: module)
    v = unit(5)
    with pytest.raises(ModelError) as err:
        knn.knn(v, v, 1, same_set=True, backend="faiss")
    assert "secret detail" not in str(err.value) and "RuntimeError" in str(err.value)


def test_unknown_backend_is_rejected() -> None:
    v = unit(5)
    with pytest.raises(ValueError, match="backend"):
        knn.knn(v, v, 1, same_set=True, backend="annoy")  # type: ignore[arg-type]


def test_identical_vectors_score_one_even_when_their_norm_is_not_exactly_one() -> None:
    """Stored float16 vectors are unit length only within 1e-3; threshold 1.0 must still hold."""
    v = unit(4, seed=21)
    v[1] = v[0]
    v = (v * np.float32(0.998)).astype(np.float32)  # inside the accepted norm tolerance
    assert float(v[0] @ v[1]) < 0.9965  # the raw inner product would miss threshold 0.9995
    idx, sims = knn.knn(v, v, 1, same_set=True)
    assert idx[0, 0] == 1 and sims[0, 0] >= 1.0 - knn.SIM_EPSILON


def test_the_callers_arrays_are_not_modified_by_normalisation() -> None:
    v = (unit(6, seed=22) * np.float32(0.999)).astype(np.float32)
    before = v.copy()
    knn.knn(v, v, 2, same_set=True)
    np.testing.assert_array_equal(v, before)


@pytest.mark.parametrize("extra", [1, 2, 7])
def test_a_base_one_row_over_a_block_and_a_partial_last_block_match_the_reference(
    extra: int,
) -> None:
    block = 16
    n = block + extra
    v = unit(n, d=12, seed=30 + extra)
    budget = block * knn.BYTES_PER_CELL * n  # exactly `block` query rows per block
    assert knn.block_rows(n, budget) == block
    idx, sims = knn.knn(v, v, 4, same_set=True, max_block_bytes=budget)
    ref_i, ref_s = reference(v, v, 4, True)
    np.testing.assert_array_equal(idx, ref_i)
    np.testing.assert_allclose(sims, ref_s, atol=1e-5)
    blocks = list(knn.iter_knn(v, v, 4, same_set=True, max_block_bytes=budget))
    assert [len(b.indices) for b in blocks] == [block, extra]


def test_faiss_is_not_imported_after_the_tensor_library_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two OpenMP runtimes abort the process on macOS, so the NumPy index is used there."""
    import sys

    sentinel = types.ModuleType("faiss")
    monkeypatch.setitem(sys.modules, "faiss", sentinel)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, knn._TENSOR_LIB, types.ModuleType(knn._TENSOR_LIB))
    assert knn._import_faiss() is None
    v = unit(5)
    with pytest.raises(ModelError, match="macOS"):
        knn.knn(v, v, 1, same_set=True, backend="faiss")
    monkeypatch.setattr(sys, "platform", "linux")
    assert knn._import_faiss() is sentinel
