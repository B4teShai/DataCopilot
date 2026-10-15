"""Real FAISS against the NumPy index. Off by default (needs the ``[embed]`` extra)."""

from __future__ import annotations

import numpy as np
import pytest

from datacopilot.embed import knn

pytestmark = pytest.mark.real_model


def test_real_faiss_matches_numpy_within_the_stated_tolerance() -> None:
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    v = rng.standard_normal((500, 64)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    exact = knn.knn(v, v, 8, same_set=True, backend="numpy")
    fast = knn.knn(v, v, 8, same_set=True, backend="faiss")
    np.testing.assert_array_equal(exact[0], fast[0])
    np.testing.assert_allclose(exact[1], fast[1], atol=knn.BACKEND_TOLERANCE)
