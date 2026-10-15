"""Real FAISS against the NumPy index. Off by default (needs the ``[embed]`` extra).

Run in a child process: on macOS, importing ``faiss`` and the tensor library into one interpreter
aborts it (two OpenMP runtimes), and the other real tests load the tensor library.
"""

from __future__ import annotations

import importlib.util
import subprocess  # noqa: S404  # runs this interpreter on a fixed script
import sys

import pytest

pytestmark = pytest.mark.real_model

SCRIPT = """
import numpy as np
from datacopilot.embed import knn
rng = np.random.default_rng(0)
v = rng.standard_normal((500, 64)).astype(np.float32)
v /= np.linalg.norm(v, axis=1, keepdims=True)
exact = knn.knn(v, v, 8, same_set=True, backend="numpy")
fast = knn.knn(v, v, 8, same_set=True, backend="faiss")
assert (exact[0] == fast[0]).all()
assert np.abs(exact[1] - fast[1]).max() <= knn.BACKEND_TOLERANCE
print("ok")
"""


def test_real_faiss_matches_numpy_within_the_stated_tolerance() -> None:
    if importlib.util.find_spec("faiss") is None:  # not importorskip: that would import it here
        pytest.skip("faiss is not installed")
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-500:]
