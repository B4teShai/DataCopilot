"""Real SigLIP2 weights (about 1.5 GB). Off by default; run with ``-m real_model``."""

from __future__ import annotations

import io

import numpy as np
import pytest

from datacopilot.embed import backends

pytestmark = pytest.mark.real_model

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def test_siglip2_embeds_deterministically_through_the_verified_load() -> None:
    for module in ("transformers", "huggingface_hub", backends.TENSOR_LIB):
        pytest.importorskip(module)
    PIL = pytest.importorskip("PIL.Image")
    yy, xx = np.mgrid[0:192, 0:256]
    scene = np.stack([xx % 256, yy * 255 // 191, (xx + yy) % 256], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    PIL.fromarray(scene).save(buf, format="PNG")
    embedder = backends.siglip2_base.create(PERMISSIVE)
    out = embedder.embed([buf.getvalue(), buf.getvalue()])
    assert out.shape == (2, backends.siglip2_base.card.dims) and np.isfinite(out).all()
    np.testing.assert_allclose(out[0], out[1], atol=1e-4)
    assert set(embedder.weights) == {"model.safetensors", "config.json"}
