"""Smoke tests with real weights. Off by default: they download models and need ``[embed]``.

Run with ``uv run --extra embed pytest -m real_model tests/real``. They are never part of the gate,
which must work without a network and without weights.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from datacopilot.embed import backends

pytestmark = pytest.mark.real_model

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def _images() -> list[bytes]:
    """A structured picture, a copy of it, a recompressed copy and two unrelated pictures."""
    PIL = pytest.importorskip("PIL.Image")
    yy, xx = np.mgrid[0:192, 0:256]
    scene = np.stack([xx * 255 // 255, yy * 255 // 191, (xx + yy) % 256], axis=-1).astype(np.uint8)
    scene[60:130, 80:170] = (250, 240, 20)  # a bright block on a gradient
    stripes = np.zeros_like(scene)
    stripes[::16] = (255, 255, 255)
    noise = np.random.default_rng(0).integers(0, 256, scene.shape, dtype=np.uint8)
    out: list[bytes] = []
    for array in (scene, scene, stripes, noise):
        buf = io.BytesIO()
        PIL.fromarray(np.ascontiguousarray(array)).save(buf, format="PNG")
        out.append(buf.getvalue())
    jpeg = io.BytesIO()
    PIL.fromarray(scene).save(jpeg, format="JPEG", quality=90)
    out.append(jpeg.getvalue())
    return out


@pytest.mark.parametrize("name", ["dinov2_small"])
def test_a_real_model_embeds_deterministically_and_keeps_a_recompressed_copy_close(
    name: str,
) -> None:
    for module in ("transformers", "huggingface_hub", backends.TENSOR_LIB):
        pytest.importorskip(module)
    plugin = getattr(backends, name)
    embedder = plugin.create(PERMISSIVE)
    images = _images()
    vectors = embedder.embed(images)
    again = embedder.embed(images[:2])
    assert vectors.shape == (5, plugin.card.dims) and np.isfinite(vectors).all()
    np.testing.assert_allclose(vectors[:2], again, atol=1e-4)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ unit[0]
    assert sims[1] > 0.999  # identical image
    assert sims[4] > 0.9  # a JPEG re-encode of the same picture stays close
    assert sims[4] > max(sims[2], sims[3])  # and closer than unrelated pictures
