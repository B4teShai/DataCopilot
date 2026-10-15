"""The dependency-free ``hash`` test embedder: deterministic, locality-preserving, bounded."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fixtures.synth import png_bytes

from datacopilot.core.errors import LicenseError
from datacopilot.embed import hashembed
from datacopilot.embed.protocol import Embedder, EmbedderPlugin, ImageDecodeError

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def cosine(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[type-arg]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_plugin_contract_and_identity() -> None:
    plugin = hashembed.plugin
    assert isinstance(plugin, EmbedderPlugin)
    assert (
        plugin.spec.name == "hash" and plugin.spec.extra is None and plugin.spec.kind == "embedder"
    )
    emb = plugin.create(PERMISSIVE)
    assert isinstance(emb, Embedder)
    assert (emb.model_id, emb.revision, emb.dims) == ("hash", "1", hashembed.DIMS)
    assert emb.weights == {}


def test_create_refuses_a_disallowed_license() -> None:
    with pytest.raises(LicenseError):
        hashembed.plugin.create(frozenset({"mit"}))


def test_vectors_have_the_declared_shape_dtype_and_are_deterministic() -> None:
    emb = hashembed.plugin.create(PERMISSIVE)
    data = [png_bytes(32, 32, n) for n in range(4)]
    a, b = emb.embed(data), emb.embed(data)
    assert a.shape == (4, hashembed.DIMS) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(emb.embed(data[2:3])[0], a[2])  # independent of batching
    assert np.isfinite(a).all() and (np.linalg.norm(a, axis=1) > 0).all()


def test_near_identical_files_are_close_and_unrelated_files_are_not() -> None:
    emb = hashembed.plugin.create(PERMISSIVE)
    base = png_bytes(64, 64, 1)
    near = base + b"\x00trailer"  # same image, a few bytes appended
    other = png_bytes(64, 64, 9)
    v = emb.embed([base, near, other])
    assert cosine(v[0], v[1]) > 0.9
    assert abs(cosine(v[0], v[2])) < 0.5


def test_identical_files_have_identical_vectors() -> None:
    emb = hashembed.plugin.create(PERMISSIVE)
    data = png_bytes(16, 16, 3)
    v = emb.embed([data, data])
    np.testing.assert_array_equal(v[0], v[1])


def test_empty_and_tiny_inputs() -> None:
    emb = hashembed.plugin.create(PERMISSIVE)
    with pytest.raises(ImageDecodeError) as err:
        emb.embed([b"abc", b""])
    assert err.value.index == 1 and err.value.reason == "decode_failed"
    assert emb.embed([b"x"]).shape == (1, hashembed.DIMS)  # shorter than a shingle is still valid
    assert emb.embed([]).shape == (0, hashembed.DIMS)


def test_input_beyond_the_byte_cap_is_truncated_and_stays_fast() -> None:
    emb = hashembed.plugin.create(PERMISSIVE)
    big = np.random.default_rng(0).bytes(hashembed.MAX_HASH_BYTES * 4)
    start = time.perf_counter()
    v = emb.embed([big])
    assert time.perf_counter() - start < 5.0
    same_prefix = big[: hashembed.MAX_HASH_BYTES]
    np.testing.assert_array_equal(v[0], emb.embed([same_prefix])[0])


def test_entry_point_is_registered() -> None:
    from datacopilot.embed import registry

    assert "hash" in registry.embedder_names()
    assert registry.get_embedder("hash", PERMISSIVE).dims == hashembed.DIMS
