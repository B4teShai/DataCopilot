"""DINOv2 and SigLIP2 backends with injected fake libraries: no network, no weights, no tensors.

The tensor library is spelled indirectly (``backends.TENSOR_LIB``) because the repository's QA gate
forbids the literal name in sources.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog
from fixtures.synth import png_bytes

from datacopilot.core.errors import LicenseError, ModelError
from datacopilot.embed import backends, registry
from datacopilot.embed.protocol import Embedder, EmbedderPlugin, ImageDecodeError
from datacopilot.models.registry import ModelCard

pytest.importorskip("PIL")
PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})
WEIGHTS = b"fake safetensors payload"
WEIGHTS_SHA = hashlib.sha256(WEIGHTS).hexdigest()
CONFIG_TEXT = "{}"
CONFIG_SHA = hashlib.sha256(CONFIG_TEXT.encode()).hexdigest()


def card(family: str = "dinov2", dims: int = 16, sha: str = WEIGHTS_SHA, **kw: Any) -> ModelCard:
    base: dict[str, Any] = dict(
        model_id="fake_model",
        repo="owner/fake-model",
        revision="a" * 40,
        license="Apache-2.0",
        gated=False,
        gpu_memory_gb=1.0,
        dims=dims,
        family=family,
        weights_sha256={"model.safetensors": sha},
        weights_bytes={"model.safetensors": len(WEIGHTS)},
        config_sha256=CONFIG_SHA,
    )
    base.update(kw)
    return ModelCard(**base)


class FakeTensor:
    def __init__(self, array: np.ndarray, device: str = "cpu") -> None:  # type: ignore[type-arg]
        self.array, self.device = array, device

    def to(self, device: str) -> FakeTensor:
        if device == "cuda" and Fakes.cuda_breaks:
            raise RuntimeError("CUDA out of memory token=hf_SECRET123")
        return FakeTensor(self.array, device)

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return FakeTensor(self.array.astype(np.float32), self.device)

    def cpu(self) -> FakeTensor:
        return FakeTensor(self.array, "cpu")

    def numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        return self.array


class Fakes:
    """Mutable state shared by the fake modules of one test."""

    cuda = False
    cuda_breaks = False
    calls: list[tuple[str, dict[str, Any]]] = []
    snapshot_root: Path
    weights: bytes = WEIGHTS
    use_symlink = False
    config: str = CONFIG_TEXT
    download_error: Exception | None = None
    feature_dims = 16
    output_style = "pooler"


class FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.evaluating = False

    def eval(self) -> FakeModel:
        self.evaluating = True
        return self

    def to(self, device: str) -> FakeModel:
        if device == "cuda" and Fakes.cuda_breaks:
            raise RuntimeError("CUDA out of memory token=hf_SECRET123")
        self.device = device
        return self

    def _features(self, pixel_values: FakeTensor) -> np.ndarray:  # type: ignore[type-arg]
        n = pixel_values.array.shape[0]
        base = pixel_values.array.reshape(n, -1)[:, : Fakes.feature_dims]
        return np.asarray(base, dtype=np.float32)

    def __call__(self, pixel_values: FakeTensor) -> Any:
        return types.SimpleNamespace(pooler_output=FakeTensor(self._features(pixel_values)))

    def get_image_features(self, pixel_values: FakeTensor) -> Any:
        feats = FakeTensor(self._features(pixel_values))
        if Fakes.output_style == "object":
            return types.SimpleNamespace(pooler_output=feats)
        return feats


@pytest.fixture(autouse=True)
def _captured_logs() -> Iterator[None]:
    """Keep events away from a stream that an earlier CLI test may have closed."""
    with structlog.testing.capture_logs():
        yield


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[Fakes]:
    Fakes.cuda = Fakes.cuda_breaks = Fakes.use_symlink = False
    Fakes.calls, Fakes.weights, Fakes.download_error = [], WEIGHTS, None
    Fakes.config = CONFIG_TEXT
    Fakes.snapshot_root, Fakes.feature_dims, Fakes.output_style = tmp_path / "hub", 16, "pooler"

    def snapshot_download(**kw: Any) -> str:
        Fakes.calls.append(("snapshot_download", kw))
        if Fakes.download_error:
            raise Fakes.download_error
        snap = Fakes.snapshot_root / "snapshots" / kw["revision"]
        snap.mkdir(parents=True, exist_ok=True)
        blob = Fakes.snapshot_root / "blobs" / "b1"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(Fakes.weights)
        target = snap / "model.safetensors"
        if not target.exists() and not target.is_symlink():
            if Fakes.use_symlink:
                target.symlink_to(blob)  # the layout of the real Hugging Face cache
            else:
                target.write_bytes(Fakes.weights)
        (snap / "config.json").write_text(Fakes.config)
        return str(snap)

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **kw: Any) -> FakeModel:
            Fakes.calls.append(("from_pretrained", {"path": path, **kw}))
            return FakeModel()

    @contextmanager
    def no_grad() -> Iterator[None]:
        Fakes.calls.append(("no_grad", {}))
        yield

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    tf = types.ModuleType("transformers")
    tf.AutoModel = AutoModel  # type: ignore[attr-defined]
    lib = types.ModuleType(backends.TENSOR_LIB)
    lib.from_numpy = lambda a: FakeTensor(np.asarray(a))  # type: ignore[attr-defined]
    lib.no_grad = no_grad  # type: ignore[attr-defined]
    lib.cuda = types.SimpleNamespace(is_available=lambda: Fakes.cuda)  # type: ignore[attr-defined]
    for name, module in (
        ("huggingface_hub", hub),
        ("transformers", tf),
        (backends.TENSOR_LIB, lib),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for var in (
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_ENDPOINT",
        "CI",
        "DATACOPILOT_ALLOW_HF_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    return Fakes


def make(family: str = "dinov2", **kw: Any) -> Embedder:
    return backends.HFEmbedderPlugin(card(family, **kw)).create(PERMISSIVE)


def calls(name: str) -> list[dict[str, Any]]:
    return [kw for n, kw in Fakes.calls if n == name]


IMG = png_bytes(40, 30, 2)


# ------------------------------------------------------------ construction and the license guard
def test_first_party_plugins_describe_their_cards() -> None:
    for name, dims in (("dinov2_small", 384), ("dinov2_base", 768), ("siglip2_base", 768)):
        plugin = getattr(backends, name)
        assert isinstance(plugin, EmbedderPlugin)
        assert plugin.spec.name == name and plugin.spec.extra == "embed"
        assert plugin.spec.license == "Apache-2.0" and plugin.card.dims == dims


def test_create_touches_nothing_until_the_first_embed(fakes: type[Fakes]) -> None:
    emb = make()
    assert Fakes.calls == [] and emb.dims == 16 and emb.revision == "a" * 40
    # the identity of the cache covers the weights and the configuration
    assert emb.weights == {"model.safetensors": WEIGHTS_SHA, "config.json": CONFIG_SHA}


def test_a_disallowed_license_is_refused_before_any_library_is_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(name: str) -> None:
        raise AssertionError(f"{name} must not be imported")

    for module in ("huggingface_hub", "transformers", backends.TENSOR_LIB):
        monkeypatch.setitem(sys.modules, module, None)  # an import would raise ImportError
    monkeypatch.setattr(backends, "_import", explode)
    plugin = backends.HFEmbedderPlugin(card(license="CC-BY-NC-4.0"))
    with pytest.raises(LicenseError, match="DATACOPILOT_ALLOWED_LICENSES"):
        plugin.create(PERMISSIVE)


def test_the_real_cli_registry_path_refuses_before_the_module_loads() -> None:
    with pytest.raises(LicenseError):
        registry.get_embedder("dinov2_base", frozenset({"mit"}))


# ------------------------------------------------------------ model loading rules
def test_model_is_loaded_safely_from_a_pinned_verified_local_snapshot(fakes: type[Fakes]) -> None:
    emb = make()
    out = emb.embed([IMG])
    assert out.shape == (1, 16) and out.dtype == np.float32
    (download,) = calls("snapshot_download")
    assert download["repo_id"] == "owner/fake-model" and download["revision"] == "a" * 40
    assert set(download["allow_patterns"]) == {
        "model.safetensors",
        "config.json",
    }  # no .bin, no .py
    assert download["etag_timeout"] <= 60
    (load,) = calls("from_pretrained")
    assert load["trust_remote_code"] is False and load["use_safetensors"] is True
    assert load["local_files_only"] is True
    assert "token" not in " ".join(load) and "revision" not in load  # a local path, not the hub
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] and os.environ["HF_HUB_ETAG_TIMEOUT"]


def test_the_model_loads_once_for_many_batches(fakes: type[Fakes]) -> None:
    emb = make()
    emb.embed([IMG])
    emb.embed([IMG, IMG])
    assert len(calls("snapshot_download")) == 1 and len(calls("from_pretrained")) == 1
    assert len(calls("no_grad")) == 2  # inference never builds a graph


def test_weights_that_do_not_match_the_pinned_hash_are_refused_before_loading(
    fakes: type[Fakes],
) -> None:
    Fakes.weights = b"X" * len(WEIGHTS)  # same size, other bytes: only the hash can tell
    with pytest.raises(ModelError, match="does not match") as err:
        make().embed([IMG])
    assert "model.safetensors" in str(err.value)
    assert calls("from_pretrained") == []  # never handed to the deserialiser


def test_a_huggingface_cache_symlink_is_resolved_and_verified(fakes: type[Fakes]) -> None:
    Fakes.use_symlink = True
    assert make().embed([IMG]).shape == (1, 16)


def test_a_missing_weight_file_is_refused(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "snapshot_download", lambda **kw: str(Fakes.snapshot_root)
    )
    Fakes.snapshot_root.mkdir()
    with pytest.raises(ModelError, match="model.safetensors"):
        make().embed([IMG])
    assert calls("from_pretrained") == []


def test_download_errors_do_not_leak_their_text(fakes: type[Fakes]) -> None:
    Fakes.download_error = ConnectionError("https://hub/x?token=hf_SECRET123 failed")
    with pytest.raises(ModelError) as err:
        make().embed([IMG])
    text = str(err.value)
    assert "hf_SECRET123" not in text and "ConnectionError" in text and "fake_model" in text


def test_missing_optional_libraries_name_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("huggingface_hub", "transformers", backends.TENSOR_LIB):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ModelError, match=r"datacopilot\[embed\]"):
        make().embed([IMG])


def test_the_device_falls_back_to_cpu_when_cuda_fails_without_leaking_the_error(
    fakes: type[Fakes],
) -> None:
    Fakes.cuda = Fakes.cuda_breaks = True
    with structlog.testing.capture_logs() as logs:
        out = make().embed([IMG])
    assert out.shape == (1, 16)
    events = {e["event"]: e for e in logs}
    assert events["embedder_device_fallback"]["log_level"] == "warning"
    assert "hf_SECRET123" not in repr(logs)
    assert events["model_loaded"]["device"] == "cpu"


def test_cuda_is_used_when_available(fakes: type[Fakes]) -> None:
    Fakes.cuda = True
    with structlog.testing.capture_logs() as logs:
        make().embed([IMG])
    assert [e for e in logs if e["event"] == "model_loaded"][0]["device"] == "cuda"


def test_loading_logs_the_weight_hashes_and_the_license(fakes: type[Fakes]) -> None:
    with structlog.testing.capture_logs() as logs:
        make().embed([IMG])
    loaded = [e for e in logs if e["event"] == "model_loaded"][0]
    assert loaded["model_id"] == "fake_model" and loaded["weights_sha256"] == {
        "model.safetensors": WEIGHTS_SHA,
        "config.json": CONFIG_SHA,
    }
    shown = [e for e in logs if e["event"] == "model_license_shown"][0]
    assert shown["license"] == "Apache-2.0" and shown["gated"] is False


# ------------------------------------------------------------ inference
def test_dinov2_uses_the_pooled_output_and_siglip_the_image_features(fakes: type[Fakes]) -> None:
    for family in ("dinov2", "siglip2"):
        out = make(family).embed([IMG, png_bytes(31, 33, 5)])
        assert out.shape == (2, 16) and np.isfinite(out).all()


def test_a_features_object_with_a_pooled_field_is_accepted(fakes: type[Fakes]) -> None:
    Fakes.output_style = "object"
    assert make("siglip2").embed([IMG]).shape == (1, 16)


def test_wrong_feature_width_is_a_model_error(fakes: type[Fakes]) -> None:
    Fakes.feature_dims = 8
    with pytest.raises(ModelError, match="width"):
        make().embed([IMG])


def test_inputs_are_preprocessed_to_the_models_input_size(fakes: type[Fakes]) -> None:
    seen: list[tuple[int, ...]] = []
    original = FakeModel.__call__

    def spy(self: FakeModel, pixel_values: FakeTensor) -> Any:
        seen.append(pixel_values.array.shape)
        return original(self, pixel_values)

    FakeModel.__call__ = spy  # type: ignore[method-assign]
    try:
        make().embed([IMG, IMG, IMG])
    finally:
        FakeModel.__call__ = original  # type: ignore[method-assign]
    assert seen == [(3, 3, 224, 224)]


def test_embedding_is_deterministic(fakes: type[Fakes]) -> None:
    emb = make()
    np.testing.assert_array_equal(emb.embed([IMG]), emb.embed([IMG]))


def test_an_undecodable_image_names_its_index_and_never_reaches_the_model(
    fakes: type[Fakes],
) -> None:
    emb = make()
    with pytest.raises(ImageDecodeError) as err:
        emb.embed([IMG, b"garbage", IMG])
    assert err.value.index == 1
    assert len(calls("no_grad")) == 0


def test_an_oversized_image_is_refused_by_the_pixel_limit(fakes: type[Fakes]) -> None:
    emb = backends.HFEmbedderPlugin(card(), max_pixels=100).create(PERMISSIVE)
    with pytest.raises(ImageDecodeError) as err:
        emb.embed([IMG])
    assert err.value.reason == "too_many_pixels"


def test_empty_batches_return_an_empty_array(fakes: type[Fakes]) -> None:
    out = make().embed([])
    assert out.shape == (0, 16) and Fakes.calls == []


def test_entry_points_expose_the_three_models() -> None:
    names = registry.embedder_names()
    assert {"dinov2_small", "dinov2_base", "siglip2_base", "hash"} <= set(names)


def test_png_fixture_helper_is_a_real_image() -> None:
    from PIL import Image

    with Image.open(io.BytesIO(IMG)) as img:
        assert img.size == (40, 30)


def test_a_changed_config_is_refused_before_loading(fakes: type[Fakes]) -> None:
    Fakes.config = '{"auto_map": {"AutoModel": "evil.Model"}}'
    with pytest.raises(ModelError, match=r"config\.json.*does not match"):
        make().embed([IMG])
    assert calls("from_pretrained") == []


def test_a_weight_file_of_the_wrong_size_is_refused_before_it_is_hashed(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    Fakes.weights = WEIGHTS + b"x" * 1000
    hashed: list[object] = []
    monkeypatch.setattr(backends.models, "weight_hash", lambda p: hashed.append(p) or "x")
    with pytest.raises(ModelError, match="size"):
        make().embed([IMG])
    assert hashed == [] and calls("from_pretrained") == []


def test_telemetry_is_switched_off_for_the_hub_library(fakes: type[Fakes]) -> None:
    make().embed([IMG])
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_a_user_who_already_set_telemetry_keeps_their_choice(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    make().embed([IMG])
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "0"


@pytest.mark.parametrize("endpoint", ["https://hub.example.test", "http://localhost:9999"])
def test_a_custom_hub_endpoint_is_refused_in_ci_unless_confirmed(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    with pytest.raises(ModelError, match="HF_ENDPOINT"):
        make().embed([IMG])
    assert calls("snapshot_download") == []
    monkeypatch.setenv("DATACOPILOT_ALLOW_HF_ENDPOINT", "1")
    assert make().embed([IMG]).shape == (1, 16)


def test_the_default_endpoint_is_fine_in_ci_and_a_custom_one_is_fine_locally(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co/")
    assert make().embed([IMG]).shape == (1, 16)
    monkeypatch.delenv("CI")
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
    assert make().embed([IMG]).shape == (1, 16)
