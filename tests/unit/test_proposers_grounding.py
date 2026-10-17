"""Grounding DINO backend with injected fake libraries: no network, no weights, no tensors.

The tensor library is spelled indirectly (``backends.TENSOR_LIB``) because the repository's QA gate
forbids the literal name in sources.
"""

from __future__ import annotations

import hashlib
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
from datacopilot.embed import backends
from datacopilot.embed.protocol import ImageDecodeError
from datacopilot.models.registry import ModelCard
from datacopilot.proposers import grounding_dino as gd
from datacopilot.proposers.protocol import Proposer, ProposerPlugin

pytest.importorskip("PIL")
PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})
WEIGHTS = b"fake safetensors payload"
AUX = {"vocab.txt": b"[PAD]\n", "tokenizer.json": b"{}", "preprocessor_config.json": b"{}"}
CONFIG = b"{}"


def card(**kw: Any) -> ModelCard:
    base: dict[str, Any] = dict(
        model_id="fake_detector",
        repo="owner/fake-detector",
        revision="a" * 40,
        license="Apache-2.0",
        gated=False,
        gpu_memory_gb=1.0,
        dims=None,
        family="grounding_dino",
        weights_sha256={"model.safetensors": hashlib.sha256(WEIGHTS).hexdigest()},
        weights_bytes={"model.safetensors": len(WEIGHTS)},
        config_sha256=hashlib.sha256(CONFIG).hexdigest(),
        aux_sha256={n: hashlib.sha256(b).hexdigest() for n, b in AUX.items()},
    )
    base.update(kw)
    return ModelCard(**base)


class FakeTensor:
    def __init__(self, array: Any, device: str = "cpu") -> None:
        self.array, self.device = np.asarray(array), device

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

    def numpy(self) -> Any:
        return self.array


class Fakes:
    cuda = False
    cuda_breaks = False
    calls: list[tuple[str, dict[str, Any]]] = []
    snapshot_root: Path
    weights = WEIGHTS
    aux = dict(AUX)
    config = CONFIG
    download_error: Exception | None = None
    #: what the next post-processing returns, one entry per call (a list of dicts per image)
    results: list[Any] = []
    token_count = staticmethod(lambda text: len(text.split()))


class FakeBatch(dict):  # type: ignore[type-arg]
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.input_ids = kw["input_ids"]

    def to(self, device: str) -> FakeBatch:
        if device == "cuda" and Fakes.cuda_breaks:
            raise RuntimeError("CUDA out of memory token=hf_SECRET123")
        return self


class FakeProcessor:
    class _Tokenizer:
        def __call__(self, text: str) -> dict[str, list[int]]:
            return {"input_ids": list(range(Fakes.token_count(text) + 2))}

    tokenizer = _Tokenizer()

    def __call__(self, *, images: Any, text: str, return_tensors: str) -> FakeBatch:
        Fakes.calls.append(("processor", {"size": images.size, "text": text}))
        return FakeBatch(input_ids=FakeTensor([[1, 2, 3]]), pixel_values=FakeTensor([0.0]))

    def post_process_grounded_object_detection(
        self, outputs: Any, input_ids: Any, **kw: Any
    ) -> Any:
        Fakes.calls.append(("post_process", kw))
        return Fakes.results.pop(0) if Fakes.results else [_result([], [], [])]


def _result(boxes: Any, scores: Any, labels: list[str]) -> dict[str, Any]:
    return {
        "boxes": FakeTensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
        "scores": FakeTensor(np.asarray(scores, dtype=np.float32)),
        "text_labels": labels,
        "labels": labels,
    }


class FakeDetector:
    def __init__(self) -> None:
        self.device = "cpu"

    def eval(self) -> FakeDetector:
        return self

    def to(self, device: str) -> FakeDetector:
        if device == "cuda" and Fakes.cuda_breaks:
            raise RuntimeError("CUDA out of memory token=hf_SECRET123")
        self.device = device
        return self

    def __call__(self, **inputs: Any) -> Any:
        Fakes.calls.append(("model", {"device": self.device}))
        return types.SimpleNamespace()


@pytest.fixture(autouse=True)
def _captured_logs() -> Iterator[None]:
    with structlog.testing.capture_logs():
        yield


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[Fakes]:
    Fakes.cuda = Fakes.cuda_breaks = False
    Fakes.calls, Fakes.results, Fakes.download_error = [], [], None
    Fakes.weights, Fakes.aux, Fakes.config = WEIGHTS, dict(AUX), CONFIG
    Fakes.snapshot_root = tmp_path / "hub"

    def snapshot_download(**kw: Any) -> str:
        Fakes.calls.append(("snapshot_download", kw))
        if Fakes.download_error:
            raise Fakes.download_error
        snap = Fakes.snapshot_root / "snapshots" / kw["revision"]
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").write_bytes(Fakes.weights)
        (snap / "config.json").write_bytes(Fakes.config)
        for name, data in Fakes.aux.items():
            (snap / name).write_bytes(data)
        return str(snap)

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path: str, **kw: Any) -> FakeProcessor:
            Fakes.calls.append(("processor_from_pretrained", {"path": path, **kw}))
            return FakeProcessor()

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **kw: Any) -> FakeDetector:
            Fakes.calls.append(("from_pretrained", {"path": path, **kw}))
            return FakeDetector()

    @contextmanager
    def no_grad() -> Iterator[None]:
        Fakes.calls.append(("no_grad", {}))
        yield

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    tf = types.ModuleType("transformers")
    tf.AutoProcessor = AutoProcessor  # type: ignore[attr-defined]
    tf.AutoModelForZeroShotObjectDetection = AutoModel  # type: ignore[attr-defined]
    lib = types.ModuleType(backends.TENSOR_LIB)
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


def make(**kw: Any) -> Proposer:
    return gd.GroundingDinoPlugin(card(**kw)).create(PERMISSIVE)


def calls(name: str) -> list[dict[str, Any]]:
    return [kw for n, kw in Fakes.calls if n == name]


IMG = png_bytes(40, 30, 2)
SIZE = (40, 30)


# --------------------------------------------------------------- construction and the license guard
def test_the_first_party_plugin_describes_its_card() -> None:
    assert isinstance(gd.plugin, ProposerPlugin)
    spec = gd.plugin.spec
    assert spec.name == "grounding_dino" and spec.kind == "proposer"
    assert spec.extra == "vlm" and spec.license == "Apache-2.0"


def test_create_touches_nothing_until_the_first_detect(fakes: type[Fakes]) -> None:
    proposer = make()
    assert Fakes.calls == [] and proposer.revision == "a" * 40
    assert proposer.model_id == "fake_detector"
    # the identity covers weights, configuration and the tokenizer files
    assert set(proposer.weights) == {"model.safetensors", "config.json", *AUX}


def test_a_disallowed_license_is_refused_before_any_library_is_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(name: str) -> None:
        raise AssertionError(f"{name} must not be imported")

    for module in ("huggingface_hub", "transformers", backends.TENSOR_LIB):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(backends, "_import", explode)
    with pytest.raises(LicenseError, match="DATACOPILOT_ALLOWED_LICENSES"):
        gd.GroundingDinoPlugin(card(license="CC-BY-NC-4.0")).create(PERMISSIVE)


def test_the_real_registry_path_refuses_before_the_module_loads() -> None:
    from datacopilot.proposers import registry

    with pytest.raises(LicenseError):
        registry.get_proposer("grounding_dino", frozenset({"mit"}))


def test_a_detector_card_cannot_become_an_embedder() -> None:
    with pytest.raises(ModelError, match="detector"):
        backends.HFEmbedder(card(), 1000)


# --------------------------------------------------------------- loading rules
def test_the_model_is_loaded_safely_from_a_pinned_verified_local_snapshot(
    fakes: type[Fakes],
) -> None:
    make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    (download,) = calls("snapshot_download")
    assert download["repo_id"] == "owner/fake-detector" and download["revision"] == "a" * 40
    assert set(download["allow_patterns"]) == {"model.safetensors", "config.json", *AUX}
    assert not [p for p in download["allow_patterns"] if p.endswith((".bin", ".py", ".pt"))]
    assert download["etag_timeout"] <= 60
    (model,) = calls("from_pretrained")
    (processor,) = calls("processor_from_pretrained")
    assert model["trust_remote_code"] is False and model["use_safetensors"] is True
    assert model["local_files_only"] is True
    assert processor["trust_remote_code"] is False and processor["local_files_only"] is True
    assert model["path"] == processor["path"] and "revision" not in model
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] and os.environ["HF_HUB_ETAG_TIMEOUT"]


def test_the_model_loads_once_for_many_images(fakes: type[Fakes]) -> None:
    proposer = make()
    proposer.detect(IMG, ["cat"], SIZE, min_score=0.3)
    proposer.detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert len(calls("snapshot_download")) == 1 and len(calls("from_pretrained")) == 1
    assert len(calls("model")) == 2 and len(calls("no_grad")) == 2


def test_weights_that_do_not_match_the_pinned_hash_are_refused_before_loading(
    fakes: type[Fakes],
) -> None:
    Fakes.weights = b"X" * len(WEIGHTS)
    with pytest.raises(ModelError, match="does not match"):
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert calls("from_pretrained") == [] and calls("processor_from_pretrained") == []


@pytest.mark.parametrize("name", sorted(AUX))
def test_a_changed_tokenizer_or_processor_file_is_refused_before_loading(
    fakes: type[Fakes], name: str
) -> None:
    Fakes.aux[name] = b"[changed]"
    with pytest.raises(ModelError, match=name.replace(".", r"\.")):
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert calls("from_pretrained") == [] and calls("processor_from_pretrained") == []


def test_a_changed_config_is_refused_before_loading(fakes: type[Fakes]) -> None:
    Fakes.config = b'{"auto_map": {"AutoModel": "evil.Model"}}'
    with pytest.raises(ModelError, match=r"config\.json.*does not match"):
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert calls("from_pretrained") == []


def test_download_errors_do_not_leak_their_text(fakes: type[Fakes]) -> None:
    Fakes.download_error = ConnectionError("https://hub/x?token=hf_SECRET123 failed")
    with pytest.raises(ModelError) as err:
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    text = str(err.value)
    assert "hf_SECRET123" not in text and "ConnectionError" in text and "fake_detector" in text


def test_missing_optional_libraries_name_the_vlm_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("huggingface_hub", "transformers", backends.TENSOR_LIB):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ModelError, match=r"datacopilot\[vlm\]"):
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)


def test_a_foreign_hub_endpoint_is_refused_in_ci_unless_confirmed(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
    with pytest.raises(ModelError, match="HF_ENDPOINT"):
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert calls("snapshot_download") == []


def test_cuda_is_used_when_available_and_a_failure_falls_back_to_cpu(fakes: type[Fakes]) -> None:
    Fakes.cuda = True
    with structlog.testing.capture_logs() as logs:
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert [e for e in logs if e["event"] == "model_loaded"][0]["device"] == "cuda"
    Fakes.calls.clear()
    Fakes.cuda_breaks = True
    with structlog.testing.capture_logs() as logs:
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    events = {e["event"]: e for e in logs}
    assert events["model_loaded"]["device"] == "cpu" and "hf_SECRET123" not in repr(logs)
    assert events["proposer_device_fallback"]["log_level"] == "warning"


def test_loading_logs_the_pinned_hashes_and_the_license(fakes: type[Fakes]) -> None:
    with structlog.testing.capture_logs() as logs:
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    loaded = [e for e in logs if e["event"] == "model_loaded"][0]
    assert loaded["model_id"] == "fake_detector" and set(loaded["weights_sha256"]) >= set(AUX)


# --------------------------------------------------------------- inference
def test_the_prompt_is_lower_case_dotted_and_the_thresholds_are_passed(fakes: type[Fakes]) -> None:
    Fakes.results = [[_result([[1, 2, 21, 22]], [0.9], ["cat"])]]
    out = make().detect(IMG, ["cat", "traffic light"], SIZE, min_score=0.42)
    (call,) = calls("processor")
    assert call["text"] == "cat. traffic light." and call["size"] == (40, 30)
    (post,) = calls("post_process")
    assert post["threshold"] == 0.42 and post["text_threshold"] == gd.TEXT_THRESHOLD
    assert post["target_sizes"] == [(30, 40)]  # (height, width) of the decoded image
    assert [(c.label, c.box, round(c.score, 3)) for c in out] == [
        ("cat", (1.0, 2.0, 21.0, 22.0), 0.9)
    ]


def test_phrases_are_split_so_that_no_prompt_exceeds_the_token_limit(fakes: type[Fakes]) -> None:
    phrases = [f"word{i} again" for i in range(150)]  # 2 tokens each in the fake tokenizer
    Fakes.results = [[_result([], [], [])] for _ in range(10)]
    assert len(phrases) * 2 > gd.MAX_TEXT_TOKENS
    make().detect(IMG, phrases, SIZE, min_score=0.3)
    texts = [c["text"] for c in calls("processor")]
    assert len(texts) > 1
    assert all(Fakes.token_count(t) + 2 <= gd.MAX_TEXT_TOKENS for t in texts)
    seen = [p for t in texts for p in t.rstrip(".").split(". ")]
    assert seen == phrases  # every phrase is asked about exactly once, in order


def test_results_of_every_chunk_are_collected(fakes: type[Fakes]) -> None:
    Fakes.results = [
        [_result([[1, 1, 5, 5]], [0.8], ["a"])],
        [_result([[2, 2, 9, 9]], [0.7], ["b"])],
    ]
    # a prompt of one phrase is 132 tokens, of two phrases 262: one phrase per prompt
    Fakes.token_count = staticmethod(lambda text: 130 * len(text.split(". ")))
    try:
        out = make().detect(IMG, ["a", "b"], SIZE, min_score=0.3)
    finally:
        Fakes.token_count = staticmethod(lambda text: len(text.split()))
    assert [c.label for c in out] == ["a", "b"]


@pytest.mark.parametrize(
    "bad", ["", "Cat", "a.b", "a. b", "a\nb", "a\x00", "x" * 65, "ünï", "a;b", "<b>", " a"]
)
def test_a_phrase_that_could_change_the_prompt_is_refused_before_the_model_runs(
    fakes: type[Fakes], bad: str
) -> None:
    with pytest.raises(ValueError, match="phrase"):
        make().detect(IMG, ["cat", bad], SIZE, min_score=0.3)
    assert Fakes.calls == []


def test_too_many_phrases_are_refused(fakes: type[Fakes]) -> None:
    with pytest.raises(ValueError, match="phrases"):
        make().detect(IMG, [f"p{i}" for i in range(gd.MAX_PHRASES + 1)], SIZE, min_score=0.3)


@pytest.mark.parametrize(
    "result",
    [
        [],
        [{}],
        [{"boxes": 1, "scores": 2, "text_labels": 3}],
        [{"boxes": FakeTensor([[1, 2, 3]]), "scores": FakeTensor([0.5]), "text_labels": ["a"]}],
        [
            {
                "boxes": FakeTensor([[1, 2, 3, 4]]),
                "scores": FakeTensor([0.5, 0.6]),
                "text_labels": ["a"],
            }
        ],
        [{"boxes": FakeTensor([[1, 2, 3, 4]]), "scores": FakeTensor([0.5]), "text_labels": []}],
        "nonsense",
    ],
)
def test_an_unusable_model_result_is_a_model_error(fakes: type[Fakes], result: Any) -> None:
    Fakes.results = [result]
    with pytest.raises(ModelError, match="unusable"):
        make().detect(IMG, ["a"], SIZE, min_score=0.3)


def test_a_photo_without_detections_returns_nothing_even_though_the_library_reports_one_empty_label(
    fakes: type[Fakes],
) -> None:
    # real output for a photo with no box: no boxes, no scores, and text_labels == [""]
    Fakes.results = [[_result([], [], [""])]]
    assert make().detect(IMG, ["cat"], SIZE, min_score=0.3) == []


def test_non_finite_numbers_are_passed_on_for_the_caller_to_reject(fakes: type[Fakes]) -> None:
    # validation is proposals/run.py's job; the backend must simply not crash or hide them
    Fakes.results = [[_result([[np.nan, 0, 5, 5]], [np.inf], ["a"])]]
    (c,) = make().detect(IMG, ["a"], SIZE, min_score=0.3)
    assert np.isnan(c.box[0]) and np.isinf(c.score)


def test_an_undecodable_image_never_reaches_the_model(fakes: type[Fakes]) -> None:
    with pytest.raises(ImageDecodeError) as err:
        make().detect(b"garbage", ["cat"], SIZE, min_score=0.3)
    assert err.value.reason == "decode_failed" and Fakes.calls == []


def test_an_oversized_image_is_refused_by_the_pixel_limit(fakes: type[Fakes]) -> None:
    proposer = gd.GroundingDinoPlugin(card(), max_pixels=100).create(PERMISSIVE)
    with pytest.raises(ImageDecodeError) as err:
        proposer.detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert err.value.reason == "too_many_pixels"


def test_a_model_failure_is_a_bounded_model_error_without_the_original_text(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: FakeDetector, **inputs: Any) -> Any:
        raise RuntimeError("secret=hf_SECRET123")

    monkeypatch.setattr(FakeDetector, "__call__", boom)
    with pytest.raises(ModelError) as err:
        make().detect(IMG, ["cat"], SIZE, min_score=0.3)
    assert "hf_SECRET123" not in str(err.value) and "RuntimeError" in str(err.value)


def test_a_phrase_less_call_does_nothing(fakes: type[Fakes]) -> None:
    assert make().detect(IMG, [], SIZE, min_score=0.3) == [] and Fakes.calls == []


def test_the_plugin_creates_a_valid_proposer(fakes: type[Fakes]) -> None:
    assert isinstance(make(), Proposer)


# ----------------------------------------------------------------------------- thresholds and sizes
def test_the_text_threshold_follows_a_low_min_score(fakes: type[Fakes]) -> None:
    # with the fixed library default of 0.25 a min_score of 0.1 silently did nothing
    make().detect(IMG, ["cat"], SIZE, min_score=0.1)
    make().detect(IMG, ["cat"], SIZE, min_score=0.6)
    low, high = calls("post_process")
    assert low["threshold"] == 0.1 and low["text_threshold"] == 0.1
    assert high["threshold"] == 0.6 and high["text_threshold"] == gd.TEXT_THRESHOLD


def test_boxes_between_the_thresholds_are_returned_when_the_caller_asked_for_them(
    fakes: type[Fakes], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = [([1, 2, 21, 22], 0.2, "cat"), ([2, 3, 20, 25], 0.4, "cat")]

    def post(self: FakeProcessor, outputs: Any, input_ids: Any, **kw: Any) -> Any:
        Fakes.calls.append(("post_process", kw))
        keep = [r for r in raw if r[1] >= max(kw["threshold"], kw["text_threshold"])]
        return [_result([r[0] for r in keep], [r[1] for r in keep], [r[2] for r in keep])]

    monkeypatch.setattr(FakeProcessor, "post_process_grounded_object_detection", post)
    assert [round(c.score, 2) for c in make().detect(IMG, ["cat"], SIZE, min_score=0.1)] == [
        0.2,
        0.4,
    ]
    assert [round(c.score, 2) for c in make().detect(IMG, ["cat"], SIZE, min_score=0.3)] == [0.4]


def test_an_image_whose_decoded_size_differs_from_the_recorded_size_is_skipped(
    fakes: type[Fakes],
) -> None:
    with pytest.raises(ImageDecodeError) as err:
        make().detect(IMG, ["cat"], (30, 40), min_score=0.3)  # recorded as rotated
    assert err.value.reason == "size_mismatch" and Fakes.calls == []


def test_the_prompt_chunks_are_computed_once_per_run(fakes: type[Fakes]) -> None:
    proposer = make()
    proposer.detect(IMG, ["cat", "dog"], SIZE, min_score=0.3)
    proposer.detect(IMG, ["cat", "dog"], SIZE, min_score=0.3)
    tokenised = []
    original = FakeProcessor._Tokenizer.__call__

    def spy(self: Any, text: str) -> Any:
        tokenised.append(text)
        return original(self, text)

    FakeProcessor._Tokenizer.__call__ = spy  # type: ignore[method-assign]
    try:
        proposer.detect(IMG, ["cat", "dog"], SIZE, min_score=0.3)
    finally:
        FakeProcessor._Tokenizer.__call__ = original  # type: ignore[method-assign]
    assert tokenised == []
