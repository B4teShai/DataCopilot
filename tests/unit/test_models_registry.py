"""Model registry and license guard (Phase 3, task 1)."""

from __future__ import annotations

import re

import pytest

from datacopilot.core.errors import LicenseError, ModelError
from datacopilot.models import registry

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def test_every_card_pins_a_full_revision_and_weight_hashes() -> None:
    cards = registry.list_models()
    assert {c.model_id for c in cards} >= {
        "dinov2_small",
        "dinov2_base",
        "siglip2_base",
        "grounding_dino",
    }
    for card in cards:
        assert re.fullmatch(r"[0-9a-f]{40}", card.revision), card.model_id
        assert card.weights_sha256, card.model_id
        for name, digest in card.weights_sha256.items():
            assert name.endswith(".safetensors"), (card.model_id, name)  # never pickle
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert (card.dims is None or card.dims > 0) and card.license
        assert (card.dims is None) == (card.family == "grounding_dino"), card.model_id
        assert re.fullmatch(r"[0-9a-f]{64}", card.config_sha256), card.model_id
        assert set(card.weights_bytes) == set(card.weights_sha256)
        assert all(n > 0 for n in card.weights_bytes.values())


def test_cards_are_listed_in_a_stable_order() -> None:
    names = [c.model_id for c in registry.list_models()]
    assert names == sorted(names)


def test_get_model_returns_the_card_for_an_allowed_license() -> None:
    card = registry.get_model("dinov2_base", PERMISSIVE)
    assert card.repo == "facebook/dinov2-base" and card.license == "Apache-2.0"


def test_license_comparison_is_case_insensitive() -> None:
    assert registry.get_model("dinov2_base", frozenset({"apache-2.0"})).model_id == "dinov2_base"


def test_disallowed_license_raises_before_anything_else_names_model_and_variable() -> None:
    with pytest.raises(LicenseError) as err:
        registry.get_model("dinov2_base", frozenset({"mit"}))
    text = str(err.value)
    assert "dinov2_base" in text and "Apache-2.0" in text
    assert "DATACOPILOT_ALLOWED_LICENSES" in text


def test_empty_allow_list_allows_nothing() -> None:
    with pytest.raises(LicenseError):
        registry.get_model("siglip2_base", frozenset())


def test_unknown_model_lists_the_known_ones() -> None:
    with pytest.raises(ModelError, match="dinov2_base"):
        registry.get_model("nope", PERMISSIVE)


@pytest.mark.parametrize("bad", ["", "../x", "A B", "x" * 200, "dinov2_base\n"])
def test_hostile_model_ids_are_rejected_without_echoing_them_unbounded(bad: str) -> None:
    with pytest.raises(ModelError) as err:
        registry.get_model(bad, PERMISSIVE)
    assert len(str(err.value)) < 400


def test_card_rejects_unpinned_or_malformed_fields() -> None:
    base = dict(
        model_id="m",
        repo="a/b",
        revision="main",
        license="MIT",
        gated=False,
        gpu_memory_gb=None,
        dims=8,
        family="dinov2",
        weights_sha256={"model.safetensors": "0" * 64},
        weights_bytes={"model.safetensors": 10},
        config_sha256="1" * 64,
    )
    with pytest.raises(ValueError):
        registry.ModelCard(**base)  # type: ignore[arg-type]  # a branch name is not a pin
    with pytest.raises(ValueError):
        registry.ModelCard(**{**base, "revision": "0" * 40, "repo": "../b"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        registry.ModelCard(
            **{**base, "revision": "0" * 40, "weights_sha256": {"pytorch_model.bin": "0" * 64}}  # type: ignore[arg-type]
        )
    registry.ModelCard(**{**base, "revision": "0" * 40})  # type: ignore[arg-type]


def test_weight_hash_streams_a_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    f = tmp_path / "w.safetensors"
    f.write_bytes(b"abc" * 100_000)
    assert registry.weight_hash(f) == hashlib.sha256(b"abc" * 100_000).hexdigest()


def test_weight_hash_refuses_symlinks_and_directories(tmp_path) -> None:  # type: ignore[no-untyped-def]
    real = tmp_path / "w.safetensors"
    real.write_bytes(b"x")
    link = tmp_path / "l.safetensors"
    link.symlink_to(real)
    with pytest.raises(ModelError):
        registry.weight_hash(link)
    with pytest.raises(ModelError):
        registry.weight_hash(tmp_path)


def test_card_rejects_a_malformed_config_digest_or_missing_sizes() -> None:
    base = dict(
        model_id="m",
        repo="a/b",
        revision="0" * 40,
        license="MIT",
        gated=False,
        gpu_memory_gb=None,
        dims=8,
        family="dinov2",
        weights_sha256={"model.safetensors": "0" * 64},
        weights_bytes={"model.safetensors": 10},
        config_sha256="1" * 64,
    )
    registry.ModelCard(**base)  # type: ignore[arg-type]
    for change in (
        dict(config_sha256="zz"),
        dict(weights_bytes={}),
        dict(weights_bytes={"model.safetensors": 0}),
        dict(weights_bytes={"other.safetensors": 5}),
    ):
        with pytest.raises(ValueError):
            registry.ModelCard(**{**base, **change})  # type: ignore[arg-type]


def test_the_detector_card_pins_every_file_it_loads() -> None:
    card = registry.get_model("grounding_dino", PERMISSIVE)
    assert card.repo == "IDEA-Research/grounding-dino-tiny" and card.license == "Apache-2.0"
    assert card.revision == "a2bb814dd30d776dcf7e30523b00659f4f141c71" and card.dims is None
    assert card.family == "grounding_dino" and not card.gated
    assert card.weights_sha256 == {
        "model.safetensors": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
    }
    assert card.weights_bytes == {"model.safetensors": 689359096}
    # the tokenizer and processor files are data, but they decide what the model is asked
    assert set(card.aux_sha256) == {
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "added_tokens.json",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in card.aux_sha256.values())


def test_the_detector_is_refused_before_anything_else_when_its_license_is_not_allowed() -> None:
    with pytest.raises(LicenseError, match="grounding_dino"):
        registry.get_model("grounding_dino", frozenset({"mit"}))


def _base() -> dict[str, object]:
    return dict(
        model_id="m",
        repo="a/b",
        revision="0" * 40,
        license="MIT",
        gated=False,
        gpu_memory_gb=None,
        dims=8,
        family="dinov2",
        weights_sha256={"model.safetensors": "0" * 64},
        weights_bytes={"model.safetensors": 10},
        config_sha256="1" * 64,
    )


def test_an_embedder_card_needs_dims_and_a_detector_card_must_not_have_them() -> None:
    with pytest.raises(ValueError, match="dims"):
        registry.ModelCard(**{**_base(), "dims": None})  # type: ignore[arg-type]
    registry.ModelCard(**{**_base(), "dims": None, "family": "grounding_dino"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dims"):
        registry.ModelCard(**{**_base(), "family": "grounding_dino"})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "aux",
    [
        {"model.bin": "0" * 64},
        {"weights.pt": "0" * 64},
        {"weights.pth": "0" * 64},
        {"x.pkl": "0" * 64},
        {"x.ckpt": "0" * 64},
        {"modeling_evil.py": "0" * 64},
        {"../vocab.txt": "0" * 64},
        {"sub/vocab.txt": "0" * 64},
        {"vocab.txt": "zz"},
        {f"f{i}.txt": "0" * 64 for i in range(17)},
    ],
)
def test_auxiliary_files_can_never_be_code_or_a_pickle(aux: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        registry.ModelCard(**{**_base(), "aux_sha256": aux})  # type: ignore[arg-type]
    registry.ModelCard(**{**_base(), "aux_sha256": {"vocab.txt": "0" * 64}})  # type: ignore[arg-type]
