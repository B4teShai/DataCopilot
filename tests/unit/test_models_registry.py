"""Model registry and license guard (Phase 3, task 1)."""

from __future__ import annotations

import re

import pytest

from datacopilot.core.errors import LicenseError, ModelError
from datacopilot.models import registry

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def test_every_card_pins_a_full_revision_and_weight_hashes() -> None:
    cards = registry.list_models()
    assert {c.model_id for c in cards} >= {"dinov2_small", "dinov2_base", "siglip2_base"}
    for card in cards:
        assert re.fullmatch(r"[0-9a-f]{40}", card.revision), card.model_id
        assert card.weights_sha256, card.model_id
        for name, digest in card.weights_sha256.items():
            assert name.endswith(".safetensors"), (card.model_id, name)  # never pickle
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert card.dims > 0 and card.license


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
