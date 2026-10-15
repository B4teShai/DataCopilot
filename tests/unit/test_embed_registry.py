"""Embedder protocol and registry: resolution by name, license enforcement before any load."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from datacopilot.core.errors import FormatError, LicenseError
from datacopilot.core.schema import PluginSpec
from datacopilot.embed import registry

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def spec(name: str, license: str = "Apache-2.0") -> PluginSpec:  # noqa: A002
    return PluginSpec(
        name=name, kind="embedder", license=license, extra=None, gpu_memory_gb=None, version="1"
    )


class FakeEmbedder:
    def __init__(self, name: str, license: str = "Apache-2.0") -> None:  # noqa: A002
        self.spec = spec(name, license)
        self.created: list[frozenset[str]] = []

    model_id = "fake"
    revision = "1"
    dims = 4

    def create(self, allowed_licenses: frozenset[str]) -> FakeEmbedder:
        self.created.append(allowed_licenses)
        return self

    def embed(self, images: list[bytes]) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros((len(images), 4), dtype=np.float32)


@dataclass
class EP:
    name: str
    value: str
    obj: object = None
    error: Exception | None = None
    loads: list[str] = field(default_factory=list)

    def load(self) -> object:
        self.loads.append(self.name)
        if self.error:
            raise self.error
        return self.obj


def install(monkeypatch: pytest.MonkeyPatch, *eps: EP) -> None:
    monkeypatch.setattr(registry, "_entry_points", lambda: list(eps))


def test_names_are_sorted_and_nothing_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = EP("zeta", "m:z", FakeEmbedder("zeta")), EP("alpha", "m:a", FakeEmbedder("alpha"))
    install(monkeypatch, a, b)
    assert registry.embedder_names() == ["alpha", "zeta"]
    assert a.loads == [] and b.loads == []


def test_resolves_by_name_and_passes_the_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = FakeEmbedder("fake")
    install(monkeypatch, EP("fake", "m:p", plugin))
    got = registry.get_embedder("fake", PERMISSIVE)
    assert got is plugin and plugin.created == [PERMISSIVE]


def test_a_disallowed_plugin_license_is_refused_before_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = FakeEmbedder("fake", license="CC-BY-NC-4.0")
    install(monkeypatch, EP("fake", "m:p", plugin))
    with pytest.raises(LicenseError, match="DATACOPILOT_ALLOWED_LICENSES"):
        registry.get_embedder("fake", PERMISSIVE)
    assert plugin.created == []  # create (the only place weights are touched) never ran


def test_a_first_party_model_is_refused_before_its_module_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ep = EP("dinov2_base", "datacopilot.embed.backends:dinov2_base", FakeEmbedder("dinov2_base"))
    install(monkeypatch, ep)
    with pytest.raises(LicenseError):
        registry.get_embedder("dinov2_base", frozenset({"mit"}))
    assert ep.loads == []  # the card's license decided; the entry point was never loaded


def test_unknown_name_lists_the_available_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, EP("fake", "m:p", FakeEmbedder("fake")))
    with pytest.raises(FormatError, match="available embedders: fake"):
        registry.get_embedder("other", PERMISSIVE)


def test_duplicate_names_from_two_packages_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, EP("fake", "pkg1:p"), EP("fake", "pkg2:p"))
    with pytest.raises(FormatError, match="duplicate embedder"):
        registry.embedder_names()


def test_missing_optional_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    err = ImportError("no pillow")
    err.name = "PIL"
    install(monkeypatch, EP("fake", "m:p", error=err))
    with pytest.raises(FormatError, match=r"datacopilot\[embed\]"):
        registry.get_embedder("fake", PERMISSIVE)


@pytest.mark.parametrize("obj", [object(), FakeEmbedder("other")])
def test_invalid_plugin_objects_are_refused(monkeypatch: pytest.MonkeyPatch, obj: object) -> None:
    install(monkeypatch, EP("fake", "m:p", obj))
    with pytest.raises(FormatError):
        registry.get_embedder("fake", PERMISSIVE)


def test_a_plugin_that_returns_a_wrong_shape_is_the_pipelines_problem_not_the_registrys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, EP("fake", "m:p", FakeEmbedder("fake")))
    assert registry.get_embedder("fake", PERMISSIVE).dims == 4


@pytest.mark.parametrize("bad", ["", "A", "a b", "../x", "x" * 100, "a\n"])
def test_hostile_names_are_refused_and_bounded(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    install(monkeypatch)
    with pytest.raises(FormatError) as err:
        registry.get_embedder(bad, PERMISSIVE)
    assert len(str(err.value)) < 400
