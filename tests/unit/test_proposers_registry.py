"""Proposer registry: resolution by name, license enforcement before any load."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import pytest
import structlog

from datacopilot.core.errors import FormatError, LicenseError
from datacopilot.core.schema import PluginSpec
from datacopilot.proposers import registry
from datacopilot.proposers.protocol import Candidate, Proposer, ProposerPlugin

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def spec(name: str, license: str = "Apache-2.0", kind: str = "proposer") -> PluginSpec:  # noqa: A002
    return PluginSpec(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        license=license,
        extra=None,
        gpu_memory_gb=None,
        version="1",
    )


class FakeProposer:
    model_id = "fake"
    revision = "1"
    weights: dict[str, str] = {}  # noqa: RUF012

    def __init__(self, name: str, license: str = "Apache-2.0", kind: str = "proposer") -> None:  # noqa: A002
        self.spec = spec(name, license, kind)
        self.created: list[frozenset[str]] = []

    def create(self, allowed_licenses: frozenset[str]) -> FakeProposer:
        self.created.append(allowed_licenses)
        return self

    def detect(
        self, image: bytes, phrases: Sequence[str], size: tuple[int, int], *, min_score: float
    ) -> list[Candidate]:
        return []


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


@pytest.fixture(autouse=True)
def _captured_logs() -> Iterator[None]:
    with structlog.testing.capture_logs():
        yield


def test_names_are_sorted_and_nothing_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = EP("zeta", "m:z", FakeProposer("zeta")), EP("alpha", "m:a", FakeProposer("alpha"))
    install(monkeypatch, a, b)
    assert registry.proposer_names() == ["alpha", "zeta"]
    assert a.loads == [] and b.loads == []


def test_resolves_by_name_and_passes_the_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = FakeProposer("fake")
    install(monkeypatch, EP("fake", "m:p", plugin))
    got = registry.get_proposer("fake", PERMISSIVE)
    assert got is plugin and plugin.created == [PERMISSIVE]
    assert isinstance(plugin, ProposerPlugin) and isinstance(got, Proposer)


def test_check_proposer_answers_the_license_question_without_creating_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = FakeProposer("fake")
    install(monkeypatch, EP("fake", "m:p", plugin))
    assert registry.check_proposer("fake", PERMISSIVE).name == "fake"
    assert plugin.created == []


def test_a_disallowed_plugin_license_is_refused_before_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = FakeProposer("fake", license="CC-BY-NC-4.0")
    install(monkeypatch, EP("fake", "m:p", plugin))
    with pytest.raises(LicenseError, match="DATACOPILOT_ALLOWED_LICENSES"):
        registry.get_proposer("fake", PERMISSIVE)
    with pytest.raises(LicenseError):
        registry.check_proposer("fake", PERMISSIVE)
    assert plugin.created == []


def test_a_first_party_model_is_refused_before_its_module_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ep = EP("grounding_dino", "datacopilot.proposers.grounding_dino:plugin", FakeProposer("x"))
    install(monkeypatch, ep)
    for call in (registry.get_proposer, registry.check_proposer):
        with pytest.raises(LicenseError):
            call("grounding_dino", frozenset({"mit"}))
    assert ep.loads == []  # the card's license decided; the entry point was never loaded


def test_unknown_name_lists_the_available_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, EP("fake", "m:p", FakeProposer("fake")))
    with pytest.raises(FormatError, match="available proposers: fake"):
        registry.get_proposer("other", PERMISSIVE)


def test_duplicate_names_from_two_packages_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, EP("fake", "pkg1:p"), EP("fake", "pkg2:p"))
    with pytest.raises(FormatError, match="duplicate proposer"):
        registry.proposer_names()


def test_missing_optional_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    err = ImportError("no transformers")
    err.name = "transformers"
    install(monkeypatch, EP("fake", "m:p", error=err))
    with pytest.raises(FormatError, match=r"datacopilot\[vlm\]"):
        registry.get_proposer("fake", PERMISSIVE)


@pytest.mark.parametrize(
    "obj",
    [object(), FakeProposer("other"), FakeProposer("fake", kind="embedder")],
)
def test_invalid_plugin_objects_are_refused(monkeypatch: pytest.MonkeyPatch, obj: object) -> None:
    install(monkeypatch, EP("fake", "m:p", obj))
    with pytest.raises(FormatError):
        registry.get_proposer("fake", PERMISSIVE)


def test_a_created_object_that_cannot_detect_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class Bad(FakeProposer):
        def create(self, allowed_licenses: frozenset[str]) -> object:  # type: ignore[override]
            return object()

    install(monkeypatch, EP("fake", "m:p", Bad("fake")))
    with pytest.raises(FormatError, match="did not create"):
        registry.get_proposer("fake", PERMISSIVE)


@pytest.mark.parametrize("bad", ["", "A", "a b", "../x", "x" * 100, "a\n"])
def test_hostile_names_are_refused_and_bounded(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    install(monkeypatch)
    for call in (registry.get_proposer, registry.check_proposer):
        with pytest.raises(FormatError) as err:
            call(bad, PERMISSIVE)
        assert len(str(err.value)) < 400


def test_the_first_party_entry_points_are_registered() -> None:
    assert {"grounding_dino", "test_boxes"} <= set(registry.proposer_names())
