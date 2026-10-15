import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import PluginSpec
from datacopilot.io import registry
from datacopilot.io.report import ConversionReport


def spec(name: str, **kw: Any) -> PluginSpec:
    base: dict[str, Any] = {
        "name": name,
        "kind": "format",
        "license": "Apache-2.0",
        "extra": None,
        "gpu_memory_gb": None,
        "version": "1",
    }
    return PluginSpec(**{**base, **kw})


class Both:
    def __init__(self, name: str) -> None:
        self.spec = spec(name)

    def read(self, path: Path) -> registry.ReadResult:
        raise NotImplementedError

    def write(self, ds: object, out: Path) -> ConversionReport:
        raise NotImplementedError


class ReadOnly:
    def __init__(self, name: str) -> None:
        self.spec = spec(name)

    def read(self, path: Path) -> registry.ReadResult:
        raise NotImplementedError


@dataclass
class FakeEntryPoint:
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


def install(monkeypatch: pytest.MonkeyPatch, *eps: FakeEntryPoint) -> None:
    monkeypatch.setattr(registry, "_entry_points", lambda: list(eps))


def test_resolves_reader_and_writer_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = Both("fake")
    install(monkeypatch, FakeEntryPoint("fake", "m:p", plugin))
    assert registry.get_reader("fake") is plugin
    assert registry.get_writer("fake") is plugin


def test_only_the_requested_plugin_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = FakeEntryPoint("a", "m:a", Both("a")), FakeEntryPoint("b", "m:b", Both("b"))
    install(monkeypatch, a, b)
    registry.get_reader("a")
    assert a.loads == ["a"] and b.loads == []


def test_unknown_format_names_the_available_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeEntryPoint("alpha", "m:a", Both("alpha")))
    with pytest.raises(FormatError, match=r"unknown format 'nope'.*alpha"):
        registry.get_reader("nope")


def test_read_only_plugin_cannot_write(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeEntryPoint("ro", "m:ro", ReadOnly("ro")))
    assert registry.get_reader("ro")
    with pytest.raises(FormatError, match="'ro' cannot write"):
        registry.get_writer("ro")


def test_duplicate_names_from_different_packages_are_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch,
        FakeEntryPoint("dup", "pkg_a:plugin", Both("dup")),
        FakeEntryPoint("dup", "pkg_b:plugin", Both("dup")),
    )
    with pytest.raises(FormatError, match="duplicate.*'dup'.*pkg_a.*pkg_b"):
        registry.get_reader("dup")


def test_the_same_entry_point_listed_twice_is_not_a_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = Both("same")
    install(
        monkeypatch,
        FakeEntryPoint("same", "pkg:plugin", plugin),
        FakeEntryPoint("same", "pkg:plugin", plugin),
    )
    assert registry.get_reader("same") is plugin


def test_plugin_without_a_valid_spec_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeEntryPoint("bad", "m:p", object()))
    with pytest.raises(FormatError, match="'bad'.*PluginSpec"):
        registry.get_reader("bad")


def test_spec_name_must_match_the_entry_point_name(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeEntryPoint("one", "m:p", Both("two")))
    with pytest.raises(FormatError, match="'one'.*'two'"):
        registry.get_reader("one")


def test_missing_optional_dependency_is_reported_with_the_format_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = ImportError("no lxml", name="lxml")
    install(monkeypatch, FakeEntryPoint("xml", "m:p", error=err))
    with pytest.raises(FormatError, match="'xml'.*lxml"):
        registry.get_reader("xml")


def test_list_formats_returns_sorted_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        monkeypatch,
        FakeEntryPoint("zeta", "m:z", ReadOnly("zeta")),
        FakeEntryPoint("alpha", "m:a", Both("alpha")),
    )
    infos = registry.list_formats()
    assert [i.spec.name for i in infos] == ["alpha", "zeta"]
    assert [(i.can_read, i.can_write) for i in infos] == [(True, True), (True, False)]


def test_importing_the_registry_does_not_import_any_adapter() -> None:
    code = (
        "import sys, datacopilot.io.registry;"
        "bad=[m for m in sys.modules if m.startswith('datacopilot.io.') and m not in "
        "('datacopilot.io.registry','datacopilot.io.report')];"
        "print(bad); sys.exit(1 if bad or 'pyarrow' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stdout + proc.stderr
