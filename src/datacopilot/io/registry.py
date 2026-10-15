"""Format registry: name -> reader/writer, resolved lazily through entry points (ADR-0005).

Plugins register under the ``datacopilot.formats`` entry-point group. Nothing is imported until a
name is requested; each plugin object exposes ``spec: PluginSpec`` and ``read`` and/or ``write``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Category, Dataset, Image, PluginSpec, Task
from datacopilot.io.report import ConversionReport

GROUP = "datacopilot.formats"


@dataclass(frozen=True)
class ReadResult:
    """Rows produced by a reader, plus what it could not carry into the data model."""

    name: str
    task: Task
    images: Sequence[Image]
    annotations: Sequence[Annotation]
    categories: Sequence[Category]
    dropped: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Reader(Protocol):
    spec: PluginSpec

    def read(self, path: Path) -> ReadResult:
        """Read the dataset rooted at ``path``; raise FormatError for malformed input."""
        ...


@runtime_checkable
class Writer(Protocol):
    spec: PluginSpec

    def write(self, ds: Dataset, out: Path) -> ConversionReport:
        """Write ``ds`` into the existing empty directory ``out``."""
        ...


@dataclass(frozen=True)
class FormatInfo:
    spec: PluginSpec
    can_read: bool
    can_write: bool


def _entry_points() -> list[object]:
    """Seam for tests: the raw entry points of the formats group."""
    return list(entry_points(group=GROUP))


def _candidates() -> dict[str, object]:
    by_name: dict[str, dict[str, object]] = {}
    for ep in _entry_points():
        name, value = ep.name, ep.value  # type: ignore[attr-defined]
        by_name.setdefault(name, {})[value] = ep
    out: dict[str, object] = {}
    for name, values in by_name.items():
        if len(values) > 1:
            raise FormatError(
                f"duplicate format name {name!r} provided by: " + ", ".join(sorted(values))
            )
        out[name] = next(iter(values.values()))
    return out


def _load(name: str) -> object:
    available = _candidates()
    if name not in available:
        known = ", ".join(sorted(available)) or "none"
        raise FormatError(f"unknown format {name!r}; available formats: {known}")
    try:
        plugin = available[name].load()  # type: ignore[attr-defined]
    except ImportError as exc:
        missing = exc.name or "a dependency"
        raise FormatError(
            f"format {name!r} needs an optional dependency that is not installed: {missing}"
        ) from None
    spec = getattr(plugin, "spec", None)
    if not isinstance(spec, PluginSpec) or spec.kind != "format":
        raise FormatError(f"format {name!r} does not expose a valid PluginSpec")
    if spec.name != name:
        raise FormatError(f"format {name!r} declares a different spec name {spec.name!r}")
    return plugin


def get_reader(name: str) -> Reader:
    """Resolve a reader by format name.

    Raises:
        FormatError: unknown name, duplicate registration, invalid plugin, missing dependency,
            or a write-only plugin.
    """
    plugin = _load(name)
    if not callable(getattr(plugin, "read", None)):
        raise FormatError(f"format {name!r} cannot read")
    return plugin  # type: ignore[return-value]


def get_writer(name: str) -> Writer:
    """Resolve a writer by format name; raises FormatError as :func:`get_reader` does."""
    plugin = _load(name)
    if not callable(getattr(plugin, "write", None)):
        raise FormatError(f"format {name!r} cannot write")
    return plugin  # type: ignore[return-value]


def list_formats() -> list[FormatInfo]:
    """All registered formats, sorted by name (loads each plugin module)."""
    infos = []
    for name in sorted(_candidates()):
        plugin = _load(name)
        infos.append(
            FormatInfo(
                spec=plugin.spec,  # type: ignore[attr-defined]
                can_read=callable(getattr(plugin, "read", None)),
                can_write=callable(getattr(plugin, "write", None)),
            )
        )
    return infos
