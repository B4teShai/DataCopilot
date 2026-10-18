"""Check registry: name -> check, resolved lazily through entry points (ADR-0005).

Checks register under the ``datacopilot.checks`` entry-point group. Nothing is imported until a
name is requested; each plugin exposes ``spec: PluginSpec`` (kind ``check``) and ``run``.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Dataset, Finding, PluginSpec

if TYPE_CHECKING:
    from datacopilot.audit.context import CheckContext

GROUP = "datacopilot.checks"
CANONICAL_ORDER: tuple[str, ...] = (
    "integrity",
    "duplicates",
    "near_duplicates",
    "leakage",
    "imbalance",
    "boxes",
    "label_noise",
    "outliers",
    "sku_confusion",
)
"""Documented checks in run order (docs/architecture/02-components.md). Checks outside this list
run after them, alphabetically."""
EMBEDDING_CHECKS: frozenset[str] = frozenset(
    {"near_duplicates", "leakage", "label_noise", "outliers"}
)
"""Checks that need image embeddings. ``--checks all`` includes them only when an embedder is
given; naming one without an embedder is a usage error."""


@runtime_checkable
class Check(Protocol):
    spec: PluginSpec

    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]:
        """Inspect ``ds`` and return findings; never mutate the dataset."""
        ...


def _entry_points() -> list[object]:
    """Seam for tests: the raw entry points of the checks group."""
    return list(entry_points(group=GROUP))


def _candidates() -> dict[str, object]:
    by_name: dict[str, dict[str, object]] = {}
    for ep in _entry_points():
        by_name.setdefault(ep.name, {})[ep.value] = ep  # type: ignore[attr-defined]
    out: dict[str, object] = {}
    for name, values in by_name.items():
        if len(values) > 1:
            raise FormatError(
                f"duplicate check name {name!r} provided by: " + ", ".join(sorted(values))
            )
        out[name] = next(iter(values.values()))
    return out


def _rank(name: str) -> tuple[int, str]:
    known = CANONICAL_ORDER.index(name) if name in CANONICAL_ORDER else len(CANONICAL_ORDER)
    return (known, name)


def check_names() -> list[str]:
    """Registered check names in run order (does not import any check).

    Raises:
        FormatError: if two packages register the same name.
    """
    return sorted(_candidates(), key=_rank)


def get_check(name: str) -> Check:
    """Resolve a check by name.

    Raises:
        FormatError: unknown or duplicate name, missing dependency, or an invalid plugin.
    """
    available = _candidates()
    if name not in available:
        known = ", ".join(sorted(available, key=_rank)) or "none"
        raise FormatError(f"unknown check {name!r}; available checks: {known}")
    try:
        plugin = available[name].load()  # type: ignore[attr-defined]
    except ImportError as exc:
        raise FormatError(
            f"check {name!r} needs an optional dependency that is not installed: "
            f"{exc.name or 'a dependency'}"
        ) from None
    spec = getattr(plugin, "spec", None)
    if not isinstance(spec, PluginSpec) or spec.kind != "check":
        raise FormatError(f"check {name!r} does not expose a valid PluginSpec")
    if spec.name != name:
        raise FormatError(f"check {name!r} declares a different spec name {spec.name!r}")
    if not callable(getattr(plugin, "run", None)):
        raise FormatError(f"check {name!r} has no run method")
    return plugin  # type: ignore[no-any-return]
