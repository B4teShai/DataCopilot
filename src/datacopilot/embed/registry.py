"""Embedder registry: name -> plugin, resolved lazily through entry points (ADR-0005).

Embedders register under ``datacopilot.embedders``. The license guard runs before any weights are
touched: first-party models are checked against their registry card *before their module is even
imported*; any plugin is checked against ``spec.license`` before ``create`` is called.
"""

from __future__ import annotations

import re
from importlib.metadata import entry_points

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import PluginSpec
from datacopilot.embed.protocol import Embedder, EmbedderPlugin
from datacopilot.models import registry as models

GROUP = "datacopilot.embedders"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _entry_points() -> list[object]:
    """Seam for tests: the raw entry points of the embedders group."""
    return list(entry_points(group=GROUP))


def _candidates() -> dict[str, object]:
    by_name: dict[str, dict[str, object]] = {}
    for ep in _entry_points():
        by_name.setdefault(ep.name, {})[ep.value] = ep  # type: ignore[attr-defined]
    out: dict[str, object] = {}
    for name, values in by_name.items():
        if len(values) > 1:
            raise FormatError(
                f"duplicate embedder name {name!r} provided by: " + ", ".join(sorted(values))
            )
        out[name] = next(iter(values.values()))
    return out


def embedder_names() -> list[str]:
    """Registered embedder names, sorted (does not import any of them).

    Raises:
        FormatError: if two packages register the same name.
    """
    return sorted(_candidates())


def get_embedder(name: str, allowed_licenses: frozenset[str]) -> Embedder:
    """Resolve ``name``, enforce the license allow-list, and create the embedder.

    Weights are not downloaded here: ``create`` returns an object that loads them on first use,
    so a caller that fails later never paid for a download it did not need.

    Raises:
        FormatError: invalid or unknown name, duplicate registration, missing optional
            dependency, or a plugin that does not satisfy the contract.
        LicenseError: the model or plugin license is not in ``allowed_licenses``.
    """
    if not _NAME.fullmatch(name):
        raise FormatError("invalid embedder name; use lower-case letters, digits and underscores")
    available = _candidates()
    if name not in available:
        known = ", ".join(sorted(available)) or "none"
        raise FormatError(f"unknown embedder {name!r}; available embedders: {known}")
    if models.has_model(name):  # decided from the reviewed card, before any import
        models.get_model(name, allowed_licenses)
    try:
        plugin = available[name].load()  # type: ignore[attr-defined]
    except ImportError as exc:
        raise FormatError(
            f"embedder {name!r} needs an optional dependency that is not installed "
            f'({exc.name or "a dependency"}): pip install "datacopilot[embed]"'
        ) from None
    spec = getattr(plugin, "spec", None)
    if not isinstance(spec, PluginSpec) or spec.kind != "embedder":
        raise FormatError(f"embedder {name!r} does not expose a valid PluginSpec")
    if spec.name != name:
        raise FormatError(f"embedder {name!r} declares a different spec name {spec.name!r}")
    models.require_license(f"embedder {name!r}", spec.license, allowed_licenses)
    if not isinstance(plugin, EmbedderPlugin):
        raise FormatError(f"embedder {name!r} has no create method")
    created = plugin.create(allowed_licenses)
    if not isinstance(created, Embedder):
        raise FormatError(f"embedder {name!r} did not create a valid embedder")
    return created
