"""Proposer registry: name -> plugin, resolved lazily through entry points (ADR-0005).

Proposers register under ``datacopilot.proposers``. The license guard runs before any weights are
touched: first-party models are checked against their registry card *before their module is even
imported*; any plugin is checked against ``spec.license`` before ``create`` is called.
"""

from __future__ import annotations

import re
from importlib.metadata import entry_points

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import PluginSpec
from datacopilot.models import registry as models
from datacopilot.proposers.protocol import Proposer, ProposerPlugin

GROUP = "datacopilot.proposers"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _entry_points() -> list[object]:
    """Seam for tests: the raw entry points of the proposers group."""
    return list(entry_points(group=GROUP))


def _candidates() -> dict[str, object]:
    by_name: dict[str, dict[str, object]] = {}
    for ep in _entry_points():
        by_name.setdefault(ep.name, {})[ep.value] = ep  # type: ignore[attr-defined]
    out: dict[str, object] = {}
    for name, values in by_name.items():
        if len(values) > 1:
            raise FormatError(
                f"duplicate proposer name {name!r} provided by: " + ", ".join(sorted(values))
            )
        out[name] = next(iter(values.values()))
    return out


def proposer_names() -> list[str]:
    """Registered proposer names, sorted (does not import any of them).

    Raises:
        FormatError: if two packages register the same name.
    """
    return sorted(_candidates())


def _plugin(name: str, allowed_licenses: frozenset[str]) -> ProposerPlugin:
    if not _NAME.fullmatch(name):
        raise FormatError("invalid proposer name; use lower-case letters, digits and underscores")
    available = _candidates()
    if name not in available:
        known = ", ".join(sorted(available)) or "none"
        raise FormatError(f"unknown proposer {name!r}; available proposers: {known}")
    if models.has_model(name):  # decided from the reviewed card, before any import
        models.get_model(name, allowed_licenses)
    try:
        plugin = available[name].load()  # type: ignore[attr-defined]
    except ImportError as exc:
        raise FormatError(
            f"proposer {name!r} needs an optional dependency that is not installed "
            f'({exc.name or "a dependency"}): install the extra "datacopilot[vlm]" '
            "(in a clone: uv sync --extra vlm)"
        ) from None
    spec = getattr(plugin, "spec", None)
    if not isinstance(spec, PluginSpec) or spec.kind != "proposer":
        raise FormatError(f"proposer {name!r} does not expose a valid PluginSpec")
    if spec.name != name:
        raise FormatError(f"proposer {name!r} declares a different spec name {spec.name!r}")
    models.require_license(f"proposer {name!r}", spec.license, allowed_licenses)
    if not isinstance(plugin, ProposerPlugin):
        raise FormatError(f"proposer {name!r} has no create method")
    return plugin


def check_proposer(name: str, allowed_licenses: frozenset[str]) -> ProposerPlugin:
    """The plugin after the license guard, without creating the proposer (``.spec`` describes it).

    ``apply`` calls this: it needs to know the proposer is installed and allowed, not to run it.

    Raises:
        FormatError: invalid or unknown name, duplicate registration, missing optional
            dependency, or a plugin that does not satisfy the contract.
        LicenseError: the model or plugin license is not in ``allowed_licenses``.
    """
    return _plugin(name, allowed_licenses)


def get_proposer(name: str, allowed_licenses: frozenset[str]) -> Proposer:
    """Resolve ``name``, enforce the license allow-list, and create the proposer.

    Weights are not downloaded here: ``create`` returns an object that loads them on first use.

    Raises:
        FormatError, LicenseError: as :func:`check_proposer`, and a ``create`` that returns
            something that cannot ``detect``.
    """
    created = _plugin(name, allowed_licenses).create(allowed_licenses)
    if not isinstance(created, Proposer):
        raise FormatError(f"proposer {name!r} did not create a valid proposer")
    return created
