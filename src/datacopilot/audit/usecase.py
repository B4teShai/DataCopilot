"""Inputs of an audit that the command line and the Python API validate the same way."""

from __future__ import annotations

import math
import re

from datacopilot.core.schema import TEST_ONLY_SUFFIX
from datacopilot.embed import registry as embed_registry
from datacopilot.embed.protocol import Embedder

EMBEDDER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def resolve_embedder(name: str, allowed_licenses: frozenset[str]) -> Embedder:
    """Create the named embedder after the license guard; no weights are touched here.

    Raises:
        ValueError: a malformed or unknown name (a usage error).
        LicenseError, FormatError: the license is not allowed, or the plugin is unusable.
    """
    if not EMBEDDER_NAME.fullmatch(name):
        raise ValueError("embedder names use lower-case letters, digits and underscores")
    available = embed_registry.embedder_names()
    if name not in available:
        raise ValueError(f"unknown embedder; available: {', '.join(available) or 'none'}")
    return embed_registry.get_embedder(name, allowed_licenses)


def embedder_label(embedder: Embedder) -> str:
    """The identity recorded in ``AuditRun.embedder`` (``-test-only`` for a test double)."""
    label = f"{embedder.model_id}@{embedder.revision[:12]}"
    return label + (TEST_ONLY_SUFFIX if getattr(embedder, "test_only", False) else "")


def check_threshold(value: float | None) -> None:
    """A cosine threshold is above 0 and at most 1.

    Raises:
        ValueError: otherwise (NaN and infinities included).
    """
    if value is not None and not (math.isfinite(value) and 0.0 < value <= 1.0):
        raise ValueError("the similarity threshold must be above 0 and at most 1")
