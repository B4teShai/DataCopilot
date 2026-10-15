"""The contract between the audit and an embedding backend.

An embedder never touches the file system: the caller (``audit/embedding.py``) resolves each image
path inside the dataset root, opens it without following symlinks, bounds its size and hands over
the encoded bytes. That keeps path containment in one reviewed place and lets a plugin be a pure
function from bytes to vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from datacopilot.core.errors import ModelError
from datacopilot.core.schema import PluginSpec


class ImageDecodeError(ModelError):
    """One image could not be decoded (corrupt, unsupported, or over the pixel limit).

    Attributes:
        index: position of the (first) offending image in the batch given to ``embed``.
        reason: short constant code (``decode_failed``, ``too_many_pixels`` or ``multi_frame``),
            never file text.
        failures: ``(index, reason)`` of every offending image the embedder found, in order, so the
            caller drops them all and retries once instead of once per bad image.
    """

    def __init__(
        self, index: int, reason: str, failures: tuple[tuple[int, str], ...] | None = None
    ) -> None:
        super().__init__(f"image {index} cannot be decoded ({reason})")
        self.index = index
        self.reason = reason
        self.failures = failures if failures else ((index, reason),)


@runtime_checkable
class Embedder(Protocol):
    """Turns encoded images into vectors.

    ``embed`` returns a ``float32`` array of shape ``(len(images), dims)``. Rows need not be
    normalised; the pipeline validates finiteness and L2-normalises. An undecodable image raises
    :class:`ImageDecodeError` (the pipeline then isolates it); any other exception is an embedder
    failure and makes the audit fail closed.
    """

    model_id: str
    revision: str
    dims: int

    def embed(self, images: Sequence[bytes]) -> NDArray[np.float32]: ...


@runtime_checkable
class EmbedderPlugin(Protocol):
    """What an entry point of ``datacopilot.embedders`` exposes.

    ``create`` is the only place weights may be downloaded or loaded; the registry calls it after
    the license guard has passed.
    """

    spec: PluginSpec

    def create(self, allowed_licenses: frozenset[str]) -> Embedder: ...
