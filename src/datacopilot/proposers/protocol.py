"""The contract between ``propose`` and a proposer backend.

A proposer never touches the file system: the caller resolves each image path inside the dataset
root, opens it without following symlinks, bounds its size and hands over the encoded bytes. What a
proposer returns is **untrusted**: ``proposals/run.py`` validates every candidate (finite numbers,
inside the image, a label that was asked for) before anything is stored.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from datacopilot.core.schema import PluginSpec

MAX_PHRASES = 512
"""Most phrases one ``detect`` call may ask about (bounds the number of model passes per image)."""
MAX_PHRASE_CHARS = 64
PHRASE = re.compile(r"^[a-z0-9][a-z0-9 '&/-]{0,63}$")
"""A phrase a text-prompted detector is asked about: lower-case ASCII words, no sentence
punctuation. The full stop and the line break are what a prompt is built from, so a class name
that contained one could split or rewrite the prompt."""


def normalize_phrase(text: str) -> str | None:
    """``text`` as a prompt phrase (NFKC, lower case, single spaces), or None when it cannot be
    one (empty, too long, non-ASCII or containing characters outside :data:`PHRASE`)."""
    phrase = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return phrase if PHRASE.fullmatch(phrase) and not phrase.endswith(" ") else None


@dataclass(frozen=True)
class Candidate:
    """One box a model claims to see: pixels of the (EXIF-oriented) image, the phrase it matched,
    and its confidence. Nothing is guaranteed about any of the three until it is validated."""

    box: tuple[float, float, float, float]
    label: str
    score: float


@runtime_checkable
class Proposer(Protocol):
    """Finds the things named by ``phrases`` in one image.

    ``detect`` returns candidates scoring at least ``min_score``, in any order. ``size`` is the
    ``(width, height)`` the dataset records for the image. An undecodable image raises
    :class:`~datacopilot.embed.protocol.ImageDecodeError` (the caller skips that image and reports
    the run as incomplete); any other exception is a model failure and ends the run.
    """

    model_id: str
    revision: str
    weights: dict[str, str]

    def detect(
        self, image: bytes, phrases: Sequence[str], size: tuple[int, int], *, min_score: float
    ) -> list[Candidate]: ...


@runtime_checkable
class ProposerPlugin(Protocol):
    """What an entry point of ``datacopilot.proposers`` exposes.

    ``create`` is the only place weights may be downloaded or loaded; the registry calls it after
    the license guard has passed.
    """

    spec: PluginSpec

    def create(self, allowed_licenses: frozenset[str]) -> Proposer: ...
