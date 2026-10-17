"""``test_boxes``: a dependency-free test proposer (boxes derived from a hash of the image bytes).

It exists so the whole flow (propose, review, apply, the command line) can be tested without model
weights, a network or Pillow. It is *not* a detector: for each of the first ``MAX_PHRASES`` phrases
it decides from ``sha256(image bytes + phrase)`` whether to report a box, where, and how confident.
Everything is integer arithmetic on digest bytes, so the result is identical on every platform.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from datacopilot.core.schema import PluginSpec
from datacopilot.models.registry import require_license
from datacopilot.proposers.protocol import Candidate, Proposer

MAX_PHRASES = 16
"""Phrases considered per image (bounds the work, whatever the caller passes)."""


class TestBoxesProposer:
    __test__ = False  # not a pytest class, whatever the name suggests
    model_id = "test_boxes"
    revision = "1"
    weights: dict[str, str] = {}  # noqa: RUF012  # nothing to download, nothing to verify
    test_only = True
    """It does not look at pictures: a person must not mistake its boxes for detections."""

    def detect(
        self, image: bytes, phrases: Sequence[str], size: tuple[int, int], *, min_score: float
    ) -> list[Candidate]:
        width, height = size
        out: list[Candidate] = []
        for phrase in phrases[:MAX_PHRASES]:
            h = hashlib.sha256(image + b"\x00" + phrase.encode("utf-8")).digest()
            score = 0.5 + h[4] / 256 * 0.49
            if h[0] >= 128 or score < min_score:
                continue
            x1 = h[1] / 256 * width * 0.5
            y1 = h[2] / 256 * height * 0.5
            x2 = min(x1 + (0.2 + h[3] / 256 * 0.3) * width, float(width))
            y2 = min(y1 + (0.2 + h[5] / 256 * 0.3) * height, float(height))
            if x2 > x1 and y2 > y1:
                out.append(Candidate((x1, y1, x2, y2), phrase, score))
        return out


class TestBoxesPlugin:
    __test__ = False
    spec = PluginSpec(
        name="test_boxes",
        kind="proposer",
        license="Apache-2.0",
        extra=None,
        gpu_memory_gb=None,
        version="1",
    )

    def create(self, allowed_licenses: frozenset[str]) -> Proposer:
        require_license("proposer 'test_boxes'", self.spec.license, allowed_licenses)
        return TestBoxesProposer()


plugin = TestBoxesPlugin()
