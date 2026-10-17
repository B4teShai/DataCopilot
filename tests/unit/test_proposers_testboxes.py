"""``test_boxes``: a dependency-free proposer for tests; its boxes mean nothing about pictures."""

from __future__ import annotations

import math

import pytest

from datacopilot.core.errors import LicenseError
from datacopilot.proposers import registry
from datacopilot.proposers.protocol import Proposer, ProposerPlugin
from datacopilot.proposers.testproposer import MAX_PHRASES, plugin

ALL = frozenset({"apache-2.0"})
PHRASES = [f"thing {i}" for i in range(40)]


def make() -> Proposer:
    return plugin.create(ALL)


def test_it_is_a_registered_license_checked_test_double() -> None:
    assert isinstance(plugin, ProposerPlugin) and plugin.spec.kind == "proposer"
    proposer = registry.get_proposer("test_boxes", ALL)
    assert proposer.model_id == "test_boxes" and proposer.weights == {}
    assert getattr(proposer, "test_only", False) is True
    with pytest.raises(LicenseError):
        plugin.create(frozenset({"mit"}))


def test_it_is_deterministic_and_depends_on_the_bytes() -> None:
    p = make()
    a = p.detect(b"image one", PHRASES, (200, 100), min_score=0.1)
    assert a == p.detect(b"image one", PHRASES, (200, 100), min_score=0.1)
    assert a != p.detect(b"image two", PHRASES, (200, 100), min_score=0.1)
    assert a  # something is produced for 40 phrases


@pytest.mark.parametrize("size", [(10, 10), (200, 100), (4000, 3)])
def test_every_box_is_inside_the_image_with_positive_extent(size: tuple[int, int]) -> None:
    out = make().detect(b"x" * 50, PHRASES, size, min_score=0.1)
    assert out
    for c in out:
        x1, y1, x2, y2 = c.box
        assert 0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]
        assert all(math.isfinite(v) for v in c.box) and 0.1 <= c.score <= 1
        assert c.label in PHRASES


def test_min_score_filters_and_the_phrase_count_is_bounded() -> None:
    p = make()
    everything = p.detect(b"abc", PHRASES, (100, 100), min_score=0.01)
    strict = p.detect(b"abc", PHRASES, (100, 100), min_score=0.9)
    assert len(strict) < len(everything) and all(c.score >= 0.9 for c in strict)
    many = p.detect(b"abc", [f"p{i}" for i in range(MAX_PHRASES * 3)], (100, 100), min_score=0.01)
    assert len(many) <= MAX_PHRASES
