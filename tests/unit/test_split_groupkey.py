"""Group keys for `split --by-group`: a field, a source field or a safe file-name pattern."""

from __future__ import annotations

import time

import pytest

from datacopilot.core.errors import ValidationError
from datacopilot.core.schema import Image
from datacopilot.splits import groupkey


def image(path: str, *, group: str | None = None, **source: object) -> Image:
    return Image(
        image_id="0123456789abcdef",
        file_path=path,
        width=4,
        height=4,
        group_id=group,
        source=source,  # type: ignore[arg-type]
    )


def test_the_default_key_is_the_group_id_field_normalised_like_the_leakage_check() -> None:
    key = groupkey.parse("group_id")
    assert key.kind == "group_id"
    assert groupkey.key_of(image("a.png", group=" Cam1 "), key) == "cam1"
    assert groupkey.key_of(image("a.png", group="Ｃam1"), key) == "cam1"  # fullwidth, NFKC
    assert groupkey.key_of(image("a.png"), key) is None
    assert groupkey.key_of(image("a.png", group="  "), key) is None


def test_a_source_field_is_a_key_when_it_is_a_string_or_an_integer() -> None:
    key = groupkey.parse("source.video_id")
    assert groupkey.key_of(image("a.png", video_id="V7"), key) == "v7"
    assert groupkey.key_of(image("a.png", video_id=12), key) == "12"
    for value in (True, 1.5, None, [1], {"a": 1}):
        assert groupkey.key_of(image("a.png", video_id=value), key) is None
    assert groupkey.key_of(image("a.png"), key) is None


def test_a_pattern_is_matched_against_the_file_name_only() -> None:
    key = groupkey.parse(r"re:^(?P<g>[^_]+)_")
    assert groupkey.key_of(image("data/train_x/cam2_0001.jpg"), key) == "cam2"
    assert groupkey.key_of(image("C:\\data\\cam3_9.jpg"), key) == "cam3"
    assert groupkey.key_of(image("nounderscore.jpg"), key) is None
    first = groupkey.parse(r"re:^([a-z]+)-[0-9]+")
    assert groupkey.key_of(image("shelf-001.png"), first) == "shelf"
    whole = groupkey.parse(r"re:^[a-z]+")
    assert groupkey.key_of(image("abc123.png"), whole) == "abc"


def test_a_file_name_longer_than_any_file_system_allows_is_refused() -> None:
    key = groupkey.parse(r"re:^(?P<g>[^_]+)_")
    with pytest.raises(ValidationError, match="255"):
        groupkey.key_of(image("a" * 300 + "_1.png"), key)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "group",
        "source.",
        "source.a/b",
        "source." + "x" * 65,
        "re:",
        "re:(",
        "re:" + "a" * 201,
    ],
)
def test_malformed_keys_are_refused(text: str) -> None:
    with pytest.raises(ValueError):
        groupkey.parse(text)


@pytest.mark.parametrize(
    "pattern",
    [
        r"(a+)+$",  # nested quantifier
        r"(a|aa)*",  # quantified alternation
        r"(?:ab)+",  # quantified group
        r"a*a*b",  # adjacent repeats that overlap
        r"\w+_\d+",  # \w can consume the underscore: no fence between the repeats
        r".*-.*",  # . consumes the separator
        r"a?a?a?a?aaaa",  # optional atoms that overlap
        r"(x+|y)z",  # a repeat inside an alternation
        r"(a)\1",  # back reference
        r"(?=a)a",  # look-around
        r"(?i)[a-z]+_",  # flags change what a class matches
        r"(?i:a)+",
    ],
)
def test_patterns_that_can_backtrack_badly_are_refused(pattern: str) -> None:
    with pytest.raises(ValueError, match="pattern"):
        groupkey.parse("re:" + pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        r"^(?P<g>[^_]+)_",
        r"^([a-z0-9]+)_\d+\.jpg$",
        r"^(?P<g>\d+)[a-z]+",
        r"-?\d+",
        r"^(cam|camera)[0-9]{2}",
        r"^(?P<g>[^/]+?)-",
        r"^[^_]*_[^_]*_",
    ],
)
def test_fenced_patterns_are_accepted(pattern: str) -> None:
    assert groupkey.parse("re:" + pattern).kind == "regex"


def test_an_accepted_pattern_stays_fast_on_hostile_names() -> None:
    hostile = ["a" * 255, "_" * 255, "a_" * 127, "1" * 254 + "!", ("ab" * 127)[:255]]
    for pattern in (r"^[^_]*_[^_]*_[^_]*_x", r"(?P<g>[a-z]+)[0-9]+[a-z]+!", r"[^-]+-[^-]+-\d+"):
        key = groupkey.parse("re:" + pattern)
        started = time.perf_counter()
        for name in hostile:
            groupkey.key_of(image(name), key)
        assert time.perf_counter() - started < 0.5, pattern
