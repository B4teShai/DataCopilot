import time

import pytest

from datacopilot.report import text


def test_markup_is_left_for_the_template_engine_to_escape() -> None:
    assert text.clean_text("<b>&\"'") == "<b>&\"'"


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("a\x00b", "a\\u0000b"),
        ("a\nb\tc\rd", "a\\u000ab\\u0009c\\u000dd"),
        ("\x1b[31mred", "\\u001b[31mred"),
        ("x‮y", "x\\u202ey"),  # right-to-left override
        ("x⁦y⁩", "x\\u2066y\\u2069"),
        ("zero​width", "zero\\u200bwidth"),
        ("﻿bom", "\\ufeffbom"),
        ("lone\ud800surrogate", "lone\\ud800surrogate"),
        ("\x85next line", "\\u0085next\\u2028line"),
        ("\x7fdel", "\\u007fdel"),
    ],
)
def test_control_bidi_and_surrogate_characters_become_visible_escapes(raw: str, shown: str) -> None:
    assert text.clean_text(raw) == shown
    text.clean_text(raw).encode("utf-8")  # always encodable


def test_ordinary_unicode_is_kept() -> None:
    assert text.clean_text("кошка 猫 🐈 café") == "кошка 猫 🐈 café"


def test_long_text_is_truncated_with_an_ellipsis_and_non_strings_are_converted() -> None:
    out = text.clean_text("x" * 1000, limit=10)
    assert out == "x" * 10 + "…"
    assert text.clean_text(None) == "None" and text.clean_text(3.5) == "3.5"


def test_output_length_is_bounded_even_when_every_character_expands() -> None:
    out = text.clean_text("\x00" * 100_000, limit=50)
    assert len(out) <= 50 * 6 + 1


def test_huge_input_is_cheap() -> None:
    start = time.perf_counter()
    text.clean_text("‮" * 5_000_000, limit=300)
    assert time.perf_counter() - start < 0.5


def test_describe_payload_is_sorted_skips_rule_and_name_and_is_bounded() -> None:
    payload = {"rule": "r", "file_name": "f", "z": 1, "a": [1, 2], "m": {"k": "<v>"}}
    assert text.describe_payload(payload) == 'a=[1,2]; m={"k":"<v>"}; z=1'
    assert len(text.describe_payload({"k": "v" * 10_000})) <= 301 * 6
