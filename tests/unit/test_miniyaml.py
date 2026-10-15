import time

import pytest

from datacopilot.core.errors import FormatError
from datacopilot.io.miniyaml import parse_mapping


def test_scalars_and_comments() -> None:
    doc = parse_mapping(
        "# c\npath: .\nnc: 3  # trailing\nflag: true\nx: 1.5\nq: 'a b'\nd: \"c\\td\"\n"
    )
    assert doc == {"path": ".", "nc": 3, "flag": True, "x": 1.5, "q": "a b", "d": "c\td"}


def test_flow_list_with_quotes_and_commas() -> None:
    assert parse_mapping("names: [\"a, b\", c, 'd']\n") == {"names": ["a, b", "c", "d"]}
    assert parse_mapping("names: []\n") == {"names": []}


def test_block_list_and_block_mapping() -> None:
    text = "names:\n  - cat\n  - dog\nother:\n  0: cat\n  1: dog\n"
    assert parse_mapping(text) == {"names": ["cat", "dog"], "other": {0: "cat", 1: "dog"}}


def test_empty_value_is_none() -> None:
    assert parse_mapping("a:\nb: 1\n") == {"a": None, "b": 1}


@pytest.mark.parametrize(
    "text",
    [
        "a: &anchor 1\nb: *anchor\n",
        "a: !!python/object:os.system x\n",
        "a: |\n  multi\n",
        "- not a mapping\n",
        "a: [1, [2]]\n",
        "a: {x: 1}\n",
        "just text\n",
        "a: 1\na: 2\n",
        "a:\n  - x\n  k: v\n",
        'a: "unterminated\n',
    ],
)
def test_unsupported_constructs_are_rejected(text: str) -> None:
    with pytest.raises(FormatError, match="data.yaml"):
        parse_mapping(text)


def test_size_and_depth_are_bounded() -> None:
    with pytest.raises(FormatError, match="data.yaml"):
        parse_mapping("a: 1\n" * 100_000)


def _fast(text: str) -> None:
    """Parse (accepting either a result or a clean rejection) and require it to be quick."""
    start = time.perf_counter()
    try:
        parse_mapping(text)
    except FormatError:
        pass
    assert time.perf_counter() - start < 0.5


def test_line_of_spaces_then_text_without_colon_is_fast() -> None:
    _fast("a" + " " * 4000 + "a\n")


def test_long_line_without_colon_is_fast() -> None:
    _fast("a" * 4000 + "\n")


def test_many_short_lines_are_fast() -> None:
    _fast("k: v\n" * 9_999)
    _fast("".join(f"k{i}: v\n" for i in range(9_999)))


def test_deep_indentation_and_tabs_are_fast_and_rejected_or_parsed() -> None:
    _fast("a:\n" + " " * 4000 + "- x\n")
    _fast("a:\n\t\t\t\t- x\n\t\t\t\t- y\n")
    _fast("\t" * 4000 + "a: 1\n")


def test_over_long_line_is_rejected_by_the_cap() -> None:
    with pytest.raises(FormatError, match=r"data.yaml.*line.*4096"):
        parse_mapping("a: " + "x" * 5000 + "\n")


def test_over_large_input_is_rejected_by_the_cap() -> None:
    with pytest.raises(FormatError, match=r"data.yaml.*larger"):
        parse_mapping(("k: v\n" * 10) + "#" * (1024 * 1024))


def test_key_with_colon_not_followed_by_space_is_part_of_the_key() -> None:
    assert parse_mapping("a:b: 1\n") == {"a:b": 1}


@pytest.mark.parametrize("text", ["a#b: 1\n", "[a: 1\n", "a:\n  b#c: 1\n"])
def test_bad_key_characters_are_rejected(text: str) -> None:
    with pytest.raises(FormatError, match="data.yaml"):
        parse_mapping(text)
