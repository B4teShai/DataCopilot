import time

import pytest

from datacopilot.core.errors import FormatError
from datacopilot.io.xmlsafe import MAX_ATTRIBUTES, MAX_DEPTH, MAX_ELEMENTS, parse_xml


def test_deeply_nested_xml_is_rejected_quickly() -> None:
    doc = b"<a>" * 5000 + b"</a>" * 5000
    start = time.perf_counter()
    with pytest.raises(FormatError, match=r"x.xml.*nested"):
        parse_xml(doc, "x.xml")
    assert time.perf_counter() - start < 0.5


def test_nesting_at_the_limit_is_accepted() -> None:
    doc = b"<a>" * MAX_DEPTH + b"</a>" * MAX_DEPTH
    assert parse_xml(doc, "x.xml").tag == "a"


def test_malformed_and_empty_documents_are_format_errors() -> None:
    for bad in (b"", b"<a>", b"not xml", b"<a></b>"):
        with pytest.raises(FormatError, match="x.xml"):
            parse_xml(bad, "x.xml")


def test_element_count_is_bounded() -> None:
    doc = b"<a>" + b"<b/>" * 11 + b"</a>"
    with pytest.raises(FormatError, match=r"x.xml.*more than 10 elements"):
        parse_xml(doc, "x.xml", max_elements=10)
    assert parse_xml(b"<a>" + b"<b/>" * 9 + b"</a>", "x.xml", max_elements=10).tag == "a"


def test_default_element_cap_stops_a_wide_document() -> None:
    doc = b"<a>" + b"<b/>" * (MAX_ELEMENTS + 1) + b"</a>"
    with pytest.raises(FormatError, match="elements"):
        parse_xml(doc, "x.xml")


def test_depth_limit_can_be_tightened_per_call() -> None:
    with pytest.raises(FormatError, match="nested more than 3 levels"):
        parse_xml(b"<a><b><c><d/></c></b></a>", "x.xml", max_depth=3)


def test_attribute_count_per_element_is_bounded() -> None:
    attrs = " ".join(f"k{i}='v'" for i in range(MAX_ATTRIBUTES + 1))
    with pytest.raises(FormatError, match="attributes"):
        parse_xml(f"<a {attrs}/>".encode(), "x.xml")


def test_entity_bomb_and_external_entity_are_rejected_without_expansion() -> None:
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaaaaaaaa">'
        b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><l>&b;&b;&b;</l>'
    )
    xxe = b'<!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]><d>&x;</d>'
    for doc in (bomb, xxe):
        start = time.perf_counter()
        with pytest.raises(FormatError, match="DOCTYPE"):
            parse_xml(doc, "x.xml")
        assert time.perf_counter() - start < 0.5


# --------------------------------------------- one huge start tag must not be built in memory
def test_a_64_mib_start_tag_of_short_attributes_is_refused_within_a_small_memory_bound() -> None:
    import tracemalloc

    from datacopilot.io.xmlsafe import MAX_PENDING_BYTES

    doc = b"<a " + b'a="" ' * (64 * 1024 * 1024 // 5) + b"/>"
    assert len(doc) >= 64 * 1024 * 1024
    tracemalloc.start()
    started = time.perf_counter()
    try:
        with pytest.raises(FormatError, match="single tag|attributes"):
            parse_xml(doc, "x.xml")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # documented bound: the pending construct (MAX_PENDING_BYTES plus one chunk) times a small
    # per-attribute factor, nowhere near the size of the tag
    assert peak < 48 * 1024 * 1024, peak
    assert MAX_PENDING_BYTES <= 1024 * 1024
    assert time.perf_counter() - started < 5.0


def test_a_large_but_legitimate_document_still_parses() -> None:
    body = b"".join(b'<b id="%d" name="n%d">text</b>' % (n, n) for n in range(60_000))
    assert len(body) > 1_000_000
    root = parse_xml(b"<a>" + body + b"</a>", "x.xml")
    assert len(root) == 60_000


def test_a_long_text_node_is_not_mistaken_for_a_tag_but_a_huge_comment_is_refused() -> None:
    text = b"x" * (3 * 1024 * 1024)
    assert parse_xml(b"<a>" + text + b"</a>", "x.xml").tag == "a"
    assert parse_xml(b"<a><!--" + b"c" * 10_000 + b"--><b/></a>", "x.xml")[0].tag == "b"
    with pytest.raises(FormatError, match="single tag or comment"):
        parse_xml(b"<a><!--" + text + b"--><b/></a>", "x.xml")


def test_a_tag_with_many_attributes_below_the_pending_cap_is_still_counted() -> None:
    attrs = " ".join(f"k{i}='v'" for i in range(MAX_ATTRIBUTES))
    assert len(parse_xml(f"<a {attrs}/>".encode(), "x.xml").attrib) == MAX_ATTRIBUTES


def test_attributes_keep_their_order_and_values_with_ordered_parsing() -> None:
    root = parse_xml(b'<a z="1" b="2&amp;3" c=""/>', "x.xml")
    assert list(root.attrib.items()) == [("z", "1"), ("b", "2&3"), ("c", "")]
