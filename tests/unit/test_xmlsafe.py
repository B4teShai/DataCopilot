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
