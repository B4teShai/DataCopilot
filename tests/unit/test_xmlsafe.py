import time

import pytest

from datacopilot.core.errors import FormatError
from datacopilot.io.xmlsafe import MAX_DEPTH, parse_xml


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
