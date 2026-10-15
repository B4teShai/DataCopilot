"""Hardened XML parsing for untrusted dataset files.

Pascal VOC annotations never need a document type declaration, so any ``<!DOCTYPE`` (and with it
every internal or external entity declaration) is rejected outright. Nothing is ever expanded or
fetched: there is no external-entity resolution, no DTD loading, and no network access. Standard
character references and the five predefined entities (``&amp;`` and friends) still work.
Nesting is limited to :data:`MAX_DEPTH` levels, the element count to :data:`MAX_ELEMENTS` and the
attributes of one element to :data:`MAX_ATTRIBUTES` while parsing, so a hostile file cannot build a
structure that later recursive code would overflow on or that exhausts memory. Callers read the
bytes through :func:`datacopilot.io.files.read_bounded`, which bounds the file size first.

The tree is built from expat events directly because the C-accelerated ``ElementTree`` parser
does not expose its expat handlers.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # only the TreeBuilder is used, not its parser
from xml.parsers import expat  # noqa: S407  # handlers below reject DOCTYPE and entities

from datacopilot.core.errors import FormatError

MAX_DEPTH = 32
"""Deepest element nesting accepted; VOC files are about four levels deep."""

MAX_ELEMENTS = 1_000_000
"""Most elements accepted in one document (a tree node costs a few hundred bytes)."""

MAX_ATTRIBUTES = 256
"""Most attributes accepted on one element."""


class _Forbidden(Exception):
    pass


class _TooDeep(Exception):
    pass


class _TooMany(Exception):
    pass


def _reject(*_args: object) -> None:
    raise _Forbidden


def _reject_ref(*_args: object) -> int:
    raise _Forbidden


def parse_xml(
    data: bytes,
    name: str,
    *,
    max_depth: int = MAX_DEPTH,
    max_elements: int = MAX_ELEMENTS,
) -> ET.Element:
    """Parse ``data`` and return the root element.

    Raises:
        FormatError: for malformed XML, a DOCTYPE, any entity declaration, excess nesting, more
            than ``max_elements`` elements or more than :data:`MAX_ATTRIBUTES` attributes on one.
    """
    builder = ET.TreeBuilder()
    depth = 0
    elements = 0

    def start(tag: str, attrs: dict[str, str]) -> None:
        nonlocal depth, elements
        depth += 1
        elements += 1
        if depth > max_depth:
            raise _TooDeep
        if elements > max_elements or len(attrs) > MAX_ATTRIBUTES:
            raise _TooMany
        builder.start(tag, attrs)

    def end(tag: str) -> None:
        nonlocal depth
        depth -= 1
        builder.end(tag)

    parser = expat.ParserCreate()
    parser.buffer_text = True
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = builder.data
    parser.StartDoctypeDeclHandler = _reject
    parser.EntityDeclHandler = _reject
    parser.ExternalEntityRefHandler = _reject_ref
    try:
        parser.Parse(data, True)
        return builder.close()
    except _Forbidden:
        raise FormatError(
            f"{name}: DOCTYPE and entity declarations are not allowed in dataset XML"
        ) from None
    except _TooDeep:
        raise FormatError(
            f"{name}: elements are nested more than {max_depth} levels deep"
        ) from None
    except _TooMany:
        raise FormatError(
            f"{name}: more than {max_elements} elements or more than {MAX_ATTRIBUTES} "
            "attributes on one element"
        ) from None
    except (expat.ExpatError, ValueError, AssertionError):
        raise FormatError(f"{name} is not well-formed XML") from None
