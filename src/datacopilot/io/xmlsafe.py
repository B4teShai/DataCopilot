"""Hardened XML parsing for untrusted dataset files.

Pascal VOC annotations never need a document type declaration, so any ``<!DOCTYPE`` (and with it
every internal or external entity declaration) is rejected outright. Nothing is ever expanded or
fetched: there is no external-entity resolution, no DTD loading, and no network access. Standard
character references and the five predefined entities (``&amp;`` and friends) still work.
Nesting is limited to :data:`MAX_DEPTH` levels while parsing, so a hostile file cannot build a
structure that later recursive code would overflow on.

The tree is built from expat events directly because the C-accelerated ``ElementTree`` parser
does not expose its expat handlers.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # only the TreeBuilder is used, not its parser
from xml.parsers import expat  # noqa: S407  # handlers below reject DOCTYPE and entities

from datacopilot.core.errors import FormatError

MAX_DEPTH = 32
"""Deepest element nesting accepted; VOC files are about four levels deep."""


class _Forbidden(Exception):
    pass


class _TooDeep(Exception):
    pass


def _reject(*_args: object) -> None:
    raise _Forbidden


def _reject_ref(*_args: object) -> int:
    raise _Forbidden


def parse_xml(data: bytes, name: str) -> ET.Element:
    """Parse ``data`` and return the root element.

    Raises:
        FormatError: for malformed XML, a DOCTYPE, any entity declaration, or excess nesting.
    """
    builder = ET.TreeBuilder()
    depth = 0

    def start(tag: str, attrs: dict[str, str]) -> None:
        nonlocal depth
        depth += 1
        if depth > MAX_DEPTH:
            raise _TooDeep
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
            f"{name}: elements are nested more than {MAX_DEPTH} levels deep"
        ) from None
    except (expat.ExpatError, ValueError, AssertionError):
        raise FormatError(f"{name} is not well-formed XML") from None
