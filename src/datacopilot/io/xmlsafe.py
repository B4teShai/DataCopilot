"""Hardened XML parsing for untrusted dataset files.

Pascal VOC annotations never need a document type declaration, so any ``<!DOCTYPE`` (and with it
every internal or external entity declaration) is rejected outright. Nothing is ever expanded or
fetched: there is no external-entity resolution, no DTD loading, and no network access. Standard
character references and the five predefined entities (``&amp;`` and friends) still work.
Nesting is limited to :data:`MAX_DEPTH` levels, the element count to :data:`MAX_ELEMENTS` and the
attributes of one element to :data:`MAX_ATTRIBUTES` while parsing, so a hostile file cannot build a
structure that later recursive code would overflow on or that exhausts memory. Callers read the
bytes through :func:`datacopilot.io.files.read_bounded`, which bounds the file size first.

Memory bound for a single construct. expat holds an incomplete start tag (or comment) in its own
buffer and the Python wrapper builds the attribute list only when the tag is complete, so one
start tag made of millions of tiny attributes would cost about a hundred bytes per attribute before
any count could be checked. The document is therefore fed in :data:`CHUNK_BYTES` pieces and parsing
stops as soon as :data:`MAX_PENDING_BYTES` bytes were fed without a completed event (a tag, text, a
comment, a processing instruction or a CDATA section): the worst case is the pending bytes plus one
chunk, about 1.1 MiB, times that per-attribute factor (measured under 48 MiB for a 64 MiB tag by
``tests/unit/test_xmlsafe.py``). Text is delivered in pieces and is not affected; a comment or a tag
larger than the cap is not valid dataset XML and is refused.

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

CHUNK_BYTES = 64 * 1024
"""Size of the pieces the document is fed to the parser in."""

MAX_PENDING_BYTES = 1024 * 1024
"""Bytes fed without a completed event before the document is refused (see the module docstring)."""


class _Forbidden(Exception):
    pass


class _TooDeep(Exception):
    pass


class _TooMany(Exception):
    pass


class _TooBig(Exception):
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
    events = 0

    def start(tag: str, attrs: list[str]) -> None:
        nonlocal depth, elements, events
        depth += 1
        elements += 1
        events += 1
        if depth > max_depth:
            raise _TooDeep
        if elements > max_elements or len(attrs) // 2 > MAX_ATTRIBUTES:
            raise _TooMany
        builder.start(tag, dict(zip(attrs[::2], attrs[1::2], strict=True)))

    def end(tag: str) -> None:
        nonlocal depth, events
        depth -= 1
        events += 1
        builder.end(tag)

    def text(chunk: str) -> None:
        nonlocal events
        events += 1
        builder.data(chunk)

    def other(*_args: object) -> None:
        nonlocal events
        events += 1

    parser = expat.ParserCreate()
    parser.ordered_attributes = True  # a flat list: no dict is built before the count is checked
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = text
    parser.CommentHandler = other
    parser.ProcessingInstructionHandler = other
    parser.StartCdataSectionHandler = other
    parser.StartDoctypeDeclHandler = _reject
    parser.EntityDeclHandler = _reject
    parser.ExternalEntityRefHandler = _reject_ref
    try:
        pending = 0
        seen = 0
        for offset in range(0, len(data), CHUNK_BYTES):
            parser.Parse(data[offset : offset + CHUNK_BYTES], False)
            if events == seen:
                pending += CHUNK_BYTES
                if pending > MAX_PENDING_BYTES:
                    raise _TooBig
            else:
                pending, seen = 0, events
        parser.Parse(b"", True)
        return builder.close()
    except _Forbidden:
        raise FormatError(
            f"{name}: DOCTYPE and entity declarations are not allowed in dataset XML"
        ) from None
    except _TooDeep:
        raise FormatError(
            f"{name}: elements are nested more than {max_depth} levels deep"
        ) from None
    except _TooBig:
        raise FormatError(
            f"{name}: a single tag or comment is larger than {MAX_PENDING_BYTES} bytes"
        ) from None
    except _TooMany:
        raise FormatError(
            f"{name}: more than {max_elements} elements or more than {MAX_ATTRIBUTES} "
            "attributes on one element"
        ) from None
    except (expat.ExpatError, ValueError, AssertionError):
        raise FormatError(f"{name} is not well-formed XML") from None
