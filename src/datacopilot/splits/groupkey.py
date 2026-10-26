"""Group keys for ``split --by-group``: which images must stay together.

A key is one of:

- ``group_id``: the image's ``group_id`` (set by ``ingest --group-by`` or by the source format);
- ``source.<name>``: a string or integer field the reader kept in ``Image.source`` (for example a
  COCO ``video_id``);
- ``re:<pattern>``: a regular expression matched against the image's *file name* (the last path
  component, with any URL query or fragment removed first, so a presigned URL's signature never
  becomes a label); the named group ``g``, else group 1, else the whole match is the key.

Labels are compared with :func:`datacopilot.core.schema.group_key`, the leakage check's
normalisation, so a split by key leaves no ``group_overlap`` for that check to find. An image
without a key is a group of its own (the split use case also keeps an existing ``group_id``
together; see ``usecase.py``).

**The pattern is untrusted work on untrusted names.** It is accepted only when
:func:`datacopilot.core.patterns.check_pattern` finds its worst case bounded (no nested or
overlapping quantifiers, at most 200 characters). The name it is matched against is at most
``MAX_FILE_NAME_CHARS`` long, counted after the query is removed; a longer one is refused rather
than searched, so a search is at most quadratic in 255 characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from datacopilot.audit.paths import base_name
from datacopilot.core.errors import ValidationError
from datacopilot.core.patterns import check_pattern
from datacopilot.core.schema import Image, group_key

MAX_FILE_NAME_CHARS = 255
"""Longest file name a pattern is matched against."""
MAX_KEY_CHARS = 1024
"""Longest group label; a longer one is refused (a group label is a short session or camera id)."""
_FIELD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")

Kind = Literal["group_id", "source", "regex"]


@dataclass(frozen=True)
class GroupKey:
    """A parsed key: ``text`` is what the user gave (it is recorded in the split report)."""

    kind: Kind
    text: str
    field: str | None = None
    pattern: re.Pattern[str] | None = None


def parse(text: str) -> GroupKey:
    """Parse ``group_id``, ``source.<name>`` or ``re:<pattern>``.

    Raises:
        ValueError: any other text, a malformed field name or an unsafe pattern.
    """
    if text == "group_id":
        return GroupKey("group_id", text)
    if text.startswith("source."):
        field = text[len("source.") :]
        if not _FIELD.fullmatch(field):
            raise ValueError("--by-group source.<name>: name is 1 to 64 letters, digits, _ . -")
        return GroupKey("source", text, field=field)
    if text.startswith("re:"):
        return GroupKey("regex", text, pattern=check_pattern(text[3:], "--by-group"))
    raise ValueError("--by-group must be group_id, source.<name> or re:<pattern>")


def label(value: object) -> str | None:
    """A normalised group label from a field value (a string or an integer), or None.

    Raises:
        ValidationError: a label over ``MAX_KEY_CHARS`` characters.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value)
    if len(text) > MAX_KEY_CHARS:
        raise ValidationError(f"a group label is longer than {MAX_KEY_CHARS} characters")
    return group_key(text) or None


def key_of(image: Image, key: GroupKey) -> str | None:
    """The normalised group label of ``image``, or None when it has none.

    Raises:
        ValidationError: a label over ``MAX_KEY_CHARS`` or, for a pattern, a file name over
            ``MAX_FILE_NAME_CHARS`` characters.
    """
    if key.kind == "group_id":
        return label(image.group_id) if image.group_id is not None else None
    if key.kind == "source":
        return label(image.source.get(key.field or ""))
    assert key.pattern is not None  # noqa: S101  # set by parse for the regex kind
    name = base_name(image.file_path)
    if len(name) > MAX_FILE_NAME_CHARS:
        raise ValidationError(
            f"image {image.image_id}: file name longer than {MAX_FILE_NAME_CHARS} characters"
        )
    match = key.pattern.search(name)
    if match is None:
        return None
    pattern = key.pattern
    value = (
        match.group("g") if "g" in pattern.groupindex else match.group(1 if pattern.groups else 0)
    )
    return label(value) if value else None
