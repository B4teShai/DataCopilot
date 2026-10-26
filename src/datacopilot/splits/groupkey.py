"""Group keys for ``split --by-group``: which images must stay together.

A key is one of:

- ``group_id``: the image's ``group_id`` (set by ``ingest --group-by`` or by the source format);
- ``source.<name>``: a string or integer field the reader kept in ``Image.source`` (for example a
  COCO ``video_id``);
- ``re:<pattern>``: a regular expression matched against the image's *file name* (the last path
  component); the named group ``g``, else group 1, else the whole match is the key.

Labels are compared with :func:`datacopilot.core.schema.group_key`, the leakage check's
normalisation, so a split by key leaves no ``group_overlap`` for that check to find. An image
without a key is a group of its own.

**The pattern is untrusted work on untrusted names.** File names come from the dataset, and
Python's regular expressions backtrack, so a pattern is accepted only when its worst case is
bounded. It is parsed with the standard library's own parser and refused unless:

1. it has at most ``MAX_PATTERN_CHARS`` characters, no flags (a case-insensitive class matches
   more than it says), no back references, look-arounds or conditionals;
2. every quantifier applies to a single character matcher (a literal, ``.``, a class): no nested
   quantifiers, no quantified group or alternation, and no quantifier inside an alternation;
3. between one variable-length quantifier and the next there is a required character the first
   one cannot match (``[^_]+_[^_]+`` is accepted, ``\\w+_\\w+`` is not because ``\\w`` matches
   ``_``), so no two quantifiers can trade characters.

With such a pattern each quantifier's extent is fixed by the next fence, a match attempt is linear
in the name and a search is at most quadratic. Names are at most ``MAX_FILE_NAME_CHARS`` long (the
usual file-system limit); a longer one is refused rather than searched.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from datacopilot.core.errors import ValidationError
from datacopilot.core.schema import Image, group_key

MAX_PATTERN_CHARS = 200
"""Longest ``re:`` pattern."""
MAX_FILE_NAME_CHARS = 255
"""Longest file name a pattern is matched against."""
MAX_KEY_CHARS = 1024
"""Longest group label; a longer one is refused (a group label is a short session or camera id)."""
SMALL_CLASS = 256
"""A character class with at most this many characters can serve as a fence."""
_FIELD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
_ATOMS = frozenset({"LITERAL", "NOT_LITERAL", "ANY", "IN"})
_CATEGORY_PATTERNS = {
    "CATEGORY_DIGIT": r"\d",
    "CATEGORY_NOT_DIGIT": r"\D",
    "CATEGORY_SPACE": r"\s",
    "CATEGORY_NOT_SPACE": r"\S",
    "CATEGORY_WORD": r"\w",
    "CATEGORY_NOT_WORD": r"\W",
}

Kind = Literal["group_id", "source", "regex"]


@dataclass(frozen=True)
class GroupKey:
    """A parsed key: ``text`` is what the user gave (it is recorded in the split report)."""

    kind: Kind
    text: str
    field: str | None = None
    pattern: re.Pattern[str] | None = None


def _parser() -> Any:
    """The standard library's regular-expression parser (``re._parser``, ``sre_parse`` on 3.10)."""
    return importlib.import_module("re._parser" if sys.version_info >= (3, 11) else "sre_parse")


def _name(op: Any) -> str:
    return str(getattr(op, "name", op))


def _refuse(why: str) -> ValueError:
    return ValueError(f"unsafe --by-group pattern: {why}")


@dataclass(frozen=True)
class _Element:
    """One position of the flattened pattern: an atom and its repeat bounds, or a branch."""

    atom: tuple[str, Any] | None  # None: an alternation (never a fence)
    low: int
    high: int

    @property
    def variable(self) -> bool:
        return self.atom is not None and self.high != self.low


def _flatten(items: Any, out: list[_Element], *, in_branch: bool) -> None:
    for op, av in items:
        name = _name(op)
        if name in _ATOMS:
            out.append(_Element((name, av), 1, 1))
        elif name == "AT":
            continue  # anchors match no character
        elif name == "SUBPATTERN":
            _group, add_flags, del_flags, body = av
            if add_flags or del_flags:
                raise _refuse("flags are not supported")
            _flatten(body, out, in_branch=in_branch)
        elif name in ("MAX_REPEAT", "MIN_REPEAT"):
            low, high, body = av
            body = list(body)
            if len(body) != 1 or _name(body[0][0]) not in _ATOMS:
                raise _refuse("a quantifier may only follow a single character or class")
            if in_branch and high != low:
                raise _refuse("no quantifier inside an alternation")
            out.append(_Element((_name(body[0][0]), body[0][1]), int(low), int(high)))
        elif name == "BRANCH":
            for alternative in av[1]:
                _flatten(alternative, [], in_branch=True)
            out.append(_Element(None, 0, 0))
        else:
            raise _refuse(f"{name.lower()} is not supported")


def _category(name: str, char: str) -> bool:
    pattern = _CATEGORY_PATTERNS.get(name)
    return True if pattern is None else re.fullmatch(pattern, char) is not None


def _matches(atom: tuple[str, Any], char: str) -> bool:
    """Whether ``atom`` can match ``char`` (``True`` when unsure, which only refuses more)."""
    kind, av = atom
    if kind == "LITERAL":
        return bool(char == chr(av))
    if kind == "NOT_LITERAL":
        return bool(char != chr(av))
    if kind == "ANY":
        return True
    negate = False
    hit = False
    for op, value in av:
        name = _name(op)
        if name == "NEGATE":
            negate = True
        elif name == "LITERAL":
            hit = hit or char == chr(value)
        elif name == "RANGE":
            hit = hit or value[0] <= ord(char) <= value[1]
        elif name == "CATEGORY":
            hit = hit or _category(_name(value), char)
        else:
            return True
    return hit != negate


def _chars(atom: tuple[str, Any]) -> set[str] | None:
    """The characters a fence candidate matches, or None when it is not a small explicit set."""
    kind, av = atom
    if kind == "LITERAL":
        return {chr(av)}
    if kind != "IN":
        return None
    chars: set[str] = set()
    for op, value in av:
        name = _name(op)
        if name == "LITERAL":
            chars.add(chr(value))
        elif name == "RANGE" and value[1] - value[0] < SMALL_CLASS:
            chars.update(chr(c) for c in range(value[0], value[1] + 1))
        else:
            return None
        if len(chars) > SMALL_CLASS:
            return None
    return chars


def _disjoint(a: tuple[str, Any], b: tuple[str, Any]) -> bool:
    """No character matches both atoms (decided over the side that is a small explicit set)."""
    for small, other in ((a, b), (b, a)):
        chars = _chars(small)
        if chars is not None:
            return not any(_matches(other, c) for c in chars)
    return False


def _fences(element: _Element, repeat: _Element) -> bool:
    """``element`` is required and ``repeat`` cannot consume any character it matches."""
    if element.atom is None or element.low < 1 or repeat.atom is None:
        return False
    return _disjoint(element.atom, repeat.atom)


def check_pattern(text: str) -> re.Pattern[str]:
    """Compile a ``re:`` pattern after the safety rules of the module docstring.

    Raises:
        ValueError: the pattern is invalid, too long or could backtrack badly (the message says
            which rule it breaks).
    """
    if not text:
        raise ValueError("--by-group re: needs a pattern")
    if len(text) > MAX_PATTERN_CHARS:
        raise _refuse(f"longer than {MAX_PATTERN_CHARS} characters")
    try:
        compiled = re.compile(text)
        parsed = _parser().parse(text)
    except (re.error, RecursionError, OverflowError) as exc:
        raise ValueError(f"invalid --by-group pattern: {getattr(exc, 'msg', exc)}") from None
    if compiled.flags & ~re.UNICODE:
        raise _refuse("flags are not supported")
    elements: list[_Element] = []
    _flatten(parsed, elements, in_branch=False)
    variable = [n for n, e in enumerate(elements) if e.variable]
    for first, second in zip(variable, variable[1:], strict=False):
        between = elements[first + 1 : second + 1]
        if not any(_fences(e, elements[first]) for e in between):
            raise _refuse(
                "two quantifiers can match the same characters; put a required separator the "
                "first one cannot match between them (for example [^_]+_ instead of \\w+_)"
            )
    return compiled


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
        return GroupKey("regex", text, pattern=check_pattern(text[3:]))
    raise ValueError("--by-group must be group_id, source.<name> or re:<pattern>")


def _label(value: object) -> str | None:
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
        return _label(image.group_id) if image.group_id is not None else None
    if key.kind == "source":
        return _label(image.source.get(key.field or ""))
    assert key.pattern is not None  # noqa: S101  # set by parse for the regex kind
    name = PurePosixPath(image.file_path.replace("\\", "/")).name
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
    return _label(value) if value else None
