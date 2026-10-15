"""A deliberately tiny YAML subset parser for YOLO ``data.yaml`` files.

DataCopilot adds no YAML dependency. This parser accepts only what dataset configs use:
top-level ``key: value`` pairs with scalar values, flow lists of scalars (``[a, "b c"]``), block
lists (``- item``) and one-level block mappings (``0: name``). Anchors, aliases, tags, block
scalars, nested flow collections and duplicate keys are rejected, so no construct can trigger
object construction or unbounded expansion.
"""

from __future__ import annotations

import json
import re

from datacopilot.core.errors import FormatError

MAX_LINES = 10_000
MAX_LINE_CHARS = 4096
MAX_TOTAL_CHARS = 1024 * 1024
_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$")
_BAD_KEY_CHARS = frozenset("#[]{}&*!|>'\"")
_NAME = "data.yaml"


def _fail(why: str) -> FormatError:
    return FormatError(f"{_NAME}: {why}")


def _split_key(text: str, *, top_level: bool) -> tuple[str, str] | None:
    """Split ``key: rest`` at the first colon followed by whitespace or the end of the line.

    Plain ``str`` scans only (linear time, no regular expression). Returns None when the text
    is not a mapping entry or the key holds characters this subset does not accept.
    """
    pos = 0
    while True:
        i = text.find(":", pos)
        if i < 0:
            return None
        if i + 1 == len(text) or text[i + 1] in " \t":
            break
        pos = i + 1
    key = text[:i].rstrip()
    if not key or key[0] in " \t" or any(c in _BAD_KEY_CHARS for c in key):
        return None
    if top_level and key[0] == "-":
        return None
    return key, text[i + 1 :].strip()


def _strip_comment(line: str) -> str:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _scalar(token: str) -> object:
    token = token.strip()
    if not token:
        return None
    if token[0] in "&*!|>{":
        raise _fail("unsupported YAML construct")
    if token[0] == '"':
        try:
            value = json.loads(token)
        except ValueError:
            raise _fail("invalid double-quoted string") from None
        if not isinstance(value, str):
            raise _fail("invalid double-quoted string")
        return value
    if token[0] == "'":
        if len(token) < 2 or token[-1] != "'":
            raise _fail("invalid single-quoted string")
        return token[1:-1].replace("''", "'")
    if token in ("true", "True"):
        return True
    if token in ("false", "False"):
        return False
    if token in ("null", "Null", "~"):
        return None
    if _INT.match(token):
        return int(token)
    if _FLOAT.match(token):
        return float(token)
    if token.startswith("[") or token.endswith("]"):
        raise _fail("unsupported YAML construct")
    return token


def _split_flow(body: str) -> list[str]:
    parts, buf, quote = [], [], ""
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if quote:
        raise _fail("unterminated quoted string")
    parts.append("".join(buf))
    return parts


def _value(token: str) -> object:
    token = token.strip()
    if token.startswith("["):
        if not token.endswith("]"):
            raise _fail("unterminated flow list")
        body = token[1:-1].strip()
        if not body:
            return []
        items = [p.strip() for p in _split_flow(body)]
        if any(i.startswith(("[", "{")) or not i for i in items):
            raise _fail("unsupported YAML construct")
        return [_scalar(i) for i in items]
    return _scalar(token)


def parse_mapping(text: str) -> dict[str, object]:
    """Parse the supported YAML subset into a dict.

    Raises:
        FormatError: for anything outside the subset, with a message naming data.yaml.
    """
    if len(text) > MAX_TOTAL_CHARS:
        raise _fail(f"file is larger than {MAX_TOTAL_CHARS} characters")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        raise _fail(f"more than {MAX_LINES} lines")
    if any(len(line) > MAX_LINE_CHARS for line in lines):
        raise _fail(f"a line is longer than {MAX_LINE_CHARS} characters (limit {MAX_LINE_CHARS})")
    out: dict[str, object] = {}
    current: str | None = None
    for raw in lines:
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if line[0] not in " \t":
            split = _split_key(line, top_level=True)
            if split is None:
                raise _fail("unsupported or malformed line")
            key, rest = split
            if key in out:
                raise _fail(f"duplicate key {key!r}")
            out[key] = _value(rest) if rest else None
            current = key if not rest else None
            continue
        if current is None:
            raise _fail("unexpected indented line")
        body = line.strip()
        container = out[current]
        if body.startswith("- ") or body == "-":
            if container is None:
                container = out[current] = []
            if not isinstance(container, list):
                raise _fail("cannot mix list items and mapping entries")
            container.append(_value(body[1:]))
        else:
            item = _split_key(body, top_level=False)
            if item is None:
                raise _fail("unsupported or malformed line")
            if container is None:
                container = out[current] = {}
            if not isinstance(container, dict):
                raise _fail("cannot mix list items and mapping entries")
            key_text = item[0]
            item_key: int | str = int(key_text) if _INT.match(key_text) else key_text
            if item_key in container:
                raise _fail(f"duplicate key {item_key!r}")
            container[item_key] = _value(item[1])
    return out
