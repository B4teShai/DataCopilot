"""Text hygiene for report output.

Escaping for the target format (HTML, JSON) is done by that format's serialiser. This module
removes what no serialiser should have to carry: control characters, bidirectional overrides and
zero-width marks (a file name can reorder the text around it), lone surrogates (they cannot be
encoded) and unbounded length.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping

MAX_TEXT_CHARS = 300
"""Default length of one displayed value."""
_INVISIBLE_RANGES = (
    (0x0000, 0x001F),  # C0 controls (tab and newline included: a table cell is one line)
    (0x007F, 0x009F),  # DEL and C1 controls
    (0x200B, 0x200F),  # zero-width and directional marks
    (0x2028, 0x202E),  # line and paragraph separators, embeddings and overrides
    (0x2060, 0x2069),  # word joiner, invisible operators, isolates
    (0xD800, 0xDFFF),  # surrogates
    (0xFEFF, 0xFEFF),  # byte order mark
    (0xFFF9, 0xFFFB),  # interlinear annotation
)


_EXTRA_INVISIBLE = frozenset({0x061C, 0x180E})  # Arabic letter mark; Mongolian vowel separator


def _is_invisible(code: int) -> bool:
    """Control, format (bidi, zero-width, tag characters) and surrogate characters."""
    if code in _EXTRA_INVISIBLE or any(lo <= code <= hi for lo, hi in _INVISIBLE_RANGES):
        return True
    return unicodedata.category(chr(code)) in ("Cc", "Cf", "Cs", "Zl", "Zp")


def clean_text(value: object, limit: int = MAX_TEXT_CHARS) -> str:
    """``value`` as one line of printable text, at most about ``limit`` characters.

    Invisible and control characters are shown as ``\\uXXXX``; text beyond ``limit`` is replaced
    by an ellipsis. Cost is proportional to ``limit``, not to the size of the input.
    """
    raw = value if isinstance(value, str) else str(value)
    out: list[str] = []
    size = 0
    for char in raw:
        if size >= limit:
            out.append("…")
            break
        code = ord(char)
        if not _is_invisible(code):
            shown = char
        elif code > 0xFFFF:
            shown = f"\\U{code:08x}"
        else:
            shown = f"\\u{code:04x}"
        out.append(shown)
        size += len(shown)
    return "".join(out)


def describe_payload(payload: Mapping[str, object], limit: int = MAX_TEXT_CHARS) -> str:
    """``key=value`` pairs of a payload, sorted by key, without ``rule`` and ``file_name``."""
    parts = []
    for key in sorted(payload):
        if key in ("rule", "file_name"):
            continue
        value = payload[key]
        shown = (
            value
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":"), ensure_ascii=True, default=str)
        )
        parts.append(f"{key}={shown}")
    return clean_text("; ".join(parts), limit)
