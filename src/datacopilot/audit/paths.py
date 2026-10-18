"""Resolution of ``Image.file_path`` values taken from a (possibly hostile) dataset.

A path is a remote URI (never fetched), unsafe (reported, never opened), external (its real path,
symlinks resolved, lies outside the dataset root: counted, never opened, so a symlink or absolute
path cannot be used to probe or hash files elsewhere on the machine), or local. Callers open
local paths with ``O_NOFOLLOW`` and accept regular files only. ``allow_external`` opts in to
images stored outside the root, which ingest records by absolute path.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Severity
from datacopilot.io import files

MAX_NAME_CHARS = 255
"""Longest ``file_name`` kept in a finding payload (the usual file-system limit)."""

Kind = Literal["local", "remote", "unsafe", "external"]


@dataclass(frozen=True)
class ResolvedPath:
    kind: Kind
    path: Path | None = None


def _is_remote(file_path: str) -> bool:
    scheme, sep, _ = file_path.partition("://")
    return (
        bool(sep)
        and len(scheme) >= 2
        and scheme.isascii()
        and all(c.isalnum() or c in "+-." for c in scheme)
    )


def resolve_image_path(file_path: str, root: Path, *, allow_external: bool = False) -> ResolvedPath:
    """Classify ``file_path``; for ``local`` return the *real* path (symlinks resolved).

    ``unsafe``: empty, NUL, a backslash, ``..`` or an absolute-looking Windows path in a relative
    value, or a relative path that resolves outside ``root``. ``external``: the real path lies
    outside ``root`` and ``allow_external`` is false, so the file is neither opened nor stat-ed
    (its existence and content stay undisclosed). ``remote``: a URI, never fetched.
    """
    if _is_remote(file_path):
        return ResolvedPath("remote")
    if "\x00" in file_path or not file_path:
        return ResolvedPath("unsafe")
    if os.path.isabs(file_path):
        candidate = Path(os.path.realpath(file_path))
    else:
        try:
            candidate = files.safe_join(root, file_path)
        except FormatError:
            return ResolvedPath("unsafe")
    if not allow_external and not candidate.is_relative_to(Path(os.path.realpath(root))):
        return ResolvedPath("external")
    return ResolvedPath("local", candidate)


def clip(value: str, limit: int = 100) -> str:
    """``value`` cut to ``limit`` characters with an ellipsis (for names shown in payloads)."""
    return value if len(value) <= limit else value[: limit - 1] + "…"


def unchecked_severity(
    unchecked: int, total: int, *, max_fraction: float, require_local: bool
) -> Severity:
    """Gate outcome for images a check could not examine.

    Any unchecked image is at least WARN, so a dataset cannot pass silently by pointing its images
    somewhere the audit will not look; CRITICAL when ``require_local`` is set or when more than
    ``max_fraction`` of the dataset was unchecked.
    """
    if require_local or (total > 0 and unchecked / total > max_fraction):
        return "CRITICAL"
    return "WARN"


def display_name(file_path: str) -> str:
    """Base name of ``file_path`` for findings (never the directory), at most 255 characters.

    A remote URI loses its query string and fragment first (a presigned URL carries a credential
    and a signature there) and shows only the last path segment, without scheme or host.
    """
    if _is_remote(file_path):
        file_path = file_path.partition("?")[0].partition("#")[0]
    return file_path.replace("\\", "/").rsplit("/", 1)[-1][:MAX_NAME_CHARS]


def safe_text(value: str, limit: int) -> str:
    """``value`` as one printable line of at most ``limit`` characters: control, bidirectional,
    format and surrogate characters become ``?``, longer text ends in an ellipsis."""
    out = []
    for char in value[: limit + 1]:
        bad = unicodedata.category(char) in ("Cc", "Cf", "Cs", "Zl", "Zp") or not char.isprintable()
        out.append("?" if bad else char)
    text = "".join(out)
    return text if len(text) <= limit else text[: limit - 1] + "…"
