"""Helpers shared by the format adapters: row construction and loss accounting."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.schema import Image, Split
from datacopilot.io import files

M = TypeVar("M", bound=BaseModel)

NATIVE_KEY = "format"
"""Key in ``source`` naming the format that produced the row. Metadata, never counted as loss."""


def natural_key(text: str) -> list[str | int]:
    """Sort key that orders ``a:2`` before ``a:10`` (digit runs compare as numbers).

    Annotation ids of YOLO and VOC end in the position of the box in its file; the writers keep
    that order so that reading their output back gives the same ids.
    """
    return [int(part) if n % 2 else part for n, part in enumerate(re.split(r"(\d+)", text))]


def build_row(model: type[M], ref: str, **fields: object) -> M:
    """Construct a schema row; a violated invariant becomes a ValidationError naming ``ref``."""
    try:
        return model(**fields)
    except PydanticValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'row'}: {e['msg']}" for e in exc.errors()[:3]
        )
        raise ValidationError(f"{ref}: {problems}") from None


def make_image(path: Path, *, split: Split, source: Mapping[str, object], ref: str) -> Image:
    """Image row from a real file: content-hash id and header dimensions.

    ``path`` is the result of :func:`files.safe_join` (already resolved inside the dataset root);
    it is opened once, without following a symlink, and both values come from that descriptor.
    """
    header, digest = files.identify(path, nofollow=True)
    width, height = header.oriented_size
    return build_row(
        Image,
        ref,
        image_id=digest,
        file_path=str(path),
        width=width,
        height=height,
        split=split,
        source=dict(source),
    )


def is_native(source: Mapping[str, object], fmt: str) -> bool:
    return source.get(NATIVE_KEY) == fmt


LAYOUT_PREFIX = "layout_"
"""``source`` keys with this prefix record where a row came from (file, reference form) so that a
write to the same format can reproduce it; they are not data and no writer counts them."""
MAX_KEY_CHARS = 64
"""Longest key of a ``source`` that may name a report column."""
MAX_COUNTERS = 200
"""Distinct columns of one report before new ones are aggregated into ``<table>.other``."""
_KEY = re.compile(r"[A-Za-z0-9_.-]+")


def safe_key(key: object) -> str:
    """``key`` when it is a plain name of at most :data:`MAX_KEY_CHARS` characters, else
    ``invalid_key``: a key read from an untrusted file becomes a report column, a log field and a
    line of ``--strict`` output, so it must not carry control characters, markup or length."""
    if isinstance(key, str) and len(key) <= MAX_KEY_CHARS and _KEY.fullmatch(key):
        return key
    return "invalid_key"


def count(counter: Counter[str], name: str, amount: int = 1) -> None:
    """Add to ``counter[name]``; once the report holds :data:`MAX_COUNTERS` distinct columns a
    new one is counted under ``<table>.other`` instead: the total is kept, the size is bounded."""
    if name not in counter and len(counter) >= MAX_COUNTERS:
        name = name.partition(".")[0] + ".other"
    counter[name] += amount


def foreign_source_drops(
    counter: Counter[str], column: str, source: Mapping[str, object], held: frozenset[str]
) -> None:
    """Count every ``source`` key of a foreign-format row that the target cannot hold."""
    for key in source:
        if key != NATIVE_KEY and key not in held and not key.startswith(LAYOUT_PREFIX):
            count(counter, f"{column}.source.{safe_key(key)}")


def require_empty_dir(out: Path) -> None:
    """Writers receive an existing, empty output directory."""
    if not out.is_dir():
        raise FormatError(f"output {out.name} is not a directory")
    if any(out.iterdir()):
        raise FormatError(f"output directory {out.name} is not empty")


def require_task(ds_task: str, wanted: str, fmt: str) -> None:
    if ds_task != wanted:
        raise FormatError(
            f"format {fmt!r} holds {wanted} datasets only; this dataset's task is {ds_task!r}"
        )


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def parse_json(data: bytes, name: str, *, unique_keys: bool = False) -> object:
    """Strict JSON: UTF-8, no NaN/Infinity literals and, with ``unique_keys``, no repeated keys
    in one object (a repeated key would otherwise let the last value win silently)."""

    def reject(token: str) -> object:
        raise ValueError(token)

    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=reject,
            object_pairs_hook=_no_duplicate_keys if unique_keys else None,
        )
    except (ValueError, UnicodeDecodeError):
        raise FormatError(f"{name} is not valid JSON") from None
    except RecursionError:
        raise FormatError(f"{name} is nested too deeply") from None
