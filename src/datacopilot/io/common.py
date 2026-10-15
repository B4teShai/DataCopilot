"""Helpers shared by the format adapters: row construction and loss accounting."""

from __future__ import annotations

import json
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
    """Image row from a real file: content-hash id and header dimensions."""
    width, height = files.image_size(path)
    return build_row(
        Image,
        ref,
        image_id=files.image_id(path),
        file_path=str(path),
        width=width,
        height=height,
        split=split,
        source=dict(source),
    )


def is_native(source: Mapping[str, object], fmt: str) -> bool:
    return source.get(NATIVE_KEY) == fmt


def foreign_source_drops(
    counter: Counter[str], column: str, source: Mapping[str, object], held: frozenset[str]
) -> None:
    """Count every ``source`` key of a foreign-format row that the target cannot hold."""
    for key in source:
        if key != NATIVE_KEY and key not in held:
            counter[f"{column}.source.{key}"] += 1


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


def parse_json(data: bytes, name: str) -> object:
    """Strict JSON: UTF-8, and no NaN/Infinity literals."""

    def reject(token: str) -> object:
        raise ValueError(token)

    try:
        return json.loads(data.decode("utf-8"), parse_constant=reject)
    except (ValueError, UnicodeDecodeError):
        raise FormatError(f"{name} is not valid JSON") from None
    except RecursionError:
        raise FormatError(f"{name} is nested too deeply") from None
