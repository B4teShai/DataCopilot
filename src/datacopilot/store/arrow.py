"""Explicit Arrow schemas for the three tables and the row <-> Arrow conversions.

Nested free-form fields (``source``, ``attributes`` of annotations) are Arrow ``map<string,
string>`` columns whose values are canonical JSON text, so scalar types (int vs float vs bool)
survive a round trip. Category attributes are plain string maps.

The annotations table has one schema per table-schema generation (ADR-0011): generation 1 is the
original; generation 2 adds the nullable ``obb_xyxyxyxy_abs`` column (eight floats) after
``bbox_xyxy_abs``. A dataset is written in the oldest generation that holds its rows
(``schema_version_for``), so a dataset without oriented boxes is written byte for byte as before.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from datacopilot.core.errors import StoreError
from datacopilot.core.schema import (
    OBB_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Annotation,
    Category,
    Image,
    schema_version_for,
)

_MAP = pa.map_(pa.string(), pa.string())

IMAGES_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("width", pa.int64(), nullable=False),
        pa.field("height", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("group_id", pa.string()),
        pa.field("source", _MAP, nullable=False),
    ]
)
_ANNOTATION_HEAD = [
    pa.field("ann_id", pa.string(), nullable=False),
    pa.field("image_id", pa.string(), nullable=False),
    pa.field("category_id", pa.int64(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),
    pa.field("bbox_xyxy_abs", pa.list_(pa.float64())),
]
_OBB_FIELD = pa.field("obb_xyxyxyxy_abs", pa.list_(pa.float64()))
_ANNOTATION_TAIL = [
    pa.field("score", pa.float64()),
    pa.field("is_crowd", pa.bool_(), nullable=False),
    pa.field("occluded", pa.bool_()),
    pa.field("truncated", pa.bool_()),
    pa.field("attributes", _MAP, nullable=False),
    pa.field("provenance", pa.string(), nullable=False),
    pa.field("source", _MAP, nullable=False),
]
ANNOTATIONS_SCHEMA = pa.schema(_ANNOTATION_HEAD + _ANNOTATION_TAIL)
"""Generation 1 (no oriented boxes)."""
ANNOTATIONS_SCHEMA_OBB = pa.schema([*_ANNOTATION_HEAD, _OBB_FIELD, *_ANNOTATION_TAIL])
"""Generation 2: generation 1 plus ``obb_xyxyxyxy_abs``."""
CATEGORIES_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("aliases", pa.list_(pa.string()), nullable=False),
        pa.field("attributes", pa.map_(pa.string(), pa.string()), nullable=False),
    ]
)


def annotations_schema(schema_version: int) -> pa.Schema:
    """The annotations schema of a table-schema generation.

    Raises:
        StoreError: a generation this package does not know.
    """
    if schema_version == SCHEMA_VERSION:
        return ANNOTATIONS_SCHEMA
    if schema_version == OBB_SCHEMA_VERSION:
        return ANNOTATIONS_SCHEMA_OBB
    raise StoreError(f"unknown schema_version {schema_version}")


def _dump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _encode_map(values: Mapping[str, object]) -> list[tuple[str, str]]:
    return [(k, _dump(v)) for k, v in values.items()]


def _decode_map(pairs: Sequence[tuple[str, str]] | None) -> dict[str, Any]:
    return {k: json.loads(v) for k, v in (pairs or [])}


def images_to_table(images: Sequence[Image]) -> pa.Table:
    rows = sorted(images, key=lambda i: i.image_id)
    return pa.table(
        {
            "image_id": [r.image_id for r in rows],
            "file_path": [r.file_path for r in rows],
            "width": [r.width for r in rows],
            "height": [r.height for r in rows],
            "split": [r.split for r in rows],
            "group_id": [r.group_id for r in rows],
            "source": pa.array([_encode_map(r.source) for r in rows], type=_MAP),
        },
        schema=IMAGES_SCHEMA,
    )


def annotations_to_table(
    annotations: Sequence[Annotation], schema_version: int | None = None
) -> pa.Table:
    """The annotations table in ``schema_version`` (default: the oldest that holds the rows).

    Raises:
        StoreError: the generation cannot hold oriented boxes but a row has one.
    """
    generation = schema_version_for(annotations) if schema_version is None else schema_version
    schema = annotations_schema(generation)
    if generation < schema_version_for(annotations):
        raise StoreError(f"schema_version {generation} cannot hold oriented boxes")
    rows = sorted(annotations, key=lambda a: a.ann_id)
    columns: dict[str, object] = {
        "ann_id": [r.ann_id for r in rows],
        "image_id": [r.image_id for r in rows],
        "category_id": [r.category_id for r in rows],
        "kind": [r.kind for r in rows],
        "bbox_xyxy_abs": pa.array(
            [list(r.bbox_xyxy_abs) if r.bbox_xyxy_abs else None for r in rows],
            type=pa.list_(pa.float64()),
        ),
    }
    if generation >= OBB_SCHEMA_VERSION:
        columns["obb_xyxyxyxy_abs"] = pa.array(
            [list(r.obb_xyxyxyxy_abs) if r.obb_xyxyxyxy_abs else None for r in rows],
            type=pa.list_(pa.float64()),
        )
    return pa.table(
        {
            **columns,
            "score": [r.score for r in rows],
            "is_crowd": [r.is_crowd for r in rows],
            "occluded": [r.occluded for r in rows],
            "truncated": [r.truncated for r in rows],
            "attributes": pa.array([_encode_map(r.attributes) for r in rows], type=_MAP),
            "provenance": [r.provenance for r in rows],
            "source": pa.array([_encode_map(r.source) for r in rows], type=_MAP),
        },
        schema=schema,
    )


def categories_to_table(categories: Sequence[Category]) -> pa.Table:
    rows = sorted(categories, key=lambda c: c.id)
    return pa.table(
        {
            "id": [r.id for r in rows],
            "name": [r.name for r in rows],
            "aliases": [list(r.aliases) for r in rows],
            "attributes": pa.array([list(r.attributes.items()) for r in rows], type=_MAP),
        },
        schema=CATEGORIES_SCHEMA,
    )


def table_to_images(table: pa.Table) -> list[Image]:
    return [Image(**{**row, "source": _decode_map(row["source"])}) for row in table.to_pylist()]


def table_to_annotations(table: pa.Table) -> list[Annotation]:
    return [
        Annotation(
            **{
                **row,
                "attributes": _decode_map(row["attributes"]),
                "source": _decode_map(row["source"]),
            }
        )
        for row in table.to_pylist()
    ]


def table_to_categories(table: pa.Table) -> list[Category]:
    return [
        Category(**{**row, "attributes": dict(row["attributes"] or [])})
        for row in table.to_pylist()
    ]
