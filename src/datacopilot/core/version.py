"""Order-independent content hash of a dataset and lineage helpers.

Procedure (docs/architecture/03-data-model.md, "Versioning and compatibility"):

1. Every row becomes canonical JSON (sorted keys, no whitespace, UTF-8, shortest round-trip
   floats, no NaN or infinity); ``images`` rows drop ``file_path``.
2. ``table_hash = sha256("".join(sorted(sha256(row_json) for each row)))``.
3. ``version = sha256("dcp:<schema_version>\\n<images>\\n<annotations>\\n<categories>")[:32]``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

from datacopilot.core.errors import ValidationError
from datacopilot.core.schema import Annotation, Category, DatasetManifest, Image

_LOCATION_COLUMNS = frozenset({"file_path"})


def _reject(value: object) -> object:
    raise ValidationError(f"value of type {type(value).__name__} cannot be hashed")


def canonical_json(row: Mapping[str, object]) -> str:
    """Serialise one row deterministically.

    Raises:
        ValidationError: if the row holds NaN, an infinity or a non-JSON value.
    """
    try:
        return json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_reject,
        )
    except ValueError as exc:
        raise ValidationError("cannot hash a row: numbers must be finite") from exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table_hash(
    rows: Iterable[Mapping[str, object]], *, exclude: frozenset[str] = frozenset()
) -> str:
    """Hash a table independently of row order; ``exclude`` columns are ignored."""
    hashes = sorted(
        _sha256(canonical_json({k: v for k, v in row.items() if k not in exclude})) for row in rows
    )
    return _sha256("".join(hashes))


def dataset_version(
    images: Iterable[Mapping[str, object]],
    annotations: Iterable[Mapping[str, object]],
    categories: Iterable[Mapping[str, object]],
    *,
    schema_version: int,
) -> str:
    """32-hex content hash of the three tables; ``file_path`` is excluded."""
    parts = (
        table_hash(images, exclude=_LOCATION_COLUMNS),
        table_hash(annotations),
        table_hash(categories),
    )
    return _sha256(f"dcp:{schema_version}\n" + "\n".join(parts))[:32]


def models_version(
    images: Sequence[Image],
    annotations: Sequence[Annotation],
    categories: Sequence[Category],
    *,
    schema_version: int,
) -> str:
    """:func:`dataset_version` over validated model rows."""
    return dataset_version(
        (i.to_row() for i in images),
        (a.to_row() for a in annotations),
        (c.to_row() for c in categories),
        schema_version=schema_version,
    )


def extend_lineage(parent: DatasetManifest, child_version: str) -> tuple[str, ...]:
    """Lineage of a child version: the parent's lineage plus the parent's version.

    Raises:
        ValidationError: if the child has the same version as its parent (nothing changed).
    """
    if child_version == parent.version:
        raise ValidationError(f"dataset unchanged: version {child_version} already exists")
    return (*parent.lineage, parent.version)


def finding_id(
    check: str, image_id: str | None, ann_id: str | None, payload: Mapping[str, object]
) -> str:
    """Deterministic id of a finding: ``sha256`` of its run-independent key, first 16 hex.

    The key is the check name, the image and annotation it concerns and the payload; the run and
    the severity are excluded so a rerun (or a changed threshold) yields the same id.
    """
    key = canonical_json(
        {"check": check, "image_id": image_id, "ann_id": ann_id, "payload": dict(payload)}
    )
    return _sha256(key)[:16]
