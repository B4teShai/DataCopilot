"""Synthetic datasets with real files on disk for the audit checks.

``build`` writes tiny PNGs (or arbitrary bytes) under ``<root>/images`` and returns an in-memory
:class:`Dataset` whose rows point at them. Nothing here exceeds a few hundred bytes per file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from datacopilot.core.schema import (
    SCHEMA_VERSION,
    Annotation,
    Category,
    Dataset,
    DatasetManifest,
    Image,
    validate_tables,
)
from datacopilot.core.version import models_version
from fixtures.synth import png_bytes

CLASSES = ("cat", "dog", "bird")


@dataclass
class Img:
    """One image to create. ``boxes`` are ``(category_index, x1, y1, x2, y2)``."""

    name: str
    width: int = 64
    height: int = 48
    split: str = "train"
    boxes: Sequence[tuple[int, float, float, float, float]] = ()
    seed: int | None = None  # pixel pattern; default is the image's position
    data: bytes | None = None  # write these bytes instead of a generated PNG
    declared: tuple[int, int] | None = None  # declared width/height when they differ from the file
    write: bool = True  # False: the row exists but the file does not
    group: str | None = None
    image_id: str | None = None  # row id when it must differ from the content hash
    path: str | None = None  # file_path override (relative, URI, hostile text ...)
    extra: dict[str, object] = field(default_factory=dict)


def image_id_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def build(
    root: Path, specs: Sequence[Img], *, validate: bool = True, task: str = "detection"
) -> Dataset:
    """Create the files and return the matching dataset (``root/images/<name>``)."""
    images: list[Image] = []
    anns: list[Annotation] = []
    for n, spec in enumerate(specs):
        data = (
            spec.data
            if spec.data is not None
            else png_bytes(spec.width, spec.height, spec.seed if spec.seed is not None else n)
        )
        target = root / "images" / spec.name
        if spec.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        w, h = spec.declared or (spec.width, spec.height)
        images.append(
            Image(
                image_id=spec.image_id or image_id_of(data),
                file_path=spec.path if spec.path is not None else str(target),
                width=w,
                height=h,
                split=spec.split,  # type: ignore[arg-type]
                group_id=spec.group,
                source={"format": "coco", **spec.extra},  # type: ignore[dict-item]
            )
        )
        for k, (cat, x1, y1, x2, y2) in enumerate(spec.boxes):
            anns.append(
                Annotation(
                    ann_id=f"{n}-{k}",
                    image_id=images[-1].image_id,
                    category_id=cat + 1,
                    kind="bbox",
                    bbox_xyxy_abs=(x1, y1, x2, y2),
                )
            )
    cats = [Category(id=i + 1, name=name) for i, name in enumerate(CLASSES)]
    return dataset(images, anns, cats, validate=validate, root=root)


def dataset(
    images: Sequence[Image],
    anns: Sequence[Annotation],
    cats: Sequence[Category],
    *,
    validate: bool = True,
    root: Path | None = None,
) -> Dataset:
    if validate:
        validate_tables(images, anns, cats, task="detection")
    manifest = DatasetManifest(
        version=models_version(images, anns, cats, schema_version=SCHEMA_VERSION),
        schema_version=SCHEMA_VERSION,
        name="demo",
        task="detection",
        categories=tuple(sorted(cats, key=lambda c: c.id)),
        image_count=len(images),
        annotation_count=len(anns),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return Dataset(manifest, tuple(images), tuple(anns), root)
