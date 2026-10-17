"""Shared test helpers: tiny synthetic datasets, built in memory."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
import structlog

from datacopilot.core.schema import (
    SCHEMA_VERSION,
    Annotation,
    Category,
    Dataset,
    DatasetManifest,
    Image,
    Task,
    validate_tables,
)
from datacopilot.core.version import models_version
from datacopilot.io.registry import ReadResult


def make_tables() -> tuple[list[Image], list[Annotation], list[Category]]:
    """Three images, four annotations, two categories; exercises nested and nullable fields."""
    images = [
        Image(
            image_id=f"{n:016x}",
            file_path=f"/data/img{n}.png",
            width=100 + n,
            height=60,
            split=("train", "val", "unassigned")[n],
            group_id=("cam1_a" if n == 0 else None),
            source={
                "format": "coco",
                "id": n,
                "file_name": f"img{n}.png",
                "extra": {"k": [1, 2.5]},
            },
        )
        for n in range(3)
    ]
    anns = [
        Annotation(
            ann_id="a1",
            image_id=images[0].image_id,
            category_id=1,
            kind="bbox",
            bbox_xyxy_abs=(10.5, 5.0, 60.25, 40.0),
            is_crowd=True,
            occluded=False,
            attributes={"difficult": True, "n": 3, "w": 0.5, "s": "x"},
            source={"format": "coco", "id": 1, "area": 1234.5, "segmentation": [[1, 2, 3, 4]]},
        ),
        Annotation(
            ann_id="a2",
            image_id=images[0].image_id,
            category_id=2,
            kind="bbox",
            bbox_xyxy_abs=(0.0, 0.0, 100.0, 60.0),
            score=0.75,
            provenance="model",
        ),
        Annotation(
            ann_id="a3",
            image_id=images[1].image_id,
            category_id=1,
            kind="bbox",
            bbox_xyxy_abs=(1.0, 2.0, 3.0, 4.0),
            truncated=True,
        ),
        Annotation(
            ann_id="a4",
            image_id=images[2].image_id,
            category_id=2,
            kind="bbox",
            bbox_xyxy_abs=(1.0, 2.0, 3.0, 4.0),
        ),
    ]
    cats = [
        Category(id=1, name="cat", aliases=("kitty", "feline"), attributes={"sku": "c-1"}),
        Category(id=2, name="dog"),
    ]
    return images, anns, cats


@pytest.fixture
def tables() -> tuple[list[Image], list[Annotation], list[Category]]:
    return make_tables()


def make_dataset(result: ReadResult) -> Dataset:
    """Validated in-memory Dataset from a reader's rows (no disk involved)."""
    return dataset_from(result.images, result.annotations, result.categories, result.task)


def dataset_from(
    images: Sequence[Image],
    annotations: Sequence[Annotation],
    categories: Sequence[Category],
    task: Task = "detection",
    name: str = "demo",
) -> Dataset:
    validate_tables(images, annotations, categories, task=task)
    manifest = DatasetManifest(
        version=models_version(images, annotations, categories, schema_version=SCHEMA_VERSION),
        schema_version=SCHEMA_VERSION,
        name=name,
        task=task,
        categories=tuple(sorted(categories, key=lambda c: c.id)),
        image_count=len(images),
        annotation_count=len(annotations),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    paths = [i.file_path for i in images if os.path.isabs(i.file_path)]
    root = Path(os.path.commonpath(paths)).parent if paths else None  # where the pictures live
    return Dataset(manifest, tuple(images), tuple(annotations), root)


def table_rows(
    ds_images: Sequence[Image], ds_anns: Sequence[Annotation]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Canonically ordered rows without ``file_path``, for field-identity comparisons."""
    images = sorted(
        ({k: v for k, v in i.to_row().items() if k != "file_path"} for i in ds_images),
        key=lambda r: str(r["image_id"]),
    )
    anns = sorted((a.to_row() for a in ds_anns), key=lambda r: str(r["ann_id"]))
    return images, anns


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """A test that configures logging (the CLI and the API do) must not leave a handle to a
    captured, soon-closed stream behind for the tests that run after it."""
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
