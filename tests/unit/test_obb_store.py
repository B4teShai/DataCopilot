"""Oriented boxes in the data model, the content version and the store (ADR-0011)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError as PydanticValidationError

from datacopilot.core.errors import StoreError, ValidationError
from datacopilot.core.schema import (
    MAX_SCHEMA_VERSION,
    OBB_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Annotation,
    Category,
    Image,
    schema_version_for,
    validate_tables,
)
from datacopilot.core.version import models_version
from datacopilot.store import DatasetStore, arrow

IMG = "0123456789abcdef"
CORNERS = (20.0, 10.0, 60.0, 30.0, 50.0, 50.0, 10.0, 30.0)


def obb(ann_id: str = "o1", corners: tuple[float, ...] = CORNERS, **kw: object) -> Annotation:
    return Annotation(
        ann_id=ann_id,
        image_id=IMG,
        category_id=0,
        kind="obb",
        obb_xyxyxyxy_abs=corners,  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


def images() -> list[Image]:
    return [Image(image_id=IMG, file_path="a.png", width=100, height=80, split="train")]


CATS = [Category(id=0, name="can")]


def test_an_oriented_box_keeps_its_corners_and_derives_the_enclosing_box() -> None:
    ann = obb()
    assert ann.obb_xyxyxyxy_abs == CORNERS
    assert ann.bbox_xyxy_abs == (10.0, 10.0, 60.0, 50.0)
    assert obb(bbox_xyxy_abs=(10.0, 10.0, 60.0, 50.0)) == ann  # an explicit, matching box


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"bbox_xyxy_abs": (10.0, 10.0, 61.0, 50.0)}, "enclosing"),
        ({"corners": (0.0, 0.0, 4.0, 2.0, 4.0, 0.0, 0.0, 2.0)}, "convex"),
        ({"corners": (-1.0, 0.0, 4.0, 0.0, 4.0, 2.0, 0.0, 2.0)}, "0 <= x_min"),
        ({"corners": (0.0, 0.0, 4.0, 0.0, float("nan"), 2.0, 0.0, 2.0)}, "finite"),
        ({"corners": (0.0, 0.0, 4.0, 0.0)}, "obb_xyxyxyxy_abs.4"),
    ],
)
def test_invalid_oriented_boxes_are_refused(fields: dict[str, object], message: str) -> None:
    corners = fields.pop("corners", CORNERS)
    with pytest.raises(PydanticValidationError, match=message):
        obb(corners=corners, **fields)  # type: ignore[arg-type]


def test_kinds_carry_only_their_own_geometry() -> None:
    with pytest.raises(PydanticValidationError, match="requires obb"):
        Annotation(ann_id="x", image_id=IMG, category_id=0, kind="obb", bbox_xyxy_abs=(0, 0, 1, 1))
    with pytest.raises(PydanticValidationError, match="only an obb"):
        Annotation(
            ann_id="x",
            image_id=IMG,
            category_id=0,
            kind="bbox",
            bbox_xyxy_abs=(10.0, 10.0, 60.0, 50.0),
            obb_xyxyxyxy_abs=CORNERS,
        )
    with pytest.raises(PydanticValidationError, match="must not have a box"):
        Annotation(
            ann_id="x", image_id=IMG, category_id=0, kind="classification", obb_xyxyxyxy_abs=CORNERS
        )


def test_a_detection_dataset_holds_one_kind_of_box_inside_its_images() -> None:
    validate_tables(images(), [obb()], CATS, task="detection")
    plain = Annotation(
        ann_id="b1", image_id=IMG, category_id=0, kind="bbox", bbox_xyxy_abs=(1, 1, 5, 5)
    )
    with pytest.raises(ValidationError, match="mixes axis-aligned"):
        validate_tables(images(), [obb(), plain], CATS, task="detection")
    with pytest.raises(ValidationError, match="dataset task is 'classification'"):
        validate_tables(images(), [obb()], CATS, task="classification")
    outside = obb(corners=(20.0, 10.0, 120.0, 30.0, 110.0, 50.0, 10.0, 30.0))
    with pytest.raises(ValidationError, match="outside image"):
        validate_tables(images(), [outside], CATS, task="detection")


def test_the_generation_is_the_oldest_that_holds_the_rows() -> None:
    plain = Annotation(
        ann_id="b1", image_id=IMG, category_id=0, kind="bbox", bbox_xyxy_abs=(1, 1, 5, 5)
    )
    assert schema_version_for([]) == SCHEMA_VERSION == 1
    assert schema_version_for([plain]) == 1
    assert schema_version_for([obb()]) == OBB_SCHEMA_VERSION == MAX_SCHEMA_VERSION == 2


def test_a_null_obb_column_does_not_change_the_version() -> None:
    plain = Annotation(
        ann_id="b1", image_id=IMG, category_id=0, kind="bbox", bbox_xyxy_abs=(1, 1, 5, 5)
    )
    row = plain.to_row()
    assert row["obb_xyxyxyxy_abs"] is None
    from datacopilot.core.version import dataset_version

    without = {k: v for k, v in row.items() if k != "obb_xyxyxyxy_abs"}
    imgs, cats = [i.to_row() for i in images()], [c.to_row() for c in CATS]
    assert dataset_version(imgs, [row], cats, schema_version=1) == dataset_version(
        imgs, [without], cats, schema_version=1
    )


def test_corner_order_is_part_of_the_version() -> None:
    turned = obb(corners=CORNERS[2:] + CORNERS[:2])
    assert turned.bbox_xyxy_abs == obb().bbox_xyxy_abs
    a = models_version(images(), [obb()], CATS, schema_version=2)
    b = models_version(images(), [turned], CATS, schema_version=2)
    assert a != b


def test_store_writes_generation_2_and_reads_it_back(tmp_path: Path) -> None:
    anns = [obb(), obb("o2", (60.0, 60.0, 90.0, 60.0, 90.0, 75.0, 60.0, 75.0))]
    store = DatasetStore.create(
        tmp_path / "o.dcp", images(), anns, CATS, name="o", task="detection"
    )
    manifest = store.manifest()
    assert manifest.schema_version == 2
    assert manifest.version == models_version(images(), anns, CATS, schema_version=2)
    schema = pq.read_schema(tmp_path / "o.dcp" / "annotations.parquet")
    assert schema.equals(arrow.ANNOTATIONS_SCHEMA_OBB)
    ds = DatasetStore.open(tmp_path / "o.dcp").read()
    assert sorted(ds.annotations, key=lambda a: a.ann_id) == anns
    # a new version without oriented boxes goes back to generation 1
    plain = [
        Annotation(ann_id="b", image_id=IMG, category_id=0, kind="bbox", bbox_xyxy_abs=(1, 1, 5, 5))
    ]
    second = store.write_version(images(), plain, CATS)
    assert second.schema_version == 1 and second.lineage == (manifest.version,)
    assert pq.read_schema(tmp_path / "o.dcp" / "annotations.parquet").equals(
        arrow.ANNOTATIONS_SCHEMA
    )
    assert DatasetStore.open(tmp_path / "o.dcp").read().annotations == tuple(plain)


def test_store_without_oriented_boxes_is_written_exactly_as_before(
    tmp_path: Path, tables: tuple[list[Image], list[Annotation], list[Category]]
) -> None:
    store = DatasetStore.create(tmp_path / "d.dcp", *tables, name="d", task="detection")
    assert store.manifest().schema_version == 1
    names = pq.read_schema(tmp_path / "d.dcp" / "annotations.parquet").names
    assert "obb_xyxyxyxy_abs" not in names


def test_a_generation_1_table_cannot_hold_oriented_boxes() -> None:
    with pytest.raises(StoreError, match="cannot hold"):
        arrow.annotations_to_table([obb()], 1)
    with pytest.raises(StoreError, match="unknown schema_version"):
        arrow.annotations_schema(3)
    table = arrow.annotations_to_table([], 2)  # an explicit newer generation is allowed
    assert table.schema.equals(arrow.ANNOTATIONS_SCHEMA_OBB)


def test_a_manifest_that_claims_the_wrong_generation_is_refused(tmp_path: Path) -> None:
    DatasetStore.create(tmp_path / "o.dcp", images(), [obb()], CATS, name="o", task="detection")
    manifest = json.loads((tmp_path / "o.dcp" / "manifest.json").read_text())
    manifest["schema_version"] = 1
    (tmp_path / "o.dcp" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="do not match the expected schema"):
        DatasetStore.open(tmp_path / "o.dcp").read()
    manifest["schema_version"] = 3
    (tmp_path / "o.dcp" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="supports up to 2"):
        DatasetStore.open(tmp_path / "o.dcp")
