from typing import Any

import pytest

from datacopilot.core.errors import DataCopilotError, ValidationError
from datacopilot.core.schema import (
    Annotation,
    Category,
    DatasetManifest,
    Image,
    PluginSpec,
    validate_tables,
)

IMG_A = "a" * 16
IMG_B = "b" * 16


def image(image_id: str = IMG_A, **kw: Any) -> Image:
    base: dict[str, Any] = {
        "image_id": image_id,
        "file_path": "/x/a.png",
        "width": 100,
        "height": 50,
    }
    return Image(**{**base, **kw})


def box(ann_id: str = "1", image_id: str = IMG_A, **kw: Any) -> Annotation:
    base: dict[str, Any] = {
        "ann_id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "kind": "bbox",
        "bbox_xyxy_abs": (10.0, 10.0, 60.0, 40.0),
    }
    return Annotation(**{**base, **kw})


CATS = [Category(id=1, name="cat"), Category(id=2, name="dog")]


def test_error_hierarchy_has_one_base() -> None:
    assert issubclass(ValidationError, DataCopilotError)


# --- Image -------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "xyz", "A" * 16, "a" * 15, "a" * 17])
def test_image_id_must_be_16_lower_hex(bad: str) -> None:
    with pytest.raises(ValueError):
        image(bad)


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1])
def test_image_dimensions_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        image(**{field: value})


def test_image_defaults_and_frozen() -> None:
    img = image()
    assert img.split == "unassigned" and img.group_id is None and img.source == {}
    with pytest.raises(ValueError):
        img.width = 5  # type: ignore[misc]


def test_image_rejects_unknown_split_and_extra_fields() -> None:
    with pytest.raises(ValueError):
        image(split="validation")
    with pytest.raises(ValueError):
        image(colour="red")


def test_source_rejects_non_finite_floats_and_non_json() -> None:
    with pytest.raises(ValueError):
        image(source={"x": float("nan")})
    with pytest.raises(ValueError):
        image(source={"x": {"y": [1, float("inf")]}})
    with pytest.raises(ValueError):
        image(source={"x": object()})


# --- Annotation --------------------------------------------------------------------------------
def test_bbox_requires_box_and_ordering() -> None:
    with pytest.raises(ValueError):
        box(bbox_xyxy_abs=None)
    with pytest.raises(ValueError):
        box(bbox_xyxy_abs=(10.0, 10.0, 10.0, 40.0))
    with pytest.raises(ValueError):
        box(bbox_xyxy_abs=(10.0, 40.0, 60.0, 10.0))
    with pytest.raises(ValueError):
        box(bbox_xyxy_abs=(-1.0, 0.0, 5.0, 5.0))
    with pytest.raises(ValueError):
        box(bbox_xyxy_abs=(0.0, 0.0, float("nan"), 5.0))


def test_classification_must_not_have_box() -> None:
    Annotation(ann_id="c", image_id=IMG_A, category_id=1, kind="classification")
    with pytest.raises(ValueError):
        Annotation(
            ann_id="c",
            image_id=IMG_A,
            category_id=1,
            kind="classification",
            bbox_xyxy_abs=(0.0, 0.0, 1.0, 1.0),
        )


def test_human_annotation_has_no_score() -> None:
    with pytest.raises(ValueError):
        box(score=0.9)
    assert box(score=0.9, provenance="model").score == 0.9


def test_attributes_hold_scalars_only() -> None:
    assert box(attributes={"difficult": True, "n": 3, "w": 0.5, "s": "x"}).attributes["n"] == 3
    with pytest.raises(ValueError):
        box(attributes={"nested": {"a": 1}})


# --- Category / manifest / plugin spec ---------------------------------------------------------
def test_category_name_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        Category(id=1, name="")


def test_manifest_version_is_32_hex_and_schema_version_positive() -> None:
    ok: dict[str, Any] = {
        "version": "0" * 32,
        "schema_version": 1,
        "name": "d",
        "task": "detection",
        "categories": CATS,
        "image_count": 1,
        "annotation_count": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    assert DatasetManifest(**ok).lineage == ()
    with pytest.raises(ValueError):
        DatasetManifest(**{**ok, "version": "xyz"})
    with pytest.raises(ValueError):
        DatasetManifest(**{**ok, "schema_version": 0})
    with pytest.raises(ValueError):
        DatasetManifest(**{**ok, "created_at": "2026-01-01T00:00:00"})  # naive datetime


def test_plugin_spec_fields() -> None:
    spec = PluginSpec(
        name="coco",
        kind="format",
        license="Apache-2.0",
        extra=None,
        gpu_memory_gb=None,
        version="1",
    )
    assert spec.name == "coco"
    with pytest.raises(ValueError):
        PluginSpec(
            name="Bad Name", kind="format", license="x", extra=None, gpu_memory_gb=None, version="1"
        )


# --- cross-row invariants (validate_tables) ----------------------------------------------------
def test_valid_tables_pass() -> None:
    validate_tables([image()], [box()], CATS, task="detection")


def test_unknown_image_is_rejected_naming_the_annotation() -> None:
    with pytest.raises(ValidationError, match="ann_id '7'.*image"):
        validate_tables([image()], [box("7", IMG_B)], CATS, task="detection")


def test_unknown_category_is_rejected_naming_the_annotation() -> None:
    with pytest.raises(ValidationError, match="ann_id '7'.*category"):
        validate_tables([image()], [box("7", category_id=99)], CATS, task="detection")


@pytest.mark.parametrize(
    "bbox",
    [(10.0, 10.0, 100.1, 40.0), (10.0, 10.0, 60.0, 50.5), (0.0, 0.0, 101.0, 10.0)],
)
def test_box_outside_image_is_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValidationError, match="ann_id '1'.*outside"):
        validate_tables([image()], [box(bbox_xyxy_abs=bbox)], CATS, task="detection")


def test_box_touching_image_edge_is_valid() -> None:
    validate_tables([image()], [box(bbox_xyxy_abs=(0.0, 0.0, 100.0, 50.0))], CATS, task="detection")


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate image_id"):
        validate_tables([image(), image()], [], CATS, task="detection")
    with pytest.raises(ValidationError, match="duplicate ann_id"):
        validate_tables([image()], [box("1"), box("1")], CATS, task="detection")
    with pytest.raises(ValidationError, match="duplicate category id"):
        validate_tables(
            [image()], [], [Category(id=1, name="a"), Category(id=1, name="b")], task="detection"
        )
    with pytest.raises(ValidationError, match="duplicate category name"):
        validate_tables(
            [image()], [], [Category(id=1, name="a"), Category(id=2, name="a")], task="detection"
        )


def test_task_governs_annotation_kind() -> None:
    cls = Annotation(ann_id="c", image_id=IMG_A, category_id=1, kind="classification")
    validate_tables([image()], [cls], CATS, task="classification")
    with pytest.raises(ValidationError, match="task"):
        validate_tables([image()], [cls], CATS, task="detection")
    with pytest.raises(ValidationError, match="task"):
        validate_tables([image()], [box()], CATS, task="classification")


def test_all_problems_are_reported_up_to_a_limit() -> None:
    anns = [box(str(i), IMG_B) for i in range(20)]
    with pytest.raises(ValidationError) as exc:
        validate_tables([image()], anns, CATS, task="detection")
    assert "ann_id '0'" in str(exc.value) and "and 15 more" in str(exc.value)
