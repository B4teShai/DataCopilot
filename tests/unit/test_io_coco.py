import json
from pathlib import Path

import pytest
from conftest import dataset_from, make_dataset, table_rows
from fixtures import synth
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files
from datacopilot.io.coco import plugin


@pytest.fixture
def coco(tmp_path: Path) -> Path:
    return synth.build_coco(tmp_path / "coco")


def write_json(root: Path, doc: object, name: str = "annotations.json") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def minimal(root: Path, **over: object) -> Path:
    """One 64x48 image 'a.png' and one valid box; ``over`` replaces top-level keys."""
    synth.write_png(root / "images" / "a.png", 64, 48, 1)
    doc: dict[str, object] = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 48}],
        "annotations": [
            {"id": 7, "image_id": 1, "category_id": 1, "bbox": [8, 6, 32, 24], "iscrowd": 0}
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    doc.update(over)
    write_json(root, doc)
    return root


# ------------------------------------------------------------------ reader
def test_reads_a_multi_split_directory(coco: Path) -> None:
    r = plugin.read(coco)
    assert r.task == "detection" and r.dropped == {} and r.skipped == {}
    assert len(r.images) == 4 and len(r.annotations) == 4
    assert {c.name for c in r.categories} == {"cat", "dog", "bird"}
    splits = {i.source["file_name"]: i.split for i in r.images}
    assert splits == {"a.png": "train", "b.png": "train", "c.png": "val", "d.png": "test"}
    assert {a.ann_id for a in r.annotations} == {
        "instances_train:101",
        "instances_train:102",
        "instances_train:103",
        "instances_test:104",
    }


def test_maps_boxes_flags_and_keeps_unmapped_fields_verbatim(coco: Path) -> None:
    r = plugin.read(coco)
    ann = next(a for a in r.annotations if a.ann_id == "instances_train:102")
    assert ann.bbox_xyxy_abs == (10.0, 12.0, 64.0, 48.0)
    assert ann.is_crowd is True and ann.kind == "bbox" and ann.provenance == "human"
    assert ann.source == {
        "format": "coco",
        "id": 102,
        "area": 54.0 * 36.0,
        "segmentation": [],
    }
    img = next(i for i in r.images if i.source["file_name"] == "a.png")
    assert img.source == {
        "format": "coco",
        "id": 1,
        "file_name": "a.png",
        "license": 1,
        "date_captured": "2020-01-01",
    }
    assert (img.width, img.height, img.image_id) == (64, 48, files.image_id(Path(img.file_path)))
    cat = next(c for c in r.categories if c.name == "cat")
    assert cat.id == 1 and cat.attributes == {"supercategory": "animal"}


def test_single_file_and_flat_directory_layouts(tmp_path: Path) -> None:
    root = minimal(tmp_path / "flat")
    for target in (root, root / "annotations.json"):
        r = plugin.read(target)
        assert [a.ann_id for a in r.annotations] == ["7"] and r.images[0].split == "unassigned"


def test_image_next_to_the_json_is_found(tmp_path: Path) -> None:
    root = tmp_path / "x"
    synth.write_png(root / "a.png", 64, 48, 1)
    write_json(root, json.loads((minimal(tmp_path / "y") / "annotations.json").read_text()))
    assert len(plugin.read(root).images) == 1


def test_missing_image_files_are_skipped_and_counted(tmp_path: Path) -> None:
    root = minimal(tmp_path / "m")
    (root / "images" / "a.png").unlink()
    r = plugin.read(root)
    assert r.images == [] and r.annotations == []
    assert r.skipped == {"images.file_missing": 1, "annotations.of_missing_image": 1}


def test_declared_size_mismatch_uses_the_file_and_is_counted(tmp_path: Path) -> None:
    root = minimal(
        tmp_path / "m", images=[{"id": 1, "file_name": "a.png", "width": 5, "height": 5}]
    )
    r = plugin.read(root)
    assert (r.images[0].width, r.images[0].height) == (64, 48)
    assert r.dropped == {"images.declared_size": 1}


def test_unmapped_category_keys_top_level_keys_and_group_id(tmp_path: Path) -> None:
    root = minimal(
        tmp_path / "m",
        categories=[{"id": 1, "name": "cat", "color": "red", "supercategory": "pet"}],
        info={"year": 2020},
        licenses=[],
        images=[{"id": 1, "file_name": "a.png", "width": 64, "height": 48, "group_id": "g1"}],
    )
    r = plugin.read(root)
    assert r.dropped == {"categories.color": 1, "dataset.info": 1, "dataset.licenses": 1}
    assert r.images[0].group_id == "g1" and "group_id" not in r.images[0].source


def test_score_marks_model_provenance(tmp_path: Path) -> None:
    root = minimal(tmp_path / "m")
    doc = json.loads((root / "annotations.json").read_text())
    doc["annotations"][0]["score"] = 0.9
    write_json(root, doc)
    a = plugin.read(root).annotations[0]
    assert a.score == 0.9 and a.provenance == "model" and "score" not in a.source


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"images": "nope"}, "'images' must be a list"),
        ({"annotations": 3}, "'annotations' must be a list"),
        ({"categories": None}, "'categories' must be a list"),
        ({"images": [{"id": 1}]}, "file_name"),
        (
            {"images": [{"id": 1, "file_name": "a.png"}, {"id": 1, "file_name": "a.png"}]},
            "duplicate image id 1",
        ),
        ({"categories": [{"id": 1}]}, "category.*name"),
        (
            {"annotations": [{"id": 7, "image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]}]},
            "unknown image id 99",
        ),
        ({"annotations": [{"id": 7, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1]}]}, "bbox"),
        (
            {"annotations": [{"id": 7, "image_id": 1, "category_id": 1, "bbox": [0, 0, "a", 1]}]},
            "bbox",
        ),
        ({"annotations": [{"id": 7, "image_id": 1, "category_id": 1}]}, "bbox"),
        ({"images": [{"id": 1, "file_name": "../evil.png"}]}, "unsafe path"),
        ({"images": [{"id": 1, "file_name": "/etc/passwd"}]}, "unsafe path"),
    ],
)
def test_invalid_files_give_one_clear_format_error(
    tmp_path: Path, over: dict[str, object], match: str
) -> None:
    with pytest.raises(FormatError, match=match):
        plugin.read(minimal(tmp_path / "bad", **over))


@pytest.mark.parametrize(
    "content", ["{not json", "[]", '{"images": [], "annotations": [], "categories": [], "x": NaN}']
)
def test_malformed_json_is_a_format_error(tmp_path: Path, content: str) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    (root / "annotations.json").write_text(content)
    with pytest.raises(FormatError, match="annotations.json"):
        plugin.read(root)


def test_missing_path_and_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="does not exist"):
        plugin.read(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FormatError, match="no COCO annotation file"):
        plugin.read(tmp_path / "empty")


def test_oversized_annotation_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = minimal(tmp_path / "m")
    monkeypatch.setattr(files, "MAX_JSON_BYTES", 10)
    with pytest.raises(FormatError, match="larger than"):
        plugin.read(root)


def test_symlink_pointing_outside_the_root_is_rejected(tmp_path: Path) -> None:
    root = minimal(tmp_path / "m")
    secret = tmp_path / "secret.png"
    synth.write_png(secret, 4, 4, 9)
    (root / "images" / "a.png").unlink()
    (root / "images" / "a.png").symlink_to(secret)
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


def test_box_outside_image_is_reported_by_validation_not_the_reader(tmp_path: Path) -> None:
    root = minimal(tmp_path / "m")
    doc = json.loads((root / "annotations.json").read_text())
    doc["annotations"][0]["bbox"] = [60, 40, 20, 20]
    write_json(root, doc)
    r = plugin.read(root)
    with pytest.raises(ValidationError, match="ann_id '7'.*outside"):
        make_dataset(r)


def test_degenerate_box_names_the_annotation(tmp_path: Path) -> None:
    root = minimal(tmp_path / "m")
    doc = json.loads((root / "annotations.json").read_text())
    doc["annotations"][0]["bbox"] = [5, 5, 0, 10]
    write_json(root, doc)
    with pytest.raises(ValidationError, match="annotation 7"):
        plugin.read(root)


# ------------------------------------------------------------------ writer
def roundtrip(src: Path, out: Path) -> tuple[object, object]:
    first = plugin.read(src)
    ds = make_dataset(first)
    out.mkdir(parents=True)
    report = plugin.write(ds, out)
    second = plugin.read(out)
    return (first, second), report


def test_same_format_round_trip_is_lossless_and_field_identical(coco: Path, tmp_path: Path) -> None:
    (first, second), report = roundtrip(coco, tmp_path / "out")  # type: ignore[misc]
    assert report.dropped == {} and report.is_lossless()  # type: ignore[attr-defined]
    assert report.rows_written == 8 and report.target_format == "coco"  # type: ignore[attr-defined]
    assert table_rows(first.images, first.annotations) == table_rows(
        second.images, second.annotations
    )  # type: ignore[attr-defined]
    assert sorted(first.categories, key=lambda c: c.id) == sorted(
        second.categories, key=lambda c: c.id
    )  # type: ignore[attr-defined]


def test_writer_layout_per_split_and_images_are_copied(coco: Path, tmp_path: Path) -> None:
    roundtrip(coco, tmp_path / "out")
    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "annotations").iterdir()) == [
        "instances_test.json",
        "instances_train.json",
        "instances_val.json",
    ]
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "a.png",
        "b.png",
        "c.png",
        "d.png",
    ]


def test_unsplit_dataset_writes_a_single_annotations_json(tmp_path: Path) -> None:
    src = minimal(tmp_path / "src")
    _, report = roundtrip(src, tmp_path / "out")
    assert (tmp_path / "out" / "annotations.json").is_file()
    assert not (tmp_path / "out" / "annotations").exists() and report.is_lossless()  # type: ignore[attr-defined]


def test_float_boxes_survive_exactly(tmp_path: Path) -> None:
    root = minimal(tmp_path / "src")
    doc = json.loads((root / "annotations.json").read_text())
    doc["annotations"][0]["bbox"] = [0.1, 0.7, 33.3, 20.1]
    write_json(root, doc)
    (first, second), _ = roundtrip(root, tmp_path / "out")  # type: ignore[misc]
    assert first.annotations[0].bbox_xyxy_abs == second.annotations[0].bbox_xyxy_abs  # type: ignore[attr-defined]


def make_foreign(tmp_path: Path) -> tuple[Path, list[Image], list[Annotation], list[Category]]:
    img_path = tmp_path / "pic.png"
    synth.write_png(img_path, 40, 30, 5)
    img = Image(
        image_id=files.image_id(img_path),
        file_path=str(img_path),
        width=40,
        height=30,
        split="val",
        group_id="g",
        source={"format": "yolo", "file_name": "pic.png", "folder": "x"},
    )
    ann = Annotation(
        ann_id="1",
        image_id=img.image_id,
        category_id=5,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 2.0, 30.0, 20.0),
        occluded=True,
        truncated=False,
        attributes={"difficult": True},
        provenance="proposal_accepted",
        source={"format": "voc", "pose": "Left"},
    )
    cat = Category(
        id=5, name="thing", aliases=("t",), attributes={"supercategory": "s", "sku": "9"}
    )
    return img_path, [img], [ann], [cat]


def test_foreign_rows_are_written_and_every_loss_is_counted(tmp_path: Path) -> None:
    _, images, anns, cats = make_foreign(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(dataset_from(images, anns, cats), out)
    assert report.dropped == {
        "annotations.occluded": 1,
        "annotations.truncated": 1,
        "annotations.attributes": 1,
        "annotations.provenance": 1,
        "annotations.source.pose": 1,
        "images.source.folder": 1,
        "categories.aliases": 1,
        "categories.attributes": 1,
    }
    back = plugin.read(out)
    assert back.images[0].group_id == "g" and back.images[0].split == "val"
    assert back.categories[0].attributes == {"supercategory": "s"}
    assert back.annotations[0].bbox_xyxy_abs == (1.0, 2.0, 30.0, 20.0)


def test_writer_refuses_a_non_empty_output_directory(coco: Path, tmp_path: Path) -> None:
    ds = make_dataset(plugin.read(coco))
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FormatError, match="not empty"):
        plugin.write(ds, out)


def test_writer_rejects_classification_datasets(tmp_path: Path) -> None:
    img_path = tmp_path / "p.png"
    synth.write_png(img_path, 8, 8, 1)
    img = Image(image_id=files.image_id(img_path), file_path=str(img_path), width=8, height=8)
    cls = Annotation(ann_id="1", image_id=img.image_id, category_id=1, kind="classification")
    ds = dataset_from([img], [cls], [Category(id=1, name="a")], task="classification")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FormatError, match="detection"):
        plugin.write(ds, out)


def test_writer_refuses_a_hostile_file_name_in_source(tmp_path: Path) -> None:
    img_path, images, anns, cats = make_foreign(tmp_path)
    evil = images[0].model_copy(
        update={"source": {"format": "coco", "id": 1, "file_name": "../../evil.png"}}
    )
    out = tmp_path / "deep" / "out"
    out.mkdir(parents=True)
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.write(dataset_from([evil], anns, cats), out)
    assert not (tmp_path / "evil.png").exists() and not (tmp_path / "deep" / "evil.png").exists()


def test_writer_fails_clearly_when_the_source_image_is_gone(tmp_path: Path) -> None:
    img_path, images, anns, cats = make_foreign(tmp_path)
    img_path.unlink()
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FormatError, match="missing"):
        plugin.write(dataset_from(images, anns, cats), out)


def test_same_basename_from_different_folders_does_not_collide(tmp_path: Path) -> None:
    images, anns = [], []
    for n, folder in enumerate(["x", "y"]):
        p = tmp_path / folder / "same.png"
        synth.write_png(p, 8, 8, n)
        images.append(
            Image(
                image_id=files.image_id(p),
                file_path=str(p),
                width=8,
                height=8,
                source={"format": "yolo", "file_name": "same.png"},
            )
        )
    out = tmp_path / "out"
    out.mkdir()
    plugin.write(dataset_from(images, anns, [Category(id=1, name="a")]), out)
    assert len(list((out / "images").iterdir())) == 2
    assert len(plugin.read(out).images) == 2


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    x=st.floats(0, 30, allow_nan=False),
    y=st.floats(0, 20, allow_nan=False),
    w=st.floats(0.001, 30, allow_nan=False),
    h=st.floats(0.001, 20, allow_nan=False),
)
def test_any_float_box_round_trips_exactly(
    tmp_path_factory: pytest.TempPathFactory, x: float, y: float, w: float, h: float
) -> None:
    base = tmp_path_factory.mktemp("prop")
    root = minimal(base / "src")
    doc = json.loads((root / "annotations.json").read_text())
    doc["annotations"][0]["bbox"] = [x, y, w, h]
    write_json(root, doc)
    (first, second), report = roundtrip(root, base / "out")  # type: ignore[misc]
    assert first.annotations[0].bbox_xyxy_abs == second.annotations[0].bbox_xyxy_abs  # type: ignore[attr-defined]
    assert report.is_lossless()  # type: ignore[attr-defined]


def test_deeply_nested_json_is_a_format_error_not_a_crash(tmp_path: Path) -> None:
    root = tmp_path / "deep"
    root.mkdir()
    (root / "annotations.json").write_text("[" * 200_000 + "]" * 200_000)
    with pytest.raises(FormatError, match="annotations.json"):
        plugin.read(root)


def test_deeply_nested_source_value_is_rejected_cleanly(tmp_path: Path) -> None:
    root = minimal(tmp_path / "deep2")
    doc = json.loads((root / "annotations.json").read_text())
    nested: object = "x"
    for _ in range(300):
        nested = [nested]
    doc["images"][0]["extra"] = nested
    write_json(root, doc)
    with pytest.raises((FormatError, ValidationError)):
        plugin.read(root)
