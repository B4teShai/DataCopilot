from pathlib import Path

import pytest
from conftest import dataset_from, make_dataset, table_rows
from fixtures import synth

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files
from datacopilot.io.imagefolder import plugin


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return synth.build_imagefolder(tmp_path / "if")


def img(tmp_path: Path, name: str, seed: int, **kw: object) -> Image:
    p = tmp_path / "src" / name
    synth.write_png(p, 8, 8, seed)
    fields: dict[str, object] = {
        "image_id": files.image_id(p),
        "file_path": str(p),
        "width": 8,
        "height": 8,
    }
    return Image(**{**fields, **kw})  # type: ignore[arg-type]


def label(image: Image, cat: int, ann_id: str = "1", **kw: object) -> Annotation:
    return Annotation(
        ann_id=ann_id, image_id=image.image_id, category_id=cat, kind="classification", **kw
    )  # type: ignore[arg-type]


# ------------------------------------------------------------------ reader
def test_reads_split_layout(folder: Path) -> None:
    r = plugin.read(folder)
    assert r.task == "classification" and r.dropped == {} and r.skipped == {}
    assert [(c.id, c.name) for c in r.categories] == [(0, "cat"), (1, "dog")]
    assert len(r.images) == 5 and len(r.annotations) == 5
    by_name = {i.source["file_name"]: i for i in r.images}
    assert by_name["c1.png"].split == "train" and by_name["c3.png"].split == "val"
    ann = next(a for a in r.annotations if a.image_id == by_name["d1.png"].image_id)
    assert ann.kind == "classification" and ann.category_id == 1 and ann.bbox_xyxy_abs is None
    assert ann.ann_id == "train/dog/d1.png" and ann.source == {"format": "imagefolder"}


def test_reads_class_only_layout_with_empty_classes(tmp_path: Path) -> None:
    root = synth.build_imagefolder(tmp_path / "flat", split_dirs=False)
    (root / "zebra").mkdir()
    (root / ".hidden").mkdir()
    r = plugin.read(root)
    assert [c.name for c in r.categories] == ["cat", "dog", "zebra"]
    assert {i.split for i in r.images} == {"unassigned"}
    assert r.images[0].source["format"] == "imagefolder"


def test_loose_files_unsupported_images_and_nested_dirs(tmp_path: Path) -> None:
    root = synth.build_imagefolder(tmp_path / "r", split_dirs=False)
    synth.write_png(root / "stray.png", 8, 8, 99)
    synth.write_png(root / "cat" / "sub" / "deep.png", 8, 8, 98)
    (root / "cat" / "notes.txt").write_text("hi")
    r = plugin.read(root)
    assert r.skipped == {"images.without_class": 1}
    assert "sub/deep.png" in {i.source["file_name"] for i in r.images}


def test_a_class_dir_named_like_a_split_with_other_dirs_is_a_class(tmp_path: Path) -> None:
    root = tmp_path / "r"
    synth.write_png(root / "train" / "a.png", 8, 8, 1)
    synth.write_png(root / "cat" / "b.png", 8, 8, 2)
    r = plugin.read(root)
    assert {c.name for c in r.categories} == {"train", "cat"}


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="does not exist"):
        plugin.read(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FormatError, match="no class directories"):
        plugin.read(tmp_path / "empty")
    bad = tmp_path / "bad"
    (bad / "cat").mkdir(parents=True)
    (bad / "cat" / "x.png").write_bytes(b"not a png")
    with pytest.raises(FormatError, match="x.png"):
        plugin.read(bad)


def test_symlink_escaping_the_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "r"
    (root / "cat").mkdir(parents=True)
    secret = tmp_path / "outside.png"
    synth.write_png(secret, 8, 8, 7)
    (root / "cat" / "link.png").symlink_to(secret)
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


# ------------------------------------------------------------------ writer
def test_split_layout_round_trip_is_lossless_and_field_identical(
    folder: Path, tmp_path: Path
) -> None:
    first = plugin.read(folder)
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(make_dataset(first), out)
    second = plugin.read(out)
    assert report.dropped == {} and report.skipped == {} and report.target_format == "imagefolder"
    assert table_rows(first.images, first.annotations) == table_rows(
        second.images, second.annotations
    )
    assert first.categories == second.categories
    assert (out / "train" / "cat" / "c1.png").is_file()


def test_class_only_round_trip_keeps_empty_classes(tmp_path: Path) -> None:
    root = synth.build_imagefolder(tmp_path / "flat", split_dirs=False)
    (root / "zebra").mkdir()
    first = plugin.read(root)
    out = tmp_path / "out"
    out.mkdir()
    plugin.write(make_dataset(first), out)
    assert (out / "zebra").is_dir() and not any((out / "zebra").iterdir())
    assert plugin.read(out).categories == first.categories


def test_foreign_rows_count_losses_and_skip_unlabelled_images(tmp_path: Path) -> None:
    a = img(
        tmp_path,
        "a.png",
        1,
        split="train",
        group_id="g",
        source={"format": "coco", "id": 1, "file_name": "x/a.png", "license": 2},
    )
    b = img(tmp_path, "b.png", 2, split="train")
    c = img(tmp_path, "c.png", 3, split="train")
    anns = [
        label(
            a,
            5,
            "1",
            is_crowd=True,
            score=0.5,
            provenance="model",
            attributes={"k": 1},
            source={"format": "coco", "id": 4},
        ),
        label(a, 6, "2"),
        label(c, 6, "3"),
    ]
    cats = [Category(id=5, name="zebra", aliases=("z",)), Category(id=6, name="ant")]
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(dataset_from([a, b, c], anns, cats, task="classification"), out)
    assert report.skipped == {"images.unlabelled": 1}
    assert report.dropped == {
        "images.group_id": 1,
        "images.source.id": 1,
        "images.source.license": 1,
        "annotations.is_crowd": 1,
        "annotations.score": 1,
        "annotations.provenance": 1,
        "annotations.attributes": 1,
        "annotations.source.id": 1,
        "annotations.multiple_labels": 1,
        "categories.id": 2,
        "categories.aliases": 1,
    }
    assert (out / "train" / "zebra" / "a.png").is_file() and (
        out / "train" / "ant" / "c.png"
    ).is_file()


def test_mixed_split_and_unassigned_images_are_an_error(tmp_path: Path) -> None:
    a = img(tmp_path, "a.png", 1, split="train")
    b = img(tmp_path, "b.png", 2)
    ds = dataset_from(
        [a, b],
        [label(a, 0, "1"), label(b, 0, "2")],
        [Category(id=0, name="x")],
        task="classification",
    )
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FormatError, match="split"):
        plugin.write(ds, out)


@pytest.mark.parametrize("name", ["../up", "a/b", "..", ".hidden", "a\\b"])
def test_unsafe_class_names_are_refused(tmp_path: Path, name: str) -> None:
    a = img(tmp_path, "a.png", 1)
    ds = dataset_from([a], [label(a, 0)], [Category(id=0, name=name)], task="classification")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FormatError, match="class name"):
        plugin.write(ds, out)
    assert not (tmp_path / "up").exists()


def test_writer_rejects_detection_and_non_empty_output(folder: Path, tmp_path: Path) -> None:
    ds = make_dataset(plugin.read(folder))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(FormatError, match="not empty"):
        plugin.write(ds, out)
    a = img(tmp_path, "z.png", 9)
    box = Annotation(
        ann_id="1",
        image_id=a.image_id,
        category_id=0,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 1.0, 2.0, 2.0),
    )
    det = dataset_from([a], [box], [Category(id=0, name="x")])
    out2 = tmp_path / "out2"
    out2.mkdir()
    with pytest.raises(FormatError, match="classification"):
        plugin.write(det, out2)


def test_same_basename_in_one_class_does_not_collide(tmp_path: Path) -> None:
    a = img(tmp_path / "p1", "same.png", 1)
    b = img(tmp_path / "p2", "same.png", 2)
    ds = dataset_from(
        [a, b],
        [label(a, 0, "1"), label(b, 0, "2")],
        [Category(id=0, name="x")],
        task="classification",
    )
    out = tmp_path / "out"
    out.mkdir()
    plugin.write(ds, out)
    assert len(list((out / "x").iterdir())) == 2


def test_an_image_in_an_unsupported_format_is_refused_not_skipped(tmp_path: Path) -> None:
    root = synth.build_imagefolder(tmp_path / "r", split_dirs=False)
    (root / "cat" / "x.webp").write_bytes(b"RIFF")
    with pytest.raises(FormatError, match=r"x\.webp: unsupported image format"):
        plugin.read(root)
