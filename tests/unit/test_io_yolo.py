import json
from pathlib import Path

import pytest
from conftest import dataset_from, make_dataset, table_rows
from fixtures import synth

from datacopilot.core.errors import FormatError, ValidationError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files
from datacopilot.io.yolo import plugin


@pytest.fixture
def yolo(tmp_path: Path) -> Path:
    return synth.build_yolo(tmp_path / "yolo")


def one_image(root: Path, label: str | None, yaml: str | None = None, split: str = "train") -> Path:
    synth.write_png(root / "images" / split / "a.png", 100, 50, 1)
    if label is not None:
        (root / "labels" / split).mkdir(parents=True)
        (root / "labels" / split / "a.txt").write_text(label)
    (root / "data.yaml").write_text(yaml or "names: [cat, dog]\n")
    return root


# ------------------------------------------------------------------ reader
def test_reads_fixture(yolo: Path) -> None:
    r = plugin.read(yolo)
    assert r.task == "detection" and r.dropped == {} and r.skipped == {}
    assert [c.name for c in r.categories] == ["cat", "dog", "bird"]
    assert [c.id for c in r.categories] == [0, 1, 2]
    assert len(r.images) == 4 and len(r.annotations) == 4
    by_name = {i.source["file_name"]: i for i in r.images}
    assert {k: v.split for k, v in by_name.items()} == {
        "a.png": "train",
        "b.png": "train",
        "c.png": "val",
        "d.png": "test",
    }
    assert by_name["c.png"].source == {"format": "yolo", "file_name": "c.png"}


def test_normalised_boxes_become_absolute_xyxy(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", "1 0.5 0.5 0.5 1\n")
    (ann,) = plugin.read(root).annotations
    assert ann.bbox_xyxy_abs == (25.0, 0.0, 75.0, 50.0)
    assert ann.category_id == 1 and ann.provenance == "human" and ann.score is None
    assert ann.ann_id == "train/a.png:0" and ann.source == {"format": "yolo"}


def test_confidence_column_becomes_score_and_model_provenance(tmp_path: Path) -> None:
    (ann,) = plugin.read(one_image(tmp_path / "y", "0 0.5 0.5 0.2 0.2 0.875\n")).annotations
    assert ann.score == 0.875 and ann.provenance == "model"


def test_missing_label_file_means_background_image(tmp_path: Path) -> None:
    r = plugin.read(one_image(tmp_path / "y", None))
    assert len(r.images) == 1 and r.annotations == []


def test_blank_lines_are_ignored_and_flat_layout_has_no_split(tmp_path: Path) -> None:
    root = tmp_path / "y"
    synth.write_png(root / "images" / "a.png", 100, 50, 1)
    (root / "labels").mkdir()
    (root / "labels" / "a.txt").write_text("\n0 0.5 0.5 0.2 0.2\n\n")
    (root / "data.yaml").write_text("names:\n  0: cat\n")
    r = plugin.read(root)
    assert r.images[0].split == "unassigned" and len(r.annotations) == 1


def test_float_noise_at_the_border_is_snapped_not_rejected(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", "0 0.5 0.5 1.0000000001 1\n")
    (ann,) = plugin.read(root).annotations
    assert ann.bbox_xyxy_abs == (0.0, 0.0, 100.0, 50.0)


def test_extra_yaml_keys_are_counted(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", "", "names: [a]\nroboflow: x\nnc: 1\n")
    assert plugin.read(root).dropped == {"dataset.roboflow": 1}


@pytest.mark.parametrize("suffix", [".webp", ".tif", ".tiff"])
def test_an_image_in_an_unsupported_format_is_refused_not_skipped(
    tmp_path: Path, suffix: str
) -> None:
    root = one_image(tmp_path / "y", "")
    (root / "images" / "train" / f"z{suffix}").write_bytes(b"RIFF")
    with pytest.raises(FormatError, match=rf"z{suffix}: unsupported image format"):
        plugin.read(root)


@pytest.mark.parametrize(
    ("label", "match"),
    [
        ("5 0.5 0.5 0.2 0.2\n", "class 5"),
        ("0 0.5 0.5 0.2\n", "a.txt:1"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4\n", "polygon"),
        ("0 a b c d\n", "a.txt:1"),
        ("0 nan 0.5 0.2 0.2\n", "a.txt:1"),
        ("0 0.5 0.5 3 0.2\n", "outside"),
        ("-1 0.5 0.5 0.2 0.2\n", "class -1"),
    ],
)
def test_invalid_labels_name_the_file_and_line(tmp_path: Path, label: str, match: str) -> None:
    with pytest.raises(FormatError, match=match):
        plugin.read(one_image(tmp_path / "y", label))


def test_zero_width_box_is_a_validation_error_naming_the_image(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="train/a.png"):
        plugin.read(one_image(tmp_path / "y", "0 0.5 0.5 0 0.2\n"))


@pytest.mark.parametrize(
    ("yaml", "match"),
    [
        (None, "data.yaml"),
        ("path: .\n", "names"),
        ("names: [a, a]\n", "duplicate"),
        ("names: [a]\nnc: 2\n", "nc"),
        ("names:\n  0: a\n  2: b\n", "contiguous"),
        ("names: 5\n", "names"),
        ("names: [1]\n", "names"),
    ],
)
def test_invalid_data_yaml(tmp_path: Path, yaml: str | None, match: str) -> None:
    root = one_image(tmp_path / "y", "")
    if yaml is None:
        (root / "data.yaml").unlink()
    else:
        (root / "data.yaml").write_text(yaml)
    with pytest.raises(FormatError, match=match):
        plugin.read(root)


def test_missing_root_or_images_dir(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="does not exist"):
        plugin.read(tmp_path / "nope")
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "data.yaml").write_text("names: [a]\n")
    with pytest.raises(FormatError, match="images"):
        plugin.read(tmp_path / "e")


def test_label_symlink_outside_root_is_rejected(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", None)
    secret = tmp_path / "secret.txt"
    secret.write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").symlink_to(secret)
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


def test_oversized_label_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = one_image(tmp_path / "y", "0 0.5 0.5 0.2 0.2\n")
    monkeypatch.setattr(files, "MAX_TEXT_BYTES", 5)
    with pytest.raises(FormatError, match="larger than"):
        plugin.read(root)


# ------------------------------------------------------------------ writer
def roundtrip(src: Path, out: Path):  # type: ignore[no-untyped-def]
    first = plugin.read(src)
    out.mkdir(parents=True)
    report = plugin.write(make_dataset(first), out)
    return first, plugin.read(out), report


def test_same_format_round_trip_is_lossless_and_field_identical(yolo: Path, tmp_path: Path) -> None:
    first, second, report = roundtrip(yolo, tmp_path / "out")
    assert report.dropped == {} and report.is_lossless() and report.target_format == "yolo"
    assert table_rows(first.images, first.annotations) == table_rows(
        second.images, second.annotations
    )
    assert first.categories == second.categories


def test_writer_layout_and_files(yolo: Path, tmp_path: Path) -> None:
    roundtrip(yolo, tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "images" / "train" / "a.png").is_file()
    assert (out / "labels" / "val" / "c.txt").read_text() == ""
    data = (out / "data.yaml").read_text()
    assert 'names: ["cat", "dog", "bird"]' in data and "nc: 3" in data
    assert "train: images/train" in data and "test: images/test" in data
    assert (out / "labels" / "train" / "a.txt").read_text().splitlines()[0].startswith("0 ")


def test_score_is_written_as_confidence_and_round_trips(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", "0 0.5 0.5 0.2 0.2 0.1\n")
    first, second, report = roundtrip(root, tmp_path / "out")
    assert second.annotations[0].score == 0.1 and report.is_lossless()


def test_precision_loss_is_counted_only_beyond_a_milli_pixel(tmp_path: Path) -> None:
    img_path = tmp_path / "p.png"
    synth.write_png(img_path, 3, 3, 1)
    img = Image(image_id=files.image_id(img_path), file_path=str(img_path), width=3, height=3)
    ok = Annotation(
        ann_id="1",
        image_id=img.image_id,
        category_id=0,
        kind="bbox",
        bbox_xyxy_abs=(0.0, 0.0, 1.0, 1.0),
    )
    assert ok.bbox_xyxy_abs
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(dataset_from([img], [ok], [Category(id=0, name="a")]), out)
    assert report.dropped == {}
    big = Image(image_id="1" * 16, file_path=str(img_path), width=2000000, height=2000000)
    shifted = Annotation(
        ann_id="2",
        image_id=big.image_id,
        category_id=0,
        kind="bbox",
        bbox_xyxy_abs=(0.123456789, 0.0, 1999999.9876543, 10.0),
    )
    out2 = tmp_path / "out2"
    out2.mkdir()
    report = plugin.write(dataset_from([big], [shifted], [Category(id=0, name="a")]), out2)
    assert report.dropped == {"annotations.bbox_xyxy_abs": 1}


def test_foreign_rows_count_every_loss(tmp_path: Path) -> None:
    img_path = tmp_path / "pic.png"
    synth.write_png(img_path, 40, 30, 5)
    img = Image(
        image_id=files.image_id(img_path),
        file_path=str(img_path),
        width=40,
        height=30,
        split="val",
        group_id="g",
        source={"format": "coco", "id": 3, "file_name": "dir/pic.png", "license": 1},
    )
    ann = Annotation(
        ann_id="1",
        image_id=img.image_id,
        category_id=7,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 2.0, 30.0, 20.0),
        is_crowd=True,
        occluded=True,
        truncated=False,
        attributes={"x": 1},
        provenance="model",
        source={"format": "coco", "id": 9, "area": 5.0},
    )
    cat = Category(id=7, name="thing", aliases=("t",), attributes={"sku": "1"})
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(dataset_from([img], [ann], [cat]), out)
    assert report.dropped == {
        "images.group_id": 1,
        "images.source.id": 1,
        "images.source.license": 1,
        "annotations.is_crowd": 1,
        "annotations.occluded": 1,
        "annotations.truncated": 1,
        "annotations.attributes": 1,
        "annotations.provenance": 1,
        "annotations.source.id": 1,
        "annotations.source.area": 1,
        "categories.id": 1,
        "categories.aliases": 1,
        "categories.attributes": 1,
    }
    assert (out / "images" / "val" / "pic.png").is_file()
    assert (out / "labels" / "val" / "pic.txt").is_file()


def test_stem_collisions_do_not_overwrite_labels(tmp_path: Path) -> None:
    images = []
    for n, ext in enumerate(["png", "bmp"]):
        p = tmp_path / f"a.{ext}"
        synth.write_png(p, 8, 8, n)
        images.append(Image(image_id=files.image_id(p), file_path=str(p), width=8, height=8))
    out = tmp_path / "out"
    out.mkdir()
    plugin.write(dataset_from(images, [], [Category(id=0, name="a")]), out)
    assert len(list((out / "labels").iterdir())) == 2
    assert len(plugin.read(out).images) == 2


def test_writer_guards(yolo: Path, tmp_path: Path) -> None:
    ds = make_dataset(plugin.read(yolo))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(FormatError, match="not empty"):
        plugin.write(ds, out)
    img = ds.images[0].model_copy(update={"source": {"format": "yolo", "file_name": "../../e.png"}})
    out2 = tmp_path / "out2"
    out2.mkdir()
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.write(dataset_from([img], [], list(ds.categories)), out2)
    assert json.dumps(sorted(p.name for p in tmp_path.iterdir())) == json.dumps(
        sorted(["yolo", "out", "out2"])
    )
