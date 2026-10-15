"""Golden round trips for every format and the cross-format dropped-field matrix.

The matrix below is the published contract (docs/architecture/03-data-model.md): a change to
any value is a deliberate, reviewed change to what a conversion can lose.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import table_rows
from fixtures import synth

from datacopilot import convert
from datacopilot.core.errors import ConversionLossError, FormatError
from datacopilot.io import registry
from datacopilot.store import DatasetStore

BUILD: dict[str, Callable[[Path], Path]] = {
    "coco": synth.build_coco,
    "yolo": synth.build_yolo,
    "voc": synth.build_voc,
    "cvat_xml": synth.build_cvat_xml,
    "imagefolder": synth.build_imagefolder,
}
DETECTION = ("coco", "yolo", "voc", "cvat_xml")

_COCO_LOSS = {
    "annotations.is_crowd": 1,
    "annotations.source.area": 4,
    "annotations.source.id": 4,
    "annotations.source.segmentation": 4,
    "categories.attributes": 3,
    "categories.id": 3,
    "images.source.date_captured": 4,
    "images.source.id": 4,
    "images.source.license": 4,
}
_VOC_LOSS = {
    "annotations.attributes": 4,
    "annotations.source.pose": 4,
    "annotations.truncated": 4,
    "images.source.depth": 4,
    "images.source.folder": 4,
    "images.source.segmented": 4,
}
_CVAT_LOSS = {
    "annotations.attributes": 1,
    "annotations.source.z_order": 4,
    "categories.attributes": 3,
    "images.source.id": 4,
}
# (source format, target format) -> dropped column counts for the synthetic fixtures
MATRIX: dict[tuple[str, str], dict[str, int]] = {
    ("coco", "coco"): {},
    ("coco", "yolo"): _COCO_LOSS,
    ("coco", "voc"): _COCO_LOSS,
    ("coco", "cvat_xml"): _COCO_LOSS,
    ("yolo", "coco"): {},
    ("yolo", "yolo"): {},
    ("yolo", "voc"): {},
    ("yolo", "cvat_xml"): {},
    ("voc", "coco"): _VOC_LOSS,
    ("voc", "yolo"): _VOC_LOSS,
    ("voc", "voc"): {},
    # VOC attributes are carried (as text); the other columns are lost as for COCO and YOLO
    ("voc", "cvat_xml"): {
        **{k: v for k, v in _VOC_LOSS.items() if k != "annotations.attributes"},
        "annotations.attributes_stringified": 4,
    },
    ("cvat_xml", "coco"): {**_CVAT_LOSS, "annotations.occluded": 4},
    ("cvat_xml", "yolo"): {**_CVAT_LOSS, "annotations.occluded": 4},
    ("cvat_xml", "voc"): _CVAT_LOSS,
    ("cvat_xml", "cvat_xml"): {},
}


@pytest.fixture(params=sorted(BUILD))
def fmt(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_every_builtin_format_is_registered_for_read_and_write() -> None:
    infos = {i.spec.name: i for i in registry.list_formats()}
    assert sorted(infos) == sorted(BUILD)
    assert all(i.can_read and i.can_write for i in infos.values())


def test_read_write_read_is_field_identical_and_lossless(fmt: str, tmp_path: Path) -> None:
    source = BUILD[fmt](tmp_path / "src")
    first = convert.ingest(source, fmt, tmp_path / "a.dcp")
    assert first.report.is_lossless()
    ds_a = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds_a, fmt, tmp_path / "out", strict=True)
    assert report.dropped == {} and report.skipped == {} and report.total_dropped == 0
    second = convert.ingest(tmp_path / "out", fmt, tmp_path / "b.dcp")
    ds_b = DatasetStore.open(tmp_path / "b.dcp").read()
    assert table_rows(ds_a.images, ds_a.annotations) == table_rows(ds_b.images, ds_b.annotations)
    assert sorted(ds_a.categories, key=lambda c: c.id) == sorted(
        ds_b.categories, key=lambda c: c.id
    )
    assert first.manifest.version == second.manifest.version


@pytest.mark.parametrize(("source", "target"), sorted(MATRIX))
def test_cross_format_dropped_field_matrix(source: str, target: str, tmp_path: Path) -> None:
    convert.ingest(BUILD[source](tmp_path / "src"), source, tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds, target, tmp_path / "out")
    assert report.dropped == MATRIX[(source, target)] and report.skipped == {}
    if MATRIX[(source, target)]:
        with pytest.raises(ConversionLossError):
            convert.export(ds, target, tmp_path / "strict", strict=True)
        assert not (tmp_path / "strict").exists()
    # whatever was kept still reads back with the same images and boxes
    again = convert.ingest(tmp_path / "out", target, tmp_path / "b.dcp")
    ds2 = DatasetStore.open(tmp_path / "b.dcp").read()
    assert again.manifest.image_count == ds.manifest.image_count
    assert {i.image_id: i.split for i in ds.images} == {i.image_id: i.split for i in ds2.images}
    boxes = sorted((a.image_id, a.bbox_xyxy_abs) for a in ds.annotations)
    boxes2 = sorted((a.image_id, a.bbox_xyxy_abs) for a in ds2.annotations)
    assert len(boxes) == len(boxes2)
    for (id_a, box_a), (id_b, box_b) in zip(boxes, boxes2, strict=True):
        assert id_a == id_b and box_a is not None and box_b is not None
        assert box_a == pytest.approx(box_b, abs=1e-3)


@pytest.mark.parametrize("target", DETECTION)
def test_classification_data_cannot_be_written_as_detection_formats(
    target: str, tmp_path: Path
) -> None:
    convert.ingest(BUILD["imagefolder"](tmp_path / "src"), "imagefolder", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    with pytest.raises(FormatError, match="detection"):
        convert.export(ds, target, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("source", DETECTION)
def test_detection_data_cannot_be_written_as_imagefolder(source: str, tmp_path: Path) -> None:
    convert.ingest(BUILD[source](tmp_path / "src"), source, tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    with pytest.raises(FormatError, match="classification"):
        convert.export(ds, "imagefolder", tmp_path / "out")


def test_ingest_twice_gives_byte_identical_tables(tmp_path: Path) -> None:
    src = synth.build_coco(tmp_path / "src")
    convert.ingest(src, "coco", tmp_path / "a.dcp")
    convert.ingest(src, "coco", tmp_path / "b.dcp")
    for name in ("images.parquet", "annotations.parquet", "categories.parquet"):
        assert (tmp_path / "a.dcp" / name).read_bytes() == (tmp_path / "b.dcp" / name).read_bytes()


def test_synthetic_fixtures_are_tiny(tmp_path: Path) -> None:
    for build in BUILD.values():
        root = build(tmp_path / build.__name__)
        sizes = [p.stat().st_size for p in root.rglob("*") if p.is_file()]
        assert max(sizes) < 100 * 1024
