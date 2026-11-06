"""YOLO-OBB round trips (ADR-0011): the label text comes back byte for byte, corner order included,
and ingest, export, ingest is a fixed point of the version."""

import random
from pathlib import Path

import pytest
from fixtures import synth

from datacopilot import convert
from datacopilot.core.errors import ConversionLossError, FormatError
from datacopilot.store import DatasetStore


def _labels(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text()
        for p in sorted((root / "labels").rglob("*"))
        if p.is_file()
    }


def test_label_text_round_trips_byte_for_byte(tmp_path: Path) -> None:
    src = synth.build_yolo_obb(tmp_path / "src")
    first = convert.ingest(src, "yolo_obb", tmp_path / "a.dcp")
    assert first.report.is_lossless() and first.manifest.schema_version == 2
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    assert {a.kind for a in ds.annotations} == {"obb"}
    report = convert.export(ds, "yolo_obb", tmp_path / "out", strict=True)
    assert report.dropped == {} and report.skipped == {}
    assert _labels(tmp_path / "out") == _labels(src)
    second = convert.ingest(tmp_path / "out", "yolo_obb", tmp_path / "b.dcp")
    assert second.manifest.version == first.manifest.version


def test_corner_order_is_kept(tmp_path: Path) -> None:
    src = synth.build_yolo_obb(tmp_path / "src")
    ds, _ = convert.load_dataset(src, "yolo_obb")
    by_id = {a.ann_id: a for a in ds.annotations}
    # "2 0.5 0.75 0.5 1 0 1 0.1 0.75" on a 64x48 image, in the given (counter-screen) order
    trapezoid = by_id["train/s1.png:2"].obb_xyxyxyxy_abs
    assert trapezoid == pytest.approx((32.0, 36.0, 32.0, 48.0, 0.0, 48.0, 6.4, 36.0))
    assert by_id["train/s1.png:2"].bbox_xyxy_abs == pytest.approx((0.0, 36.0, 32.0, 48.0))
    prediction = by_id["val/s2.png:0"]
    assert prediction.score == 0.875 and prediction.provenance == "model"


def _hairline_fixture(root: Path) -> Path:
    """Seven-decimal-plus text and corners a hair outside the image, as real files have."""
    rng = random.Random(11)
    lines = ["0 -0.0000004 0.1 1.0000003 0.1 1.0000003 0.6 -0.0000004 0.6"]
    for _ in range(20):
        cx, cy, w, h = (rng.uniform(0.3, 0.7) for _ in range(4))
        dx, dy = w / 4, h / 4
        corners = (cx - dx, cy, cx, cy - dy, cx + dx, cy, cx, cy + dy)  # a diamond
        lines.append("1 " + " ".join(f"{v:.9f}" for v in corners))
    synth.write_png(root / "images" / "train" / "h.png", 640, 480, 3)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "h.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text("names: [a, b]\n")
    return root


def test_ingest_export_ingest_is_a_fixed_point_for_off_grid_text(tmp_path: Path) -> None:
    src = _hairline_fixture(tmp_path / "src")
    first = convert.ingest(src, "yolo_obb", tmp_path / "a.dcp")
    assert first.report.is_lossless()
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    corners = [v for a in ds.annotations for v in a.obb_xyxyxyxy_abs or ()]
    assert min(corners) == 0.0 and max(corners) <= 640.0  # the hairline overshoot was snapped
    convert.export(ds, "yolo_obb", tmp_path / "out", strict=True)
    second = convert.ingest(tmp_path / "out", "yolo_obb", tmp_path / "b.dcp")
    assert second.manifest.version == first.manifest.version


def test_other_formats_keep_the_enclosing_box_and_strict_refuses(tmp_path: Path) -> None:
    ds, _ = convert.load_dataset(synth.build_yolo_obb(tmp_path / "src"), "yolo_obb")
    report = convert.export(ds, "coco", tmp_path / "coco")
    assert report.dropped == {"annotations.obb_xyxyxyxy_abs": 4}
    with pytest.raises(ConversionLossError):
        convert.export(ds, "yolo", tmp_path / "strict", strict=True)
    assert not (tmp_path / "strict").exists()
    back, _ = convert.load_dataset(tmp_path / "coco", "coco")
    assert {a.kind for a in back.annotations} == {"bbox"}
    assert sorted(a.bbox_xyxy_abs for a in back.annotations) == pytest.approx(  # type: ignore[type-var]
        sorted(a.bbox_xyxy_abs for a in ds.annotations)  # type: ignore[type-var]
    )


def test_axis_aligned_boxes_written_as_yolo_obb_come_back_as_their_corners(tmp_path: Path) -> None:
    ds, _ = convert.load_dataset(synth.build_yolo(tmp_path / "src"), "yolo")
    report = convert.export(ds, "yolo_obb", tmp_path / "obb", strict=True)
    assert report.dropped == {}
    back, _ = convert.load_dataset(tmp_path / "obb", "yolo_obb")
    assert {a.kind for a in back.annotations} == {"obb"}
    pairs = zip(
        sorted((a.image_id, a.bbox_xyxy_abs) for a in ds.annotations),
        sorted((a.image_id, a.bbox_xyxy_abs) for a in back.annotations),
        strict=True,
    )
    for (ia, ba), (ib, bb) in pairs:
        assert ia == ib and ba == pytest.approx(bb, abs=1e-3)


def test_each_yolo_flavour_points_at_the_other(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="yolo_obb format"):
        convert.load_dataset(synth.build_yolo_obb(tmp_path / "obb"), "yolo")
    with pytest.raises(FormatError, match="yolo format"):
        convert.load_dataset(synth.build_yolo(tmp_path / "aa"), "yolo_obb")
