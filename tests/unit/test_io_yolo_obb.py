"""The ``yolo_obb`` reader and writer on single label lines (ADR-0011)."""

from pathlib import Path

import pytest
from conftest import dataset_from
from fixtures import synth

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files
from datacopilot.io.yolo_obb import plugin


def one_image(root: Path, label: str) -> Path:
    synth.write_png(root / "images" / "train" / "a.png", 100, 50, 1)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text(label)
    (root / "data.yaml").write_text("names: [cat, dog]\n")
    return root


def test_corners_become_absolute_pixels_in_their_order(tmp_path: Path) -> None:
    (ann,) = plugin.read(one_image(tmp_path / "y", "1 0.1 0.2 0.5 0.1 0.6 0.9 0.2 1\n")).annotations
    assert ann.kind == "obb" and ann.category_id == 1
    assert ann.obb_xyxyxyxy_abs == pytest.approx((10, 10, 50, 5, 60, 45, 20, 50))
    assert ann.bbox_xyxy_abs == pytest.approx((10, 5, 60, 50))
    assert ann.source == {"format": "yolo_obb"} and ann.ann_id == "train/a.png:0"
    assert ann.provenance == "human" and ann.score is None


def test_confidence_makes_a_model_row(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", "0 0.1 0.2 0.5 0.1 0.6 0.9 0.2 1 0.25\n")
    (ann,) = plugin.read(root).annotations
    assert ann.score == 0.25 and ann.provenance == "model"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("0 0.5 0.5 0.2 0.2", "yolo format"),
        ("0 0.1 0.2 0.5 0.1 0.6 0.9", "expected 'class x1"),
        ("0 0.1 0.2 0.5 0.1 0.6 0.9 0.2 1 0.5 0.5", "expected 'class x1"),
        ("0 0.1 0.2 0.5 0.1 0.6 0.9 0.2 1.01", "outside the image"),
        ("0 -0.01 0.2 0.5 0.1 0.6 0.9 0.2 1", "outside the image"),
        ("0 0.1 0.1 0.5 0.5 0.5 0.1 0.1 0.5", "convex quadrilateral"),  # bow-tie
        ("0 0.1 0.1 0.3 0.1 0.5 0.1 0.1 0.5", "convex quadrilateral"),  # collinear
        ("0 0.1 0.1 0.1 0.1 0.5 0.5 0.1 0.5", "convex quadrilateral"),  # repeated
        ("2 0.1 0.2 0.5 0.1 0.6 0.9 0.2 1", "class 2 is not defined"),
        ("x 0.1 0.2 0.5 0.1 0.6 0.9 0.2 1", "class must be an integer"),
        ("0 0.1 0.2 0.5 0.1 0.6 nan 0.2 1", "finite"),
    ],
)
def test_malformed_lines_name_the_file_and_line(tmp_path: Path, line: str, message: str) -> None:
    with pytest.raises(FormatError, match=message) as info:
        plugin.read(one_image(tmp_path / "y", "\n" + line + "\n"))
    assert "a.txt:2" in str(info.value)


def test_a_hairline_overshoot_is_snapped_onto_the_edge(tmp_path: Path) -> None:
    root = one_image(tmp_path / "y", "0 -0.000001 0 1.000001 0 1 1 0 1\n")
    (ann,) = plugin.read(root).annotations
    assert ann.obb_xyxyxyxy_abs == (0.0, 0.0, 100.0, 0.0, 100.0, 50.0, 0.0, 50.0)


def _write_one(
    tmp_path: Path, width: int, corners: tuple[float, ...]
) -> tuple[dict[str, int], str]:
    image_path = tmp_path / "a.png"
    synth.write_png(image_path, 7, 3, 1)
    _, digest = files.identify(image_path)
    image = Image(image_id=digest, file_path=str(image_path), width=width, height=3, split="train")
    ann = Annotation(
        ann_id="x",
        image_id=digest,
        category_id=0,
        kind="obb",
        obb_xyxyxyxy_abs=corners,  # type: ignore[arg-type]
    )
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(dataset_from([image], [ann], [Category(id=0, name="c")]), out)
    return report.dropped, (out / "labels" / "train" / "a.txt").read_text()


def test_writer_prints_seven_decimals(tmp_path: Path) -> None:
    dropped, text = _write_one(tmp_path, 7, (1.0, 1.0, 6.0, 1.0, 6.0, 2.0, 1.0, 2.0))
    assert dropped == {}
    assert text == (
        "0 0.1428571 0.3333333 0.8571429 0.3333333 0.8571429 0.6666667 0.1428571 0.6666667\n"
    )


def test_writer_counts_corners_that_move_off_the_7_decimal_grid(tmp_path: Path) -> None:
    corners = (12345.678, 1.0, 20000.0, 1.0, 20000.0, 2.0, 12345.678, 2.0)
    dropped, _ = _write_one(tmp_path, 100_000, corners)  # 5e-8 of 100,000 px is 0.005 px
    assert dropped == {"annotations.obb_xyxyxyxy_abs": 1}
