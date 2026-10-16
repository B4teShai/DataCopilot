"""An image with more than ten boxes must read back in the same order after a same-format round
trip (annotation ids end in the position of the box, and ``10`` sorts before ``2`` as text)."""

import re
from pathlib import Path

import pytest
from conftest import table_rows
from fixtures import synth

from datacopilot import convert
from datacopilot.store import DatasetStore

BOXES = 12


def _crowd_yolo(root: Path) -> None:
    label = root / "labels" / "train" / "a.txt"
    lines = [f"{n % 3} {0.1 + n * 0.05:.4f} 0.5 0.08 0.2" for n in range(BOXES)]
    label.write_text("\n".join(lines) + "\n")


def _crowd_voc(root: Path) -> None:
    xml_file = root / "Annotations" / "a.xml"
    objects = "".join(
        f"<object><name>{synth.CATEGORIES[n % 3]}</name><bndbox><xmin>{n * 4}</xmin><ymin>5</ymin>"
        f"<xmax>{n * 4 + 3}</xmax><ymax>20</ymax></bndbox></object>"
        for n in range(BOXES)
    )
    text = xml_file.read_text()
    xml_file.write_text(re.sub(r"<object>.*</object>", objects, text, flags=re.S))


CASES = {
    "yolo": (synth.build_yolo, _crowd_yolo),
    "voc": (synth.build_voc, _crowd_voc),
}


@pytest.mark.parametrize("fmt", sorted(CASES))
def test_more_than_ten_boxes_survive_a_same_format_round_trip(fmt: str, tmp_path: Path) -> None:
    build, crowd = CASES[fmt]
    source = build(tmp_path / "src")
    crowd(source)
    first = convert.ingest(source, fmt, tmp_path / "a.dcp")
    ds_a = DatasetStore.open(tmp_path / "a.dcp").read()
    assert sum(1 for a in ds_a.annotations if a.ann_id.startswith(("train/a.png:", "a:"))) >= BOXES
    convert.export(ds_a, fmt, tmp_path / "out", strict=True)
    second = convert.ingest(tmp_path / "out", fmt, tmp_path / "b.dcp")
    ds_b = DatasetStore.open(tmp_path / "b.dcp").read()
    assert table_rows(ds_a.images, ds_a.annotations) == table_rows(ds_b.images, ds_b.annotations)
    assert first.manifest.version == second.manifest.version
