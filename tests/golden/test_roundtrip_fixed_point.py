"""ingest, export, ingest must give the version of the first ingest (a fixed point).

The fixture mimics what real label files do: six-decimal YOLO text, boxes touching or overshooting
all four image edges by a hair (the reader snaps those back), tiny boxes and more than ten boxes on
one image. Every format must come back with the same tables and the same ``manifest.version``.
"""

import random
from pathlib import Path

import pytest
from conftest import table_rows
from fixtures import synth

from datacopilot import convert
from datacopilot.store import DatasetStore

FORMATS = ("yolo", "coco", "voc", "cvat_xml", "label_studio")
WIDTH, HEIGHT = 640, 480


def _edge_fixture(root: Path) -> Path:
    rng = random.Random(7)
    lines_for: dict[str, list[str]] = {}
    for name, count in (("a", 3), ("b", 12), ("c", 4)):
        lines = [f"{rng.randrange(3)} 0.5 0.5 1.000001 1.000001"] if name == "a" else []
        lines += ["0 0.0005 0.5 0.001 0.2", "1 0.9995 0.5 0.001 0.2"]  # touching left and right
        lines += ["2 0.5 0.0001 0.3 0.0002", "0 0.5 0.9999 0.3 0.0002"]  # top and bottom, thin
        lines += ["1 0.123457 0.654321 0.003 0.004"]  # tiny
        lines += ["2 0.391581 0.556305 0.546862 0.887391"]  # real COCO label: overshoots the bottom
        for _ in range(6):  # six-decimal boxes ending a hair beyond the right or bottom edge
            side = round(rng.uniform(0.3, 0.9), 6)
            lines.append(f"0 {round(1 - side / 2 + rng.uniform(0, 4e-7), 6)} 0.4 {side} 0.2")
            lines.append(f"1 0.4 {round(1 - side / 2 + rng.uniform(0, 4e-7), 6)} 0.2 {side}")
        while len(lines) < count + 5:
            lines.append(
                f"{rng.randrange(3)} {rng.uniform(0.2, 0.8):.6f} {rng.uniform(0.2, 0.8):.6f} "
                f"{rng.uniform(0.05, 0.3):.6f} {rng.uniform(0.05, 0.3):.6f}"
            )
        lines_for[name] = lines
    for n, (name, lines) in enumerate(lines_for.items()):
        synth.write_png(root / "images" / "train" / f"{name}.png", WIDTH, HEIGHT, n)
        label = root / "labels" / "train" / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nnc: 3\nnames: [a, b, c]\n")
    return root


def _version(path: Path, fmt: str, out: Path) -> tuple[str, DatasetStore]:
    result = convert.ingest(path, fmt, out)
    return result.manifest.version, DatasetStore.open(out)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ingest_export_ingest_is_a_fixed_point(fmt: str, tmp_path: Path) -> None:
    yolo = _edge_fixture(tmp_path / "yolo")
    first_version, store = _version(yolo, "yolo", tmp_path / "y0.dcp")
    ds0 = store.read()
    if fmt != "yolo":  # start from the format under test
        convert.export(ds0, fmt, tmp_path / "src")
        first_version, store = _version(tmp_path / "src", fmt, tmp_path / "f0.dcp")
        ds0 = store.read()
    convert.export(ds0, fmt, tmp_path / "out")
    second_version, store2 = _version(tmp_path / "out", fmt, tmp_path / "f1.dcp")
    ds1 = store2.read()
    assert table_rows(ds0.images, ds0.annotations) == table_rows(ds1.images, ds1.annotations)
    assert second_version == first_version
