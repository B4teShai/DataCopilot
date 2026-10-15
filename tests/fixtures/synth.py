"""Tiny synthetic datasets in every supported format, generated at test time.

Images are small grayscale PNGs written with the standard library; each image's pixels depend on
its index so every file has a distinct content hash. No real photographs, nothing over 1 KB.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

CATEGORIES = ["cat", "dog", "bird"]


def png_bytes(width: int, height: int, seed: int) -> bytes:
    """A valid 8-bit grayscale PNG whose pixel values depend on ``seed``."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(
        b"\x00" + bytes((seed * 37 + x * 3 + y * 5) % 256 for x in range(width))
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def write_png(path: Path, width: int, height: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(width, height, seed))


# name, width, height, split, boxes as (category index, x1, y1, x2, y2)
IMAGES: list[tuple[str, int, int, str, list[tuple[int, float, float, float, float]]]] = [
    ("a.png", 64, 48, "train", [(0, 8.0, 6.0, 40.0, 30.0), (1, 10.0, 12.0, 64.0, 48.0)]),
    ("b.png", 80, 60, "train", [(2, 0.0, 0.0, 80.0, 60.0)]),
    ("c.png", 64, 48, "val", []),
    ("d.png", 50, 50, "test", [(0, 12.5, 7.25, 31.5, 44.0)]),
]


def build_coco(root: Path, *, with_extras: bool = True) -> Path:
    """COCO layout: annotations/instances_<split>.json plus images/<name>. Returns ``root``."""
    by_split: dict[str, list[int]] = {}
    for n, (_, _, _, split, _) in enumerate(IMAGES):
        by_split.setdefault(split, []).append(n)
    categories = [
        {"id": i + 1, "name": name, **({"supercategory": "animal"} if with_extras else {})}
        for i, name in enumerate(CATEGORIES)
    ]
    ann_id = 100
    for split, members in by_split.items():
        images, annotations = [], []
        for n in members:
            name, w, h, _, boxes = IMAGES[n]
            write_png(root / "images" / name, w, h, n)
            entry: dict[str, object] = {"id": n + 1, "file_name": name, "width": w, "height": h}
            if with_extras:
                entry["license"] = 1
                entry["date_captured"] = "2020-01-01"
            images.append(entry)
            for cat, x1, y1, x2, y2 in boxes:
                ann_id += 1
                ann: dict[str, object] = {
                    "id": ann_id,
                    "image_id": n + 1,
                    "category_id": cat + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": 1 if (with_extras and ann_id == 102) else 0,
                }
                if with_extras:
                    ann["area"] = (x2 - x1) * (y2 - y1)
                    ann["segmentation"] = []
                annotations.append(ann)
        out = root / "annotations" / f"instances_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": categories})
        )
    return root


def _norm(box: tuple[float, float, float, float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    vals = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
    return " ".join(f"{v:.6f}".rstrip("0").rstrip(".") for v in vals)


def build_yolo(root: Path) -> Path:
    """YOLO layout: data.yaml, images/<split>/x.png, labels/<split>/x.txt."""
    for n, (name, w, h, split, boxes) in enumerate(IMAGES):
        write_png(root / "images" / split / name, w, h, n)
        lines = [f"{cat} {_norm((x1, y1, x2, y2), w, h)}" for cat, x1, y1, x2, y2 in boxes]
        label = root / "labels" / split / (Path(name).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CATEGORIES))
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 3\n"
        f"names:\n{names}\n"
    )
    return root


def build_voc(root: Path) -> Path:
    """Pascal VOC layout: Annotations/*.xml, JPEGImages/*, ImageSets/Main/<split>.txt."""
    sets: dict[str, list[str]] = {}
    for n, (name, w, h, split, boxes) in enumerate(IMAGES):
        write_png(root / "JPEGImages" / name, w, h, n)
        stem = Path(name).stem
        sets.setdefault(split, []).append(stem)
        objects = "".join(
            f"<object><name>{CATEGORIES[cat]}</name><pose>Left</pose><truncated>{int(n == 0)}"
            f"</truncated><difficult>{int(i == 1)}</difficult><bndbox><xmin>{x1:g}</xmin>"
            f"<ymin>{y1:g}</ymin><xmax>{x2:g}</xmax><ymax>{y2:g}</ymax></bndbox></object>"
            for i, (cat, x1, y1, x2, y2) in enumerate(boxes)
        )
        xml = (
            f"<annotation><folder>JPEGImages</folder><filename>{name}</filename>"
            f"<size><width>{w}</width><height>{h}</height><depth>1</depth></size>"
            f"<segmented>0</segmented>{objects}</annotation>"
        )
        (root / "Annotations").mkdir(parents=True, exist_ok=True)
        (root / "Annotations" / f"{stem}.xml").write_text(xml)
    (root / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
    for split, stems in sets.items():
        (root / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(stems) + "\n")
    (root / "labelmap.txt").write_text("\n".join(CATEGORIES) + "\n")
    return root


def build_cvat_xml(root: Path) -> Path:
    """CVAT for images 1.1: annotations/<split>.xml with meta labels (name and colour) and
    ``images/<name>``. Boxes carry ``z_order`` and ``occluded``; the first box has an attribute."""
    by_split: dict[str, list[int]] = {}
    for n, (_, _, _, split, _) in enumerate(IMAGES):
        by_split.setdefault(split, []).append(n)
    colours = ("#ff0000", "#00ff00", "#0000ff")
    labels = "".join(
        f"<label><name>{name}</name><color>{colours[i]}</color><type>any</type><attributes/></label>"
        for i, name in enumerate(CATEGORIES)
    )
    for split, members in by_split.items():
        body = ""
        for frame, n in enumerate(members):
            name, w, h, _, boxes = IMAGES[n]
            write_png(root / "images" / name, w, h, n)
            shapes = "".join(
                f'<box label="{CATEGORIES[cat]}" source="manual" occluded="{int(i == 1)}" '
                f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0">'
                + ('<attribute name="colour">red</attribute>' if (n == 0 and i == 0) else "")
                + "</box>"
                for i, (cat, x1, y1, x2, y2) in enumerate(boxes)
            )
            body += f'<image id="{frame}" name="{name}" width="{w}" height="{h}">{shapes}</image>'
        xml = (
            '<?xml version="1.0" encoding="utf-8"?><annotations><version>1.1</version><meta><task>'
            f"<name>demo</name><size>{len(members)}</size><mode>annotation</mode>"
            f"<subset>{split}</subset><labels>{labels}</labels></task></meta>{body}</annotations>"
        )
        out = root / "annotations" / f"{split}.xml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(xml)
    return root


def ls_rect(
    label: str, box: tuple[float, float, float, float], size: tuple[int, int], rid: str
) -> dict[str, object]:
    """A Label Studio ``rectanglelabels`` result for a pixel box (percent of the original size)."""
    x1, y1, x2, y2 = box
    w, h = size
    return {
        "original_width": w,
        "original_height": h,
        "image_rotation": 0,
        "value": {
            "x": x1 * 100 / w,
            "y": y1 * 100 / h,
            "width": (x2 - x1) * 100 / w,
            "height": (y2 - y1) * 100 / h,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "id": rid,
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
    }


def build_label_studio(root: Path) -> Path:
    """Label Studio JSON export: annotations/<split>.json (one task per image) plus
    ``images/<name>``. Every task has one annotation with an ``id``, ``completed_by`` (user id) and
    ``lead_time``; every result has an ``id``."""
    by_split: dict[str, list[int]] = {}
    for n, (_, _, _, split, _) in enumerate(IMAGES):
        by_split.setdefault(split, []).append(n)
    for split, members in by_split.items():
        tasks = []
        for n in members:
            name, w, h, _, boxes = IMAGES[n]
            write_png(root / "images" / name, w, h, n)
            results = [
                ls_rect(CATEGORIES[cat], (x1, y1, x2, y2), (w, h), f"r{n}_{i}")
                for i, (cat, x1, y1, x2, y2) in enumerate(boxes)
            ]
            tasks.append(
                {
                    "id": n + 1,
                    "data": {"image": f"images/{name}"},
                    "annotations": [
                        {"id": n + 10, "completed_by": 1, "lead_time": 1.5, "result": results}
                    ],
                    "predictions": [],
                }
            )
        out = root / "annotations" / f"{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tasks))
    return root


# ImageFolder fixture: split -> class -> image names
FOLDER_IMAGES = [
    ("train", "cat", "c1.png", 40, 30),
    ("train", "cat", "c2.png", 40, 30),
    ("train", "dog", "d1.png", 32, 32),
    ("val", "cat", "c3.png", 40, 30),
    ("val", "dog", "d2.png", 32, 32),
]


def build_imagefolder(root: Path, *, split_dirs: bool = True) -> Path:
    for n, (split, cls, name, w, h) in enumerate(FOLDER_IMAGES):
        target = root / split / cls / name if split_dirs else root / cls / name
        write_png(target, w, h, 50 + n)
    return root


def jpeg_bytes(width: int, height: int, orientation: int | None = None, segments: int = 0) -> bytes:
    """A header-only JPEG: SOI, optional EXIF orientation, empty COM segments, SOF0."""
    out = b"\xff\xd8"
    if orientation is not None:
        tiff = b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", 1)
        tiff += struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + b"\x00\x00\x00\x00"
        exif = b"Exif\x00\x00" + tiff
        out += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    out += (b"\xff\xfe" + struct.pack(">H", 2)) * segments
    sof = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    return out + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9"
