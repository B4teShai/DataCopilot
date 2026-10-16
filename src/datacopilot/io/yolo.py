"""YOLO detection reader and writer (txt labels plus ``data.yaml``).

Layout: ``data.yaml`` (class names), ``images/[train|val|test/]<name>`` and the parallel
``labels/[split/]<name>.txt`` with lines ``class cx cy w h [conf]`` in normalised coordinates.
A missing label file means an image without objects. Polygon (segmentation) lines are rejected.

Boxes are written with 7 decimals, so a box is reproduced to within 5e-8 of the image size; a
box that moves by more than 1e-3 pixel is counted in ``annotations.bbox_xyxy_abs``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

from datacopilot.core import geometry
from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Category, Dataset, Image, PluginSpec, Split
from datacopilot.io import files
from datacopilot.io.common import (
    NATIVE_KEY,
    build_row,
    foreign_source_drops,
    is_native,
    make_image,
    natural_key,
    require_empty_dir,
    require_task,
)
from datacopilot.io.miniyaml import parse_mapping
from datacopilot.io.registry import ReadResult
from datacopilot.io.report import ConversionReport

NAME = "yolo"
SPLITS = ("train", "val", "test")
CLIP_TOLERANCE_PX = 1e-3
_CANONICAL_ROUNDS = 8
PRECISION_TOLERANCE_PX = 1e-3
_YAML_KEYS = frozenset({"path", "train", "val", "test", "nc", "names"})
_NAME_HINTS = frozenset({"file_name", "filename"})


def _class_names(doc: dict[str, object]) -> list[str]:
    names = doc.get("names")
    if isinstance(names, dict):
        if sorted(names) != list(range(len(names))) or not names:
            raise FormatError("data.yaml: names must have contiguous integer keys from 0")
        ordered: list[object] = [names[i] for i in range(len(names))]
    elif isinstance(names, list) and names:
        ordered = list(names)
    else:
        raise FormatError("data.yaml: 'names' must be a non-empty list or mapping of class names")
    if not all(isinstance(n, str) and n for n in ordered):
        raise FormatError("data.yaml: class names must be non-empty strings")
    result = [str(n) for n in ordered]
    if len(set(result)) != len(result):
        raise FormatError("data.yaml: duplicate class names")
    nc = doc.get("nc")
    if nc is not None and nc != len(result):
        raise FormatError(f"data.yaml: nc is {nc} but {len(result)} names are listed")
    return result


def _number(token: str, where: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise FormatError(f"{where}: {token!r} is not a number") from None
    if not math.isfinite(value):
        raise FormatError(f"{where}: numbers must be finite")
    return value


def _fmt(value: float) -> str:
    text = f"{value:.7f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


class YoloFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path) -> ReadResult:
        root = Path(path)
        if not root.is_dir():
            raise FormatError(f"{root.name} does not exist or is not a directory")
        yaml_file = root / "data.yaml"
        if not yaml_file.is_file():
            raise FormatError(f"data.yaml not found in {root.name}")
        doc = parse_mapping(
            files.read_bounded(yaml_file, files.MAX_TEXT_BYTES).decode("utf-8", "replace")
        )
        names = _class_names(doc)
        dropped: Counter[str] = Counter(f"dataset.{k}" for k in doc if k not in _YAML_KEYS)
        skipped: Counter[str] = Counter()
        image_root = root / "images"
        if not image_root.is_dir():
            raise FormatError(f"images directory not found in {root.name}")
        images: list[Image] = []
        annotations: list[Annotation] = []
        for file in sorted(p for p in image_root.rglob("*") if p.is_file()):
            suffix = file.suffix.lower()
            files.refuse_unsupported(file.name)
            if suffix not in files.IMAGE_EXTENSIONS:
                continue
            rel = files.safe_join(root, file.relative_to(root).as_posix())
            parts = rel.relative_to(root.resolve() / "images").parts
            split: Split = parts[0] if len(parts) > 1 and parts[0] in SPLITS else "unassigned"  # type: ignore[assignment]
            file_name = "/".join(parts[1:] if split != "unassigned" else parts)
            rel_key = "/".join(parts)
            image = make_image(
                rel, split=split, source={NATIVE_KEY: NAME, "file_name": file_name}, ref=rel_key
            )
            images.append(image)
            annotations.extend(self._labels(root, parts, image, rel_key, names))
        return ReadResult(
            name=root.name or NAME,
            task="detection",
            images=images,
            annotations=annotations,
            categories=[Category(id=i, name=n) for i, n in enumerate(names)],
            dropped=dict(dropped),
            skipped=dict(skipped),
        )

    @staticmethod
    def _labels(
        root: Path, parts: tuple[str, ...], image: Image, rel_key: str, names: list[str]
    ) -> list[Annotation]:
        label_rel = "labels/" + "/".join(parts[:-1] + (PurePosixPath(parts[-1]).stem + ".txt",))
        label = files.safe_join(root, label_rel)
        if not label.is_file():
            return []
        out: list[Annotation] = []
        text = files.read_bounded(label, files.MAX_TEXT_BYTES).decode("utf-8", "replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{label.name}:{number}"
            tokens = line.split()
            if len(tokens) > 6:
                raise FormatError(f"{where}: polygon/segmentation labels are not supported")
            if len(tokens) < 5:
                raise FormatError(f"{where}: expected 'class cx cy w h [conf]'")
            try:
                cls = int(tokens[0])
            except ValueError:
                raise FormatError(f"{where}: class must be an integer") from None
            if not 0 <= cls < len(names):
                raise FormatError(f"{where}: class {cls} is not defined in data.yaml")
            values = [_number(t, where) for t in tokens[1:]]
            try:
                box = _canonical(
                    geometry.clip_to_image(
                        geometry.cxcywh_norm_to_xyxy(
                            (values[0], values[1], values[2], values[3]), image.width, image.height
                        ),
                        image.width,
                        image.height,
                        tol=CLIP_TOLERANCE_PX,
                    ),
                    image,
                )
            except ValueError:
                raise FormatError(f"{where}: box lies outside the image") from None
            score = values[4] if len(values) == 5 else None
            out.append(
                build_row(
                    Annotation,
                    f"{rel_key}",
                    ann_id=f"{rel_key}:{len(out)}",
                    image_id=image.image_id,
                    category_id=cls,
                    kind="bbox",
                    bbox_xyxy_abs=box,
                    score=score,
                    provenance="model" if score is not None else "human",
                    source={NATIVE_KEY: NAME},
                )
            )
        return out

    # ----------------------------------------------------------------- write
    def write(self, ds: Dataset, out: Path) -> ConversionReport:
        out = Path(out)
        require_empty_dir(out)
        require_task(ds.manifest.task, "detection", NAME)
        dropped: Counter[str] = Counter()
        categories = sorted(ds.categories, key=lambda c: c.id)
        index = {c.id: i for i, c in enumerate(categories)}
        for i, cat in enumerate(categories):
            for column, lost in (
                ("id", cat.id != i),
                ("aliases", bool(cat.aliases)),
                ("attributes", bool(cat.attributes)),
            ):
                if lost:
                    dropped[f"categories.{column}"] += 1

        used: dict[str, set[str]] = {}
        placed: dict[str, str] = {}
        for image in sorted(ds.images, key=lambda i: (_hint(i), i.image_id)):
            placed[image.image_id] = _unique_stem(
                _hint(image), used.setdefault(image.split, set()), image.image_id[:8]
            )
            if image.group_id is not None:
                dropped["images.group_id"] += 1
            if not is_native(image.source, NAME):
                foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: natural_key(a.ann_id)):
            anns_of.setdefault(ann.image_id, []).append(ann)

        for image in ds.images:
            folder = "" if image.split == "unassigned" else image.split + "/"
            name = placed[image.image_id]
            files.copy_file(Path(image.file_path), files.safe_join(out / "images", folder + name))
            label = files.safe_join(
                out / "labels", folder + str(PurePosixPath(name).with_suffix(".txt"))
            )
            label.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                self._line(a, image, index[a.category_id], dropped)
                for a in anns_of.get(image.image_id, [])
            ]
            label.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

        present = sorted({i.split for i in ds.images if i.split != "unassigned"}, key=SPLITS.index)
        yaml_lines = ["path: ."]
        yaml_lines += [f"{s}: images/{s}" for s in present] or ["train: images"]
        yaml_lines += [
            f"nc: {len(categories)}",
            "names: [" + ", ".join(_quote(c.name) for c in categories) + "]",
        ]
        (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        rows = len(ds.images) + len(ds.annotations)
        return ConversionReport(
            source_format=NAME,
            target_format=NAME,
            rows_read=rows,
            rows_written=rows,
            dropped=dict(dropped),
        )

    @staticmethod
    def _line(ann: Annotation, image: Image, cls: int, dropped: Counter[str]) -> str:
        if ann.bbox_xyxy_abs is None:
            raise FormatError(f"annotation {ann.ann_id!r} has no box")
        norm = geometry.xyxy_to_cxcywh_norm(ann.bbox_xyxy_abs, image.width, image.height)
        text = [_fmt(v) for v in norm]
        back = geometry.cxcywh_norm_to_xyxy(
            (float(text[0]), float(text[1]), float(text[2]), float(text[3])),
            image.width,
            image.height,
        )
        if (
            max(abs(a - b) for a, b in zip(back, ann.bbox_xyxy_abs, strict=True))
            > PRECISION_TOLERANCE_PX
        ):
            dropped["annotations.bbox_xyxy_abs"] += 1
        for column, present in (
            ("is_crowd", ann.is_crowd),
            ("occluded", ann.occluded is not None),
            ("truncated", ann.truncated is not None),
            ("attributes", bool(ann.attributes)),
            (
                "provenance",
                ann.provenance == "proposal_accepted"
                or (ann.provenance == "model" and ann.score is None),
            ),
        ):
            if present:
                dropped[f"annotations.{column}"] += 1
        if not is_native(ann.source, NAME):
            foreign_source_drops(dropped, "annotations", ann.source, frozenset())
        fields = [str(cls), *text]
        if ann.score is not None:
            fields.append(repr(ann.score))
        return " ".join(fields)


def _canonical(box: geometry.Box, image: Image) -> geometry.Box:
    """The box this format can represent exactly: what the writer would print, read back.

    A box that was snapped onto the image edge no longer sits on the 7-decimal grid of the text
    files, so writing it and reading it again would move it by a few hundred-thousandths of a
    pixel. Applying "print, read, snap" until nothing changes (at most a few rounds) makes
    ingest, export, ingest a fixed point. A box that was already on the grid is returned unchanged.
    """
    for _ in range(_CANONICAL_ROUNDS):
        text = [_fmt(v) for v in geometry.xyxy_to_cxcywh_norm(box, image.width, image.height)]
        back = geometry.clip_to_image(
            geometry.cxcywh_norm_to_xyxy(
                (float(text[0]), float(text[1]), float(text[2]), float(text[3])),
                image.width,
                image.height,
            ),
            image.width,
            image.height,
            tol=CLIP_TOLERANCE_PX,
        )
        if back == box:
            break
        box = back
    return box


def _quote(name: str) -> str:
    return json.dumps(name, ensure_ascii=False)


def _hint(image: Image) -> str:
    """Preferred file name below ``images/<split>/``."""
    if is_native(image.source, NAME):
        name = image.source.get("file_name")
        if isinstance(name, str):
            return name
    for key in _NAME_HINTS:
        value = image.source.get(key)
        if isinstance(value, str) and value:
            return PurePosixPath(value).name
    return PurePosixPath(image.file_path).name


def _unique_stem(name: str, used: set[str], salt: str) -> str:
    """Name whose extension-less form is unused in this split, so label files never collide."""
    path = PurePosixPath(name)
    candidate = name
    if str(path.with_suffix("")) in used:
        candidate = str(path.with_name(f"{path.stem}_{salt}{path.suffix}"))
    used.add(str(PurePosixPath(candidate).with_suffix("")))
    return candidate


plugin = YoloFormat()
