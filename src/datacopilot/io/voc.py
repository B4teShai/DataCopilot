"""Pascal VOC detection reader and writer.

Layout: ``Annotations/<stem>.xml``, images in ``JPEGImages/`` (or ``images/``),
``ImageSets/Main/{train,val,test}.txt`` for splits, and an optional ``labelmap.txt`` (one class
name per line, id = line index; otherwise ids follow the sorted set of object names).

Mapped: bndbox (taken as given, no 1-based offset), ``truncated``, ``occluded``, ``difficult``
(attribute ``difficult``), class name, image size. Every other element is kept verbatim in
``source`` (nested elements as dicts, repeated elements as lists) and re-emitted for VOC rows.
XML is parsed by :mod:`datacopilot.io.xmlsafe`, which rejects DOCTYPE and entities.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: S405  # used only to build output, never to parse
from collections import Counter
from pathlib import Path, PurePosixPath

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
from datacopilot.io.registry import ReadResult
from datacopilot.io.report import ConversionReport
from datacopilot.io.xmlsafe import parse_xml

NAME = "voc"
SPLITS = ("train", "val", "test")
_OBJECT_MAPPED = frozenset({"name", "bndbox", "truncated", "occluded", "difficult"})
_NAME_HINTS = frozenset({"file_name", "filename"})
_IMAGE_DIRS = ("JPEGImages", "images", "")

MAX_EMIT_DEPTH = 8
MAX_EMIT_ELEMENTS = 1000
MAX_NAME_CHARS = 64
_ILLEGAL_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_NAME_START = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
_NAME_REST = _NAME_START | frozenset("0123456789.-")

Value = str | dict[str, "Value"] | list["Value"]


def _to_value(el: ET.Element) -> Value:
    """Element -> str (leaf), dict (children) with repeated tags collected into lists."""
    children = list(el)
    if not children:
        return (el.text or "").strip()
    out: dict[str, Value] = {}
    for child in children:
        value = _to_value(child)
        if child.tag in out:
            existing = out[child.tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[child.tag] = [existing, value]
        else:
            out[child.tag] = value
    return out


def _text(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    return None if child is None else (child.text or "").strip()


def _flag(el: ET.Element, tag: str, where: str) -> bool | None:
    raw = _text(el, tag)
    if raw is None:
        return None
    if raw in ("0", "false", "False"):
        return False
    if raw in ("1", "true", "True"):
        return True
    raise FormatError(f"{where}: {tag} must be 0 or 1")


def _float(el: ET.Element | None, tag: str, where: str) -> float:
    raw = None if el is None else _text(el, tag)
    try:
        value = float(raw) if raw is not None else None
    except ValueError:
        value = None
    if value is None or value != value or value in (float("inf"), float("-inf")):
        raise FormatError(f"{where}: {tag} is missing or not a finite number")
    return value


def _fmt(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


class VocFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path) -> ReadResult:
        root = Path(path)
        if not root.is_dir():
            raise FormatError(f"{root.name} does not exist or is not a directory")
        xml_files = sorted((root / "Annotations").glob("*.xml"))
        if not xml_files:
            raise FormatError(f"no Annotations/*.xml found in {root.name}")
        splits = self._splits(root)
        parsed = []
        for file in xml_files:
            tree = parse_xml(files.read_bounded(file, files.MAX_XML_BYTES), file.name)
            if tree.tag != "annotation":
                raise FormatError(f"{file.name}: root element must be <annotation>")
            parsed.append((file, tree))
        labelmap = self._labelmap(root)
        if labelmap is None:
            names = sorted(
                {(_text(o, "name") or "") for _, t in parsed for o in t.findall("object")} - {""}
            )
            labelmap = names
        ids = {name: i for i, name in enumerate(labelmap)}
        dropped: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        images: list[Image] = []
        annotations: list[Annotation] = []
        for file, tree in parsed:
            stem = file.stem
            split = splits.get(stem, "unassigned")
            image = self._image(root, file, tree, split, dropped)
            n_objects = len(tree.findall("object"))
            if image is None:
                skipped["images.file_missing"] += 1
                if n_objects:
                    skipped["annotations.of_missing_image"] += n_objects
                continue
            images.append(image)
            for index, el in enumerate(tree.findall("object")):
                annotations.append(self._object(el, f"{stem}:{index}", image, ids, file.name))
        return ReadResult(
            name=root.name or NAME,
            task="detection",
            images=images,
            annotations=annotations,
            categories=[Category(id=i, name=n) for n, i in ids.items()],
            dropped=dict(dropped),
            skipped=dict(skipped),
        )

    @staticmethod
    def _labelmap(root: Path) -> list[str] | None:
        file = root / "labelmap.txt"
        if not file.is_file():
            return None
        names = [
            ln.strip()
            for ln in files.read_bounded(file, files.MAX_TEXT_BYTES)
            .decode("utf-8", "replace")
            .splitlines()
            if ln.strip()
        ]
        if len(set(names)) != len(names):
            raise FormatError("labelmap.txt: duplicate class names")
        return names

    @staticmethod
    def _splits(root: Path) -> dict[str, Split]:
        out: dict[str, Split] = {}
        for split in SPLITS:
            file = root / "ImageSets" / "Main" / f"{split}.txt"
            if not file.is_file():
                continue
            text = files.read_bounded(file, files.MAX_TEXT_BYTES).decode("utf-8", "replace")
            for line in text.splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] in out and out[tokens[0]] != split:
                    raise FormatError(f"image {tokens[0]!r} is listed in more than one split")
                out[tokens[0]] = split  # type: ignore[assignment]
        return out

    def _image(
        self, root: Path, file: Path, tree: ET.Element, split: Split, dropped: Counter[str]
    ) -> Image | None:
        filename = _text(tree, "filename")
        if filename:
            candidates = [
                files.safe_join(root / d, filename) if d else files.safe_join(root, filename)
                for d in _IMAGE_DIRS
            ]
        else:
            candidates = [
                files.safe_join(root / d, file.stem + ext)
                if d
                else files.safe_join(root, file.stem + ext)
                for d in _IMAGE_DIRS
                for ext in sorted(files.IMAGE_EXTENSIONS)
            ]
        located = next((c for c in candidates if c.is_file()), None)
        if located is None:
            return None
        size = tree.find("size")
        declared = None
        if size is not None:
            declared = (
                int(_float(size, "width", file.name)),
                int(_float(size, "height", file.name)),
            )
        source: dict[str, Value] = {}
        for child in tree:
            if child.tag in ("object",):
                continue
            if child.tag == "size":
                depth = _text(child, "depth")
                if depth is not None:
                    source["depth"] = depth
                continue
            source[child.tag] = _to_value(child)
        image = make_image(
            located, split=split, source={NATIVE_KEY: NAME, **source}, ref=f"image {file.stem}"
        )
        if declared is not None and declared != (image.width, image.height):
            dropped["images.declared_size"] += 1
        return image

    @staticmethod
    def _object(
        el: ET.Element, ann_id: str, image: Image, ids: dict[str, int], file: str
    ) -> Annotation:
        where = f"{file} object {ann_id}"
        name = _text(el, "name")
        if not name:
            raise FormatError(f"{where}: object has no name")
        if name not in ids:
            raise FormatError(f"{where}: class {name!r} is not in the label map")
        box = el.find("bndbox")
        if box is None:
            raise FormatError(f"{where}: object has no bndbox")
        coords = tuple(_float(box, tag, where) for tag in ("xmin", "ymin", "xmax", "ymax"))
        difficult = _flag(el, "difficult", where)
        extras = {c.tag: _to_value(c) for c in el if c.tag not in _OBJECT_MAPPED}
        return build_row(
            Annotation,
            where,
            ann_id=ann_id,
            image_id=image.image_id,
            category_id=ids[name],
            kind="bbox",
            bbox_xyxy_abs=coords,
            truncated=_flag(el, "truncated", where),
            occluded=_flag(el, "occluded", where),
            attributes={} if difficult is None else {"difficult": difficult},
            source={NATIVE_KEY: NAME, **extras},
        )

    # ----------------------------------------------------------------- write
    def write(self, ds: Dataset, out: Path) -> ConversionReport:
        out = Path(out)
        require_empty_dir(out)
        require_task(ds.manifest.task, "detection", NAME)
        dropped: Counter[str] = Counter()
        categories = sorted(ds.categories, key=lambda c: c.id)
        for i, cat in enumerate(categories):
            for column, lost in (
                ("id", cat.id != i),
                ("aliases", bool(cat.aliases)),
                ("attributes", bool(cat.attributes)),
            ):
                if lost:
                    dropped[f"categories.{column}"] += 1
        clean = {c.id: _ILLEGAL_XML.sub("\ufffd", c.name) for c in categories}
        for c in categories:
            if clean[c.id] != c.name:
                dropped["categories.text.invalid_xml_chars"] += 1
        labels = [clean[c.id] for c in categories]
        if len(set(labels)) != len(labels):
            raise FormatError("class names collide after replacing characters illegal in XML")
        name_of = clean
        if any("\n" in n or "\r" in n for n in labels):
            raise FormatError("class names with line breaks cannot be written to labelmap.txt")

        names: dict[str, str] = {}
        stems: set[str] = set()
        for image in sorted(ds.images, key=lambda i: (_hint(i), i.image_id)):
            name = _hint(image)
            files.safe_join(out / "JPEGImages", name)
            stem = PurePosixPath(name).stem
            if stem in stems:
                p = PurePosixPath(name)
                name = str(p.with_name(f"{p.stem}_{image.image_id[:8]}{p.suffix}"))
                stem = PurePosixPath(name).stem
            stems.add(stem)
            names[image.image_id] = name
            if image.group_id is not None:
                dropped["images.group_id"] += 1
            if not is_native(image.source, NAME):
                foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: natural_key(a.ann_id)):
            anns_of.setdefault(ann.image_id, []).append(ann)

        (out / "Annotations").mkdir()
        listed: dict[str, list[str]] = {}
        for image in ds.images:
            name = names[image.image_id]
            stem = PurePosixPath(name).stem
            files.copy_file(Path(image.file_path), files.safe_join(out / "JPEGImages", name))
            root_el = self._annotation_xml(
                image, name, anns_of.get(image.image_id, []), name_of, dropped
            )
            ET.indent(root_el)
            data = ET.tostring(root_el, encoding="utf-8", xml_declaration=False)
            (out / "Annotations" / f"{stem}.xml").write_bytes(data + b"\n")
            if image.split != "unassigned":
                listed.setdefault(image.split, []).append(stem)
        if listed:
            (out / "ImageSets" / "Main").mkdir(parents=True)
            for split, stems_in in listed.items():
                (out / "ImageSets" / "Main" / f"{split}.txt").write_text(
                    "".join(f"{s}\n" for s in sorted(stems_in)), encoding="utf-8"
                )
        (out / "labelmap.txt").write_text("".join(f"{n}\n" for n in labels), encoding="utf-8")
        rows = len(ds.images) + len(ds.annotations)
        return ConversionReport(
            source_format=NAME,
            target_format=NAME,
            rows_read=rows,
            rows_written=rows,
            dropped=dict(dropped),
        )

    @staticmethod
    def _annotation_xml(
        image: Image,
        name: str,
        anns: list[Annotation],
        name_of: dict[int, str],
        dropped: Counter[str],
    ) -> ET.Element:
        root = ET.Element("annotation")
        native = is_native(image.source, NAME)
        emit = _Emit(dropped, "images")
        extras = {
            k: v
            for k, v in image.source.items()
            if native and k not in (NATIVE_KEY, "filename", "depth")
        }
        if "folder" not in extras:
            emit.sub(root, "folder", "JPEGImages")
        emit.sub(root, "filename", name)
        for key, value in extras.items():
            emit.emit(root, key, value)
        size = emit.sub(root, "size")
        emit.sub(size, "width", image.width)
        emit.sub(size, "height", image.height)
        depth = image.source.get("depth") if native else None
        emit.sub(size, "depth", depth if isinstance(depth, str) else "3")
        if "segmented" not in extras:
            emit.sub(root, "segmented", "0")
        for ann in anns:
            root.append(VocFormat._object_xml(ann, name_of, dropped))
        return root

    @staticmethod
    def _object_xml(ann: Annotation, name_of: dict[int, str], dropped: Counter[str]) -> ET.Element:
        if ann.bbox_xyxy_abs is None:
            raise FormatError(f"annotation {ann.ann_id!r} has no box")
        el = ET.Element("object")
        emit = _Emit(dropped, "annotations")
        emit.sub(el, "name", name_of[ann.category_id])
        native = is_native(ann.source, NAME)
        for key, value in ann.source.items():
            if native and key != NATIVE_KEY:
                emit.emit(el, key, value)
        for tag, flag in (("truncated", ann.truncated), ("occluded", ann.occluded)):
            if flag is not None:
                emit.sub(el, tag, int(flag))
        difficult = ann.attributes.get("difficult")
        if isinstance(difficult, bool):
            emit.sub(el, "difficult", int(difficult))
        box = emit.sub(el, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), ann.bbox_xyxy_abs, strict=True):
            emit.sub(box, tag, _fmt(value))
        if set(ann.attributes) - {"difficult"} or (
            "difficult" in ann.attributes and not isinstance(difficult, bool)
        ):
            dropped["annotations.attributes"] += 1
        for column, present in (
            ("is_crowd", ann.is_crowd),
            ("score", ann.score is not None),
            ("provenance", ann.provenance != "human"),
        ):
            if present:
                dropped[f"annotations.{column}"] += 1
        if not native:
            foreign_source_drops(dropped, "annotations", ann.source, frozenset())
        return el


class _Emit:
    """Writes untrusted ``source`` content as XML under hard limits, counting what it skips.

    * Element names must match ``[A-Za-z_][A-Za-z0-9_.-]*`` (at most 64 characters); any other
      key is not written and counted as ``<table>.source.invalid_key``.
    * Characters illegal in XML 1.0 are replaced by U+FFFD in every text value; each affected
      value is counted as ``<table>.text.invalid_xml_chars``.
    * Nesting deeper than 8 levels, or more than 1000 elements per image or object, is not
      written and is counted as ``<table>.source.overflow``.
    """

    def __init__(self, dropped: Counter[str], table: str) -> None:
        self.dropped = dropped
        self.table = table
        self.elements = 0

    def text(self, value: object) -> str:
        text = str(value)
        if _ILLEGAL_XML.search(text):
            self.dropped[f"{self.table}.text.invalid_xml_chars"] += 1
            return _ILLEGAL_XML.sub("\ufffd", text)
        return text

    def sub(self, parent: ET.Element, tag: str, text: object | None = None) -> ET.Element:
        """Trusted, fixed-name child element (counts toward the element budget)."""
        self.elements += 1
        el = ET.SubElement(parent, tag)
        if text is not None:
            el.text = self.text(text)
        return el

    def emit(self, parent: ET.Element, tag: str, value: object, depth: int = 1) -> None:
        if isinstance(value, list):
            for item in value:
                self.emit(parent, tag, item, depth)
            return
        if not self._valid(tag):
            self.dropped[f"{self.table}.source.invalid_key"] += 1
            return
        if depth > MAX_EMIT_DEPTH or self.elements >= MAX_EMIT_ELEMENTS:
            self.dropped[f"{self.table}.source.overflow"] += 1
            return
        el = self.sub(parent, tag)
        if isinstance(value, dict):
            for key, item in value.items():
                self.emit(el, key, item, depth + 1)
        else:
            el.text = self.text(value)

    @staticmethod
    def _valid(tag: str) -> bool:
        return (
            0 < len(tag) <= MAX_NAME_CHARS
            and tag[0] in _NAME_START
            and all(c in _NAME_REST for c in tag)
        )


def _hint(image: Image) -> str:
    if is_native(image.source, NAME):
        name = image.source.get("filename")
        if isinstance(name, str) and name:
            return name
    for key in _NAME_HINTS:
        value = image.source.get(key)
        if isinstance(value, str) and value:
            return PurePosixPath(value).name
    return PurePosixPath(image.file_path).name


plugin = VocFormat()
