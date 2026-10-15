"""CVAT for images 1.1 XML reader and writer (detection).

Layout read: ``annotations.xml``, or ``annotations/*.xml`` (one document per split), or one XML
file given directly. Images are looked up next to the annotation root and under ``images/``. The
writer emits ``annotations.xml`` (no splits) or ``annotations/<split>.xml`` plus
``images/<name>``; every document carries ``<meta><task>`` with the labels and the subset.

Mapped: ``<image>`` (``name`` kept as ``source.file_name``, ``id`` kept as ``source.id``, sizes read
from the file header), ``<box>`` (``xtl ytl xbr ybr``, ``label``, ``occluded``, ``<attribute>``
children as attributes, ``source`` as provenance: ``manual`` is human, ``auto`` is model), the
labels of ``<meta>`` (name, ``color`` and ``type`` as category attributes) and the task ``subset``
as the split. Every other attribute of an image or box is kept verbatim in ``source`` and
re-emitted for CVAT rows.

Never mapped, always counted (``skipped`` ``annotations.unsupported.<shape>``): polygons,
polylines, points, ellipses, masks, skeletons, cuboids, tags, rotated boxes and every ``<track>``.
Label attribute definitions that carry more than a name are counted in ``categories.attributes``.
Meta fields that describe the export rather than the data (ids, dates, owner, size) are neither
kept nor counted.

Parsing goes through :mod:`datacopilot.io.xmlsafe` (no DOCTYPE, no entities, bounded depth,
element and attribute counts) after a bounded read; every number is finite and validated and every
image path is contained in the dataset root.
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
    require_empty_dir,
    require_task,
)
from datacopilot.io.registry import ReadResult
from datacopilot.io.report import ConversionReport
from datacopilot.io.xmlsafe import parse_xml

NAME = "cvat_xml"
MAX_DOCUMENT_ELEMENTS = 1_000_000
MAX_SHAPES_PER_IMAGE = 100_000
MAX_TEXT_CHARS = 4096
MAX_ATTRIBUTE_NAME_CHARS = 256
MAX_ATTRIBUTES_PER_BOX = 256
MAX_LABELS = 100_000
_SPLIT = re.compile(r"(?<![a-z])(train|val|test)(?![a-z])")
_UINT = re.compile(r"[0-9]{1,9}")
_SPLITS = ("train", "val", "test")
_UNSUPPORTED_SHAPES = frozenset(
    {"polygon", "polyline", "points", "ellipse", "mask", "skeleton", "cuboid", "tag"}
)
_LABEL_KEYS = frozenset({"id", "name", "color", "type", "attributes"})
_LABEL_UNSUPPORTED = frozenset({"parent", "svg", "sublabels"})
_BOX_RESERVED = frozenset({"label", "occluded", "xtl", "ytl", "xbr", "ybr", "source"})
_IMAGE_RESERVED = frozenset({"id", "name", "width", "height"})
_ATTR_SPEC_TAGS = frozenset({"name", "mutable", "input_type", "default_value", "values"})
_NAME_HINTS = frozenset({"file_name", "filename"})
_ILLEGAL_XML = re.compile("[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\ud800-\udfff￾￿]")
_NAME_START = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
_NAME_REST = _NAME_START | frozenset("0123456789.-")


def _split_of(stem: str) -> Split:
    match = _SPLIT.search(stem.lower())
    return match.group(1) if match else "unassigned"  # type: ignore[return-value]


def _text(el: ET.Element | None) -> str:
    text = "" if el is None else (el.text or "")
    if len(text) > MAX_TEXT_CHARS:
        raise FormatError(f"<{el.tag if el is not None else ''}> text is too long")
    return text


def _whole_number(raw: str | None, what: str, *, positive: bool) -> int:
    if raw is None or not _UINT.fullmatch(raw) or (positive and int(raw) == 0):
        raise FormatError(
            f"{what} must be a {'positive' if positive else 'non-negative'} whole number"
        )
    return int(raw)


def _coordinate(box: ET.Element, key: str, what: str) -> float:
    raw = box.get(key)
    try:
        value = float(raw) if raw is not None else None
    except ValueError:
        value = None
    if value is None or value != value or value in (float("inf"), float("-inf")):
        raise FormatError(f"{what}: {key} is missing or not a finite number")
    return value


def _flag(raw: str | None, what: str) -> bool | None:
    if raw is None:
        return None
    if raw in ("0", "false", "False"):
        return False
    if raw in ("1", "true", "True"):
        return True
    raise FormatError(f"{what}: occluded must be 0 or 1")


def _checked(value: str, what: str) -> str:
    if len(value) > MAX_TEXT_CHARS:
        raise FormatError(f"{what} is too long")
    return value


def _fmt(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


def _annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    single = path / "annotations.xml"
    if single.is_file():
        return [single]
    folder = path / "annotations"
    found = sorted(folder.glob("*.xml")) if folder.is_dir() else []
    if not found:
        raise FormatError(f"no CVAT annotation file found in {path.name}")
    return found


def _image_roots(annotation_file: Path, root: Path) -> list[Path]:
    roots = [root, root / "images"]
    if annotation_file.parent.name == "annotations":
        roots += [annotation_file.parent.parent, annotation_file.parent.parent / "images"]
    return roots


def _trivial_spec(spec: ET.Element) -> bool:
    """A label attribute definition that says nothing beyond the attribute's name."""
    tags = [child.tag for child in spec]
    return (
        set(tags) <= _ATTR_SPEC_TAGS
        and len(tags) == len(set(tags))
        and _text(spec.find("mutable")).strip() in ("", "False", "false")
        and _text(spec.find("input_type")).strip() in ("", "text")
        and _text(spec.find("default_value")) == ""
        and _text(spec.find("values")) == ""
    )


def _clean(text: str) -> str:
    return _ILLEGAL_XML.sub("�", text)


class CvatFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path) -> ReadResult:
        path = Path(path)
        if not path.exists():
            raise FormatError(f"{path.name} does not exist")
        root = path if path.is_dir() else path.parent
        documents = []
        for file in _annotation_files(path):
            tree = parse_xml(
                files.read_bounded(file, files.MAX_CVAT_BYTES),
                file.name,
                max_elements=MAX_DOCUMENT_ELEMENTS,
            )
            if tree.tag != "annotations":
                raise FormatError(f"{file.name}: root element must be <annotations>")
            documents.append((file, tree))
        dropped: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        categories = self._categories(documents, dropped)
        ids = {c.name: c.id for c in categories}
        prefix = len(documents) > 1
        images: list[Image] = []
        annotations: list[Annotation] = []
        for file, tree in documents:
            self._rows(file, tree, root, prefix, ids, images, annotations, dropped, skipped)
        return ReadResult(
            name=root.name or NAME,
            task="detection",
            images=images,
            annotations=annotations,
            categories=categories,
            dropped=dict(dropped),
            skipped=dict(skipped),
        )

    @staticmethod
    def _label_elements(tree: ET.Element) -> list[ET.Element]:
        found: list[ET.Element] = []
        for parent in ("task", "project"):
            found += tree.findall(f"meta/{parent}/labels/label")
        return found

    def _categories(
        self, documents: list[tuple[Path, ET.Element]], dropped: Counter[str]
    ) -> list[Category]:
        by_name: dict[str, Category] = {}
        for file, tree in documents:
            for label in self._label_elements(tree):
                name = _text(label.find("name"))
                if not name:
                    raise FormatError(f"{file.name}: a label has no name")
                attributes: dict[str, str] = {}
                for key in ("color", "type"):
                    value = _text(label.find(key))
                    if value:
                        attributes[key] = value
                if name in by_name:
                    if by_name[name].attributes != attributes:
                        raise FormatError(f"label {name!r} is defined differently in {file.name}")
                    continue
                if len(by_name) >= MAX_LABELS:
                    raise FormatError(f"{file.name}: more than {MAX_LABELS} labels")
                rich = [
                    spec
                    for spec in label.findall("attributes/attribute")
                    if not _trivial_spec(spec)
                ]
                if rich:
                    dropped["categories.attributes"] += 1
                for child in label:
                    if child.tag in _LABEL_UNSUPPORTED:
                        dropped[f"categories.unsupported.{child.tag}"] += 1
                    elif child.tag not in _LABEL_KEYS:
                        dropped["categories.unsupported.other"] += 1
                by_name[name] = build_row(
                    Category,
                    f"label {name[:40]!r}",
                    id=len(by_name),
                    name=name,
                    attributes=attributes,
                )
        if not by_name:  # exports without <meta> labels: the classes the boxes use, sorted
            used = {
                box.get("label")
                for _, tree in documents
                for box in tree.iter("box")
                if box.get("label")
            }
            for name in sorted(n for n in used if n):
                by_name[name] = Category(id=len(by_name), name=name)
        return list(by_name.values())

    def _rows(
        self,
        file: Path,
        tree: ET.Element,
        root: Path,
        prefix: bool,
        ids: dict[str, int],
        images: list[Image],
        annotations: list[Annotation],
        dropped: Counter[str],
        skipped: Counter[str],
    ) -> None:
        split = self._split(file, tree, dropped)
        seen: set[int] = set()
        for child in tree:
            if child.tag == "track":
                skipped["annotations.unsupported.track"] += 1
            elif child.tag not in ("version", "meta", "image"):
                skipped["dataset.unsupported_element"] += 1
        for el in tree.findall("image"):
            image_id = _whole_number(el.get("id"), "image id", positive=False)
            if image_id in seen:
                raise FormatError(f"{file.name}: duplicate image id {image_id}")
            seen.add(image_id)
            name = el.get("name")
            if not name:
                raise FormatError(f"{file.name}: image {image_id} has no name")
            shapes = [c for c in el if c.tag != "attribute"]
            if len(shapes) > MAX_SHAPES_PER_IMAGE:
                raise FormatError(f"{file.name}: image {image_id} has too many shapes")
            declared = None
            if el.get("width") is not None or el.get("height") is not None:
                declared = (
                    _whole_number(el.get("width"), f"image {image_id} width", positive=True),
                    _whole_number(el.get("height"), f"image {image_id} height", positive=True),
                )
            located = self._locate(name, _image_roots(file, root))
            if located is None:
                skipped["images.file_missing"] += 1
                if shapes:
                    skipped["annotations.of_missing_image"] += len(shapes)
                continue
            extras: dict[str, str] = {}
            for key, value in el.attrib.items():
                if key not in _IMAGE_RESERVED:
                    extras[key] = _checked(value, f"image {image_id} attribute {key}")
            source = {NATIVE_KEY: NAME, "id": image_id, "file_name": name} | extras
            image = make_image(located, split=split, source=source, ref=f"image {image_id}")
            if declared is not None and declared != (image.width, image.height):
                dropped["images.declared_size"] += 1
            images.append(image)
            index = 0
            for shape in shapes:
                if shape.tag in _UNSUPPORTED_SHAPES:
                    skipped[f"annotations.unsupported.{shape.tag}"] += 1
                    continue
                if shape.tag != "box":
                    skipped["annotations.unsupported.other"] += 1
                    continue
                stem = f"{file.stem}:" if prefix else ""
                ann_id = f"{stem}{image_id}:{index:06d}"
                row = self._box(shape, ann_id, image, ids, skipped, dropped)
                if row is not None:
                    annotations.append(row)
                    index += 1

    @staticmethod
    def _split(file: Path, tree: ET.Element, dropped: Counter[str]) -> Split:
        subset = _text(tree.find("meta/task/subset")).strip().lower()
        if subset in _SPLITS:
            return subset  # type: ignore[return-value]
        if subset and subset != "default":
            dropped["images.subset"] += 1
        return _split_of(file.stem)

    @staticmethod
    def _box(
        el: ET.Element,
        ann_id: str,
        image: Image,
        ids: dict[str, int],
        skipped: Counter[str],
        dropped: Counter[str],
    ) -> Annotation | None:
        where = f"image {image.source.get('id')} box"
        label = el.get("label")
        if not label or label not in ids:
            raise FormatError(f"{where}: label {(label or '')[:40]!r} is not defined")
        rotation = el.get("rotation")
        if rotation is not None:
            try:
                turned = float(rotation) != 0.0
            except ValueError:
                raise FormatError(f"{where}: rotation is not a number") from None
            if turned:
                skipped["annotations.unsupported.rotated_box"] += 1
                return None
        coords = tuple(_coordinate(el, tag, where) for tag in ("xtl", "ytl", "xbr", "ybr"))
        attributes: dict[str, str] = {}
        for child in el:
            if child.tag != "attribute":
                dropped["annotations.unsupported_child"] += 1
                continue
            key = _checked(child.get("name") or "", f"{where} attribute name")
            if not key:
                raise FormatError(f"{where}: an attribute has no name")
            if key in attributes:
                raise FormatError(f"{where}: duplicate attribute {key[:40]!r}")
            if len(attributes) >= MAX_ATTRIBUTES_PER_BOX:
                raise FormatError(f"{where}: too many attributes")
            attributes[key] = _text(child)
        extras = {
            k: _checked(v, f"{where} attribute {k}")
            for k, v in el.attrib.items()
            if k not in _BOX_RESERVED
        }
        provenance = "human"
        origin = el.get("source")
        if origin == "auto":
            provenance = "model"
        elif origin not in (None, "manual"):
            extras["source"] = _checked(origin, f"{where} source")
        return build_row(
            Annotation,
            where,
            ann_id=ann_id,
            image_id=image.image_id,
            category_id=ids[label],
            kind="bbox",
            bbox_xyxy_abs=coords,
            occluded=_flag(el.get("occluded"), where),
            attributes=attributes,
            provenance=provenance,
            source={NATIVE_KEY: NAME, **extras},
        )

    @staticmethod
    def _locate(name: str, roots: list[Path]) -> Path | None:
        found: Path | None = None
        for base in roots:
            candidate = files.safe_join(base, name)
            if found is None and candidate.is_file():
                found = candidate
        return found

    # ----------------------------------------------------------------- write
    def write(self, ds: Dataset, out: Path) -> ConversionReport:
        out = Path(out)
        require_empty_dir(out)
        require_task(ds.manifest.task, "detection", NAME)
        dropped: Counter[str] = Counter()
        categories = sorted(ds.categories, key=lambda c: c.id)
        labels = self._labels(categories, dropped)
        names = self._names(ds)
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: a.ann_id):
            anns_of.setdefault(ann.image_id, []).append(ann)
        by_split: dict[str, list[Image]] = {}
        for image in sorted(ds.images, key=lambda i: (names[i.image_id], i.image_id)):
            by_split.setdefault(image.split, []).append(image)

        used_attrs: dict[int, set[str]] = {}
        for ann in ds.annotations:
            used_attrs.setdefault(ann.category_id, set()).update(ann.attributes)
        documents: dict[str, ET.Element] = {}
        for split, members in by_split.items():
            documents[split] = self._document(
                ds, split, members, names, anns_of, labels, used_attrs, dropped
            )
        if set(documents) <= {"unassigned"}:
            targets = {"unassigned": out / "annotations.xml"}
        else:
            (out / "annotations").mkdir()
            targets = {s: out / "annotations" / f"{s}.xml" for s in documents}
        for split, element in documents.items():
            ET.indent(element)
            targets[split].write_bytes(
                b'<?xml version="1.0" encoding="utf-8"?>\n'
                + ET.tostring(element, encoding="utf-8", xml_declaration=False)
                + b"\n"
            )
        for image in ds.images:
            files.copy_file(
                Path(image.file_path), files.safe_join(out / "images", names[image.image_id])
            )
        rows = len(ds.images) + len(ds.annotations)
        return ConversionReport(
            source_format=NAME,
            target_format=NAME,
            rows_read=rows,
            rows_written=rows,
            dropped=dict(dropped),
        )

    @staticmethod
    def _labels(categories: list[Category], dropped: Counter[str]) -> dict[int, Category]:
        clean: dict[int, Category] = {}
        for position, cat in enumerate(categories):
            for column, lost in (
                ("id", cat.id != position),
                ("aliases", bool(cat.aliases)),
                ("attributes", bool(set(cat.attributes) - {"color", "type"})),
            ):
                if lost:
                    dropped[f"categories.{column}"] += 1
            name = _clean(cat.name)
            if name != cat.name:
                dropped["categories.text.invalid_xml_chars"] += 1
            clean[cat.id] = cat.model_copy(update={"name": name})
        names = [c.name for c in clean.values()]
        if len(set(names)) != len(names):
            raise FormatError("class names collide after replacing characters illegal in XML")
        return clean

    @staticmethod
    def _names(ds: Dataset) -> dict[str, str]:
        used: set[str] = set()
        names: dict[str, str] = {}
        for image in sorted(ds.images, key=lambda i: (_hint(i), i.image_id)):
            names[image.image_id] = files.unique_name(_hint(image), used, image.image_id[:8])
            files.safe_join(Path("images"), names[image.image_id])
        return names

    def _document(
        self,
        ds: Dataset,
        split: str,
        members: list[Image],
        names: dict[str, str],
        anns_of: dict[str, list[Annotation]],
        labels: dict[int, Category],
        used_attrs: dict[int, set[str]],
        dropped: Counter[str],
    ) -> ET.Element:
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        task = ET.SubElement(ET.SubElement(root, "meta"), "task")
        ET.SubElement(task, "name").text = _clean(ds.manifest.name)
        ET.SubElement(task, "size").text = str(len(members))
        ET.SubElement(task, "mode").text = "annotation"
        if split != "unassigned":
            ET.SubElement(task, "subset").text = split
        label_root = ET.SubElement(task, "labels")
        for cat in labels.values():
            label = ET.SubElement(label_root, "label")
            ET.SubElement(label, "name").text = cat.name
            for key in ("color", "type"):
                if key in cat.attributes:
                    ET.SubElement(label, key).text = _clean(cat.attributes[key])
            specs = ET.SubElement(label, "attributes")
            for attr in sorted(used_attrs.get(cat.id, ())):
                spec = ET.SubElement(specs, "attribute")
                ET.SubElement(spec, "name").text = _clean(attr)
                ET.SubElement(spec, "mutable").text = "False"
                ET.SubElement(spec, "input_type").text = "text"
                ET.SubElement(spec, "default_value").text = ""
                ET.SubElement(spec, "values").text = ""
        taken = {
            i.source["id"]
            for i in members
            if is_native(i.source, NAME) and isinstance(i.source.get("id"), int)
        }
        next_id = max((v for v in taken if isinstance(v, int)), default=-1) + 1
        used: set[int] = set()
        for image in members:
            native = is_native(image.source, NAME)
            wanted = image.source.get("id") if native else None
            if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted not in used:
                image_id = wanted
            else:
                while next_id in used or next_id in taken:
                    next_id += 1
                image_id = next_id
            used.add(image_id)
            el = ET.SubElement(root, "image")
            el.set("id", str(image_id))
            el.set("name", _clean(names[image.image_id]))
            el.set("width", str(image.width))
            el.set("height", str(image.height))
            if native:
                for key, value in image.source.items():
                    if key in (NATIVE_KEY, "id", "file_name"):
                        continue
                    if _valid_name(key) and key not in _IMAGE_RESERVED and isinstance(value, str):
                        el.set(key, _clean(value))
                    else:
                        dropped[f"images.source.{_safe_key(key)}"] += 1
            else:
                foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
            if image.group_id is not None:
                dropped["images.group_id"] += 1
            for ann in anns_of.get(image.image_id, []):
                el.append(self._box_xml(ann, labels, dropped))
        return root

    @staticmethod
    def _box_xml(ann: Annotation, labels: dict[int, Category], dropped: Counter[str]) -> ET.Element:
        if ann.bbox_xyxy_abs is None:
            raise FormatError(f"annotation {ann.ann_id!r} has no box")
        el = ET.Element("box")
        el.set("label", labels[ann.category_id].name)
        native = is_native(ann.source, NAME)
        origin = ann.source.get("source") if native else None
        el.set(
            "source",
            origin
            if isinstance(origin, str)
            else ("auto" if ann.provenance == "model" else "manual"),
        )
        if ann.occluded is not None:
            el.set("occluded", str(int(ann.occluded)))
        for tag, coord in zip(("xtl", "ytl", "xbr", "ybr"), ann.bbox_xyxy_abs, strict=True):
            el.set(tag, _fmt(coord))
        if native:
            for key, extra in ann.source.items():
                if key in (NATIVE_KEY, "source"):
                    continue
                if _valid_name(key) and key not in _BOX_RESERVED and isinstance(extra, str):
                    el.set(key, _clean(extra))
                else:
                    dropped[f"annotations.source.{_safe_key(key)}"] += 1
        else:
            foreign_source_drops(dropped, "annotations", ann.source, frozenset())
        stringified = False
        for key, attr in ann.attributes.items():
            child = ET.SubElement(el, "attribute")
            child.set("name", _clean(key))
            if isinstance(attr, bool):
                text, stringified = ("true" if attr else "false"), True
            elif isinstance(attr, str):
                text = attr
            else:
                text, stringified = repr(attr), True
            child.text = _clean(text)
        for column, present in (
            ("attributes_stringified", stringified),
            ("is_crowd", ann.is_crowd),
            ("truncated", ann.truncated is not None),
            ("score", ann.score is not None),
            ("provenance", ann.provenance == "proposal_accepted"),
        ):
            if present:
                dropped[f"annotations.{column}"] += 1
        return el


def _valid_name(tag: str) -> bool:
    return 0 < len(tag) <= 64 and tag[0] in _NAME_START and all(c in _NAME_REST for c in tag)


def _safe_key(key: str) -> str:
    return key if _valid_name(key) else "invalid_key"


def _hint(image: Image) -> str:
    if is_native(image.source, NAME):
        name = image.source.get("file_name")
        if isinstance(name, str) and name:
            return name
    for key in _NAME_HINTS:
        value = image.source.get(key)
        if isinstance(value, str) and value:
            return PurePosixPath(value).name
    return PurePosixPath(image.file_path).name


plugin = CvatFormat()
