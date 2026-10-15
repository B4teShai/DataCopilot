"""COCO detection reader and writer.

Layout read: a single ``annotations.json``, or ``annotations/*.json`` (one file per split, the
split inferred from a ``train``/``val``/``test`` word in the file name), or one JSON file given
directly. Images are looked up next to the annotation root and under ``images/``. The writer
emits ``annotations.json`` (no splits) or ``annotations/instances_<split>.json`` plus
``images/<file_name>``.

Mapped: bbox (xywh <-> xyxy), category id, ``iscrowd``, ``score``, image size, image ``group_id``,
category ``supercategory``. Everything else on an image or annotation is kept verbatim in
``source`` and re-emitted when the row came from COCO.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

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
    parse_json,
    require_empty_dir,
    require_task,
)
from datacopilot.io.registry import ReadResult
from datacopilot.io.report import ConversionReport

NAME = "coco"
_SPLIT = re.compile(r"(?<![a-z])(train|val|test)(?![a-z])")
_IMAGE_RESERVED = frozenset({"width", "height", "group_id"})
_ANN_RESERVED = frozenset({"image_id", "category_id", "bbox", "iscrowd", "score"})
_CATEGORY_KEYS = frozenset({"id", "name", "supercategory"})
_TOP_LEVEL = frozenset({"images", "annotations", "categories"})
_NAME_HINTS = frozenset({"file_name", "filename"})


def _split_of(stem: str) -> Split:
    match = _SPLIT.search(stem.lower())
    return match.group(1) if match else "unassigned"  # type: ignore[return-value]


def _annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    single = path / "annotations.json"
    if single.is_file():
        return [single]
    found = sorted((path / "annotations").glob("*.json")) if (path / "annotations").is_dir() else []
    if not found:
        raise FormatError(f"no COCO annotation file found in {path.name}")
    return found


def _image_roots(annotation_file: Path, root: Path) -> list[Path]:
    roots = [root, root / "images"]
    if annotation_file.parent.name == "annotations":
        roots += [annotation_file.parent.parent, annotation_file.parent.parent / "images"]
    return roots


def _need(obj: object, key: str, kind: type | tuple[type, ...], what: str) -> Any:
    """``obj[key]`` checked to be present and of ``kind`` (bool never counts as a number)."""
    if not isinstance(obj, dict) or key not in obj:
        raise FormatError(f"{what} is missing {key!r}")
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise FormatError(f"{what}: {key!r} has the wrong type")
    return value


def _bbox(ann: dict[str, object], what: str) -> geometry.Box:
    raw = ann.get("bbox")
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw)
    ):
        raise FormatError(f"{what}: bbox must be a list of four numbers")
    x, y, w, h = (float(v) for v in raw)
    return geometry.xywh_to_xyxy((x, y, w, h))


class CocoFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path) -> ReadResult:
        path = Path(path)
        if not path.exists():
            raise FormatError(f"{path.name} does not exist")
        root = path if path.is_dir() else path.parent
        annotation_files = _annotation_files(path)
        prefix_ids = len(annotation_files) > 1
        images: list[Image] = []
        annotations: list[Annotation] = []
        categories: dict[int, Category] = {}
        dropped: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        for file in annotation_files:
            doc = parse_json(files.read_bounded(file, files.MAX_JSON_BYTES), file.name)
            if not isinstance(doc, dict):
                raise FormatError(f"{file.name}: top level must be a JSON object")
            for key in doc:
                if key not in _TOP_LEVEL:
                    dropped[f"dataset.{key}"] += 1
            self._categories(doc, file, categories, dropped)
            self._rows(
                doc,
                file,
                root,
                _split_of(file.stem),
                prefix_ids,
                images,
                annotations,
                dropped,
                skipped,
            )
        return ReadResult(
            name=root.name or "coco",
            task="detection",
            images=images,
            annotations=annotations,
            categories=sorted(categories.values(), key=lambda c: c.id),
            dropped=dict(dropped),
            skipped=dict(skipped),
        )

    def _categories(
        self,
        doc: dict[str, object],
        file: Path,
        out: dict[int, Category],
        dropped: Counter[str],
    ) -> None:
        raw = doc.get("categories")
        if not isinstance(raw, list):
            raise FormatError(f"{file.name}: 'categories' must be a list")
        for item in raw:
            cid = _need(item, "id", int, "category")
            name = _need(item, "name", str, f"category {cid}")
            attributes: dict[str, str] = {}
            sup = item.get("supercategory")
            if isinstance(sup, str):
                attributes["supercategory"] = sup
            elif sup is not None:
                dropped["categories.supercategory"] += 1
            for key in item:
                if key not in _CATEGORY_KEYS:
                    dropped[f"categories.{key}"] += 1
            cat = build_row(Category, f"category {cid}", id=cid, name=name, attributes=attributes)
            if cid in out and out[cid] != cat:
                raise FormatError(f"category id {cid} is defined differently in {file.name}")
            if cid not in out:
                out[cid] = cat

    def _rows(
        self,
        doc: dict[str, object],
        file: Path,
        root: Path,
        split: Split,
        prefix_ids: bool,
        images: list[Image],
        annotations: list[Annotation],
        dropped: Counter[str],
        skipped: Counter[str],
    ) -> None:
        raw_images, raw_anns = doc.get("images"), doc.get("annotations")
        if not isinstance(raw_images, list):
            raise FormatError(f"{file.name}: 'images' must be a list")
        if not isinstance(raw_anns, list):
            raise FormatError(f"{file.name}: 'annotations' must be a list")
        by_id: dict[object, Image | None] = {}
        for item in raw_images:
            iid = _need(item, "id", (int, str), "image")
            name = _need(item, "file_name", str, f"image {iid}")
            if iid in by_id:
                raise FormatError(f"{file.name}: duplicate image id {iid}")
            located = self._locate(name, _image_roots(file, root))
            if located is None:
                skipped["images.file_missing"] += 1
                by_id[iid] = None
                continue
            group = item.get("group_id")
            if group is not None and not isinstance(group, str):
                raise FormatError(f"image {iid}: 'group_id' must be a string")
            source = {NATIVE_KEY: NAME} | {
                k: v for k, v in item.items() if k not in _IMAGE_RESERVED
            }
            image = make_image(located, split=split, source=source, ref=f"image {iid}")
            declared = (item.get("width"), item.get("height"))
            if any(d is not None for d in declared) and declared != (image.width, image.height):
                dropped["images.declared_size"] += 1
            if group is not None:
                image = image.model_copy(update={"group_id": group})
            by_id[iid] = image
            images.append(image)
        for item in raw_anns:
            aid = _need(item, "id", (int, str), "annotation")
            what = f"annotation {aid}"
            image_ref = _need(item, "image_id", (int, str), what)
            cat = _need(item, "category_id", int, what)
            if image_ref not in by_id:
                raise FormatError(f"{what} references unknown image id {image_ref}")
            target = by_id[image_ref]
            if target is None:
                skipped["annotations.of_missing_image"] += 1
                continue
            box = _bbox(item, what)
            score = item.get("score")
            source = {NATIVE_KEY: NAME} | {k: v for k, v in item.items() if k not in _ANN_RESERVED}
            annotations.append(
                build_row(
                    Annotation,
                    what,
                    ann_id=f"{file.stem}:{aid}" if prefix_ids else str(aid),
                    image_id=target.image_id,
                    category_id=cat,
                    kind="bbox",
                    bbox_xyxy_abs=box,
                    score=score,
                    is_crowd=bool(item.get("iscrowd", 0)),
                    provenance="model" if score is not None else "human",
                    source=source,
                )
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
        names = self._names(ds)
        categories = []
        for cat in sorted(ds.categories, key=lambda c: c.id):
            entry: dict[str, object] = {"id": cat.id, "name": cat.name}
            extra = {k: v for k, v in cat.attributes.items() if k != "supercategory"}
            if "supercategory" in cat.attributes:
                entry["supercategory"] = cat.attributes["supercategory"]
            if cat.aliases:
                dropped["categories.aliases"] += 1
            if extra:
                dropped["categories.attributes"] += 1
            categories.append(entry)

        used_image_ids = {
            i.source["id"]
            for i in ds.images
            if is_native(i.source, NAME) and isinstance(i.source.get("id"), int)
        }
        next_image_id = max((v for v in used_image_ids if isinstance(v, int)), default=0) + 1
        by_split: dict[str, list[Image]] = {}
        for image in sorted(ds.images, key=lambda i: (names[i.image_id], i.image_id)):
            by_split.setdefault(image.split, []).append(image)
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: a.ann_id):
            anns_of.setdefault(ann.image_id, []).append(ann)
        used_ann_ids = {
            a.source["id"]
            for a in ds.annotations
            if is_native(a.source, NAME) and isinstance(a.source.get("id"), int)
        }
        next_ann_id = max((v for v in used_ann_ids if isinstance(v, int)), default=0) + 1

        documents: dict[str, dict[str, object]] = {}
        for split, members in by_split.items():
            doc_images: list[dict[str, object]] = []
            doc_anns: list[dict[str, object]] = []
            for image in members:
                native = is_native(image.source, NAME)
                if native and "id" in image.source:
                    image_id: object = image.source["id"]
                else:
                    image_id = next_image_id
                    next_image_id += 1
                entry = {
                    "id": image_id,
                    "file_name": names[image.image_id],
                    "width": image.width,
                    "height": image.height,
                }
                if native:
                    for key, value in image.source.items():
                        if key not in (NATIVE_KEY, "id", "file_name") and key not in entry:
                            entry[key] = value
                else:
                    foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
                if image.group_id is not None:
                    entry["group_id"] = image.group_id
                doc_images.append(entry)
                for ann in anns_of.get(image.image_id, []):
                    if is_native(ann.source, NAME) and "id" in ann.source:
                        ann_id: object = ann.source["id"]
                    else:
                        ann_id = next_ann_id
                        next_ann_id += 1
                    doc_anns.append(self._annotation(ann, ann_id, image_id, dropped))
            documents[split] = {
                "images": doc_images,
                "annotations": doc_anns,
                "categories": categories,
            }

        if set(documents) <= {"unassigned"}:
            targets = {"unassigned": out / "annotations.json"}
        else:
            (out / "annotations").mkdir()
            targets = {s: out / "annotations" / f"instances_{s}.json" for s in documents}
        for split, doc in documents.items():
            targets[split].write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False))
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
    def _names(ds: Dataset) -> dict[str, str]:
        used: set[str] = set()
        names: dict[str, str] = {}
        ordered = sorted(ds.images, key=lambda i: (_hint(i), i.image_id))
        for image in ordered:
            hint = _hint(image)
            names[image.image_id] = files.unique_name(hint, used, image.image_id[:8])
            files.safe_join(Path("images"), names[image.image_id])
        return names

    @staticmethod
    def _annotation(
        ann: Annotation, ann_id: object, image_id: object, dropped: Counter[str]
    ) -> dict[str, object]:
        if ann.bbox_xyxy_abs is None:
            raise FormatError(f"annotation {ann.ann_id!r} has no box")
        x, y, w, h = geometry.xyxy_to_xywh(ann.bbox_xyxy_abs)
        entry: dict[str, object] = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann.category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": int(ann.is_crowd),
            "segmentation": [],
        }
        if ann.score is not None:
            entry["score"] = ann.score
        if is_native(ann.source, NAME):
            for key, value in ann.source.items():
                if key not in (NATIVE_KEY, "id") and key not in _ANN_RESERVED:
                    entry[key] = value
        else:
            foreign_source_drops(dropped, "annotations", ann.source, frozenset())
        for column, present in (
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
        return entry


def _hint(image: Image) -> str:
    """Preferred output file name: the source's own name when it has one, else the file name."""
    if is_native(image.source, NAME):
        name = image.source.get("file_name")
        if isinstance(name, str):
            return name
    for key in _NAME_HINTS:
        value = image.source.get(key)
        if isinstance(value, str) and value:
            return PurePosixPath(value).name
    return PurePosixPath(image.file_path).name


plugin = CocoFormat()
