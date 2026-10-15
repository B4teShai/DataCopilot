"""Label Studio JSON export reader and writer (detection and classification).

Layout read: ``annotations.json``, or ``annotations/*.json`` (one file per split, the split
inferred from a ``train``/``val``/``test`` word in the file name), or one JSON file given
directly. The file is the "JSON" export: an array of tasks. Images are looked up next to the
annotation root and under ``images/``. The writer emits ``annotations.json`` (no splits) or
``annotations/<split>.json`` plus ``images/<name>``.

Mapped: ``rectanglelabels`` results (percent of the original image size, converted to pixels of
the real image; boxes are rounded to 1e-6 pixel) and ``choices`` results (classification). A task
becomes an image; ``annotations`` results are human rows, results of the first ``predictions``
entry are model rows (the result's or the prediction's ``score``). The task type is detection when
any box was read, otherwise classification. Categories are the sorted label names in use (the
labelling configuration is never read, so classes without a row are not carried).

Multiple annotators: one annotation set per task is read, never merged: the first annotation
marked ``ground_truth``, else the lowest ``id``. Cancelled annotations and every other annotation
or prediction are counted (``skipped``: ``annotations.cancelled``,
``annotations.of_other_annotators``, ``annotations.of_other_predictions``). Everything else on a
task, annotation, prediction or result is kept verbatim in ``source`` and re-emitted for Label
Studio rows. Personal fields of ``completed_by`` (anything but the numeric user id) are never
stored.

Never mapped, always counted (``skipped`` ``annotations.unsupported.<type>``): polygons, brushes,
keypoints, text areas, relations, every other result type, rectangles with several labels, rotated
rectangles and rotated images. Images that are not local files (``http``, ``s3``, ``data:`` and
so on) are counted as ``images.remote_uri`` and never fetched.

The JSON is read through a bounded descriptor, duplicate keys and non-finite numbers are refused,
every percentage and size is validated and every image path is contained in the dataset root.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlsplit

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Category, Dataset, Image, PluginSpec, Split, Task
from datacopilot.io import files
from datacopilot.io.common import (
    NATIVE_KEY,
    build_row,
    foreign_source_drops,
    is_native,
    make_image,
    parse_json,
    require_empty_dir,
)
from datacopilot.io.registry import ReadResult
from datacopilot.io.report import ConversionReport

NAME = "label_studio"
MAX_TASKS = 1_000_000
MAX_ANNOTATIONS_PER_TASK = 1000
MAX_RESULTS_PER_TASK = 100_000
MAX_CHOICES_PER_RESULT = 1000
MAX_REF_CHARS = 4096
MAX_LABEL_CHARS = 4096
MAX_FIELD_CHARS = 256
MAX_DIMENSION = 1_000_000_000
SNAP_PIXELS = 1e-3
_SPLIT = re.compile(r"(?<![a-z])(train|val|test)(?![a-z])")
_REMOTE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*://|data:)")
_NAME_HINTS = frozenset({"file_name", "filename"})
_TASK_RESERVED = frozenset({"id", "data", "annotations", "predictions"})
_TASK_INTERNAL = frozenset({NATIVE_KEY, "file_name", "annotation", "prediction", "unlabeled"})
_RESULT_RESERVED = frozenset(
    {"original_width", "original_height", "image_rotation", "value", "id", "from_name", "to_name"}
    | {"type"}
)
_VALUE_RESERVED = frozenset({"x", "y", "width", "height", "rotation", "rectanglelabels"})
_KNOWN_UNSUPPORTED = frozenset(
    {
        "polygonlabels",
        "polygon",
        "brushlabels",
        "keypointlabels",
        "ellipselabels",
        "rectangle",
        "textarea",
        "taxonomy",
        "rating",
        "number",
        "datetime",
        "labels",
        "hypertextlabels",
        "paragraphlabels",
        "timeserieslabels",
        "videorectangle",
        "relation",
        "pairwise",
    }
)
_DEFAULT_FROM = {"rectanglelabels": "label", "choices": "choice"}


def _split_of(stem: str) -> Split:
    match = _SPLIT.search(stem.lower())
    return match.group(1) if match else "unassigned"  # type: ignore[return-value]


def _annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    single = path / "annotations.json"
    if single.is_file():
        return [single]
    folder = path / "annotations"
    found = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not found:
        raise FormatError(f"no Label Studio annotation file found in {path.name}")
    return found


def _image_roots(annotation_file: Path, root: Path) -> list[Path]:
    roots = [root, root / "images"]
    if annotation_file.parent.name == "annotations":
        roots += [annotation_file.parent.parent, annotation_file.parent.parent / "images"]
    return roots


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatError(f"{what} must be a finite number")
    if not math.isfinite(value):
        raise FormatError(f"{what} must be a finite number")
    return float(value)


def _percent(value: dict[str, object], key: str, what: str, *, positive: bool) -> float:
    number = _number(value.get(key), f"{what}: {key} (a percent from 0 to 100)")
    low_ok = number > 0 if positive else number >= 0
    if not low_ok or number > 100:
        raise FormatError(f"{what}: {key} must be a finite number between 0 and 100 percent")
    return number


def _dimension(value: object, key: str, what: str) -> float:
    number = _number(value, f"{what}: {key}")
    if not 0 < number <= MAX_DIMENSION:
        raise FormatError(f"{what}: {key} must be a positive number (the original image size)")
    return number


def _text(value: object, what: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise FormatError(f"{what} must be a text of 1 to {limit} characters")
    return value


def _sort_id(value: object) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return math.inf


@dataclass
class _Raw:
    """One supported result, before categories and the dataset task are known."""

    image: Image
    prefix: str
    task_id: int
    origin: str  # "a" for annotations, "p" for predictions
    labels: list[str]
    box: tuple[float, float, float, float] | None
    score: float | None
    source: dict[str, object] = field(default_factory=dict)


@dataclass
class _Entry:
    meta: dict[str, object]
    results: list[object]


def _entries(items: object, what: str, noun: str) -> list[_Entry]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise FormatError(f"{what}: {noun}s must be a list")
    if len(items) > MAX_ANNOTATIONS_PER_TASK:
        raise FormatError(f"{what}: too many {noun}s")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise FormatError(f"{what}: a {noun} is not an object")
        results = item.get("result")
        if not isinstance(results, list):
            raise FormatError(f"{what}: a {noun} has no result list")
        if len(results) > MAX_RESULTS_PER_TASK:
            raise FormatError(f"{what}: too many results")
        out.append(_Entry({k: v for k, v in item.items() if k != "result"}, results))
    return out


def _clean_meta(meta: dict[str, object]) -> dict[str, object]:
    """Annotation or prediction fields to keep; ``completed_by`` keeps only a numeric user id."""
    out = dict(meta)
    who = out.get("completed_by")
    if isinstance(who, dict):
        who = who.get("id")
    if isinstance(who, int) and not isinstance(who, bool):
        out["completed_by"] = who
    else:
        out.pop("completed_by", None)
    return out


class LabelStudioFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path) -> ReadResult:
        path = Path(path)
        if not path.exists():
            raise FormatError(f"{path.name} does not exist")
        root = path if path.is_dir() else path.parent
        documents = _annotation_files(path)
        prefix = len(documents) > 1
        images: list[Image] = []
        raws: list[_Raw] = []
        dropped: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        for file in documents:
            doc = parse_json(
                files.read_bounded(file, files.MAX_LABEL_STUDIO_BYTES), file.name, unique_keys=True
            )
            if not isinstance(doc, list):
                raise FormatError(f"{file.name}: top level must be a JSON array of tasks")
            if len(doc) > MAX_TASKS:
                raise FormatError(f"{file.name}: too many tasks")
            seen: set[int] = set()
            for item in doc:
                self._task(
                    item, file, root, f"{file.stem}:" if prefix else "", seen, images, raws,
                    dropped, skipped,
                )  # fmt: skip
        return self._finish(root, images, raws, dropped, skipped)

    def _task(
        self,
        item: object,
        file: Path,
        root: Path,
        prefix: str,
        seen: set[int],
        images: list[Image],
        raws: list[_Raw],
        dropped: Counter[str],
        skipped: Counter[str],
    ) -> None:
        if not isinstance(item, dict):
            raise FormatError(f"{file.name}: a task is not an object")
        tid = item.get("id")
        if isinstance(tid, bool) or not isinstance(tid, int) or tid < 0:
            raise FormatError(f"{file.name}: a task has no valid id")
        what = f"task {tid}"
        if tid in seen:
            raise FormatError(f"{file.name}: duplicate task id {tid}")
        seen.add(tid)
        data = item.get("data")
        if not isinstance(data, dict):
            raise FormatError(f"{what}: data must be an object")
        ref = data.get("image")
        if not isinstance(ref, str):
            raise FormatError(f"{what}: data.image must be a text")
        annotations = _entries(item.get("annotations"), what, "annotation")
        predictions = _entries(item.get("predictions"), what, "prediction")
        located, name, remote = self._locate(ref, _image_roots(file, root), root)
        if located is None:
            skipped["images.remote_uri" if remote else "images.file_missing"] += 1
            total = sum(len(e.results) for e in annotations + predictions)
            if total:
                skipped["annotations.of_missing_image"] += total
            return
        chosen = self._choose(annotations, skipped)
        first = predictions[0] if predictions else None
        for other in predictions[1:]:
            skipped["annotations.of_other_predictions"] += len(other.results)
        source: dict[str, object] = {NATIVE_KEY: NAME, "id": tid, "file_name": name}
        extra_data = {k: v for k, v in data.items() if k != "image"}
        if extra_data:
            source["data"] = extra_data
        for key, value in item.items():
            if key in _TASK_RESERVED:
                continue
            if key in _TASK_INTERNAL:
                dropped[f"images.source.{key}"] += 1
            else:
                source[key] = value
        if chosen is None:
            source["unlabeled"] = True
        elif meta := _clean_meta(chosen.meta):
            source["annotation"] = meta
        if first is not None and (pmeta := _clean_meta(first.meta)):
            source["prediction"] = pmeta
        image = make_image(located, split=_split_of(file.stem), source=source, ref=what)
        sets: list[tuple[str, _Entry, float | None]] = []
        if chosen is not None:
            sets.append(("a", chosen, None))
        if first is not None:
            sets.append(("p", first, self._score(first.meta, what)))
        mismatch = False
        for origin, entry, score in sets:
            for index, result in enumerate(entry.results):
                mismatch |= self._result(
                    result, f"{what} result {index}", image, tid, prefix, origin, score, raws,
                    dropped, skipped,
                )  # fmt: skip
        if mismatch:
            dropped["images.declared_size"] += 1
        images.append(image)

    @staticmethod
    def _score(meta: dict[str, object], what: str) -> float | None:
        if "score" not in meta:
            return None
        return _number(meta["score"], f"{what}: prediction score")

    @staticmethod
    def _choose(entries: list[_Entry], skipped: Counter[str]) -> _Entry | None:
        live = []
        for entry in entries:
            if entry.meta.get("was_cancelled") is True:
                skipped["annotations.cancelled"] += len(entry.results)
            else:
                live.append(entry)
        if not live:
            return None
        pool = [e for e in live if e.meta.get("ground_truth") is True] or live
        chosen = min(enumerate(pool), key=lambda pair: (_sort_id(pair[1].meta.get("id")), pair[0]))[
            1
        ]
        for entry in live:
            if entry is not chosen:
                skipped["annotations.of_other_annotators"] += len(entry.results)
        return chosen

    @staticmethod
    def _locate(ref: str, roots: list[Path], root: Path) -> tuple[Path | None, str, bool]:
        """``(path, name under images/, remote)``; the path is None for a missing or remote file."""
        if len(ref) > MAX_REF_CHARS:
            raise FormatError("an image reference is too long")
        if _REMOTE.match(ref):
            return None, ref, True
        candidates: list[Path]
        if ref.startswith("/data/local-files/"):
            try:
                query = parse_qs(urlsplit(ref).query)
            except ValueError:
                raise FormatError("an image reference is malformed") from None
            given = query.get("d", [])
            if len(given) != 1:
                raise FormatError("a local-files reference needs exactly one d parameter")
            name = given[0]
            candidates = [files.safe_join(root, name)]
        elif ref.startswith("/data/upload/"):
            parts = ref.split("/", 4)
            name = unquote(parts[4]) if len(parts) == 5 else ""
            candidates = [files.safe_join(root / "images", name), files.safe_join(root, name)]
        else:
            name = ref
            candidates = [files.safe_join(base, ref) for base in roots]
        shown = name[len("images/") :] if name.startswith("images/") and name != "images/" else name
        found = next((c for c in candidates if c.is_file()), None)
        return found, shown, False

    def _result(
        self,
        result: object,
        what: str,
        image: Image,
        tid: int,
        prefix: str,
        origin: str,
        prediction_score: float | None,
        raws: list[_Raw],
        dropped: Counter[str],
        skipped: Counter[str],
    ) -> bool:
        """Read one result; True when it declares an original size that differs from the file."""
        if not isinstance(result, dict):
            raise FormatError(f"{what} is not an object")
        kind = result.get("type")
        if kind not in ("rectanglelabels", "choices"):
            key = kind if isinstance(kind, str) and kind in _KNOWN_UNSUPPORTED else "other"
            skipped[f"annotations.unsupported.{key}"] += 1
            return False
        value = result.get("value")
        if not isinstance(value, dict):
            raise FormatError(f"{what}: value must be an object")
        source: dict[str, object] = {
            NATIVE_KEY: NAME,
            "from_name": _text(
                result.get("from_name", _DEFAULT_FROM[kind]), f"{what}: from_name", MAX_FIELD_CHARS
            ),
            "to_name": _text(result.get("to_name", "image"), f"{what}: to_name", MAX_FIELD_CHARS),
        }
        if "id" in result:
            source["id"] = _text(result["id"], f"{what}: id", MAX_FIELD_CHARS)
        reserved = _RESULT_RESERVED | ({"score"} if origin == "p" else frozenset())
        for key, item in result.items():
            if key == NATIVE_KEY:
                dropped["annotations.source.format"] += 1
            elif key not in reserved:
                source[key] = item
        score = prediction_score
        if origin == "p" and "score" in result:
            score = _number(result["score"], f"{what}: score")
        raw = _Raw(image, prefix, tid, origin, [], None, score, source)
        if kind == "choices":
            choices = value.get("choices")
            if not isinstance(choices, list) or len(choices) > MAX_CHOICES_PER_RESULT:
                raise FormatError(f"{what}: choices must be a short list")
            raw.labels = [_text(c, f"{what}: choice label", MAX_LABEL_CHARS) for c in choices]
            if not raw.labels:
                skipped["annotations.unsupported.empty_choices"] += 1
                return False
            extras = {k: v for k, v in value.items() if k != "choices"}
            if extras:
                source["value"] = extras
            raws.append(raw)
            return False
        labels = value.get("rectanglelabels")
        if not isinstance(labels, list):
            raise FormatError(f"{what}: rectanglelabels must be a list")
        if len(labels) != 1:
            skipped["annotations.unsupported.multi_label_rectangle"] += 1
            return False
        if _number(value.get("rotation", 0), f"{what}: rotation") != 0:
            skipped["annotations.unsupported.rotated_rectangle"] += 1
            return False
        if _number(result.get("image_rotation", 0), f"{what}: image_rotation") != 0:
            skipped["annotations.unsupported.image_rotation"] += 1
            return False
        raw.labels = [_text(labels[0], f"{what}: label", MAX_LABEL_CHARS)]
        x = _percent(value, "x", what, positive=False)
        y = _percent(value, "y", what, positive=False)
        w = _percent(value, "width", what, positive=True)
        h = _percent(value, "height", what, positive=True)
        mismatch = False
        if "original_width" in result or "original_height" in result:
            ow = _dimension(result.get("original_width"), "original_width", what)
            oh = _dimension(result.get("original_height"), "original_height", what)
            mismatch = (ow, oh) != (float(image.width), float(image.height))
        raw.box = self._pixels(x, y, w, h, image, what, dropped)
        extras = {k: v for k, v in value.items() if k not in _VALUE_RESERVED}
        if extras:
            source["value"] = extras
        raws.append(raw)
        return mismatch

    @staticmethod
    def _pixels(
        x: float, y: float, w: float, h: float, image: Image, what: str, dropped: Counter[str]
    ) -> tuple[float, float, float, float]:
        width, height = float(image.width), float(image.height)
        x1, y1 = round(x * width / 100, 6), round(y * height / 100, 6)
        x2 = round(x * width / 100 + w * width / 100, 6)
        y2 = round(y * height / 100 + h * height / 100, 6)
        snapped = False
        if x2 > width:
            if x2 - width > SNAP_PIXELS:
                raise FormatError(f"{what}: the box extends beyond the image")
            x2, snapped = width, True
        if y2 > height:
            if y2 - height > SNAP_PIXELS:
                raise FormatError(f"{what}: the box extends beyond the image")
            y2, snapped = height, True
        if snapped:
            dropped["annotations.bbox_snapped"] += 1
        return (x1, y1, x2, y2)

    def _finish(
        self,
        root: Path,
        images: list[Image],
        raws: list[_Raw],
        dropped: Counter[str],
        skipped: Counter[str],
    ) -> ReadResult:
        task: Task = (
            "detection"
            if any(r.box is not None for r in raws) or not any(r.labels for r in raws)
            else "classification"
        )
        wanted = [r for r in raws if (r.box is not None) == (task == "detection")]
        for raw in raws:
            if raw not in wanted:
                skipped["annotations.unsupported.choices"] += 1
        names = sorted({label for r in wanted for label in r.labels})
        ids = {name: i for i, name in enumerate(names)}
        counters: Counter[tuple[str, str]] = Counter()
        annotations: list[Annotation] = []
        for raw in wanted:
            for label in raw.labels:
                key = (raw.prefix + str(raw.task_id), raw.origin)
                if counters[key] > 999_999:
                    raise FormatError(f"task {raw.task_id}: too many rows")
                ann_id = f"{raw.prefix}{raw.task_id}:{raw.origin}{counters[key]:06d}"
                counters[key] += 1
                human = raw.origin == "a"
                annotations.append(
                    build_row(
                        Annotation,
                        f"task {raw.task_id}",
                        ann_id=ann_id,
                        image_id=raw.image.image_id,
                        category_id=ids[label],
                        kind="bbox" if task == "detection" else "classification",
                        bbox_xyxy_abs=raw.box,
                        score=None if human else raw.score,
                        provenance="human" if human else "model",
                        source=dict(raw.source),
                    )
                )
        return ReadResult(
            name=root.name or NAME,
            task=task,
            images=images,
            annotations=annotations,
            categories=[Category(id=i, name=n) for n, i in ids.items()],
            dropped=dict(dropped),
            skipped=dict(skipped),
        )

    # ----------------------------------------------------------------- write
    def write(self, ds: Dataset, out: Path) -> ConversionReport:
        out = Path(out)
        require_empty_dir(out)
        dropped: Counter[str] = Counter()
        label_of: dict[int, str] = {}
        for rank, cat in enumerate(sorted(ds.categories, key=lambda c: c.name)):
            if not 0 < len(cat.name) <= MAX_LABEL_CHARS:
                raise FormatError(f"class name is longer than {MAX_LABEL_CHARS} characters")
            for column, lost in (
                ("id", cat.id != rank),
                ("aliases", bool(cat.aliases)),
                ("attributes", bool(cat.attributes)),
            ):
                if lost:
                    dropped[f"categories.{column}"] += 1
            label_of[cat.id] = cat.name
        names = self._names(ds)
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: a.ann_id):
            anns_of.setdefault(ann.image_id, []).append(ann)
        by_split: dict[str, list[Image]] = {}
        for image in sorted(ds.images, key=lambda i: (names[i.image_id], i.image_id)):
            by_split.setdefault(image.split, []).append(image)
        documents = {
            split: self._tasks(members, names, anns_of, label_of, dropped)
            for split, members in by_split.items()
        }
        if set(documents) <= {"unassigned"}:
            targets = {"unassigned": out / "annotations.json"}
        else:
            (out / "annotations").mkdir()
            targets = {s: out / "annotations" / f"{s}.json" for s in documents}
        for split, tasks in documents.items():
            targets[split].write_text(
                json.dumps(tasks, ensure_ascii=False, allow_nan=False), encoding="utf-8"
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
    def _names(ds: Dataset) -> dict[str, str]:
        used: set[str] = set()
        names: dict[str, str] = {}
        for image in sorted(ds.images, key=lambda i: (_hint(i), i.image_id)):
            names[image.image_id] = files.unique_name(_hint(image), used, image.image_id[:8])
            files.safe_join(Path("images"), names[image.image_id])
        return names

    def _tasks(
        self,
        members: list[Image],
        names: dict[str, str],
        anns_of: dict[str, list[Annotation]],
        label_of: dict[int, str],
        dropped: Counter[str],
    ) -> list[dict[str, object]]:
        taken = {
            i.source["id"]
            for i in members
            if is_native(i.source, NAME) and isinstance(i.source.get("id"), int)
        }
        next_id = max((v for v in taken if isinstance(v, int)), default=-1) + 1
        used: set[int] = set()
        tasks: list[dict[str, object]] = []
        for image in members:
            native = is_native(image.source, NAME)
            wanted = image.source.get("id") if native else None
            if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted not in used:
                tid = wanted
            else:
                while next_id in used or next_id in taken:
                    next_id += 1
                tid = next_id
            used.add(tid)
            data: dict[str, object] = {}
            extra_data = image.source.get("data") if native else None
            if isinstance(extra_data, dict):
                data.update(extra_data)
            data["image"] = f"images/{names[image.image_id]}"
            task: dict[str, object] = {}
            if native:
                for key, value in image.source.items():
                    if key not in _TASK_RESERVED | _TASK_INTERNAL:
                        task[key] = value
            else:
                foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
            task.update({"id": tid, "data": data})
            if image.group_id is not None:
                dropped["images.group_id"] += 1
            humans: list[dict[str, object]] = []
            models: list[dict[str, object]] = []
            for ann in anns_of.get(image.image_id, []):
                result = self._result_json(ann, image, label_of, dropped)
                (models if ann.provenance == "model" else humans).append(result)
            meta = image.source.get("annotation") if native else None
            unlabeled = native and image.source.get("unlabeled") is True
            if humans or not unlabeled:
                task["annotations"] = [
                    {**(meta if isinstance(meta, dict) else {}), "result": humans}
                ]
            else:
                task["annotations"] = []
            if models:
                pmeta = image.source.get("prediction") if native else None
                task["predictions"] = [
                    {**(pmeta if isinstance(pmeta, dict) else {}), "result": models}
                ]
            tasks.append(task)
        return tasks

    @staticmethod
    def _result_json(
        ann: Annotation, image: Image, label_of: dict[int, str], dropped: Counter[str]
    ) -> dict[str, object]:
        native = is_native(ann.source, NAME)
        label = label_of[ann.category_id]
        if ann.bbox_xyxy_abs is None:
            kind = "choices"
            value: dict[str, object] = {}
            result: dict[str, object] = {}
        else:
            x1, y1, x2, y2 = ann.bbox_xyxy_abs
            kind = "rectanglelabels"
            value = {}
            result = {
                "original_width": image.width,
                "original_height": image.height,
                "image_rotation": 0,
            }
        extra_value = ann.source.get("value") if native else None
        if isinstance(extra_value, dict):
            value.update(extra_value)
        if kind == "choices":
            value["choices"] = [label]
        else:
            value.update(
                {
                    "x": x1 * 100 / image.width,
                    "y": y1 * 100 / image.height,
                    "width": (x2 - x1) * 100 / image.width,
                    "height": (y2 - y1) * 100 / image.height,
                    "rotation": 0,
                    "rectanglelabels": [label],
                }
            )
        if native:
            for key, item in ann.source.items():
                if key not in {NATIVE_KEY, "value", "from_name", "to_name", "id"}:
                    result[key] = item
        else:
            foreign_source_drops(dropped, "annotations", ann.source, frozenset())
        from_name = ann.source.get("from_name") if native else None
        to_name = ann.source.get("to_name") if native else None
        result.update(
            {
                "value": value,
                "from_name": from_name if isinstance(from_name, str) else _DEFAULT_FROM[kind],
                "to_name": to_name if isinstance(to_name, str) else "image",
                "type": kind,
            }
        )
        rid = ann.source.get("id") if native else None
        if isinstance(rid, str):
            result["id"] = rid
        if ann.provenance == "model" and ann.score is not None:
            result["score"] = ann.score
        for column, present in (
            ("is_crowd", ann.is_crowd),
            ("occluded", ann.occluded is not None),
            ("truncated", ann.truncated is not None),
            ("attributes", bool(ann.attributes)),
            ("provenance", ann.provenance == "proposal_accepted"),
        ):
            if present:
                dropped[f"annotations.{column}"] += 1
        return result


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


plugin = LabelStudioFormat()
