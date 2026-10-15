"""Label Studio JSON export reader and writer (detection and classification).

Layout read: ``annotations.json``, or ``annotations/*.json`` (one file per split, the split
inferred from a ``train``/``val``/``test`` word in the file name), or one JSON file given
directly. The file is the "JSON" export: an array of tasks. Images are looked up next to the
annotation root and under ``images/``. The writer emits ``annotations.json`` (no splits) or one
document per original file (``annotations/<name>.json``, or ``annotations/<split>.json`` for rows
that came from elsewhere) plus ``images/<name>``.

Mapped: ``rectanglelabels`` results (percent of the original image size, converted to pixels of the
real image; boxes are rounded to 1e-6 pixel) and ``choices`` results (classification). A task
becomes an image; ``annotations`` results are human rows, results of the first ``predictions``
entry are model rows (the result's or the prediction's ``score``; a null score is no score). The
task type is decided by the human rows first (boxes, else choices) and only without any human row by
the model rows; rows of the other kind are counted. Categories are the sorted label names in use
(the labelling configuration is never read), so a category that no row uses is not carried and the
writer counts it (``categories.unused``).

Multiple annotators: one annotation set per task is read, never merged: the first annotation
marked ``ground_truth``, else the lowest ``id``. Cancelled annotations and every other annotation
or prediction are counted (``skipped``: ``annotations.cancelled``,
``annotations.of_other_annotators``, ``annotations.of_other_predictions``). Predictions of a task
without a kept annotation are not read as rows (they would be unreviewed model output standing in
for labels): counted as ``annotations.prediction_of_unlabeled_task``.

Personal and free-form data. Exports carry usernames, e-mail addresses, comment authors, upload
names and presigned URLs in task, annotation, prediction and result fields. Only an allow-list is
read into ``source`` and re-emitted: the task fields ``created_at``, ``updated_at`` and
``inner_id``; the annotation fields ``id``, ``ground_truth``, ``was_cancelled``, ``lead_time``,
``created_at`` and ``updated_at`` (short scalars); the prediction fields ``id``,
``model_version``, ``score``, ``created_at`` and ``updated_at``; the result fields ``id``,
``from_name``, ``to_name`` and ``origin``. Every other key is dropped and counted by name
(``images.source.<key>``, ``images.source.annotation.<key>``,
``images.source.prediction.<key>``, ``annotations.source.<key>``; the names are bounded, see
``common.safe_key``), including ``completed_by``, ``created_username``, ``updated_by``,
``comment_authors``, ``meta`` and ``file_upload``. Task ``data`` fields other than ``image`` are
dropped and counted too (``images.source.data.<key>``) unless ``keep_task_data`` is set
(``datacopilot ingest --keep-task-data``); kept data is stored and re-emitted as it is, so use it
only for data you have checked. The allow-list protects the stored dataset, the reports and the
re-exported JSON; it cannot know that a label name, an image file name or an allowed text field
holds personal data.

Where a row came from is kept in ``layout_`` keys of ``source`` (never counted by another format's
writer): the original document name (when read from ``annotations/``) and the original form of the
image reference, so a same-format write reproduces both. Never mapped, always counted
(``skipped`` ``annotations.unsupported.<type>``): polygons, brushes, keypoints, text areas,
relations, every other result type, rectangles with several labels, rotated rectangles and
rotated images. Images that are not local files (``http``, ``s3``, ``data:`` and so on) are
counted as ``images.remote_uri`` and never fetched.

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
    count,
    foreign_source_drops,
    is_native,
    make_image,
    parse_json,
    require_empty_dir,
    safe_key,
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
_TASK_KEEP = frozenset({"created_at", "updated_at", "inner_id"})
_ANNOTATION_KEEP = frozenset(
    {"id", "ground_truth", "was_cancelled", "lead_time", "created_at", "updated_at"}
)
_PREDICTION_KEEP = frozenset({"id", "model_version", "score", "created_at", "updated_at"})
_RESULT_KEEP = frozenset({"origin"})
_RESULT_RESERVED = frozenset(
    {"original_width", "original_height", "image_rotation", "value", "id", "from_name", "to_name"}
    | {"type"}
)
LAYOUT_DOCUMENT = "layout_document"
LAYOUT_REF = "layout_image_ref"
MAX_SCALAR_CHARS = 256
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


def _scalar(value: object) -> bool:
    """A short JSON scalar (text, finite number or flag): the only shape of a kept field."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, str) and len(value) <= MAX_SCALAR_CHARS


def _keep(
    meta: dict[str, object], allowed: frozenset[str], table: str, dropped: Counter[str]
) -> dict[str, object]:
    """The allow-listed, scalar fields of ``meta``; every other key is dropped and counted.

    A null is an absent value, not a loss. ``table`` names the report column family.
    """
    kept: dict[str, object] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if key in allowed and _scalar(value):
            kept[key] = value
        else:
            count(dropped, f"{table}.{safe_key(key)}")
    return kept


def _skip(counter: Counter[str], key: str, amount: int) -> None:
    """Count skipped rows; a zero is nothing (a zero entry would make an invalid report)."""
    if amount > 0:
        count(counter, key, amount)


class LabelStudioFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path, *, keep_task_data: bool = False) -> ReadResult:
        path = Path(path)
        if not path.exists():
            raise FormatError(f"{path.name} does not exist")
        root = path if path.is_dir() else path.parent
        documents = _annotation_files(path)
        folder = path.is_dir() and not (path / "annotations.json").is_file()
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
                    item, file, root, folder, keep_task_data, seen, images, raws, dropped, skipped
                )
        return self._finish(root, images, raws, dropped, skipped)

    def _task(
        self,
        item: object,
        file: Path,
        root: Path,
        folder: bool,
        keep_task_data: bool,
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
            count(skipped, "images.remote_uri" if remote else "images.file_missing")
            _skip(
                skipped,
                "annotations.of_missing_image",
                sum(len(e.results) for e in annotations + predictions),
            )
            return
        chosen = self._choose(annotations, skipped)
        first = predictions[0] if predictions else None
        for other in predictions[1:]:
            _skip(skipped, "annotations.of_other_predictions", len(other.results))
        if chosen is None and first is not None:
            _skip(skipped, "annotations.prediction_of_unlabeled_task", len(first.results))
            first = None
        source: dict[str, object] = {NATIVE_KEY: NAME, "id": tid, "file_name": name}
        if ref != f"images/{name}":
            source[LAYOUT_REF] = ref
        if folder:
            source[LAYOUT_DOCUMENT] = file.name
        for key, value in data.items():
            if key == "image":
                continue
            if keep_task_data:
                source.setdefault("data", {})[key] = value  # type: ignore[index]
            else:
                count(dropped, f"images.source.data.{safe_key(key)}")
        task_extras = {k: v for k, v in item.items() if k not in _TASK_RESERVED}
        source.update(_keep(task_extras, _TASK_KEEP, "images.source", dropped))
        if chosen is None:
            source["unlabeled"] = True
        else:
            source["annotation"] = _keep(
                chosen.meta, _ANNOTATION_KEEP, "images.source.annotation", dropped
            )
        if first is not None:
            source["prediction"] = _keep(
                first.meta, _PREDICTION_KEEP, "images.source.prediction", dropped
            )
        image = make_image(located, split=_split_of(file.stem), source=source, ref=what)
        sets: list[tuple[str, _Entry, float | None]] = []
        if chosen is not None:
            sets.append(("a", chosen, None))
        if first is not None:
            sets.append(("p", first, self._score(first.meta, what)))
        prefix = f"{file.stem}:" if folder else ""
        mismatch = False
        for origin, entry, score in sets:
            for index, result in enumerate(entry.results):
                mismatch |= self._result(
                    result, f"{what} result {index}", image, tid, prefix, origin, score, raws,
                    dropped, skipped,
                )  # fmt: skip
        if mismatch:
            count(dropped, "images.declared_size")
        images.append(image)

    @staticmethod
    def _score(meta: dict[str, object], what: str) -> float | None:
        value = meta.get("score")
        return None if value is None else _number(value, f"{what}: prediction score")

    @staticmethod
    def _choose(entries: list[_Entry], skipped: Counter[str]) -> _Entry | None:
        live = []
        for entry in entries:
            if entry.meta.get("was_cancelled") is True:
                _skip(skipped, "annotations.cancelled", len(entry.results))
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
                _skip(skipped, "annotations.of_other_annotators", len(entry.results))
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
            count(skipped, f"annotations.unsupported.{key}")
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
        extras = {k: v for k, v in result.items() if k not in reserved}
        source.update(_keep(extras, _RESULT_KEEP, "annotations.source", dropped))
        score = prediction_score
        if origin == "p" and result.get("score") is not None:
            score = _number(result["score"], f"{what}: score")
        raw = _Raw(image, prefix, tid, origin, [], None, score, source)
        if kind == "choices":
            choices = value.get("choices")
            if not isinstance(choices, list) or len(choices) > MAX_CHOICES_PER_RESULT:
                raise FormatError(f"{what}: choices must be a short list")
            raw.labels = [_text(c, f"{what}: choice label", MAX_LABEL_CHARS) for c in choices]
            if not raw.labels:
                count(skipped, "annotations.unsupported.empty_choices")
                return False
            for key in value:
                if key != "choices":
                    count(dropped, f"annotations.source.value.{safe_key(key)}")
            raws.append(raw)
            return False
        labels = value.get("rectanglelabels")
        if not isinstance(labels, list):
            raise FormatError(f"{what}: rectanglelabels must be a list")
        if len(labels) != 1:
            count(skipped, "annotations.unsupported.multi_label_rectangle")
            return False
        if _number(value.get("rotation", 0), f"{what}: rotation") != 0:
            count(skipped, "annotations.unsupported.rotated_rectangle")
            return False
        if _number(result.get("image_rotation", 0), f"{what}: image_rotation") != 0:
            count(skipped, "annotations.unsupported.image_rotation")
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
        for key in value:
            if key not in _VALUE_RESERVED:
                count(dropped, f"annotations.source.value.{safe_key(key)}")
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
            count(dropped, "annotations.bbox_snapped")
        return (x1, y1, x2, y2)

    def _finish(
        self,
        root: Path,
        images: list[Image],
        raws: list[_Raw],
        dropped: Counter[str],
        skipped: Counter[str],
    ) -> ReadResult:
        by_people = [r for r in raws if r.origin == "a"]
        decisive = by_people or raws
        task: Task = (
            "detection"
            if any(r.box is not None for r in decisive) or not any(r.labels for r in decisive)
            else "classification"
        )
        wanted = [r for r in raws if (r.box is not None) == (task == "detection")]
        for raw in raws:
            if raw not in wanted:
                kind = "choices" if task == "detection" else "rectanglelabels"
                count(skipped, f"annotations.unsupported.{kind}")
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
        used = {a.category_id for a in ds.annotations}
        # a reader numbers the categories that rows use, by name: rank exactly those
        for rank, cat in enumerate(
            sorted((c for c in ds.categories if c.id in used), key=_by_name)
        ):
            if not 0 < len(cat.name) <= MAX_LABEL_CHARS:
                raise FormatError(f"class name is longer than {MAX_LABEL_CHARS} characters")
            for column, lost in (
                ("id", cat.id != rank),
                ("aliases", bool(cat.aliases)),
                ("attributes", bool(cat.attributes)),
            ):
                if lost:
                    count(dropped, f"categories.{column}")
            label_of[cat.id] = cat.name
        for cat in ds.categories:
            if cat.id not in used:  # no row names it, and the format has no list of classes
                count(dropped, "categories.unused")
        names = self._names(ds)
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: a.ann_id):
            anns_of.setdefault(ann.image_id, []).append(ann)
        groups: dict[tuple[str, str], list[Image]] = {}
        for image in sorted(ds.images, key=lambda i: (names[i.image_id], i.image_id)):
            groups.setdefault(_document_of(image), []).append(image)
        splits_of: dict[str, set[str]] = {}
        for document, split in groups:
            splits_of.setdefault(document, set()).add(split)
        files_out: dict[str, list[dict[str, object]]] = {}
        for (document, split), members in groups.items():
            target = document
            if len(splits_of[document]) > 1:  # one original file whose images changed split
                target = f"{document.removesuffix('.json')}_{split}.json"
            tasks = self._tasks(members, names, anns_of, label_of, dropped)
            files_out.setdefault(target, []).extend(tasks)
        single = set(files_out) <= {"unassigned.json"} and not any(
            is_native(i.source, NAME) and LAYOUT_DOCUMENT in i.source for i in ds.images
        )
        if single:
            targets = {name: out / "annotations.json" for name in files_out}
        else:
            (out / "annotations").mkdir()
            targets = {name: out / "annotations" / name for name in files_out}
        for name, tasks in files_out.items():
            targets[name].write_text(
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
                if native:
                    count(dropped, "images.source.id")
            used.add(tid)
            name = names[image.image_id]
            ref = f"images/{name}"
            if native:
                original = image.source.get(LAYOUT_REF)
                if name != image.source.get("file_name"):
                    count(dropped, "images.source.file_name")  # renamed to keep names unique
                elif isinstance(original, str):
                    ref = original
            data: dict[str, object] = {}
            extra_data = image.source.get("data") if native else None
            if isinstance(extra_data, dict):
                data.update(extra_data)
            data["image"] = ref
            task: dict[str, object] = {}
            if native:
                task.update({k: v for k, v in image.source.items() if k in _TASK_KEEP})
            else:
                foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
            task.update({"id": tid, "data": data})
            if image.group_id is not None:
                count(dropped, "images.group_id")
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
            pmeta = image.source.get("prediction") if native else None
            if models or isinstance(pmeta, dict):
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
            result.update({k: v for k, v in ann.source.items() if k in _RESULT_KEEP})
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
                count(dropped, f"annotations.{column}")
        return result


_DOCUMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.json")


def _by_name(category: Category) -> str:
    return category.name


def _document_of(image: Image) -> tuple[str, str]:
    """``(output document name, split)``: the original file when the row came from one."""
    if is_native(image.source, NAME):
        document = image.source.get(LAYOUT_DOCUMENT)
        if isinstance(document, str) and _DOCUMENT_NAME.fullmatch(document):
            return document, image.split
    return f"{image.split}.json", image.split


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
