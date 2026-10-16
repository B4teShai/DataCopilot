"""ImageFolder reader and writer (classification).

Layout: ``<root>/<class>/<image>`` or, when every top-level directory is named ``train``,
``val`` or ``test``, ``<root>/<split>/<class>/<image>``. Class ids follow the sorted class
names; empty class directories are kept. One image carries one classification annotation.
Images outside a class directory are skipped and counted.
"""

from __future__ import annotations

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

NAME = "imagefolder"
SPLITS = ("train", "val", "test")
_NAME_HINTS = frozenset({"file_name", "filename"})


def _dirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


class ImageFolderFormat:
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    # ------------------------------------------------------------------ read
    def read(self, path: Path) -> ReadResult:
        root = Path(path)
        if not root.is_dir():
            raise FormatError(f"{root.name} does not exist or is not a directory")
        top = _dirs(root)
        has_splits = bool(top) and all(d.name in SPLITS for d in top)
        groups: list[tuple[Split, Path]] = (
            [(d.name, d) for d in top] if has_splits else [("unassigned", root)]  # type: ignore[misc]
        )
        classes = sorted({c.name for _, base in groups for c in _dirs(base)})
        if not classes:
            raise FormatError(f"no class directories found in {root.name}")
        ids = {name: i for i, name in enumerate(classes)}
        skipped: Counter[str] = Counter()
        images: list[Image] = []
        annotations: list[Annotation] = []
        for split, base in groups:
            for loose in (p for p in base.iterdir() if p.is_file()):
                if loose.suffix.lower() in files.IMAGE_EXTENSIONS:
                    skipped["images.without_class"] += 1
            for class_dir in _dirs(base):
                for file in sorted(p for p in class_dir.rglob("*") if p.is_file()):
                    suffix = file.suffix.lower()
                    files.refuse_unsupported(file.name)
                    if suffix not in files.IMAGE_EXTENSIONS:
                        continue
                    rel = file.relative_to(root).as_posix()
                    located = files.safe_join(root, rel)
                    file_name = file.relative_to(class_dir).as_posix()
                    image = make_image(
                        located,
                        split=split,
                        source={NATIVE_KEY: NAME, "file_name": file_name},
                        ref=rel,
                    )
                    images.append(image)
                    annotations.append(
                        build_row(
                            Annotation,
                            rel,
                            ann_id=rel,
                            image_id=image.image_id,
                            category_id=ids[class_dir.name],
                            kind="classification",
                            source={NATIVE_KEY: NAME},
                        )
                    )
        return ReadResult(
            name=root.name or NAME,
            task="classification",
            images=images,
            annotations=annotations,
            categories=[Category(id=i, name=n) for n, i in ids.items()],
            skipped=dict(skipped),
        )

    # ----------------------------------------------------------------- write
    def write(self, ds: Dataset, out: Path) -> ConversionReport:
        out = Path(out)
        require_empty_dir(out)
        require_task(ds.manifest.task, "classification", NAME)
        splits = {i.split for i in ds.images}
        if "unassigned" in splits and len(splits) > 1:
            raise FormatError(
                "imagefolder cannot hold both split and unassigned images; assign every split first"
            )
        dropped: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        ranked = sorted(ds.categories, key=lambda c: c.name)
        for rank, cat in enumerate(ranked):
            name = cat.name
            if "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
                raise FormatError(f"class name {name!r} cannot be used as a directory name")
            files.safe_join(out, name)
            for column, lost in (
                ("id", cat.id != rank),
                ("aliases", bool(cat.aliases)),
                ("attributes", bool(cat.attributes)),
            ):
                if lost:
                    dropped[f"categories.{column}"] += 1
        class_of = {c.id: c.name for c in ds.categories}
        anns_of: dict[str, list[Annotation]] = {}
        for ann in sorted(ds.annotations, key=lambda a: a.ann_id):
            anns_of.setdefault(ann.image_id, []).append(ann)

        base = {s: ("" if s == "unassigned" else s + "/") for s in splits} or {"unassigned": ""}
        for prefix in base.values():
            for cat in ds.categories:
                (out / (prefix + cat.name)).mkdir(parents=True, exist_ok=True)

        used: dict[str, set[str]] = {}
        written = 0
        for image in sorted(ds.images, key=lambda i: (_hint(i), i.image_id)):
            anns = anns_of.get(image.image_id)
            if not anns:
                skipped["images.unlabelled"] += 1
                continue
            ann = anns[0]
            if len(anns) > 1:
                dropped["annotations.multiple_labels"] += len(anns) - 1
            directory = base[image.split] + class_of[ann.category_id]
            name = files.unique_name(
                _hint(image), used.setdefault(directory, set()), image.image_id[:8]
            )
            files.copy_file(Path(image.file_path), files.safe_join(out / directory, name))
            written += 1
            self._count(image, ann, dropped)
        rows = len(ds.images) + len(ds.annotations)
        return ConversionReport(
            source_format=NAME,
            target_format=NAME,
            rows_read=rows,
            rows_written=written * 2,
            dropped=dict(dropped),
            skipped=dict(skipped),
        )

    @staticmethod
    def _count(image: Image, ann: Annotation, dropped: Counter[str]) -> None:
        if image.group_id is not None:
            dropped["images.group_id"] += 1
        if not is_native(image.source, NAME):
            foreign_source_drops(dropped, "images", image.source, _NAME_HINTS)
        for column, present in (
            ("is_crowd", ann.is_crowd),
            ("occluded", ann.occluded is not None),
            ("truncated", ann.truncated is not None),
            ("attributes", bool(ann.attributes)),
            ("score", ann.score is not None),
            ("provenance", ann.provenance != "human"),
        ):
            if present:
                dropped[f"annotations.{column}"] += 1
        if not is_native(ann.source, NAME):
            foreign_source_drops(dropped, "annotations", ann.source, frozenset())


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


plugin = ImageFolderFormat()
