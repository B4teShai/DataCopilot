"""The integrity invariant of the CVAT and Label Studio adapters, judged by behaviour.

For every input in a hostile but valid corpus (a fixed regression list and a generated one) a
same-format read, write, read cycle must end in exactly one of two ways:

1. the rows are identical (ann_id, image_id, names, categories, every ``source`` field) and the
   ``manifest.version`` is equal; or
2. the reader's or the writer's ``ConversionReport`` counts a difference, so that ``--strict``
   fails.

A third outcome (the data changed and every report is lossless) is a bug, whatever the cause. A
clean refusal (a typed ``DataCopilotError`` before anything is written) is the only other
acceptable end, and only for the inputs the corpus marks as invalid.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import pytest
from conftest import dataset_from, table_rows
from fixtures import synth
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from datacopilot import convert
from datacopilot.core.errors import DataCopilotError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io.files import image_id as real_id
from datacopilot.store import DatasetStore

FORMATS = ("cvat_xml", "label_studio")
W, H = 64, 48


@dataclass
class Box:
    label: str
    xyxy: tuple[float, float, float, float] = (8.0, 6.0, 40.0, 30.0)
    attrs: dict[str, str] = field(default_factory=dict)  # CVAT: attribute children
    extra: dict[str, str] = field(default_factory=dict)  # CVAT: XML attributes of the box


@dataclass
class Item:
    folder: str  # directory below images/
    name: str
    ident: int  # CVAT frame id, Label Studio task id
    document: str  # the file the item is listed in
    boxes: list[Box] = field(default_factory=list)
    extra: dict[str, str] = field(default_factory=dict)  # CVAT: XML attributes of the image


@dataclass
class Case:
    items: list[Item]
    expect: str = "cycle"  # or "refused": a typed error is the right answer


def png(root: Path, item: Item, seed: int) -> None:
    synth.write_png(root / "images" / item.folder / item.name, W, H, seed)


def relative(item: Item) -> str:
    return f"{item.folder}/{item.name}" if item.folder else item.name


# ----------------------------------------------------------------------------------- builders
def build_cvat(root: Path, case: Case) -> None:
    labels = sorted({b.label for i in case.items for b in i.boxes if b.label})
    meta = "".join(f"<label><name>{escape(n)}</name></label>" for n in labels)
    documents: dict[str, str] = {}
    for seed, item in enumerate(case.items, start=1):
        png(root, item, seed)
        attrs = "".join(f" {k}={quoteattr(v)}" for k, v in item.extra.items())
        boxes = "".join(
            f'<box label={quoteattr(b.label)} source="manual" occluded="0" xtl="{b.xyxy[0]}" '
            f'ytl="{b.xyxy[1]}" xbr="{b.xyxy[2]}" ybr="{b.xyxy[3]}"'
            + "".join(f" {k}={quoteattr(v)}" for k, v in b.extra.items())
            + ">"
            + "".join(
                f"<attribute name={quoteattr(k)}>{escape(v)}</attribute>"
                for k, v in b.attrs.items()
            )
            + "</box>"
            for b in item.boxes
        )
        documents[item.document] = documents.get(item.document, "") + (
            f'<image id="{item.ident}" name={quoteattr(relative(item))} width="{W}" height="{H}"'
            f"{attrs}>{boxes}</image>"
        )
    for document, body in documents.items():
        text = (
            '<?xml version="1.0" encoding="utf-8"?><annotations><version>1.1</version>'
            f"<meta><task><labels>{meta}</labels></task></meta>{body}</annotations>"
        )
        target = root / "annotations" / document
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def build_ls(root: Path, case: Case) -> None:
    documents: dict[str, list[dict[str, Any]]] = {}
    for seed, item in enumerate(case.items, start=1):
        png(root, item, seed)
        results = [
            synth.ls_rect(b.label, b.xyxy, (W, H), f"r{n}") for n, b in enumerate(item.boxes)
        ]
        documents.setdefault(item.document, []).append(
            {
                "id": item.ident,
                "data": {"image": f"images/{relative(item)}"},
                "annotations": [{"id": 1, "result": results}],
                **item.extra,
            }
        )
    for document, tasks in documents.items():
        target = root / "annotations" / document
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")


BUILD: dict[str, Callable[[Path, Case], None]] = {"cvat_xml": build_cvat, "label_studio": build_ls}
DOCUMENT = {"cvat_xml": ".xml", "label_studio": ".json"}


# ---------------------------------------------------------------------------- the invariant
def lossless(*reports: Any) -> bool:
    return all(r.is_lossless() for r in reports)


def cycle(fmt: str, case: Case, work: Path) -> str:
    """Run the cycle and assert the invariant; returns "same", "counted" or "refused"."""
    source = work / "src"
    BUILD[fmt](source, case)
    try:
        first = convert.ingest(source, fmt, work / "a.dcp")
        ds = DatasetStore.open(work / "a.dcp").read()
        written = convert.export(ds, fmt, work / "out")
        second = convert.ingest(work / "out", fmt, work / "b.dcp")
        ds2 = DatasetStore.open(work / "b.dcp").read()
    except DataCopilotError:
        assert case.expect == "refused", "a valid input was refused"
        return "refused"
    assert case.expect == "cycle", "an invalid input went through"
    same = (
        table_rows(ds.images, ds.annotations) == table_rows(ds2.images, ds2.annotations)
        and sorted(ds.categories, key=lambda c: c.id) == sorted(ds2.categories, key=lambda c: c.id)
        and first.manifest.version == second.manifest.version
    )
    if same:
        return "same"
    assert not lossless(first.report, written), (
        "the data changed and every report is lossless: "
        f"read={first.report!r} write={written!r} "
        f"before={table_rows(ds.images, ds.annotations)!r} "
        f"after={table_rows(ds2.images, ds2.annotations)!r}"
    )
    return "counted"


def item(folder: str, name: str, ident: int, document: str, *boxes: Box, **extra: str) -> Item:
    return Item(folder, name, ident, document, list(boxes), dict(extra))


def corpus(fmt: str) -> dict[str, Case]:
    doc = f"annotations{DOCUMENT[fmt]}"
    one, two = f"job1{DOCUMENT[fmt]}", f"job2{DOCUMENT[fmt]}"
    cases = {
        "plain": Case([item("", "a.png", 0, doc, Box("cat"))]),
        "same name in two folders": Case(
            [
                item("d1", "a.png", 0, doc, Box("cat")),
                item("d2", "a.png", 1, doc, Box("dog")),
            ]
        ),
        "same id in two files": Case(
            [item("", "a.png", 0, one, Box("cat")), item("", "b.png", 0, two, Box("dog"))]
        ),
        "boxes at the exact border": Case(
            [
                item(
                    "",
                    "a.png",
                    0,
                    doc,
                    Box("cat", (0.0, 0.0, float(W), float(H))),
                    Box("dog", (float(W) - 1, float(H) - 1, float(W), float(H))),
                )
            ]
        ),
        "unicode and markup labels": Case(
            [item("", "é.png", 0, doc, Box("猫"), Box("a&b<c>\"'"), Box("x" * 300), Box("ünï"))]
        ),
        "names with spaces and unicode": Case(
            [item("d 1", "a b.png", 0, doc, Box("cat")), item("", "ü.png", 1, doc, Box("cat"))]
        ),
        "an image without boxes": Case(
            [item("", "a.png", 0, doc), item("", "b.png", 1, doc, Box("cat"))]
        ),
        "very long value": Case([item("", "a.png", 0, doc, Box("cat"))]),
        "empty label": Case([item("", "a.png", 0, doc, Box(""))], expect="refused"),
    }
    if fmt == "cvat_xml":
        cases["attributes named like metadata"] = Case(
            [
                item(
                    "", "a.png", 0, doc,
                    Box("cat", attrs={"format": "x", "file_name": "y", "id": "z", "source": "w"},
                        extra={"format": "coco", "z_order": "0"}),
                    format="coco", file_name="evil.png", task="7",
                )
            ]
        )  # fmt: skip
        cases["long attribute value"] = Case(
            [item("", "a.png", 0, doc, Box("cat", attrs={"k": "v" * 3000}))]
        )
        cases["too long a value"] = Case(
            [item("", "a.png", 0, doc, Box("cat", attrs={"k": "v" * 5000}))], expect="refused"
        )
    else:
        cases["task fields outside the allow-list"] = Case(
            [item("", "a.png", 0, doc, Box("cat"), format="coco", file_name="evil.png",
                  created_username="alice@corp.example, 3", meta="x" * 3000)]  # type: ignore[arg-type]
        )  # fmt: skip
    return cases


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_regression_corpus_obeys_the_invariant(fmt: str, tmp_path: Path) -> None:
    outcomes = {}
    for n, (name, case) in enumerate(corpus(fmt).items()):
        work = tmp_path / f"case{n}"
        work.mkdir()
        outcomes[name] = cycle(fmt, case, work)
    assert outcomes["empty label"] == "refused"
    for name in ("plain", "same name in two folders", "same id in two files",
                 "boxes at the exact border", "unicode and markup labels",
                 "names with spaces and unicode", "an image without boxes"):  # fmt: skip
        assert outcomes[name] == "same", name
    from datacopilot.io.registry import get_reader

    if fmt == "cvat_xml":
        work = tmp_path / "meta"
        work.mkdir()
        build_cvat(work, corpus(fmt)["attributes named like metadata"])
        read = get_reader(fmt).read(work)
        assert read.dropped == {
            "images.source.format": 1,
            "images.source.file_name": 1,
            "annotations.source.format": 1,
        }  # counted on the way in, so the rows that are stored survive the cycle unchanged
        assert outcomes["attributes named like metadata"] in ("same", "counted")
    else:
        work = tmp_path / "meta"
        work.mkdir()
        build_ls(work, corpus(fmt)["task fields outside the allow-list"])
        read = get_reader(fmt).read(work)
        assert {"images.source.format", "images.source.created_username"} <= set(read.dropped)
        assert outcomes["task fields outside the allow-list"] in ("same", "counted")


# ------------------------------------------------------------------------- generated corpus
STEMS = st.sampled_from(["a", "b", "é", "a b", "猫", "x" * 40])
FOLDERS = st.sampled_from(["", "d1", "d 2", "é"])
LABELS = st.sampled_from(["cat", "dog", "猫", "a&b", "<t>", "q'\"", "z" * 120])
KEYS = st.sampled_from(["format", "file_name", "id", "name", "k", "z_order", "rotation_note"])
VALUES = st.one_of(
    st.sampled_from(["", "0", "x", "ü", "a<b", "v" * 500]),
    st.text(max_size=20).filter(lambda t: all(c.isprintable() and c not in "\r" for c in t)),
)


@st.composite
def boxes(draw: st.DrawFn, fmt: str) -> Box:
    x1 = draw(st.sampled_from([0.0, 1.5, 8.0, 20.25]))
    y1 = draw(st.sampled_from([0.0, 2.0, 6.0]))
    x2 = draw(st.sampled_from([float(W), 40.0, 30.5]))
    y2 = draw(st.sampled_from([float(H), 30.0, 25.0]))
    box = Box(draw(LABELS), (x1, y1, x2, y2))
    if fmt == "cvat_xml":
        box.attrs = draw(
            st.dictionaries(st.sampled_from(["k", "format", "file_name", "id"]), VALUES, max_size=2)
        )
        box.extra = draw(st.dictionaries(KEYS, VALUES, max_size=2))
    return box


@st.composite
def cases(draw: st.DrawFn, fmt: str) -> Case:
    ext = DOCUMENT[fmt]
    documents = draw(st.lists(st.sampled_from([f"annotations{ext}", f"job1{ext}", f"job2{ext}"]),
                              min_size=1, max_size=4))  # fmt: skip
    seen: set[tuple[str, str]] = set()
    items: list[Item] = []
    for document in documents:
        folder, stem = draw(FOLDERS), draw(STEMS)
        if (folder, stem) in seen:
            continue
        seen.add((folder, stem))
        extra = draw(st.dictionaries(KEYS, VALUES, max_size=2)) if fmt == "cvat_xml" else {}
        extra = {k: v for k, v in extra.items() if k not in ("id", "name", "width", "height")}
        items.append(Item(folder, stem + ".png", draw(st.integers(0, 3)), document,
                          draw(st.lists(boxes(fmt), max_size=3)), extra))  # fmt: skip
    single = len({i.document for i in items}) == 1 and items[0].document == f"annotations{ext}"
    if not single:  # a folder layout holds no top-level annotations.* file
        for it in items:
            if it.document == f"annotations{ext}":
                it.document = f"job1{ext}"
    # one task or frame id may appear in several files, never twice in one file
    fixed: list[Item] = []
    taken: set[tuple[str, int]] = set()
    for it in items:
        ident = it.ident
        while (it.document, ident) in taken:
            ident += 1
        taken.add((it.document, ident))
        fixed.append(Item(it.folder, it.name, ident, it.document, it.boxes, it.extra))
    return Case(fixed)


@pytest.mark.parametrize("fmt", FORMATS)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_generated_inputs_obey_the_invariant(fmt: str, data: st.DataObject) -> None:
    case = data.draw(cases(fmt))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        outcome = cycle_or_refuse(fmt, case, work)
    assert outcome in ("same", "counted", "refused")


def cycle_or_refuse(fmt: str, case: Case, work: Path) -> str:
    """Generated inputs may legitimately be refused (an empty label cannot occur, but a value over
    the length limit can): a refusal is fine, a silent change is not."""
    try:
        return cycle(fmt, case, work)
    except AssertionError as exc:
        if "a valid input was refused" in str(exc):
            return "refused"
        raise


# ---------------------------------------------- foreign origin: hostile rows written to the format
HOSTILE_NAMES = ["we\x01ird.png", "tab\there.png", "a b.png", "é.png", "x" * 200 + ".png"]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_hostile_rows_from_another_format_are_written_faithfully_or_counted(
    fmt: str, name: str, tmp_path: Path
) -> None:
    synth.write_png(tmp_path / "i.png", W, H, 1)
    image = Image(
        image_id=real_id(tmp_path / "i.png"),
        file_path=str(tmp_path / "i.png"),
        width=W,
        height=H,
        source={"format": "coco", "id": 1, "file_name": name},
    )
    ann = Annotation(
        ann_id="a", image_id=image.image_id, category_id=0, kind="bbox",
        bbox_xyxy_abs=(0.0, 0.0, float(W), float(H)), attributes={"k\x02": "v\x03", "n": 3},
    )  # fmt: skip
    cats = [Category(id=0, name="ca\x01t"), Category(id=1, name="unused")]
    ds = dataset_from([image], [ann], cats)
    report = convert.export(ds, fmt, tmp_path / "out")
    back = convert.ingest(tmp_path / "out", fmt, tmp_path / "b.dcp")
    ds2 = DatasetStore.open(tmp_path / "b.dcp").read()
    names_before = {name}
    names_after = {str(i.source.get("file_name")) for i in ds2.images}
    cat_names = {c.name for c in ds2.categories}
    attrs_after = {a.attributes.get("n") for a in ds2.annotations}
    unchanged = (
        names_after == names_before
        and cat_names == {"ca\x01t"}
        and attrs_after == {3}
        and {c.id for c in ds2.categories} == {0}
    )
    assert back.manifest.image_count == 1  # the image is found under the name that was written
    assert back.report.skipped == {}
    if not unchanged:
        assert not report.is_lossless(), (names_after, cat_names, attrs_after)
