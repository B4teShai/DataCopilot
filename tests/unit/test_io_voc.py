from pathlib import Path

import pytest
from conftest import dataset_from, make_dataset, table_rows
from fixtures import synth

from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files
from datacopilot.io.voc import plugin


@pytest.fixture
def voc(tmp_path: Path) -> Path:
    return synth.build_voc(tmp_path / "voc")


def xml_doc(objects: str = "", filename: str = "a.png", extra: str = "", size: str = "") -> str:
    size = size or "<size><width>64</width><height>48</height><depth>1</depth></size>"
    return (
        f"<annotation><folder>JPEGImages</folder><filename>{filename}</filename>{extra}"
        f"{size}{objects}</annotation>"
    )


def obj(
    name: str = "cat",
    box: str = "<xmin>8</xmin><ymin>6</ymin><xmax>40</xmax><ymax>30</ymax>",
    extra: str = "",
) -> str:
    return f"<object><name>{name}</name>{extra}<bndbox>{box}</bndbox></object>"


def one(root: Path, xml: str, labelmap: str | None = "cat\ndog\n") -> Path:
    synth.write_png(root / "JPEGImages" / "a.png", 64, 48, 1)
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    (root / "Annotations" / "a.xml").write_text(xml)
    if labelmap is not None:
        (root / "labelmap.txt").write_text(labelmap)
    return root


# ------------------------------------------------------------------ reader
def test_reads_fixture_with_flags_and_splits(voc: Path) -> None:
    r = plugin.read(voc)
    assert r.task == "detection" and r.dropped == {} and r.skipped == {}
    assert [(c.id, c.name) for c in r.categories] == [(0, "cat"), (1, "dog"), (2, "bird")]
    assert {i.source["filename"]: i.split for i in r.images} == {
        "a.png": "train",
        "b.png": "train",
        "c.png": "val",
        "d.png": "test",
    }
    a = next(i for i in r.images if i.source["filename"] == "a.png")
    assert a.source == {
        "format": "voc",
        "filename": "a.png",
        "folder": "JPEGImages",
        "segmented": "0",
        "depth": "1",
    }
    boxes = {x.ann_id: x for x in r.annotations}
    first, second = boxes["a:0"], boxes["a:1"]
    assert first.bbox_xyxy_abs == (8.0, 6.0, 40.0, 30.0) and first.category_id == 0
    assert first.truncated is True and first.occluded is None
    assert first.attributes == {"difficult": False} and second.attributes == {"difficult": True}
    assert first.source == {"format": "voc", "pose": "Left"}
    assert boxes["d:0"].truncated is False and boxes["d:0"].bbox_xyxy_abs == (
        12.5,
        7.25,
        31.5,
        44.0,
    )


def test_occluded_flag_and_unknown_object_children_are_kept(tmp_path: Path) -> None:
    root = one(
        tmp_path / "v",
        xml_doc(obj(extra="<occluded>1</occluded><part><name>head</name></part><score>x</score>")),
    )
    (ann,) = plugin.read(root).annotations
    assert ann.occluded is True
    assert ann.source == {"format": "voc", "part": {"name": "head"}, "score": "x"}


def test_categories_default_to_sorted_object_names_without_a_labelmap(tmp_path: Path) -> None:
    root = one(tmp_path / "v", xml_doc(obj("zebra") + obj("ant")), labelmap=None)
    assert [(c.id, c.name) for c in plugin.read(root).categories] == [(0, "ant"), (1, "zebra")]


def test_file_found_by_stem_when_filename_element_is_missing(tmp_path: Path) -> None:
    xml = "<annotation><size><width>64</width><height>48</height></size></annotation>"
    root = one(tmp_path / "v", xml)
    r = plugin.read(root)
    assert len(r.images) == 1 and r.images[0].width == 64


def test_missing_image_is_skipped_and_counted(tmp_path: Path) -> None:
    root = one(tmp_path / "v", xml_doc(obj()))
    (root / "JPEGImages" / "a.png").unlink()
    r = plugin.read(root)
    assert r.skipped == {"images.file_missing": 1, "annotations.of_missing_image": 1}


def test_an_image_in_an_unsupported_format_is_refused_not_counted_as_missing(
    tmp_path: Path,
) -> None:
    root = one(tmp_path / "v", xml_doc(obj(), filename="a.webp"))
    (root / "JPEGImages" / "a.webp").write_bytes(b"RIFF")
    with pytest.raises(FormatError, match=r"a\.webp: unsupported image format"):
        plugin.read(root)


def test_declared_size_mismatch_is_counted(tmp_path: Path) -> None:
    size = "<size><width>5</width><height>5</height></size>"
    r = plugin.read(one(tmp_path / "v", xml_doc(size=size)))
    assert (r.images[0].width, r.images[0].height) == (64, 48) and r.dropped == {
        "images.declared_size": 1
    }


@pytest.mark.parametrize(
    ("xml", "match"),
    [
        ("<annotation", "a.xml"),
        ("<other/>", "annotation"),
        (xml_doc(obj("bird")), "'bird'"),
        (xml_doc("<object><name>cat</name></object>"), "bndbox"),
        (xml_doc("<object><bndbox><xmin>1</xmin></bndbox></object>"), "name"),
        (xml_doc(obj(box="<xmin>a</xmin><ymin>1</ymin><xmax>3</xmax><ymax>3</ymax>")), "xmin"),
        (xml_doc(obj(box="<xmin>1</xmin><ymin>1</ymin><xmax>3</xmax>")), "ymax"),
        (xml_doc(obj(extra="<truncated>maybe</truncated>")), "truncated"),
        (xml_doc(filename="../evil.png"), "unsafe path"),
        (xml_doc(size="<size><width>x</width><height>5</height></size>"), "width"),
    ],
)
def test_invalid_xml_gives_one_format_error(tmp_path: Path, xml: str, match: str) -> None:
    with pytest.raises(FormatError, match=match):
        plugin.read(one(tmp_path / "v", xml))


def test_box_outside_image_is_a_validation_error(tmp_path: Path) -> None:
    from datacopilot.core.errors import ValidationError

    root = one(
        tmp_path / "v",
        xml_doc(obj(box="<xmin>1</xmin><ymin>1</ymin><xmax>99</xmax><ymax>3</ymax>")),
    )
    with pytest.raises(ValidationError, match="outside"):
        make_dataset(plugin.read(root))


def test_image_in_two_splits_is_an_error(voc: Path) -> None:
    (voc / "ImageSets" / "Main" / "val.txt").write_text("a\nc\n")
    with pytest.raises(FormatError, match="'a'.*more than one split"):
        plugin.read(voc)


def test_split_lists_may_carry_class_flags(tmp_path: Path) -> None:
    root = one(tmp_path / "v", xml_doc(obj()))
    (root / "ImageSets" / "Main").mkdir(parents=True)
    (root / "ImageSets" / "Main" / "train.txt").write_text("a  1\n")
    assert plugin.read(root).images[0].split == "train"


def test_missing_dirs_and_empty_dataset(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="does not exist"):
        plugin.read(tmp_path / "nope")
    (tmp_path / "e").mkdir()
    with pytest.raises(FormatError, match="Annotations"):
        plugin.read(tmp_path / "e")


def test_oversized_xml_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = one(tmp_path / "v", xml_doc())
    monkeypatch.setattr(files, "MAX_XML_BYTES", 10)
    with pytest.raises(FormatError, match="larger than"):
        plugin.read(root)


# ------------------------------------------------------------------ XXE and entity expansion
CANARY = "TOP-CANARY-CONTENT-9d41"


@pytest.mark.parametrize(
    "prolog",
    [
        '<!DOCTYPE annotation [<!ENTITY xxe SYSTEM "file://{secret}">]>',
        '<!DOCTYPE annotation SYSTEM "http://127.0.0.1:9/evil.dtd">',
        '<!DOCTYPE annotation [<!ENTITY % p SYSTEM "http://127.0.0.1:9/x"> %p;]>',
        '<!DOCTYPE annotation [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]>',
        "<!doctype annotation []>",
    ],
)
def test_doctype_and_entities_are_rejected_and_inert(tmp_path: Path, prolog: str) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text(CANARY)
    body = xml_doc(extra="<path>&xxe;</path>")
    root = one(tmp_path / "v", prolog.format(secret=secret) + body)
    with pytest.raises(FormatError, match="DOCTYPE|entit|well-formed") as exc:
        plugin.read(root)
    assert CANARY not in str(exc.value)


def test_xxe_in_utf16_document_is_rejected(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text(CANARY)
    doc = (
        f'<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE a [<!ENTITY x SYSTEM "file://{secret}">]>'
        + xml_doc(extra="<path>&x;</path>")
    )
    root = one(tmp_path / "v", "")
    (root / "Annotations" / "a.xml").write_bytes(doc.encode("utf-16"))
    with pytest.raises(FormatError, match="DOCTYPE|entit|a.xml"):
        plugin.read(root)


def test_plain_text_entities_are_still_fine(tmp_path: Path) -> None:
    root = one(tmp_path / "v", xml_doc(obj("cat"), extra="<path>a &amp; b</path>"))
    (img,) = plugin.read(root).images
    assert img.source["path"] == "a & b"


# ------------------------------------------------------------------ writer
def roundtrip(src: Path, out: Path):  # type: ignore[no-untyped-def]
    first = plugin.read(src)
    out.mkdir(parents=True)
    report = plugin.write(make_dataset(first), out)
    return first, plugin.read(out), report


def test_same_format_round_trip_is_lossless_and_field_identical(voc: Path, tmp_path: Path) -> None:
    first, second, report = roundtrip(voc, tmp_path / "out")
    assert report.dropped == {} and report.is_lossless() and report.target_format == "voc"
    assert table_rows(first.images, first.annotations) == table_rows(
        second.images, second.annotations
    )
    assert first.categories == second.categories


def test_writer_layout(voc: Path, tmp_path: Path) -> None:
    roundtrip(voc, tmp_path / "out")
    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "Annotations").iterdir()) == [
        "a.xml",
        "b.xml",
        "c.xml",
        "d.xml",
    ]
    assert (out / "JPEGImages" / "a.png").is_file()
    assert (out / "ImageSets" / "Main" / "train.txt").read_text() == "a\nb\n"
    assert (out / "labelmap.txt").read_text() == "cat\ndog\nbird\n"


def test_special_characters_are_escaped_not_injected(tmp_path: Path) -> None:
    p = tmp_path / "p.png"
    synth.write_png(p, 8, 8, 1)
    img = Image(image_id=files.image_id(p), file_path=str(p), width=8, height=8)
    ann = Annotation(
        ann_id="1",
        image_id=img.image_id,
        category_id=0,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 1.0, 5.0, 5.0),
    )
    cat = Category(id=0, name="</name><x>&<![CDATA[")
    out = tmp_path / "out"
    out.mkdir()
    plugin.write(dataset_from([img], [ann], [cat]), out)
    back = plugin.read(out)
    assert back.categories[0].name == "</name><x>&<![CDATA[" and len(back.annotations) == 1


def test_foreign_rows_count_every_loss(tmp_path: Path) -> None:
    p = tmp_path / "pic.png"
    synth.write_png(p, 40, 30, 5)
    img = Image(
        image_id=files.image_id(p),
        file_path=str(p),
        width=40,
        height=30,
        split="val",
        group_id="g",
        source={"format": "coco", "id": 3, "file_name": "x/pic.png", "license": 1},
    )
    ann = Annotation(
        ann_id="1",
        image_id=img.image_id,
        category_id=7,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 2.0, 30.5, 20.0),
        is_crowd=True,
        occluded=True,
        truncated=False,
        attributes={"difficult": True, "other": 1},
        score=0.5,
        provenance="model",
        source={"format": "coco", "id": 9},
    )
    cat = Category(id=7, name="thing", aliases=("t",), attributes={"sku": "1"})
    out = tmp_path / "out"
    out.mkdir()
    report = plugin.write(dataset_from([img], [ann], [cat]), out)
    assert report.dropped == {
        "images.group_id": 1,
        "images.source.id": 1,
        "images.source.license": 1,
        "annotations.is_crowd": 1,
        "annotations.score": 1,
        "annotations.provenance": 1,
        "annotations.attributes": 1,
        "annotations.source.id": 1,
        "categories.id": 1,
        "categories.aliases": 1,
        "categories.attributes": 1,
    }
    back = plugin.read(out)
    (a,) = back.annotations
    assert a.occluded is True and a.truncated is False and a.attributes == {"difficult": True}
    assert back.images[0].split == "val" and (out / "JPEGImages" / "pic.png").is_file()


def test_writer_guards(voc: Path, tmp_path: Path) -> None:
    ds = make_dataset(plugin.read(voc))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(FormatError, match="not empty"):
        plugin.write(ds, out)
    evil = ds.images[0].model_copy(update={"source": {"format": "voc", "filename": "../../e.png"}})
    out2 = tmp_path / "out2"
    out2.mkdir()
    with pytest.raises(FormatError, match="unsafe path|file name"):
        plugin.write(dataset_from([evil], [], list(ds.categories)), out2)
    assert not (tmp_path / "e.png").exists()


# --- writer: untrusted source content
def _voc_dataset(
    tmp_path: Path,
    image_source: dict,
    ann_source: dict | None = None,  # type: ignore[type-arg]
    cat_name: str = "cat",
) -> tuple[object, Path]:
    p = tmp_path / "pic.png"
    synth.write_png(p, 40, 30, 5)
    image = Image(
        image_id=files.image_id(p),
        file_path=str(p),
        width=40,
        height=30,
        source={"format": "voc", "filename": "pic.png", **image_source},
    )
    ann = Annotation(
        ann_id="1",
        image_id=image.image_id,
        category_id=0,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 2.0, 30.0, 20.0),
        source={"format": "voc", **(ann_source or {})},
    )
    out = tmp_path / "out"
    out.mkdir()
    return dataset_from([image], [ann], [Category(id=0, name=cat_name)]), out


@pytest.mark.parametrize(
    "key",
    [
        "a><injected/><b",
        "a<b",
        "a>b",
        "a&b",
        'a"b',
        "a'b",
        "a b",
        "a:b",
        "1abc",
        "",
        "-x",
        "a" * 200,
        "é",
    ],
)
def test_invalid_element_names_are_not_emitted_and_are_counted(tmp_path: Path, key: str) -> None:
    ds, out = _voc_dataset(tmp_path, {key: "v1"}, {key: {"x": "y"}})
    report = plugin.write(ds, out)  # type: ignore[arg-type]
    assert report.dropped == {
        "images.source.invalid_key": 1,
        "annotations.source.invalid_key": 1,
    }
    raw = (out / "Annotations" / "pic.xml").read_text()
    assert "injected" not in raw
    back = plugin.read(out)  # the hardened reader proves the output is well-formed
    assert key not in back.images[0].source and len(back.annotations) == 1


def test_valid_names_with_dots_dashes_underscores_are_kept(tmp_path: Path) -> None:
    ds, out = _voc_dataset(tmp_path, {"a_b.c-d": "1", "_x": "2"})
    report = plugin.write(ds, out)  # type: ignore[arg-type]
    assert report.dropped == {}
    assert plugin.read(out).images[0].source["a_b.c-d"] == "1"


@pytest.mark.parametrize("char", ["\x00", "\x01", "\x08", "\x0b", "\x0c", "\x0e", "\x1f", "￾", "￿"])
def test_illegal_xml_characters_are_replaced_and_counted(tmp_path: Path, char: str) -> None:
    ds, out = _voc_dataset(
        tmp_path, {"note": f"a{char}b"}, {"pose": f"x{char}"}, cat_name=f"c{char}t"
    )
    report = plugin.write(ds, out)  # type: ignore[arg-type]
    assert report.dropped["images.text.invalid_xml_chars"] == 1
    assert report.dropped["annotations.text.invalid_xml_chars"] == 1  # pose
    assert report.dropped["categories.text.invalid_xml_chars"] == 1  # class name
    back = plugin.read(out)
    assert back.images[0].source["note"] == "a�b"
    assert back.annotations[0].source["pose"] == "x�"


def test_tab_newline_and_astral_characters_are_legal(tmp_path: Path) -> None:
    ds, out = _voc_dataset(tmp_path, {"note": "a\tb\nc \U0001f600"})
    report = plugin.write(ds, out)  # type: ignore[arg-type]
    assert report.dropped == {}
    assert plugin.read(out).images[0].source["note"] == "a\tb\nc \U0001f600"


def test_deep_and_huge_source_values_are_capped_and_counted(tmp_path: Path) -> None:
    deep: object = "leaf"
    for _ in range(50):
        deep = {"n": deep}
    wide = {f"k{i}": "v" for i in range(5000)}
    ds, out = _voc_dataset(tmp_path, {"deep": deep, "wide": wide})
    report = plugin.write(ds, out)  # type: ignore[arg-type]
    assert report.dropped.get("images.source.overflow", 0) >= 1
    back = plugin.read(out)
    assert back.images[0].width == 40  # still well-formed and readable
    assert len((out / "Annotations" / "pic.xml").read_bytes()) < 200_000


@pytest.mark.parametrize(
    "name", ["../evil.png", "/abs/evil.png", "a\\..\\evil.png", "a\x00b.png", "x/../../evil.png"]
)
def test_hostile_file_names_never_leave_the_output_directory(tmp_path: Path, name: str) -> None:
    ds, out = _voc_dataset(tmp_path, {"filename": name})
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.write(ds, out)  # type: ignore[arg-type]
    assert not (tmp_path / "evil.png").exists() and not (out.parent.parent / "evil.png").exists()
    assert list(out.rglob("*.png")) == []


def test_hostile_foreign_file_name_hint_is_rejected_too(tmp_path: Path) -> None:
    p = tmp_path / "pic.png"
    synth.write_png(p, 8, 8, 1)
    image = Image(
        image_id=files.image_id(p),
        file_path=str(p),
        width=8,
        height=8,
        source={"format": "coco", "file_name": "a\\..\\evil.png"},
    )
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.write(dataset_from([image], [], [Category(id=0, name="a")]), out)


def test_lone_surrogates_cannot_enter_the_model_at_all() -> None:
    with pytest.raises(ValueError):
        Category(id=0, name="c\ud800t")
