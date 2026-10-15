import time
from pathlib import Path

import pytest
from conftest import dataset_from, make_dataset, table_rows
from fixtures import synth

from datacopilot import convert
from datacopilot.core.errors import ConversionLossError, FormatError, ValidationError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files, registry
from datacopilot.io.cvat import plugin
from datacopilot.store import DatasetStore

LABELS = "<labels><label><name>cat</name></label><label><name>dog</name></label></labels>"
BOX = 'label="cat" source="manual" occluded="0" xtl="8" ytl="6" xbr="40" ybr="30"'


def doc(body: str = "", meta: str = f"<task>{LABELS}</task>", head: str = "") -> str:
    return (
        f'<?xml version="1.0" encoding="utf-8"?>{head}<annotations><version>1.1</version>'
        f"<meta>{meta}</meta>{body}</annotations>"
    )


def image(shapes: str = f"<box {BOX}/>", name: str = "a.png", extra: str = 'id="0"') -> str:
    return f'<image {extra} name="{name}" width="64" height="48">{shapes}</image>'


def one(root: Path, xml: str) -> Path:
    synth.write_png(root / "images" / "a.png", 64, 48, 1)
    (root / "annotations.xml").write_text(xml)
    return root


@pytest.fixture
def cvat(tmp_path: Path) -> Path:
    return synth.build_cvat_xml(tmp_path / "cvat")


# ------------------------------------------------------------------ reader
def test_registered_for_read_and_write() -> None:
    info = {i.spec.name: i for i in registry.list_formats()}["cvat_xml"]
    assert info.can_read and info.can_write


def test_reads_fixture(cvat: Path) -> None:
    r = plugin.read(cvat)
    assert r.task == "detection" and r.dropped == {} and r.skipped == {}
    assert [(c.id, c.name, c.attributes) for c in r.categories] == [
        (0, "cat", {"color": "#ff0000", "type": "any"}),
        (1, "dog", {"color": "#00ff00", "type": "any"}),
        (2, "bird", {"color": "#0000ff", "type": "any"}),
    ]
    assert {i.source["file_name"]: i.split for i in r.images} == {
        "a.png": "train",
        "b.png": "train",
        "c.png": "val",
        "d.png": "test",
    }
    a = next(i for i in r.images if i.source["file_name"] == "a.png")
    assert a.source == {
        "format": "cvat_xml",
        "id": 0,
        "file_name": "a.png",
        "layout_document": "train.xml",
    }
    assert (a.width, a.height) == (64, 48)
    boxes = sorted((x for x in r.annotations if x.image_id == a.image_id), key=lambda x: x.ann_id)
    assert [b.ann_id for b in boxes] == ["train:0:000000", "train:0:000001"]
    assert boxes[0].bbox_xyxy_abs == (8.0, 6.0, 40.0, 30.0)
    assert boxes[0].attributes == {"colour": "red"} and boxes[0].occluded is False
    assert boxes[1].occluded is True and boxes[1].provenance == "human"
    assert boxes[0].source == {"format": "cvat_xml", "z_order": "0"}


def test_single_file_path_and_subset_decide_the_split(tmp_path: Path) -> None:
    meta = f"<task><subset>Train</subset>{LABELS}</task>"
    root = one(tmp_path / "r", doc(image(), meta))
    r = plugin.read(root / "annotations.xml")
    assert [i.split for i in r.images] == ["train"] and r.dropped == {}


def test_unknown_subset_is_counted_and_labels_are_derived_without_meta(tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image(), "<task><subset>Validation</subset></task>"))
    r = plugin.read(root)
    assert r.dropped == {"images.subset": 1}
    assert [c.name for c in r.categories] == ["cat"] and r.images[0].split == "unassigned"


@pytest.mark.parametrize(
    ("shape", "key"),
    [
        ('<polygon label="cat" points="1,1;5,5;9,1"/>', "polygon"),
        ('<polyline label="cat" points="1,1;5,5"/>', "polyline"),
        ('<points label="cat" points="1,1"/>', "points"),
        ('<ellipse label="cat" cx="5" cy="5" rx="2" ry="2"/>', "ellipse"),
        ('<mask label="cat" rle="1,2" left="0" top="0" width="2" height="2"/>', "mask"),
        ('<skeleton label="cat"><points label="a" points="1,1"/></skeleton>', "skeleton"),
        ('<cuboid label="cat" xtl1="1" ytl1="1"/>', "cuboid"),
        ('<tag label="cat"/>', "tag"),
        (f'<box {BOX} rotation="15"/>', "rotated_box"),
        ("<weird/>", "other"),
    ],
)
def test_unsupported_shapes_are_counted_never_dropped_silently(
    shape: str, key: str, tmp_path: Path
) -> None:
    root = one(tmp_path / "r", doc(image(f"<box {BOX}/>{shape}")))
    r = plugin.read(root)
    assert r.skipped == {f"annotations.unsupported.{key}": 1}
    assert len(r.annotations) == 1


def test_tracks_and_unknown_top_level_elements_are_counted(tmp_path: Path) -> None:
    track = '<track id="0" label="cat"><box frame="0" xtl="1" ytl="1" xbr="5" ybr="5"/></track>'
    root = one(tmp_path / "r", doc(image() + track + track + "<extra/>"))
    r = plugin.read(root)
    assert r.skipped == {"annotations.unsupported.track": 2, "dataset.unsupported_element": 1}


def test_missing_image_files_skip_the_image_and_its_shapes(tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image() + image(name="gone.png", extra='id="1"')))
    r = plugin.read(root)
    assert r.skipped == {"images.file_missing": 1, "annotations.of_missing_image": 1}
    assert len(r.images) == 1


def test_declared_size_mismatch_is_counted_and_the_header_wins(tmp_path: Path) -> None:
    xml = doc('<image id="0" name="a.png" width="99" height="48"><box ' + BOX + "/></image>")
    r = plugin.read(one(tmp_path / "r", xml))
    assert r.dropped == {"images.declared_size": 1} and r.images[0].width == 64


def test_label_attribute_definitions_are_counted_unless_they_only_name_the_attribute(
    tmp_path: Path,
) -> None:
    trivial = (
        "<attribute><name>k</name><mutable>False</mutable><input_type>text</input_type>"
        "<default_value></default_value><values></values></attribute>"
    )
    rich = (
        "<attribute><name>k</name><mutable>True</mutable><input_type>select</input_type>"
        "<default_value>a</default_value><values>a\nb</values></attribute>"
    )
    meta = (
        f"<task><labels><label><name>cat</name><attributes>{trivial}</attributes></label>"
        f"<label><name>dog</name><attributes>{rich}</attributes><svg>x</svg></label></labels></task>"
    )
    r = plugin.read(one(tmp_path / "r", doc(image(), meta)))
    assert r.dropped == {"categories.attributes": 1, "categories.unsupported.svg": 1}


def test_source_attribute_maps_to_provenance(tmp_path: Path) -> None:
    shapes = (
        '<box label="cat" source="auto" xtl="1" ytl="1" xbr="5" ybr="5"/>'
        '<box label="cat" source="semi-auto" xtl="2" ytl="2" xbr="6" ybr="6"/>'
        '<box label="dog" xtl="3" ytl="3" xbr="7" ybr="7"/>'
    )
    r = plugin.read(one(tmp_path / "r", doc(image(shapes))))
    rows = sorted(r.annotations, key=lambda a: a.ann_id)
    assert [a.provenance for a in rows] == ["model", "human", "human"]
    assert rows[1].source["source"] == "semi-auto" and rows[0].score is None


def test_undefined_label_is_a_format_error(tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image('<box label="bird" xtl="1" ytl="1" xbr="5" ybr="5"/>')))
    with pytest.raises(FormatError, match="not defined"):
        plugin.read(root)


def test_duplicate_image_id_and_duplicate_attribute_are_format_errors(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="duplicate image id"):
        plugin.read(one(tmp_path / "a", doc(image() + image())))
    dup = '<attribute name="k">1</attribute><attribute name="k">2</attribute>'
    with pytest.raises(FormatError, match="duplicate attribute"):
        plugin.read(one(tmp_path / "b", doc(image(f"<box {BOX}>{dup}</box>"))))


def test_wrong_root_missing_directory_and_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "annotations.xml").write_text("<other/>")
    with pytest.raises(FormatError, match="annotations"):
        plugin.read(tmp_path / "w")
    with pytest.raises(FormatError, match="does not exist"):
        plugin.read(tmp_path / "nope")
    (tmp_path / "e").mkdir()
    with pytest.raises(FormatError, match="no CVAT annotation file"):
        plugin.read(tmp_path / "e")


# ------------------------------------------------------- hostile input
@pytest.mark.parametrize(
    "head",
    [
        '<!DOCTYPE a [<!ENTITY x SYSTEM "file:///etc/passwd">]>',
        '<!DOCTYPE a [<!ENTITY x SYSTEM "http://127.0.0.1:9/x">]>',
        '<!DOCTYPE a [<!ENTITY % p SYSTEM "http://127.0.0.1:9/e.dtd"> %p;]>',
        '<!DOCTYPE a [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>',
    ],
)
def test_xxe_and_entity_bombs_are_rejected_without_expansion(head: str, tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image(), head=head))
    started = time.perf_counter()
    with pytest.raises(FormatError, match="DOCTYPE"):
        plugin.read(root)
    assert time.perf_counter() - started < 1.0


def test_nesting_bomb_is_rejected_quickly(tmp_path: Path) -> None:
    deep = "<g>" * 20_000 + "</g>" * 20_000
    root = one(tmp_path / "r", doc(image(deep)))
    started = time.perf_counter()
    with pytest.raises(FormatError, match="nested"):
        plugin.read(root)
    assert time.perf_counter() - started < 1.0


def test_element_count_bomb_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datacopilot.io.cvat.MAX_DOCUMENT_ELEMENTS", 50)
    root = one(tmp_path / "r", doc(image("<tag label='cat'/>" * 60)))
    with pytest.raises(FormatError, match="more than 50 elements"):
        plugin.read(root)


def test_huge_shape_count_per_image_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("datacopilot.io.cvat.MAX_SHAPES_PER_IMAGE", 3)
    root = one(tmp_path / "r", doc(image(f"<box {BOX}/>" * 4)))
    with pytest.raises(FormatError, match="too many shapes"):
        plugin.read(root)


def test_attribute_flood_on_one_element_is_rejected(tmp_path: Path) -> None:
    flood = " ".join(f"k{i}='v'" for i in range(300))
    root = one(tmp_path / "r", doc(image(f"<box {BOX} {flood}/>")))
    with pytest.raises(FormatError, match="attributes"):
        plugin.read(root)


def test_oversized_file_is_refused_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = one(tmp_path / "r", doc(image()))
    monkeypatch.setattr(files, "MAX_CVAT_BYTES", 100)
    with pytest.raises(FormatError, match="larger than 100 bytes"):
        plugin.read(root)


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "1e999", "abc", ""])
def test_non_finite_and_malformed_coordinates_are_rejected(value: str, tmp_path: Path) -> None:
    shape = f'<box label="cat" xtl="{value}" ytl="1" xbr="5" ybr="5"/>'
    with pytest.raises(FormatError, match="xtl"):
        plugin.read(one(tmp_path / "r", doc(image(shape))))


@pytest.mark.parametrize(
    "coords",
    [
        'xtl="-1" ytl="1" xbr="5" ybr="5"',  # negative origin
        'xtl="5" ytl="1" xbr="5" ybr="5"',  # zero width
        'xtl="6" ytl="1" xbr="5" ybr="5"',  # inverted
    ],
)
def test_invalid_box_geometry_is_a_validation_error(coords: str, tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image(f'<box label="cat" {coords}/>')))
    with pytest.raises(ValidationError):
        plugin.read(root)


def test_box_outside_the_image_is_rejected_at_ingest(tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image('<box label="cat" xtl="1" ytl="1" xbr="500" ybr="5"/>')))
    with pytest.raises(ValidationError, match="outside image"):
        convert.ingest(root, "cvat_xml", tmp_path / "x.dcp")


@pytest.mark.parametrize(
    "size",
    [
        'width="-5" height="48"',
        'width="0" height="48"',
        'width="1e3" height="48"',
        'width="99999999999" height="48"',
        'width="" height="48"',
        'width="64"',
    ],
)
def test_negative_zero_and_malformed_sizes_are_rejected(size: str, tmp_path: Path) -> None:
    xml = doc(f'<image id="0" name="a.png" {size}><box {BOX}/></image>')
    with pytest.raises(FormatError, match="whole number"):
        plugin.read(one(tmp_path / "r", xml))


@pytest.mark.parametrize("ident", ["-1", "x", "", "1e2"])
def test_malformed_image_ids_are_rejected(ident: str, tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="image id"):
        plugin.read(one(tmp_path / "r", doc(image(extra=f'id="{ident}"'))))


@pytest.mark.parametrize("name", ["../outside.png", "/etc/passwd", "a/../../b.png", "..\\x.png"])
def test_image_paths_cannot_leave_the_dataset(name: str, tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image(name=name)))
    (tmp_path / "outside.png").write_bytes(synth.png_bytes(4, 4, 9))
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


def test_symlinked_image_pointing_outside_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "r"
    (root / "images").mkdir(parents=True)
    (tmp_path / "secret.png").write_bytes(synth.png_bytes(4, 4, 3))
    (root / "images" / "a.png").symlink_to(tmp_path / "secret.png")
    (root / "annotations.xml").write_text(doc(image()))
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


def test_overlong_text_is_rejected(tmp_path: Path) -> None:
    xml = doc(image(f'<box {BOX}><attribute name="k">{"x" * 5000}</attribute></box>'))
    with pytest.raises(FormatError, match="too long"):
        plugin.read(one(tmp_path / "r", xml))


def test_markup_in_label_names_is_data_and_round_trips_escaped(tmp_path: Path) -> None:
    hostile = "<script>alert(1)</script>&\"'"
    labels = (
        "<labels><label><name>&lt;script&gt;alert(1)&lt;/script&gt;&amp;\"'</name></label></labels>"
    )
    shape = (
        '<box label="&lt;script&gt;alert(1)&lt;/script&gt;&amp;&quot;\'" '
        'xtl="8" ytl="6" xbr="40" ybr="30"/>'
    )
    root = one(tmp_path / "r", doc(image(shape), f"<task>{labels}</task>"))
    first = convert.ingest(root, "cvat_xml", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    assert [c.name for c in ds.categories] == [hostile]
    convert.export(ds, "cvat_xml", tmp_path / "out", strict=True)
    text = (tmp_path / "out" / "annotations.xml").read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text
    again = convert.ingest(tmp_path / "out", "cvat_xml", tmp_path / "b.dcp")
    assert again.manifest.version == first.manifest.version


# ------------------------------------------------------------------ writer
def test_round_trip_is_lossless_and_field_identical(cvat: Path, tmp_path: Path) -> None:
    first = convert.ingest(cvat, "cvat_xml", tmp_path / "a.dcp")
    assert first.report.is_lossless()
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds, "cvat_xml", tmp_path / "out", strict=True)
    assert report.dropped == {} and report.skipped == {}
    second = convert.ingest(tmp_path / "out", "cvat_xml", tmp_path / "b.dcp")
    assert second.report.is_lossless()
    ds2 = DatasetStore.open(tmp_path / "b.dcp").read()
    assert table_rows(ds.images, ds.annotations) == table_rows(ds2.images, ds2.annotations)
    assert first.manifest.version == second.manifest.version
    assert sorted(p.name for p in (tmp_path / "out" / "annotations").iterdir()) == [
        "test.xml",
        "train.xml",
        "val.xml",
    ]


def test_unassigned_dataset_writes_a_single_document(tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image()))
    convert.ingest(root, "cvat_xml", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    convert.export(ds, "cvat_xml", tmp_path / "out")
    assert (tmp_path / "out" / "annotations.xml").is_file()
    assert (tmp_path / "out" / "images" / "a.png").is_file()
    text = (tmp_path / "out" / "annotations.xml").read_text()
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    assert "<version>1.1</version>" in text


def test_ingest_strict_fails_when_a_shape_was_skipped(tmp_path: Path) -> None:
    root = one(
        tmp_path / "r", doc(image(f'<box {BOX}/><polygon label="cat" points="1,1;5,5;9,1"/>'))
    )
    with pytest.raises(ConversionLossError, match="annotations.unsupported.polygon=1"):
        convert.ingest(root, "cvat_xml", tmp_path / "a.dcp", strict=True)
    assert not (tmp_path / "a.dcp").exists()


def test_model_provenance_and_attribute_specs_round_trip_without_loss(tmp_path: Path) -> None:
    shape = (
        '<box label="cat" source="auto" xtl="1" ytl="1" xbr="5" ybr="5">'
        '<attribute name="k">v</attribute></box>'
    )
    root = one(tmp_path / "r", doc(image(shape)))
    first = convert.ingest(root, "cvat_xml", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    convert.export(ds, "cvat_xml", tmp_path / "out", strict=True)
    text = (tmp_path / "out" / "annotations.xml").read_text()
    assert 'source="auto"' in text and "<input_type>text</input_type>" in text
    again = convert.ingest(tmp_path / "out", "cvat_xml", tmp_path / "b.dcp", strict=True)
    assert again.manifest.version == first.manifest.version


def test_foreign_rows_are_written_and_losses_counted(tmp_path: Path) -> None:
    images = [
        Image(
            image_id=f"{n:016x}",
            file_path=str(tmp_path / f"i{n}.png"),
            width=64,
            height=48,
            group_id="g" if n == 0 else None,
            source={"format": "coco", "id": n, "license": 1},
        )
        for n in range(2)
    ]
    for n, im in enumerate(images):
        synth.write_png(Path(im.file_path), 64, 48, n)
    anns = [
        Annotation(
            ann_id="a",
            image_id=images[0].image_id,
            category_id=5,
            kind="bbox",
            bbox_xyxy_abs=(1.0, 2.0, 30.0, 40.0),
            is_crowd=True,
            truncated=True,
            attributes={"difficult": True, "n": 3, "s": "x"},
            source={"format": "coco", "area": 3},
        ),
        Annotation(
            ann_id="b",
            image_id=images[1].image_id,
            category_id=5,
            kind="bbox",
            bbox_xyxy_abs=(1.5, 2.0, 30.0, 40.0),
            score=0.5,
            provenance="model",
        ),
        Annotation(
            ann_id="c",
            image_id=images[1].image_id,
            category_id=5,
            kind="bbox",
            bbox_xyxy_abs=(2.5, 2.0, 30.0, 40.0),
            provenance="proposal_accepted",
        ),
    ]
    cats = [Category(id=5, name="cat", aliases=("kitty",), attributes={"supercategory": "pet"})]
    ds = dataset_from(images, anns, cats)
    report = convert.export(ds, "cvat_xml", tmp_path / "out")
    assert report.dropped == {
        "annotations.attributes_stringified": 1,
        "annotations.is_crowd": 1,
        "annotations.provenance": 1,
        "annotations.score": 1,
        "annotations.source.area": 1,
        "annotations.truncated": 1,
        "categories.aliases": 1,
        "categories.attributes": 1,
        "categories.id": 1,
        "images.group_id": 1,
        "images.source.id": 2,
        "images.source.license": 2,
    }
    with pytest.raises(ConversionLossError):
        convert.export(ds, "cvat_xml", tmp_path / "strict", strict=True)
    back = plugin.read(tmp_path / "out")
    assert back.dropped == {} and len(back.annotations) == 3
    assert {a.source.get("source") for a in back.annotations} == {None}
    assert sorted(a.provenance for a in back.annotations) == ["human", "human", "model"]
    text = (tmp_path / "out" / "annotations.xml").read_text()
    assert '<attribute name="difficult">true</attribute>' in text


def test_names_with_characters_illegal_in_xml_are_replaced_and_counted(tmp_path: Path) -> None:
    root = one(tmp_path / "r", doc(image()))
    convert.ingest(root, "cvat_xml", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    ds = dataset_from(
        ds.images,
        ds.annotations,
        [
            Category(id=0, name="ca\x01t"),
            ds.categories[0].model_copy(update={"id": 1, "name": "x"}),
        ][:1],
    )
    report = convert.export(ds, "cvat_xml", tmp_path / "out")
    assert report.dropped["categories.text.invalid_xml_chars"] == 1
    assert plugin.read(tmp_path / "out").categories[0].name == "ca�t"


def test_names_that_collide_after_cleaning_are_refused(tmp_path: Path) -> None:
    synth.write_png(tmp_path / "i.png", 8, 8, 1)
    img = Image(image_id=f"{1:016x}", file_path=str(tmp_path / "i.png"), width=8, height=8)
    ann = Annotation(
        ann_id="a",
        image_id=img.image_id,
        category_id=0,
        kind="bbox",
        bbox_xyxy_abs=(1.0, 1.0, 4.0, 4.0),
    )
    ds = dataset_from([img], [ann], [Category(id=0, name="a\x01"), Category(id=1, name="a\x02")])
    with pytest.raises(FormatError, match="collide"):
        convert.export(ds, "cvat_xml", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_classification_datasets_are_refused(tmp_path: Path) -> None:
    convert.ingest(synth.build_imagefolder(tmp_path / "src"), "imagefolder", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    with pytest.raises(FormatError, match="detection"):
        convert.export(ds, "cvat_xml", tmp_path / "out")


def test_in_memory_dataset_from_read_result_writes(cvat: Path, tmp_path: Path) -> None:
    ds = make_dataset(plugin.read(cvat))
    report = plugin.write(ds, _empty(tmp_path / "w"))
    assert report.is_lossless()


def _empty(path: Path) -> Path:
    path.mkdir()
    return path


# ------------------------------------------------------------- accounting and reserved keys
def test_attribute_children_of_an_image_are_counted(tmp_path: Path) -> None:
    body = (
        '<image id="0" name="a.png" width="64" height="48">'
        '<attribute name="k">v</attribute><attribute name="j">w</attribute>'
        f"<box {BOX}/></image>"
    )
    r = plugin.read(one(tmp_path / "r", doc(body)))
    assert r.dropped == {"images.attributes": 2} and len(r.annotations) == 1


def test_attributes_cannot_replace_the_metadata_of_a_row(tmp_path: Path) -> None:
    body = (
        '<image id="0" name="a.png" width="64" height="48" format="coco" file_name="evil.png">'
        f'<box {BOX} format="yolo"/></image>'
    )
    r = plugin.read(one(tmp_path / "r", doc(body)))
    assert r.images[0].source["format"] == "cvat_xml" and r.images[0].source["file_name"] == "a.png"
    assert r.annotations[0].source["format"] == "cvat_xml"
    assert r.dropped == {
        "images.source.format": 1,
        "images.source.file_name": 1,
        "annotations.source.format": 1,
    }


def test_labels_derived_from_boxes_are_bounded_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shapes = "".join(f'<box label="c{n}" xtl="1" ytl="1" xbr="5" ybr="5"/>' for n in range(5))
    root = one(tmp_path / "a", doc(image(shapes), "<task></task>"))
    monkeypatch.setattr("datacopilot.io.cvat.MAX_LABELS", 3)
    with pytest.raises(FormatError, match="more than 3 labels"):
        plugin.read(root)
    monkeypatch.undo()
    long = "x" * 5000
    root = one(
        tmp_path / "b",
        doc(image(f'<box label="{long}" xtl="1" ytl="1" xbr="5" ybr="5"/>'), "<task/>"),
    )
    with pytest.raises(FormatError, match="label name"):
        plugin.read(root)


def test_a_label_name_over_the_limit_in_the_meta_is_a_format_error(tmp_path: Path) -> None:
    meta = f"<task><labels><label><name>{'y' * 5000}</name></label></labels></task>"
    with pytest.raises(FormatError, match="label name|too long"):
        plugin.read(one(tmp_path / "r", doc(image(), meta)))


def test_every_replacement_of_an_illegal_character_is_counted_and_the_file_matches(
    tmp_path: Path,
) -> None:
    synth.write_png(tmp_path / "i.png", 8, 8, 1)
    img = Image(
        image_id=f"{1:016x}",
        file_path=str(tmp_path / "i.png"),
        width=8,
        height=8,
        source={
            "format": "cvat_xml",
            "id": 0,
            "file_name": "we\x01ird.png",
            "tag\x02": "v",
            "note": "a\x03b",
        },
    )
    ann = Annotation(
        ann_id="a", image_id=img.image_id, category_id=0, kind="bbox",
        bbox_xyxy_abs=(1.0, 1.0, 4.0, 4.0), attributes={"k\x04": "v\x05"},
        source={"format": "cvat_xml", "z\x06": "1", "z_order": "0\x07"},
    )  # fmt: skip
    cats = [Category(id=0, name="ca\x01t", attributes={"color": "#f\x08f"})]
    ds = dataset_from([img], [ann], cats, name="na\x0bme")
    report = convert.export(ds, "cvat_xml", tmp_path / "out")
    assert report.dropped["dataset.text.invalid_xml_chars"] == 1
    assert report.dropped["categories.text.invalid_xml_chars"] == 2  # name and colour
    assert report.dropped["images.text.invalid_xml_chars"] == 2  # file name and attribute value
    assert report.dropped["annotations.text.invalid_xml_chars"] >= 3
    assert report.dropped["images.source.invalid_key"] == 1
    assert report.dropped["annotations.source.invalid_key"] == 1
    names = [p.name for p in (tmp_path / "out" / "images").iterdir()]
    assert names == ["we�ird.png"]
    back = plugin.read(tmp_path / "out")  # the name in the document is the name of the file
    assert back.skipped == {} and len(back.images) == 1


# ------------------------------------------ same-format round trips with several documents
def multi_file(root: Path) -> Path:
    synth.write_png(root / "images" / "a.png", 64, 48, 1)
    synth.write_png(root / "images" / "b.png", 64, 48, 2)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "job1.xml").write_text(doc(image()))
    (root / "annotations" / "job2.xml").write_text(doc(image(name="b.png")))  # the same image id
    (root / "annotations" / "job3.xml").write_text(doc(""))  # a document without images
    return root


def test_several_documents_of_one_split_keep_their_ids_through_the_round_trip(
    tmp_path: Path,
) -> None:
    root = multi_file(tmp_path / "cv")
    first = convert.ingest(root, "cvat_xml", tmp_path / "a.dcp", strict=True)
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    assert sorted(a.ann_id for a in ds.annotations) == ["job1:0:000000", "job2:0:000000"]
    report = convert.export(ds, "cvat_xml", tmp_path / "out", strict=True)
    assert report.dropped == {}
    second = convert.ingest(tmp_path / "out", "cvat_xml", tmp_path / "b.dcp", strict=True)
    ds2 = DatasetStore.open(tmp_path / "b.dcp").read()
    assert table_rows(ds.images, ds.annotations) == table_rows(ds2.images, ds2.annotations)
    assert first.manifest.version == second.manifest.version
    assert sorted(p.name for p in (tmp_path / "out" / "annotations").iterdir()) == [
        "job1.xml",
        "job2.xml",
    ]


def test_a_single_document_in_the_annotations_folder_keeps_its_prefix(tmp_path: Path) -> None:
    synth.write_png(tmp_path / "cv" / "images" / "a.png", 64, 48, 1)
    (tmp_path / "cv" / "annotations").mkdir(parents=True)
    (tmp_path / "cv" / "annotations" / "only.xml").write_text(doc(image()))
    convert.ingest(tmp_path / "cv", "cvat_xml", tmp_path / "a.dcp", strict=True)
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    assert [a.ann_id for a in ds.annotations] == ["only:0:000000"]
    convert.export(ds, "cvat_xml", tmp_path / "out", strict=True)
    ds2 = dc_load(tmp_path / "out")
    assert [a.ann_id for a in ds2.annotations] == ["only:0:000000"]


def dc_load(path: Path):  # type: ignore[no-untyped-def]
    import datacopilot as dc

    return dc.load(path, format="cvat_xml", strict=True)


def test_a_renumbered_image_id_is_counted(tmp_path: Path) -> None:
    from datacopilot.core.schema import Image as Img

    synth.write_png(tmp_path / "i.png", 8, 8, 1)
    synth.write_png(tmp_path / "j.png", 8, 8, 2)
    imgs = [
        Img(
            image_id=f"{n:016x}",
            file_path=str(tmp_path / f),
            width=8,
            height=8,
            source={"format": "cvat_xml", "id": 5, "file_name": f},
        )  # fmt: skip
        for n, f in ((1, "i.png"), (2, "j.png"))
    ]
    ds = dataset_from(imgs, [], [Category(id=0, name="c")])
    report = convert.export(ds, "cvat_xml", tmp_path / "out")
    assert report.dropped == {"images.source.id": 1}
