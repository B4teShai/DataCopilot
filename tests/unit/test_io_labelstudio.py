import json
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import dataset_from, table_rows
from fixtures import synth
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from datacopilot import convert
from datacopilot.core.errors import ConversionLossError, FormatError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files, registry
from datacopilot.io.labelstudio import plugin
from datacopilot.store import DatasetStore

SIZE = (64, 48)


def rect(
    label: str = "cat", box: tuple[float, float, float, float] = (8, 6, 40, 30), **extra: Any
) -> dict[str, Any]:
    return {**synth.ls_rect(label, box, SIZE, "r"), **extra}


def task(
    results: list[dict[str, Any]] | None = None, tid: int = 1, **fields: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": tid,
        "data": {"image": "images/a.png"},
        "annotations": [{"id": 5, "result": results if results is not None else [rect()]}],
    }
    return body | fields


def one(root: Path, tasks: Any) -> Path:
    synth.write_png(root / "images" / "a.png", *SIZE, 1)
    (root / "annotations.json").write_text(json.dumps(tasks))
    return root


@pytest.fixture
def ls(tmp_path: Path) -> Path:
    return synth.build_label_studio(tmp_path / "ls")


# ------------------------------------------------------------------ reader
def test_registered_for_read_and_write() -> None:
    info = {i.spec.name: i for i in registry.list_formats()}["label_studio"]
    assert info.can_read and info.can_write


def test_reads_fixture(ls: Path) -> None:
    r = plugin.read(ls)
    assert r.task == "detection" and r.dropped == {} and r.skipped == {}
    assert [(c.id, c.name) for c in r.categories] == [(0, "bird"), (1, "cat"), (2, "dog")]
    assert {i.source["file_name"]: i.split for i in r.images} == {
        "a.png": "train",
        "b.png": "train",
        "c.png": "val",
        "d.png": "test",
    }
    a = next(i for i in r.images if i.source["file_name"] == "a.png")
    assert a.source == {
        "format": "label_studio",
        "id": 1,
        "file_name": "a.png",
        "layout_document": "train.json",
        "annotation": {"id": 10, "was_cancelled": False, "lead_time": 1.5},
    }
    boxes = sorted((x for x in r.annotations if x.image_id == a.image_id), key=lambda x: x.ann_id)
    assert [b.ann_id for b in boxes] == ["train:1:a000000", "train:1:a000001"]
    assert boxes[0].bbox_xyxy_abs == (8.0, 6.0, 40.0, 30.0)
    assert boxes[0].category_id == 1 and boxes[0].provenance == "human"
    assert boxes[0].source == {
        "format": "label_studio",
        "id": "r0_0",
        "from_name": "label",
        "to_name": "image",
    }


def test_percentages_are_applied_to_the_real_image_size(tmp_path: Path) -> None:
    result = rect(box=(16, 12, 32, 24))
    result["original_width"], result["original_height"] = 640, 480  # a larger original
    r = plugin.read(one(tmp_path / "r", [task([result])]))
    assert r.annotations[0].bbox_xyxy_abs == pytest.approx((16, 12, 32, 24), abs=1e-6)
    assert r.dropped == {"images.declared_size": 1}


def test_tiny_overshoot_is_snapped_and_counted_larger_is_an_error(tmp_path: Path) -> None:
    result = rect()
    result["value"].update(x=50.0, width=50.001)  # 64 px wide image: 0.00064 px over
    r = plugin.read(one(tmp_path / "a", [task([result])]))
    assert r.annotations[0].bbox_xyxy_abs is not None
    assert r.annotations[0].bbox_xyxy_abs[2] == 64.0
    assert r.dropped == {"annotations.bbox_snapped": 1}
    result["value"].update(width=51.0)
    with pytest.raises(FormatError, match="beyond the image"):
        plugin.read(one(tmp_path / "b", [task([result])]))


def test_choices_make_a_classification_dataset(tmp_path: Path) -> None:
    choice = {
        "value": {"choices": ["dog"]},
        "from_name": "c",
        "to_name": "image",
        "type": "choices",
    }
    r = plugin.read(one(tmp_path / "r", [task([choice])]))
    assert r.task == "classification" and r.skipped == {}
    assert [c.name for c in r.categories] == ["dog"]
    ann = r.annotations[0]
    assert ann.kind == "classification" and ann.bbox_xyxy_abs is None and ann.category_id == 0


def test_boxes_win_over_choices_and_the_choices_are_counted(tmp_path: Path) -> None:
    choice = {
        "value": {"choices": ["dog", "cat"]},
        "from_name": "c",
        "to_name": "image",
        "type": "choices",
    }
    r = plugin.read(one(tmp_path / "r", [task([rect(), choice])]))
    assert r.task == "detection" and len(r.annotations) == 1
    assert r.skipped == {"annotations.unsupported.choices": 1}


def test_predictions_are_model_rows_with_scores(tmp_path: Path) -> None:
    pred = {"id": 7, "model_version": "m1", "score": 0.5, "result": [rect(), rect(score=0.9)]}
    r = plugin.read(one(tmp_path / "r", [task([], predictions=[pred])]))
    rows = sorted(r.annotations, key=lambda a: a.ann_id)
    assert [a.provenance for a in rows] == ["model", "model"]
    assert [a.score for a in rows] == [0.5, 0.9]
    assert rows[0].ann_id == "1:p000000"
    assert r.images[0].source["prediction"] == {"id": 7, "model_version": "m1", "score": 0.5}
    assert r.images[0].source["annotation"] == {"id": 5}


def test_multiple_annotators_policy_is_one_set_and_the_rest_is_counted(tmp_path: Path) -> None:
    t = task()
    t["annotations"] = [
        {"id": 9, "result": [rect("dog")]},
        {"id": 3, "result": [rect("cat"), rect("cat", (1, 1, 5, 5))]},
        {"id": 4, "ground_truth": True, "result": [rect("dog", (2, 2, 9, 9))]},
        {"id": 1, "was_cancelled": True, "result": [rect()]},
    ]
    r = plugin.read(one(tmp_path / "a", [t]))
    assert [a.category_id for a in r.annotations] == [0]  # dog: the only label read
    assert r.skipped == {"annotations.of_other_annotators": 3, "annotations.cancelled": 1}
    t["annotations"][2]["ground_truth"] = False  # without ground truth the lowest id wins
    r = plugin.read(one(tmp_path / "b", [t]))
    assert len(r.annotations) == 2 and r.annotations[0].bbox_xyxy_abs == (8.0, 6.0, 40.0, 30.0)
    assert r.skipped == {"annotations.of_other_annotators": 2, "annotations.cancelled": 1}


def test_only_the_first_prediction_is_read(tmp_path: Path) -> None:
    preds = [{"result": [rect()]}, {"result": [rect(), rect()]}]
    r = plugin.read(one(tmp_path / "r", [task([], predictions=preds)]))
    assert len(r.annotations) == 1
    assert r.skipped == {"annotations.of_other_predictions": 2}


@pytest.mark.parametrize(
    ("result", "key"),
    [
        ({"type": "polygonlabels", "value": {"points": [[1, 1], [2, 2]]}}, "polygonlabels"),
        ({"type": "brushlabels", "value": {"rle": [1]}}, "brushlabels"),
        ({"type": "keypointlabels", "value": {"x": 1, "y": 1}}, "keypointlabels"),
        ({"type": "textarea", "value": {"text": ["x"]}}, "textarea"),
        ({"type": "relation", "from_id": "a", "to_id": "b"}, "relation"),
        ({"type": "madeup<script>", "value": {}}, "other"),
        ({"value": {}}, "other"),
    ],
)
def test_unsupported_result_types_are_counted(
    result: dict[str, Any], key: str, tmp_path: Path
) -> None:
    r = plugin.read(one(tmp_path / "r", [task([rect(), result])]))
    assert r.skipped == {f"annotations.unsupported.{key}": 1} and len(r.annotations) == 1


def test_rotated_multi_label_and_rotated_image_rectangles_are_counted(tmp_path: Path) -> None:
    rotated = rect()
    rotated["value"]["rotation"] = 12.5
    multi = rect()
    multi["value"]["rectanglelabels"] = ["cat", "dog"]
    turned = rect()
    turned["image_rotation"] = 90
    r = plugin.read(one(tmp_path / "r", [task([rect(), rotated, multi, turned])]))
    assert r.skipped == {
        "annotations.unsupported.rotated_rectangle": 1,
        "annotations.unsupported.multi_label_rectangle": 1,
        "annotations.unsupported.image_rotation": 1,
    }
    assert len(r.annotations) == 1


@pytest.mark.parametrize(
    "ref",
    [
        "http://example.com/a.png",
        "https://x/a.png",
        "s3://b/a.png",
        "gs://b/a.png",
        "data:image/png;base64,AAAA",
        "azure-blob://c/a.png",
    ],
)
def test_remote_images_are_skipped_never_fetched(ref: str, tmp_path: Path) -> None:
    r = plugin.read(one(tmp_path / "r", [task(data={"image": ref})]))
    assert r.images == () or len(r.images) == 0
    assert r.skipped == {"images.remote_uri": 1, "annotations.of_missing_image": 1}


def test_local_files_and_upload_references_are_resolved(tmp_path: Path) -> None:
    root = one(tmp_path / "r", [])
    tasks = [
        task(tid=1, data={"image": "/data/local-files/?d=images/a.png"}),
        task(tid=2, data={"image": "/data/upload/3/a.png"}),
    ]
    (root / "annotations.json").write_text(json.dumps(tasks))
    r = plugin.read(root)
    assert len(r.images) == 2 and r.skipped == {}


def test_missing_files_skip_the_task_and_are_counted(tmp_path: Path) -> None:
    root = one(tmp_path / "r", [task(), task(tid=2, data={"image": "images/gone.png"})])
    r = plugin.read(root)
    assert r.skipped == {"images.file_missing": 1, "annotations.of_missing_image": 1}


def test_unlabelled_tasks_are_marked_and_their_predictions_are_not_read_as_rows(
    tmp_path: Path,
) -> None:
    bare = {"id": 2, "data": {"image": "images/b.png"}, "annotations": []}
    pre = {"id": 3, "data": {"image": "images/c.png"}, "annotations": [],
           "predictions": [{"result": [rect(), rect()]}]}  # fmt: skip
    root = one(tmp_path / "r", [task(), bare, pre])
    synth.write_png(root / "images" / "b.png", 40, 40, 2)
    synth.write_png(root / "images" / "c.png", 40, 40, 3)
    r = plugin.read(root)
    by_name = {i.source["file_name"]: i for i in r.images}
    assert by_name["b.png"].source["unlabeled"] is True and by_name["c.png"].source["unlabeled"]
    assert len(r.annotations) == 1  # only the human box of a.png
    assert r.skipped == {"annotations.prediction_of_unlabeled_task": 2}


def test_task_fields_outside_the_allow_list_are_dropped_and_counted(tmp_path: Path) -> None:
    t = task(file_upload="a.png", meta={"k": [1, 2]}, created_at="2024-01-01", inner_id=3,
             data={"image": "images/a.png", "text": "hi"})  # fmt: skip
    r = plugin.read(one(tmp_path / "r", [t]))
    source = r.images[0].source
    assert source["created_at"] == "2024-01-01" and source["inner_id"] == 3
    assert "data" not in source and "file_upload" not in source and "meta" not in source
    assert r.dropped == {
        "images.source.file_upload": 1,
        "images.source.meta": 1,
        "images.source.data.text": 1,
    }


def test_task_data_fields_are_kept_only_on_request(tmp_path: Path) -> None:
    t = task(data={"image": "images/a.png", "text": "hi"})
    r = plugin.read(one(tmp_path / "r", [t]), keep_task_data=True)
    assert r.images[0].source["data"] == {"text": "hi"} and r.dropped == {}


def test_single_file_path_and_missing_inputs(tmp_path: Path) -> None:
    root = one(tmp_path / "r", [task()])
    assert len(plugin.read(root / "annotations.json").images) == 1
    with pytest.raises(FormatError, match="does not exist"):
        plugin.read(tmp_path / "nope")
    (tmp_path / "e").mkdir()
    with pytest.raises(FormatError, match="no Label Studio"):
        plugin.read(tmp_path / "e")


# ------------------------------------------------------- hostile input
@pytest.mark.parametrize(
    "document",
    [
        "{}",
        "[1]",
        '[{"data": {"image": "images/a.png"}}]',  # no id
        '[{"id": true, "data": {"image": "images/a.png"}}]',
        '[{"id": -1, "data": {"image": "images/a.png"}}]',
        '[{"id": 1}]',
        '[{"id": 1, "data": []}]',
        '[{"id": 1, "data": {"image": 5}}]',
        '[{"id": 1, "data": {"image": "images/a.png"}, "annotations": {}}]',
        '[{"id": 1, "data": {"image": "images/a.png"}, "annotations": [{"id": 1}]}]',
        '[{"id": 1, "data": {"image": "images/a.png"}, "annotations": [5]}]',
        '[{"id": 1, "data": {"image": "images/a.png"}, "annotations": [{"result": [5]}]}]',
    ],
)
def test_malformed_structure_is_a_format_error(document: str, tmp_path: Path) -> None:
    synth.write_png(tmp_path / "r" / "images" / "a.png", *SIZE, 1)
    (tmp_path / "r" / "annotations.json").write_text(document)
    with pytest.raises(FormatError):
        plugin.read(tmp_path / "r")


def test_duplicate_task_ids_and_duplicate_json_keys_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="duplicate task id"):
        plugin.read(one(tmp_path / "a", [task(), task()]))
    synth.write_png(tmp_path / "b" / "images" / "a.png", *SIZE, 1)
    (tmp_path / "b" / "annotations.json").write_text(
        '[{"id": 1, "id": 2, "data": {"image": "images/a.png"}, "annotations": []}]'
    )
    with pytest.raises(FormatError, match="not valid JSON"):
        plugin.read(tmp_path / "b")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_literals_are_rejected(literal: str, tmp_path: Path) -> None:
    text = json.dumps([task()]).replace('"x": 12.5', f'"x": {literal}')
    root = one(tmp_path / "r", [])
    (root / "annotations.json").write_text(text)
    with pytest.raises(FormatError, match="not valid JSON"):
        plugin.read(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", 1e999),
        ("x", "12"),
        ("x", True),
        ("x", None),
        ("x", -0.5),
        ("x", 100.5),
        ("y", -1),
        ("y", 101),
        ("width", 0),
        ("width", -5),
        ("width", 100.5),
        ("height", 0),
        ("height", float("1e400")),
    ],
)
def test_invalid_percentages_are_rejected(field: str, value: object, tmp_path: Path) -> None:
    result = rect()
    result["value"][field] = value
    root = one(tmp_path / "r", [])
    text = json.dumps([task([result])]).replace("Infinity", "1e999")
    (root / "annotations.json").write_text(text)
    with pytest.raises(FormatError, match="percent|finite|number"):
        plugin.read(root)


@pytest.mark.parametrize("value", [0, -5, "64", True, 1e999, 10**12, None])
def test_invalid_original_sizes_are_rejected(value: object, tmp_path: Path) -> None:
    result = rect()
    result["original_width"] = value
    root = one(tmp_path / "r", [])
    (root / "annotations.json").write_text(
        json.dumps([task([result])]).replace("Infinity", "1e999")
    )
    with pytest.raises(FormatError, match="original"):
        plugin.read(root)


@pytest.mark.parametrize(
    "ref",
    [
        "../outside.png",
        "/etc/passwd",
        "a/../../b.png",
        "..\\x.png",
        "images/a\x00.png",
        "/data/local-files/?d=../../etc/passwd",
        "/data/local-files/?d=/etc/passwd",
        "/data/upload/1/..%2F..%2Fx.png",
        "/data/upload/1/%2Fetc%2Fpasswd",
    ],
)
def test_image_paths_cannot_leave_the_dataset(ref: str, tmp_path: Path) -> None:
    root = one(tmp_path / "r", [task(data={"image": ref})])
    (tmp_path / "outside.png").write_bytes(synth.png_bytes(4, 4, 9))
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


def test_symlinked_image_pointing_outside_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "r"
    (root / "images").mkdir(parents=True)
    (tmp_path / "secret.png").write_bytes(synth.png_bytes(4, 4, 3))
    (root / "images" / "a.png").symlink_to(tmp_path / "secret.png")
    (root / "annotations.json").write_text(json.dumps([task()]))
    with pytest.raises(FormatError, match="unsafe path"):
        plugin.read(root)


def test_overlong_reference_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="too long"):
        plugin.read(one(tmp_path / "r", [task(data={"image": "a" * 5000})]))


def test_nesting_bomb_is_rejected_quickly(tmp_path: Path) -> None:
    root = one(tmp_path / "r", [])
    (root / "annotations.json").write_text("[" * 200_000 + "]" * 200_000)
    started = time.perf_counter()
    with pytest.raises(FormatError, match="nested too deeply|not valid JSON"):
        plugin.read(root)
    assert time.perf_counter() - started < 2.0


def test_oversized_file_is_refused_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = one(tmp_path / "r", [task()])
    monkeypatch.setattr(files, "MAX_LABEL_STUDIO_BYTES", 100)
    with pytest.raises(FormatError, match="larger than 100 bytes"):
        plugin.read(root)


def test_huge_counts_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datacopilot.io.labelstudio.MAX_RESULTS_PER_TASK", 3)
    with pytest.raises(FormatError, match="too many results"):
        plugin.read(one(tmp_path / "a", [task([rect()] * 4)]))
    monkeypatch.setattr("datacopilot.io.labelstudio.MAX_ANNOTATIONS_PER_TASK", 2)
    t = task()
    t["annotations"] = [{"id": i, "result": []} for i in range(3)]
    with pytest.raises(FormatError, match="too many annotations"):
        plugin.read(one(tmp_path / "b", [t]))
    monkeypatch.setattr("datacopilot.io.labelstudio.MAX_TASKS", 1)
    with pytest.raises(FormatError, match="too many tasks"):
        plugin.read(one(tmp_path / "c", [task(), task(tid=2)]))


def test_label_and_field_names_are_bounded_strings(tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="label"):
        plugin.read(one(tmp_path / "a", [task([rect("x" * 5000)])]))
    with pytest.raises(FormatError, match="label"):
        plugin.read(one(tmp_path / "b", [task([rect("")])]))
    bad = rect()
    bad["from_name"] = 5
    with pytest.raises(FormatError, match="from_name"):
        plugin.read(one(tmp_path / "c", [task([bad])]))


def test_markup_in_label_names_is_data_and_survives_a_round_trip(tmp_path: Path) -> None:
    hostile = "<img src=x onerror=alert(1)>&\"'"
    root = one(tmp_path / "r", [task([rect(hostile)])])
    first = convert.ingest(root, "label_studio", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    assert [c.name for c in ds.categories] == [hostile]
    convert.export(ds, "label_studio", tmp_path / "out", strict=True)
    again = convert.ingest(tmp_path / "out", "label_studio", tmp_path / "b.dcp")
    assert again.manifest.version == first.manifest.version


def test_markup_in_a_label_is_escaped_in_the_html_report(tmp_path: Path) -> None:
    hostile = "<script>alert(1)</script>"
    convert.ingest(one(tmp_path / "r", [task([rect(hostile)])]), "label_studio", tmp_path / "a.dcp")
    from datacopilot.audit import engine
    from datacopilot.audit.context import CheckContext
    from datacopilot.report import render

    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    ctx = CheckContext(
        run_id="00000000-0000-4000-8000-000000000000",
        root=tmp_path,
        cache_dir=tmp_path / "cache",
        workers=1,
        use_hash_cache=False,
        dataset_dir=tmp_path / "a.dcp",
    )
    result = engine.run_audit(ds, engine.select_checks("all", with_embedder=False), ctx, ())
    render(result, html=tmp_path / "r.html", json=tmp_path / "r.json", sarif=tmp_path / "r.sarif")
    page = (tmp_path / "r.html").read_text()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


# ------------------------------------------------------------------ writer
def test_round_trip_is_lossless_and_field_identical(ls: Path, tmp_path: Path) -> None:
    first = convert.ingest(ls, "label_studio", tmp_path / "a.dcp")
    assert first.report.is_lossless()
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds, "label_studio", tmp_path / "out", strict=True)
    assert report.dropped == {} and report.skipped == {}
    second = convert.ingest(tmp_path / "out", "label_studio", tmp_path / "b.dcp", strict=True)
    ds2 = DatasetStore.open(tmp_path / "b.dcp").read()
    assert table_rows(ds.images, ds.annotations) == table_rows(ds2.images, ds2.annotations)
    assert first.manifest.version == second.manifest.version
    assert sorted(p.name for p in (tmp_path / "out" / "annotations").iterdir()) == [
        "test.json",
        "train.json",
        "val.json",
    ]


def test_written_document_is_plain_label_studio_json(ls: Path, tmp_path: Path) -> None:
    convert.ingest(ls, "label_studio", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    convert.export(ds, "label_studio", tmp_path / "out")
    tasks = json.loads((tmp_path / "out" / "annotations" / "train.json").read_text())
    first = tasks[0]
    assert first["data"] == {"image": "images/a.png"} and "predictions" not in first
    result = first["annotations"][0]["result"][0]
    assert result["type"] == "rectanglelabels" and result["value"]["rectanglelabels"] == ["cat"]
    assert 0 <= result["value"]["x"] <= 100 and result["value"]["rotation"] == 0
    assert (tmp_path / "out" / "images" / "a.png").is_file()


def test_unassigned_dataset_writes_a_single_document(tmp_path: Path) -> None:
    convert.ingest(one(tmp_path / "r", [task()]), "label_studio", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    convert.export(ds, "label_studio", tmp_path / "out")
    assert (tmp_path / "out" / "annotations.json").is_file()


def test_predictions_and_unlabelled_tasks_round_trip(tmp_path: Path) -> None:
    pred = {"id": 7, "model_version": "m1", "result": [rect(score=0.25)]}
    bare = {"id": 2, "data": {"image": "images/b.png"}, "annotations": []}
    root = one(tmp_path / "r", [task(predictions=[pred]), bare])
    synth.write_png(root / "images" / "b.png", 40, 40, 2)
    first = convert.ingest(root, "label_studio", tmp_path / "a.dcp", strict=True)
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    convert.export(ds, "label_studio", tmp_path / "out", strict=True)
    written = json.loads((tmp_path / "out" / "annotations.json").read_text())
    by_id = {t["id"]: t for t in written}
    assert by_id[1]["predictions"][0]["result"][0]["score"] == 0.25
    assert by_id[1]["predictions"][0]["model_version"] == "m1"
    assert by_id[2]["annotations"] == []
    again = convert.ingest(tmp_path / "out", "label_studio", tmp_path / "b.dcp", strict=True)
    assert again.manifest.version == first.manifest.version


def test_classification_round_trip_from_imagefolder(tmp_path: Path) -> None:
    first = convert.ingest(
        synth.build_imagefolder(tmp_path / "src"), "imagefolder", tmp_path / "a.dcp"
    )
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds, "label_studio", tmp_path / "out")
    assert report.skipped == {}
    back = plugin.read(tmp_path / "out")
    assert (
        back.task == "classification" and len(back.annotations) == first.manifest.annotation_count
    )
    assert {c.name for c in back.categories} == {"cat", "dog"}
    names = {c.id: c.name for c in back.categories}
    by_image = {a.image_id: names[a.category_id] for a in back.annotations}
    assert by_image == {
        a.image_id: {c.id: c.name for c in ds.categories}[a.category_id] for a in ds.annotations
    }
    assert {i.split for i in back.images} == {"train", "val"}


def _foreign(tmp_path: Path) -> Any:
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
            occluded=False,
            attributes={"difficult": True},
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
    return dataset_from(images, anns, cats)


def test_foreign_rows_are_written_and_every_loss_is_counted(tmp_path: Path) -> None:
    ds = _foreign(tmp_path)
    report = convert.export(ds, "label_studio", tmp_path / "out")
    assert report.dropped == {
        "annotations.attributes": 1,
        "annotations.is_crowd": 1,
        "annotations.occluded": 1,
        "annotations.provenance": 1,
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
        convert.export(ds, "label_studio", tmp_path / "strict", strict=True)
    back = plugin.read(tmp_path / "out")
    assert back.dropped == {} and len(back.annotations) == 3
    assert sorted(a.provenance for a in back.annotations) == ["human", "human", "model"]
    assert [a.score for a in back.annotations if a.provenance == "model"] == [0.5]


def test_names_and_directories_of_unsafe_images_are_contained(tmp_path: Path) -> None:
    ds = _foreign(tmp_path)
    ds = dataset_from(
        [
            i.model_copy(
                update={"source": {"format": "label_studio", "file_name": "../../evil.png"}}
            )
            for i in ds.images
        ],
        ds.annotations,
        ds.categories,
    )
    with pytest.raises(FormatError, match="unsafe path"):
        convert.export(ds, "label_studio", tmp_path / "out")
    assert not (tmp_path / "out").exists()


@settings(max_examples=60, deadline=None)
@given(w=st.integers(1, 200), h=st.integers(1, 200), data=st.data())
def test_boxes_survive_the_writer_and_reader_exactly(w: int, h: int, data: st.DataObject) -> None:
    import tempfile

    x1 = round(data.draw(st.floats(0, w - 0.01, allow_nan=False)), 6)
    y1 = round(data.draw(st.floats(0, h - 0.01, allow_nan=False)), 6)
    x2 = round(data.draw(st.floats(x1 + 0.001, w, allow_nan=False)), 6)
    y2 = round(data.draw(st.floats(y1 + 0.001, h, allow_nan=False)), 6)
    assume(x1 < x2 <= w and y1 < y2 <= h)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        synth.write_png(root / "i.png", w, h, 1)
        image = Image(image_id=f"{1:016x}", file_path=str(root / "i.png"), width=w, height=h)
        ann = Annotation(
            ann_id="a",
            image_id=image.image_id,
            category_id=0,
            kind="bbox",
            bbox_xyxy_abs=(x1, y1, x2, y2),
        )
        ds = dataset_from([image], [ann], [Category(id=0, name="cat")])
        (root / "out").mkdir()
        plugin.write(ds, root / "out")
        back = plugin.read(root / "out")
        assert back.annotations[0].bbox_xyxy_abs == (x1, y1, x2, y2)
