"""Label Studio: personal data, skipped-row accounting, lossless same-format round trips."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import dataset_from, table_rows
from fixtures import synth

import datacopilot as dc
from datacopilot import convert
from datacopilot.cli import app as cli
from datacopilot.core.errors import ConversionLossError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io.labelstudio import plugin
from datacopilot.store import DatasetStore

SIZE = (64, 48)


def rect(
    label: str = "cat", box: tuple[float, float, float, float] = (8, 6, 40, 30)
) -> dict[str, Any]:
    return synth.ls_rect(label, box, SIZE, "r")


def choice(label: str) -> dict[str, Any]:
    return {"value": {"choices": [label]}, "from_name": "c", "to_name": "image", "type": "choices"}


def task(
    results: list[Any] | None = None, tid: int = 1, image: str = "images/a.png", **fields: Any
) -> dict[str, Any]:
    body = {
        "id": tid,
        "data": {"image": image},
        "annotations": [{"id": 5, "result": [rect()] if results is None else results}],
    }
    return body | fields


def one(root: Path, tasks: Any, name: str = "annotations.json") -> Path:
    synth.write_png(root / "images" / "a.png", *SIZE, 1)
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(tasks))
    return root


def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


# ------------------------------------------------------ K1: skipped tasks with empty results
def skipped_export(tmp_path: Path) -> Path:
    t = task()
    t["annotations"] = [
        {"id": 1, "was_cancelled": True, "result": []},
        {"id": 2, "result": [rect()]},
        {"id": 3, "result": []},  # a second annotator who has not drawn anything
    ]
    t["predictions"] = [{"result": [rect("dog")]}, {"result": []}]
    return one(tmp_path / "ls", [t])


def test_skipped_entries_with_no_results_are_not_counted_as_zero(tmp_path: Path) -> None:
    result = plugin.read(skipped_export(tmp_path))
    assert result.skipped == {} and result.dropped == {}
    assert len(result.annotations) == 2


def test_ingest_through_the_command_line_survives_empty_skipped_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = skipped_export(tmp_path)
    capsys.readouterr()
    assert run(monkeypatch, "ingest", root, "-f", "label_studio", "-o", tmp_path / "a.dcp") == 0
    assert len(capsys.readouterr().out.strip()) == 32
    code = run(
        monkeypatch, "ingest", root, "-f", "label_studio", "-o", tmp_path / "b.dcp", "--strict"
    )
    assert code == 0, "nothing was lost, so --strict passes"


def test_load_through_the_api_survives_empty_skipped_entries(tmp_path: Path) -> None:
    ds = dc.load(skipped_export(tmp_path), format="label_studio", strict=True)
    assert ds.manifest.annotation_count == 2


def test_a_reader_that_lets_a_row_error_escape_is_a_format_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import BaseModel

    class Row(BaseModel):
        x: int

    def broken(self: object, path: Path, **_kw: object) -> None:
        Row(x="not a number")  # type: ignore[arg-type]

    monkeypatch.setattr(type(plugin), "read", broken)
    with pytest.raises(dc.FormatError, match="label_studio"):
        convert.ingest(tmp_path, "label_studio", tmp_path / "a.dcp")


# ---------------------------------------------------------------- K2: null scores
def test_a_null_score_is_no_score(tmp_path: Path) -> None:
    pred = {"score": None, "result": [{**rect(), "score": None}, {**rect("dog"), "score": 0.5}]}
    result = plugin.read(one(tmp_path / "ls", [task(predictions=[pred])]))
    models = sorted(
        (a for a in result.annotations if a.provenance == "model"), key=lambda a: a.ann_id
    )
    assert [a.score for a in models] == [None, 0.5]
    assert "score" not in result.images[0].source["prediction"]


# --------------------------------------- K3: same-format round trips are lossless or counted
def round_trip(root: Path, tmp_path: Path) -> tuple[Any, Any, Any]:
    first = convert.ingest(root, "label_studio", tmp_path / "a.dcp", strict=True)
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds, "label_studio", tmp_path / "out", strict=True)
    second = convert.ingest(tmp_path / "out", "label_studio", tmp_path / "b.dcp", strict=True)
    ds2 = DatasetStore.open(tmp_path / "b.dcp").read()
    assert table_rows(ds.images, ds.annotations) == table_rows(ds2.images, ds2.annotations)
    assert first.manifest.version == second.manifest.version
    return report, ds, ds2


def test_a_prediction_with_an_empty_result_survives_the_round_trip(tmp_path: Path) -> None:
    pred = {"id": 7, "model_version": "m1", "result": []}
    root = one(tmp_path / "ls", [task(predictions=[pred])])
    _, ds, _ = round_trip(root, tmp_path)
    assert ds.images[0].source["prediction"] == {"id": 7, "model_version": "m1"}
    written = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert written[0]["predictions"] == [{"id": 7, "model_version": "m1", "result": []}]


def multi_file_export(root: Path) -> Path:
    synth.write_png(root / "images" / "a.png", *SIZE, 1)
    synth.write_png(root / "images" / "b.png", *SIZE, 2)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "job1.json").write_text(json.dumps([task(tid=1)]))
    (root / "annotations" / "job2.json").write_text(
        json.dumps([task(tid=1, image="images/b.png")])  # the same task id in another export
    )
    (root / "annotations" / "job3.json").write_text("[]")  # an export with no tasks
    return root


def test_several_files_of_one_split_keep_their_ids_through_the_round_trip(tmp_path: Path) -> None:
    root = multi_file_export(tmp_path / "ls")
    _, ds, _ = round_trip(root, tmp_path)
    assert sorted(a.ann_id for a in ds.annotations) == ["job1:1:a000000", "job2:1:a000000"]
    assert sorted(p.name for p in (tmp_path / "out" / "annotations").iterdir()) == [
        "job1.json",
        "job2.json",
    ]


def test_a_single_file_inside_the_annotations_folder_keeps_its_prefix(tmp_path: Path) -> None:
    synth.write_png(tmp_path / "ls" / "images" / "a.png", *SIZE, 1)
    (tmp_path / "ls" / "annotations").mkdir(parents=True)
    (tmp_path / "ls" / "annotations" / "only.json").write_text(json.dumps([task()]))
    _, ds, _ = round_trip(tmp_path / "ls", tmp_path)
    assert [a.ann_id for a in ds.annotations] == ["only:1:a000000"]


@pytest.mark.parametrize(
    "ref",
    ["images/a.png", "a.png", "/data/local-files/?d=images/a.png", "/data/upload/3/a.png"],
)
def test_the_form_of_the_image_reference_survives_the_round_trip(ref: str, tmp_path: Path) -> None:
    root = one(tmp_path / "ls", [task(image=ref)])
    _, _, _ = round_trip(root, tmp_path)
    written = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert written[0]["data"]["image"] == ref


def test_a_renamed_image_is_counted_because_its_reference_changed(tmp_path: Path) -> None:
    synth.write_png(tmp_path / "ls" / "images" / "a.png", *SIZE, 1)
    synth.write_png(tmp_path / "ls" / "a.png", *SIZE, 2)  # the same name, another folder
    tasks = [task(tid=1), task(tid=2, image="a.png")]
    (tmp_path / "ls" / "annotations.json").write_text(json.dumps(tasks))
    convert.ingest(tmp_path / "ls", "label_studio", tmp_path / "a.dcp")
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    report = convert.export(ds, "label_studio", tmp_path / "out")
    assert report.dropped == {"images.source.file_name": 1}
    with pytest.raises(ConversionLossError):
        convert.export(ds, "label_studio", tmp_path / "strict", strict=True)


# ------------------------------------------------------------ K4: categories no row uses
def test_a_category_no_row_uses_is_counted_and_ids_are_ranked_among_the_used(
    tmp_path: Path,
) -> None:
    synth.write_png(tmp_path / "i.png", 8, 8, 1)
    image = Image(image_id=f"{1:016x}", file_path=str(tmp_path / "i.png"), width=8, height=8)
    ann = Annotation(ann_id="a", image_id=image.image_id, category_id=1, kind="bbox",
                     bbox_xyxy_abs=(1.0, 1.0, 4.0, 4.0))  # fmt: skip
    cats = [Category(id=0, name="bird"), Category(id=1, name="cat"), Category(id=2, name="dog")]
    ds = dataset_from([image], [ann], cats)
    report = convert.export(ds, "label_studio", tmp_path / "out")
    # the reader will number only `cat` (0): its id 1 changes; bird and dog are not carried
    assert report.dropped == {"categories.unused": 2, "categories.id": 1}
    with pytest.raises(ConversionLossError, match="categories.unused=2"):
        convert.export(ds, "label_studio", tmp_path / "strict", strict=True)
    ds2 = dc.load(tmp_path / "out", format="label_studio")
    assert [(c.id, c.name) for c in ds2.categories] == [(0, "cat")]


# ----------------------------------------------- K5: the task type follows the human rows
def test_human_choices_win_over_model_boxes(tmp_path: Path) -> None:
    t = task([choice("dog")], predictions=[{"result": [rect()]}])
    result = plugin.read(one(tmp_path / "ls", [t]))
    assert result.task == "classification"
    assert [a.provenance for a in result.annotations] == ["human"]
    assert result.skipped == {"annotations.unsupported.rectanglelabels": 1}


def test_human_boxes_win_over_model_choices(tmp_path: Path) -> None:
    t = task([rect()], predictions=[{"result": [choice("dog")]}])
    result = plugin.read(one(tmp_path / "ls", [t]))
    assert result.task == "detection" and result.skipped == {"annotations.unsupported.choices": 1}


# ------------------------------------------- F2 and K6: personal data never reaches anything
EMAIL = "alice@corp.example"
PRESIGNED = "https://bucket.s3.amazonaws.com/a.png?X-Amz-Signature=deadbeef0123456789"


def realistic_export(root: Path) -> Path:
    result = rect()
    result.update(
        {
            "origin": "manual",
            "meta": {"text": [f"checked by {EMAIL}"]},
            "created_by": EMAIL,
            "readonly": False,
        }  # fmt: skip
    )
    result["value"]["text"] = [f"note from {EMAIL}"]
    t = task(
        [result],
        created_at="2024-05-01T10:00:00Z",
        inner_id=12,
        file_upload=f"{EMAIL.split('@')[0]}-a.png",
        meta={"owner": EMAIL},
        updated_by=[{"user_id": 7, "email": EMAIL}],
        created_username=f"{EMAIL}, 7",
        comment_authors=[{"email": EMAIL, "first_name": "Alice"}],
        comment_count=1,
        data={"image": "images/a.png", "caption": f"taken by {EMAIL}", "download": PRESIGNED},
    )
    t["annotations"][0].update(
        {
            "completed_by": {"id": 7, "email": EMAIL, "first_name": "Alice", "last_name": "A"},
            "created_username": f"{EMAIL}, 7",
            "updated_by": 7,
            "draft_created_at": "2024-05-01",
            "lead_time": 3.5,
            "ground_truth": False,
            "was_cancelled": False,
        }
    )
    t["predictions"] = [
        {
            "id": 9,
            "model_version": "v2",
            "score": 0.9,
            "created_ago": f"by {EMAIL}",
            "result": [{**rect("dog"), "score": 0.9}],
        }  # fmt: skip
    ]
    return one(root, [t])


def every_text(tmp_path: Path) -> str:
    chunks = []
    for path in tmp_path.rglob("*"):
        if path.is_file() and "ls" != path.parent.name and "images" not in path.parts:
            chunks.append(path.read_bytes().decode("latin-1"))
    return "\n".join(chunks)


def test_personal_data_in_a_realistic_export_never_reaches_a_stored_or_written_file(
    tmp_path: Path,
) -> None:
    root = realistic_export(tmp_path / "ls")
    first = convert.ingest(root, "label_studio", tmp_path / "work" / "a.dcp")
    dropped = first.report.dropped
    for key in ("created_username", "updated_by", "comment_authors", "meta", "file_upload",
                "comment_count", "data.caption", "data.download", "annotation.completed_by",
                "annotation.created_username", "annotation.updated_by",
                "annotation.draft_created_at", "prediction.created_ago"):  # fmt: skip
        assert dropped[f"images.source.{key}"] == 1, key
    assert dropped["annotations.source.meta"] == 1 and dropped["annotations.source.created_by"] == 1
    assert dropped["annotations.source.value.text"] == 1
    ds = DatasetStore.open(tmp_path / "work" / "a.dcp").read()
    source = ds.images[0].source
    assert source["created_at"] == "2024-05-01T10:00:00Z" and source["inner_id"] == 12
    assert source["annotation"] == {"id": 5, "lead_time": 3.5, "ground_truth": False,
                                    "was_cancelled": False}  # fmt: skip
    assert source["prediction"] == {"id": 9, "model_version": "v2", "score": 0.9}
    convert.export(ds, "label_studio", tmp_path / "work" / "out")
    dc.audit(ds, fail_on=())  # logs and reports are produced from the same rows
    from datacopilot.audit.context import CheckContext
    from datacopilot.audit.engine import run_audit
    from datacopilot.report import render

    ctx = CheckContext(run_id="00000000-0000-4000-8000-000000000000", root=root,
                       cache_dir=tmp_path / "cache", workers=1, use_hash_cache=False)  # fmt: skip
    run_ = run_audit(ds, "all", ctx, ())
    render(run_, html=tmp_path / "work" / "r.html", json=tmp_path / "work" / "r.json",
           sarif=tmp_path / "work" / "r.sarif")  # fmt: skip
    leaked = [
        str(p.relative_to(tmp_path))
        for p in (tmp_path / "work").rglob("*")
        if p.is_file()
        and p.suffix != ".png"
        and (
            EMAIL.encode() in p.read_bytes()
            or b"X-Amz-Signature" in p.read_bytes()
            or b"alice" in p.read_bytes().lower()
        )
    ]
    assert leaked == []
    assert EMAIL not in json.dumps([i.model_dump() for i in ds.images], default=str)


def test_task_data_is_kept_only_with_the_explicit_option_and_then_re_emitted(
    tmp_path: Path,
) -> None:
    root = realistic_export(tmp_path / "ls")
    kept = convert.ingest(
        root, "label_studio", tmp_path / "a.dcp", reader_options={"keep_task_data": True}
    )
    assert "images.source.data.caption" not in kept.report.dropped
    ds = DatasetStore.open(tmp_path / "a.dcp").read()
    assert ds.images[0].source["data"]["caption"] == f"taken by {EMAIL}"
    convert.export(ds, "label_studio", tmp_path / "out")
    written = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert written[0]["data"]["caption"] == f"taken by {EMAIL}"


def test_keep_task_data_through_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = realistic_export(tmp_path / "ls")
    assert run(monkeypatch, "ingest", root, "-f", "label_studio", "-o", tmp_path / "a.dcp",
               "--keep-task-data") == 0  # fmt: skip
    assert "caption" in DatasetStore.open(tmp_path / "a.dcp").read().images[0].source["data"]
    capsys.readouterr()
    code = run(
        monkeypatch, "ingest", root, "-f", "label_studio", "-o", tmp_path / "b.dcp", "--strict"
    )
    err = capsys.readouterr().err
    assert code == 1 and "images.source.created_username=1" in err and EMAIL not in err
    coco = synth.build_coco(tmp_path / "coco")
    assert (
        run(monkeypatch, "ingest", coco, "-f", "coco", "-o", tmp_path / "c.dcp", "--keep-task-data")
        == 2
    )


def test_hostile_key_names_become_a_bounded_column(tmp_path: Path) -> None:
    t = task()
    t["\x1b[31m::error::x"] = 1
    for n in range(400):
        t[f"k{n}"] = n
    result = plugin.read(one(tmp_path / "ls", [t]))
    assert "images.source.invalid_key" in result.dropped
    assert len(result.dropped) <= 202 and "images.other" in result.dropped
    assert sum(result.dropped.values()) == 401
