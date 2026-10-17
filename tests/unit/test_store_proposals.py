"""Proposal sets in dataset.dcp/proposals/ and a new dataset that records its parent."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from datacopilot.core.errors import StoreError, ValidationError
from datacopilot.core.proposals import Proposal, ProposalSet
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.store import DatasetStore, dataset_store

Tables = tuple[list[Image], list[Annotation], list[Category]]
VERSION = "c" * 32


def make_set(tables: Tables, **kw: object) -> ProposalSet:
    images, _, _ = tables
    proposal = Proposal(
        proposal_id="1" * 16,
        image_id=images[0].image_id,
        category_id=1,
        bbox_xyxy_abs=(1.0, 1.0, 20.0, 20.0),
        score=0.7,
        prompt="cat",
    )
    fields: dict[str, object] = {
        "dataset_version": VERSION,
        "proposer": "fake",
        "model_id": "fake_model",
        "model_revision": "r1",
        "weights": {},
        "prompt_hash": "d" * 64,
        "min_score": 0.3,
        "class_min_score": {},
        "overlap_iou": 0.5,
        "images_total": 3,
        "images_done": 3,
        "incomplete_reason": None,
        "rejected": {},
        "proposals": [proposal],
    }
    return ProposalSet.build(**{**fields, **kw})  # type: ignore[arg-type]


def open_store(tmp_path: Path, tables: Tables) -> DatasetStore:
    images, anns, cats = tables
    return DatasetStore.create(
        tmp_path / "ds.dcp", images, anns, cats, name="demo", task="detection"
    )


def test_a_set_round_trips_and_is_written_under_its_id(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    ps = make_set(tables)
    path = store.write_proposals(ps)
    assert path == tmp_path / "ds.dcp" / "proposals" / f"{ps.set_id}.json"
    assert store.read_proposals(ps.set_id) == ps
    assert not [p for p in path.parent.iterdir() if ".tmp-" in p.name]


def test_writing_the_same_set_twice_is_a_no_op(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    ps = make_set(tables)
    first = store.write_proposals(ps)
    stamp = first.stat().st_mtime_ns
    assert store.write_proposals(ps) == first and first.stat().st_mtime_ns == stamp


def test_writing_a_set_does_not_touch_the_dataset_version(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    root = tmp_path / "ds.dcp"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    store.write_proposals(make_set(tables))
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert before == after
    assert store.read().manifest.version == store.manifest().version


@pytest.mark.parametrize(
    "bad", ["", "../x", "latest", "A" * 32, "a" * 31, "a" * 33, "a/b" + "c" * 29]
)
def test_a_set_id_is_32_lower_hex_and_never_a_path(
    tmp_path: Path, tables: Tables, bad: str
) -> None:
    store = open_store(tmp_path, tables)
    with pytest.raises(StoreError, match="set id"):
        store.read_proposals(bad)


def test_a_missing_set_is_a_store_error(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    with pytest.raises(StoreError, match="no proposal set"):
        store.read_proposals("a" * 32)


def test_a_tampered_file_is_refused(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    ps = make_set(tables)
    path = store.write_proposals(ps)
    doc = json.loads(path.read_text())
    doc["proposals"][0]["score"] = 0.99
    path.write_text(json.dumps(doc))
    with pytest.raises(StoreError, match="invalid"):
        store.read_proposals(ps.set_id)


def test_a_valid_set_stored_under_another_name_is_refused(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    path = store.write_proposals(make_set(tables))
    other = make_set(tables, min_score=0.5)
    (path.parent / f"{other.set_id}.json").write_text(path.read_text())
    with pytest.raises(StoreError, match="does not match"):
        store.read_proposals(other.set_id)


@pytest.mark.parametrize("payload", ["", "{", "[]", "null", '{"set_id": 1}', "\xff", "1e999"])
def test_hostile_content_is_a_store_error_not_a_crash(
    tmp_path: Path, tables: Tables, payload: str
) -> None:
    store = open_store(tmp_path, tables)
    (tmp_path / "ds.dcp" / "proposals").mkdir()
    (tmp_path / "ds.dcp" / "proposals" / ("a" * 32 + ".json")).write_bytes(
        payload.encode("latin-1")
    )
    with pytest.raises(StoreError, match="invalid"):
        store.read_proposals("a" * 32)


def test_an_oversized_file_is_refused_before_it_is_parsed(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = open_store(tmp_path, tables)
    ps = make_set(tables)
    path = store.write_proposals(ps)
    monkeypatch.setattr(dataset_store, "MAX_PROPOSAL_BYTES", path.stat().st_size - 1)
    with pytest.raises(StoreError, match="too large"):
        store.read_proposals(ps.set_id)


def test_a_symlinked_file_or_directory_is_refused(tmp_path: Path, tables: Tables) -> None:
    store = open_store(tmp_path, tables)
    ps = make_set(tables)
    path = store.write_proposals(ps)
    real = tmp_path / "real.json"
    path.rename(real)
    os.symlink(real, path)
    with pytest.raises(StoreError, match="regular file"):
        store.read_proposals(ps.set_id)
    path.unlink()
    real.rename(path)
    moved = tmp_path / "elsewhere"
    path.parent.rename(moved)
    os.symlink(moved, tmp_path / "ds.dcp" / "proposals")
    with pytest.raises(StoreError, match="symlink"):
        store.read_proposals(ps.set_id)
    with pytest.raises(StoreError, match="symlink"):
        store.write_proposals(make_set(tables, min_score=0.5))


# --- a new dataset that names its parent --------------------------------------------------------
def test_create_records_the_given_lineage(tmp_path: Path, tables: Tables) -> None:
    images, anns, cats = tables
    parent = open_store(tmp_path, tables).manifest()
    child = DatasetStore.create(
        tmp_path / "child.dcp",
        images,
        anns[:-1],
        cats,
        name="demo",
        task="detection",
        lineage=(*parent.lineage, parent.version),
    )
    assert child.manifest().lineage == (parent.version,)
    assert DatasetStore.open(tmp_path / "child.dcp").read().manifest.lineage == (parent.version,)


@pytest.mark.parametrize("lineage", [("xyz",), ("A" * 32,), ("a" * 32, "a" * 32)])
def test_create_validates_the_lineage(
    tmp_path: Path, tables: Tables, lineage: tuple[str, ...]
) -> None:
    images, anns, cats = tables
    with pytest.raises(ValidationError, match="lineage"):
        DatasetStore.create(
            tmp_path / "x.dcp", images, anns, cats, name="d", task="detection", lineage=lineage
        )
    assert not (tmp_path / "x.dcp").exists()


def test_a_lineage_that_contains_the_new_version_is_refused(tmp_path: Path, tables: Tables) -> None:
    from datacopilot.core.version import models_version

    images, anns, cats = tables
    own = models_version(images, anns, cats, schema_version=1)
    with pytest.raises(ValidationError, match="lineage"):
        DatasetStore.create(
            tmp_path / "x.dcp", images, anns, cats, name="d", task="detection", lineage=(own,)
        )
