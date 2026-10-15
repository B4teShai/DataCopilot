import json
import os
from pathlib import Path

import pytest
from test_audit_schema import RUN, finding, run_of

from datacopilot.core.errors import StoreError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.store import DatasetStore

Tables = tuple[list[Image], list[Annotation], list[Category]]


def make_store(tmp_path: Path, tables: Tables) -> DatasetStore:
    return DatasetStore.create(tmp_path / "ds.dcp", *tables, name="d", task="detection")


def test_write_audit_creates_audits_run_json_atomically(tmp_path: Path, tables: Tables) -> None:
    store = make_store(tmp_path, tables)
    run = run_of(finding("WARN"))
    path = store.write_audit(run)
    assert path == tmp_path / "ds.dcp" / "audits" / f"{RUN}.json"
    assert json.loads(path.read_text())["run_id"] == RUN
    assert [p.name for p in path.parent.iterdir()] == [f"{RUN}.json"]  # no temp files left


def test_audit_round_trips_through_the_store(tmp_path: Path, tables: Tables) -> None:
    store = make_store(tmp_path, tables)
    run = run_of(finding("WARN"), finding("INFO", image_id="1" * 16))
    store.write_audit(run)
    assert store.read_audit(RUN) == run


def test_write_audit_refuses_to_overwrite(tmp_path: Path, tables: Tables) -> None:
    store = make_store(tmp_path, tables)
    store.write_audit(run_of())
    with pytest.raises(StoreError, match="already exists"):
        store.write_audit(run_of(finding("INFO")))


def test_audit_does_not_change_the_dataset_version(tmp_path: Path, tables: Tables) -> None:
    store = make_store(tmp_path, tables)
    before = store.manifest().version
    store.write_audit(run_of())
    assert DatasetStore.open(tmp_path / "ds.dcp").read().manifest.version == before


@pytest.mark.parametrize(
    "run_id", ["../evil", "a/b", "..", "x\x00y", "C:\\x", "/abs", "", "G" * 36, RUN + "/x"]
)
def test_read_audit_rejects_ids_that_are_not_uuids(
    tmp_path: Path, tables: Tables, run_id: str
) -> None:
    store = make_store(tmp_path, tables)
    with pytest.raises(StoreError, match="run_id"):
        store.read_audit(run_id)


def test_read_audit_rejects_oversized_and_malformed_files(tmp_path: Path, tables: Tables) -> None:
    store = make_store(tmp_path, tables)
    audits = tmp_path / "ds.dcp" / "audits"
    audits.mkdir()
    (audits / f"{RUN}.json").write_text("{not json")
    with pytest.raises(StoreError, match="invalid"):
        store.read_audit(RUN)
    big = "2222bbbb-2222-4333-8444-5555bbbb5555"
    with (audits / f"{big}.json").open("wb") as handle:
        handle.truncate(300 * 1024 * 1024)
    with pytest.raises(StoreError, match="too large"):
        store.read_audit(big)
    with pytest.raises(StoreError, match="no audit"):
        store.read_audit("3333cccc-2222-4333-8444-5555bbbb5555")


def test_write_audit_does_not_follow_a_symlinked_audits_directory(
    tmp_path: Path, tables: Tables
) -> None:
    store = make_store(tmp_path, tables)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "ds.dcp" / "audits")
    with pytest.raises(StoreError, match="audits"):
        store.write_audit(run_of())
    assert list(outside.iterdir()) == []


def test_failed_write_leaves_no_partial_file(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path, tables)

    def boom(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("datacopilot.store.dataset_store.os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.write_audit(run_of())
    assert list((tmp_path / "ds.dcp" / "audits").iterdir()) == []
