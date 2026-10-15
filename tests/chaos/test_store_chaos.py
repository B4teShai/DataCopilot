"""An interrupted write must never leave a dataset that looks complete."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.store import DatasetStore

pytestmark = pytest.mark.chaos
Tables = tuple[list[Image], list[Annotation], list[Category]]


def _fail_on_nth_write(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    import pyarrow.parquet as pq

    real = pq.write_table
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == n:
            raise OSError("disk full")
        real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("datacopilot.store.dataset_store.pq.write_table", flaky)


@pytest.mark.parametrize("nth", [1, 2, 3])
def test_failed_create_leaves_nothing_behind(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch, nth: int
) -> None:
    _fail_on_nth_write(monkeypatch, nth)
    with pytest.raises(OSError, match="disk full"):
        DatasetStore.create(tmp_path / "ds.dcp", *tables, name="d", task="detection")
    assert list(tmp_path.iterdir()) == []


def test_failed_write_version_keeps_the_current_version(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DatasetStore.create(tmp_path / "ds.dcp", *tables, name="d", task="detection")
    before = store.manifest()
    images, anns, cats = tables
    _fail_on_nth_write(monkeypatch, 2)
    changed = [images[0].model_copy(update={"split": "test"}), *images[1:]]
    with pytest.raises(OSError, match="disk full"):
        store.write_version(changed, anns, cats)
    reopened = DatasetStore.open(tmp_path / "ds.dcp")
    assert reopened.read().manifest.version == before.version
    assert sorted(p.name for p in (tmp_path / "ds.dcp" / "versions").iterdir()) == [before.version]


KILLED_WRITER = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {tests!r})
    from conftest import make_tables
    import pyarrow.parquet as pq
    from datacopilot.store import DatasetStore

    real = pq.write_table
    n = {{"calls": 0}}
    def killer(*a, **k):
        real(*a, **k)
        n["calls"] += 1
        if n["calls"] == 2:
            os._exit(9)  # simulates kill -9 after two tables are on disk
    import datacopilot.store.dataset_store as m
    m.pq.write_table = killer
    DatasetStore.create(sys.argv[1], *make_tables(), name="d", task="detection")
    """
)


def test_killed_process_leaves_no_partial_dataset_and_a_retry_succeeds(
    tmp_path: Path, tables: Tables
) -> None:
    target = tmp_path / "ds.dcp"
    code = KILLED_WRITER.format(tests=str(Path(__file__).parent.parent))
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(target)], capture_output=True, check=False, timeout=120
    )
    assert proc.returncode == 9
    assert not target.exists()
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers and all(".tmp-" in name for name in leftovers)
    from datacopilot.core.errors import StoreError

    with pytest.raises(StoreError):
        DatasetStore.open(target)
    store = DatasetStore.create(target, *tables, name="d", task="detection")
    assert store.read().manifest.version == store.manifest().version
