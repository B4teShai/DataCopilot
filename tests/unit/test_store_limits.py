import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datacopilot.core.errors import StoreError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.store import DatasetStore, StoreLimits

Tables = tuple[list[Image], list[Annotation], list[Category]]


def make(tmp_path: Path, tables: Tables) -> Path:
    DatasetStore.create(tmp_path / "ds.dcp", *tables, name="d", task="detection")
    return tmp_path / "ds.dcp"


def read(root: Path, **limits: int) -> None:
    DatasetStore.open(root, limits=StoreLimits(**limits)).read()


def test_row_limit_is_exercised_at_the_boundary(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)  # 4 annotations, 3 images
    read(root, max_rows=4)
    with pytest.raises(StoreError, match=r"annotations\.parquet.*4 rows.*limit of 3"):
        read(root, max_rows=3)


def _uncompressed(path: Path) -> int:
    meta = pq.ParquetFile(path).metadata
    return sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))


def test_uncompressed_size_limit_is_exercised_at_the_boundary(
    tmp_path: Path, tables: Tables
) -> None:
    root = make(tmp_path, tables)
    largest = max(_uncompressed(root / f) for f in ("images.parquet", "annotations.parquet"))
    read(root, max_uncompressed_bytes=largest)
    with pytest.raises(StoreError, match=r"uncompressed.*limit of " + str(largest - 1)):
        read(root, max_uncompressed_bytes=largest - 1)


def test_compressed_file_size_limit(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    with pytest.raises(StoreError, match=r"images\.parquet.*file size"):
        read(root, max_file_bytes=10)


def test_limits_are_checked_before_any_data_is_read(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make(tmp_path, tables)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("data was read before the limit check")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", boom)
    monkeypatch.setattr(pq, "read_table", boom)
    with pytest.raises(StoreError, match="limit"):
        read(root, max_rows=1)


def test_read_is_incremental_over_batches(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make(tmp_path, tables)
    calls: list[int] = []
    real = pq.ParquetFile.iter_batches

    def spy(self: pq.ParquetFile, batch_size: int = 65536, **kw: object) -> object:
        calls.append(batch_size)
        return real(self, batch_size=1, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    ds = DatasetStore.open(root).read()
    assert len(calls) == 3 and len(ds.annotations) == 4


def _replace(root: Path, name: str, table: pa.Table) -> None:
    pq.write_table(table, root / name)


def test_wrong_column_type_fails_cleanly(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    good = pq.read_table(root / "categories.parquet")
    bad = good.set_column(0, "id", pa.array([str(i) for i in good["id"].to_pylist()]))
    _replace(root, "categories.parquet", bad)
    with pytest.raises(StoreError, match=r"categories\.parquet.*schema"):
        DatasetStore.open(root).read()


def test_missing_column_fails_cleanly(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    _replace(root, "images.parquet", pq.read_table(root / "images.parquet").drop(["source"]))
    with pytest.raises(StoreError, match=r"images\.parquet.*schema"):
        DatasetStore.open(root).read()


def test_nullable_column_that_must_not_be_null_fails_cleanly(
    tmp_path: Path, tables: Tables
) -> None:
    root = make(tmp_path, tables)
    t = pq.read_table(root / "categories.parquet")
    nullable = pa.schema([pa.field(f.name, f.type, nullable=True) for f in t.schema])
    _replace(root, "categories.parquet", t.cast(nullable))
    with pytest.raises(StoreError, match=r"categories\.parquet.*schema"):
        DatasetStore.open(root).read()


def test_null_in_a_required_value_fails_cleanly(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    t = pq.read_table(root / "categories.parquet")
    nullable = pa.schema([pa.field(f.name, f.type, nullable=True) for f in t.schema])
    t = t.cast(nullable).set_column(1, pa.field("name", pa.string()), pa.array([None, "dog"]))
    _replace(root, "categories.parquet", t)
    with pytest.raises(StoreError, match="categories.parquet"):
        DatasetStore.open(root).read()


@pytest.mark.parametrize("field", ["image_count", "annotation_count"])
def test_manifest_counts_must_equal_actual_rows(tmp_path: Path, tables: Tables, field: str) -> None:
    root = make(tmp_path, tables)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest[field] > 0
    manifest[field] += 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match=rf"{field}.*claims"):
        DatasetStore.open(root).read()


def test_manifest_counts_are_checked_even_without_verify(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["image_count"] = 99
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="image_count"):
        DatasetStore.open(root).read(verify=False)


def test_write_version_updates_counts(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    store = DatasetStore.open(root)
    images, anns, cats = tables
    m = store.write_version(images, anns[:2], cats)
    assert (m.image_count, m.annotation_count) == (3, 2)
    assert DatasetStore.open(root).read().manifest.annotation_count == 2


def test_a_box_that_is_not_four_numbers_fails_cleanly(tmp_path: Path, tables: Tables) -> None:
    root = make(tmp_path, tables)
    t = pq.read_table(root / "annotations.parquet")
    boxes = [[1.0, 2.0, 3.0] for _ in range(t.num_rows)]
    bad = t.set_column(
        t.schema.get_field_index("bbox_xyxy_abs"),
        arrow_field(),
        pa.array(boxes, type=pa.list_(pa.float64())),
    )
    pq.write_table(bad, root / "annotations.parquet")
    with pytest.raises(StoreError, match=r"annotations\.parquet"):
        DatasetStore.open(root).read()


def arrow_field() -> pa.Field:
    return pa.field("bbox_xyxy_abs", pa.list_(pa.float64()))
