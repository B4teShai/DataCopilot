"""A dataset written before oriented boxes existed (generation 1, by the code of that time) still
reads, verifies and keeps its version; writing the same rows today gives the same version and the
same annotations schema (ADR-0011)."""

from pathlib import Path

import pyarrow.parquet as pq

from datacopilot.store import DatasetStore, arrow

LEGACY = Path(__file__).parent / "data" / "legacy_v1.dcp"
LEGACY_VERSION = "59f516b835ffc25ce059a670f345ec0c"


def test_a_generation_1_store_still_reads_and_verifies() -> None:
    store = DatasetStore.open(LEGACY)
    manifest = store.manifest()
    assert (manifest.version, manifest.schema_version) == (LEGACY_VERSION, 1)
    ds = store.read(verify=True)
    assert {a.kind for a in ds.annotations} == {"bbox"}
    assert all(a.obb_xyxyxyxy_abs is None for a in ds.annotations)


def test_rewriting_a_generation_1_store_keeps_its_version_and_schema(tmp_path: Path) -> None:
    ds = DatasetStore.open(LEGACY).read()
    again = DatasetStore.create(
        tmp_path / "again.dcp",
        ds.images,
        ds.annotations,
        ds.categories,
        name="legacy",
        task="detection",
    )
    assert again.manifest().version == LEGACY_VERSION
    assert again.manifest().schema_version == 1
    for table in ("images.parquet", "annotations.parquet", "categories.parquet"):
        assert pq.read_schema(tmp_path / "again.dcp" / table).equals(pq.read_schema(LEGACY / table))
    assert pq.read_schema(LEGACY / "annotations.parquet").equals(arrow.ANNOTATIONS_SCHEMA)
