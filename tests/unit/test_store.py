import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from datacopilot.core.errors import StoreError, ValidationError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.core.version import models_version
from datacopilot.store import DatasetStore, arrow

Tables = tuple[list[Image], list[Annotation], list[Category]]
FILES = ("manifest.json", "images.parquet", "annotations.parquet", "categories.parquet")


def create(path: Path, tables: Tables) -> DatasetStore:
    images, anns, cats = tables
    return DatasetStore.create(path, images, anns, cats, name="demo", task="detection")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_create_writes_layout_and_a_verified_manifest(tmp_path: Path, tables: Tables) -> None:
    store = create(tmp_path / "ds.dcp", tables)
    root = tmp_path / "ds.dcp"
    manifest = store.manifest()
    assert all((root / f).is_file() for f in FILES)
    assert all((root / "versions" / manifest.version / f).is_file() for f in FILES)
    assert manifest.version == models_version(*tables, schema_version=1)
    assert manifest.schema_version == 1 and manifest.lineage == ()
    assert json.loads((root / "manifest.json").read_text())["version"] == manifest.version
    assert {c.name for c in manifest.categories} == {"cat", "dog"}
    assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]


def test_read_round_trips_every_field(tmp_path: Path, tables: Tables) -> None:
    create(tmp_path / "ds.dcp", tables)
    ds = DatasetStore.open(tmp_path / "ds.dcp").read()
    images, anns, cats = tables
    assert sorted(ds.images, key=lambda i: i.image_id) == sorted(images, key=lambda i: i.image_id)
    assert sorted(ds.annotations, key=lambda a: a.ann_id) == sorted(anns, key=lambda a: a.ann_id)
    assert sorted(ds.categories, key=lambda c: c.id) == sorted(cats, key=lambda c: c.id)
    by_id = {a.ann_id: a for a in ds.annotations}
    assert by_id["a1"].attributes == {"difficult": True, "n": 3, "w": 0.5, "s": "x"}
    assert (
        isinstance(by_id["a1"].attributes["n"], int) and by_id["a1"].attributes["difficult"] is True
    )
    assert by_id["a1"].source["segmentation"] == [[1, 2, 3, 4]]


def test_tables_use_explicit_arrow_schemas(tmp_path: Path, tables: Tables) -> None:
    create(tmp_path / "ds.dcp", tables)
    root = tmp_path / "ds.dcp"
    assert pq.read_schema(root / "images.parquet").equals(arrow.IMAGES_SCHEMA)
    assert pq.read_schema(root / "annotations.parquet").equals(arrow.ANNOTATIONS_SCHEMA)
    assert pq.read_schema(root / "categories.parquet").equals(arrow.CATEGORIES_SCHEMA)


def test_images_are_referenced_never_copied(tmp_path: Path, tables: Tables) -> None:
    (tmp_path / "pic.png").write_bytes(b"not really a png")
    images, anns, cats = tables
    images = [images[0].model_copy(update={"file_path": str(tmp_path / "pic.png")}), *images[1:]]
    create(tmp_path / "ds.dcp", (images, anns, cats))
    names = {p.name for p in (tmp_path / "ds.dcp").rglob("*") if p.is_file()}
    assert "pic.png" not in names and names <= set(FILES)


def test_create_refuses_to_overwrite_and_leaves_existing_untouched(
    tmp_path: Path, tables: Tables
) -> None:
    root = tmp_path / "ds.dcp"
    create(root, tables)
    before = {f: sha(root / f) for f in FILES}
    with pytest.raises(StoreError, match="already exists"):
        create(root, tables)
    assert {f: sha(root / f) for f in FILES} == before
    (tmp_path / "other").mkdir()
    with pytest.raises(StoreError, match="already exists"):
        create(tmp_path / "other", tables)


def test_create_validates_before_writing_anything(tmp_path: Path, tables: Tables) -> None:
    images, anns, cats = tables
    bad = [*anns, anns[0].model_copy(update={"ann_id": "zz", "category_id": 99})]
    with pytest.raises(ValidationError, match="unknown category"):
        DatasetStore.create(tmp_path / "ds.dcp", images, bad, cats, name="d", task="detection")
    assert list(tmp_path.iterdir()) == []


def test_output_is_byte_deterministic_under_input_reordering(
    tmp_path: Path, tables: Tables
) -> None:
    a = create(tmp_path / "a.dcp", tables)
    images, anns, cats = tables
    rnd = random.Random(1)
    rnd.shuffle(images), rnd.shuffle(anns), rnd.shuffle(cats)
    b = create(tmp_path / "b.dcp", (images, anns, cats))
    assert a.manifest().version == b.manifest().version
    for f in FILES[1:]:
        assert sha(tmp_path / "a.dcp" / f) == sha(tmp_path / "b.dcp" / f)


def test_write_version_appends_lineage_and_never_touches_old_snapshots(
    tmp_path: Path, tables: Tables
) -> None:
    root = tmp_path / "ds.dcp"
    store = create(root, tables)
    v1 = store.manifest()
    snapshot = {f: sha(root / "versions" / v1.version / f) for f in FILES}
    images, anns, cats = tables
    split = [images[2].model_copy(update={"split": "test"}), *images[:2]]
    v2 = store.write_version(split, anns, cats)
    assert v2.version != v1.version and v2.lineage == (v1.version,)
    assert store.manifest().version == v2.version
    assert DatasetStore.open(root).manifest().version == v2.version
    assert {f: sha(root / "versions" / v1.version / f) for f in FILES} == snapshot
    assert (root / "versions" / v2.version / "manifest.json").is_file()
    with pytest.raises(StoreError, match="already exists"):  # back to v1 content
        store.write_version(images, anns, cats)


def test_write_version_rejects_unchanged_content(tmp_path: Path, tables: Tables) -> None:
    store = create(tmp_path / "ds.dcp", tables)
    with pytest.raises(ValidationError, match="unchanged"):
        store.write_version(*tables)


def test_open_rejects_missing_or_foreign_directories(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="not a dataset"):
        DatasetStore.open(tmp_path / "nope.dcp")
    (tmp_path / "empty.dcp").mkdir()
    with pytest.raises(StoreError, match="manifest.json"):
        DatasetStore.open(tmp_path / "empty.dcp")


def test_open_refuses_a_newer_schema_version(tmp_path: Path, tables: Tables) -> None:
    root = tmp_path / "ds.dcp"
    create(root, tables)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["schema_version"] = 99
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="schema_version 99"):
        DatasetStore.open(root)


def test_open_rejects_corrupt_manifest_and_oversized_manifest(
    tmp_path: Path, tables: Tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ds.dcp"
    create(root, tables)
    (root / "manifest.json").write_text("{not json")
    with pytest.raises(StoreError, match="manifest.json"):
        DatasetStore.open(root)
    (root / "manifest.json").write_text("{}")
    with pytest.raises(StoreError, match="manifest.json"):
        DatasetStore.open(root)
    monkeypatch.setattr("datacopilot.store.dataset_store.MAX_MANIFEST_BYTES", 1)
    with pytest.raises(StoreError, match="too large"):
        DatasetStore.open(root)


def test_read_detects_tables_that_do_not_match_the_manifest(tmp_path: Path, tables: Tables) -> None:
    root = tmp_path / "ds.dcp"
    create(root, tables)
    flipped = [a.model_copy(update={"category_id": 3 - a.category_id}) for a in tables[1]]
    other = create(tmp_path / "other.dcp", (tables[0], flipped, tables[2]))
    swap = (tmp_path / "other.dcp" / "annotations.parquet").read_bytes()
    (root / "annotations.parquet").write_bytes(swap)
    assert other.manifest().version != DatasetStore.open(root).manifest().version
    with pytest.raises(StoreError, match="do not match"):
        DatasetStore.open(root).read()


def test_read_wraps_a_corrupt_parquet_file(tmp_path: Path, tables: Tables) -> None:
    root = tmp_path / "ds.dcp"
    create(root, tables)
    (root / "annotations.parquet").write_bytes(b"garbage")
    with pytest.raises(StoreError, match="annotations.parquet"):
        DatasetStore.open(root).read()


def test_leftover_temp_directories_are_ignored(tmp_path: Path, tables: Tables) -> None:
    root = tmp_path / "ds.dcp"
    store = create(root, tables)
    (root / "versions" / "deadbeef.tmp-123").mkdir()
    (tmp_path / "ds.dcp.tmp-999").mkdir()
    assert DatasetStore.open(root).read().manifest.version == store.manifest().version


def test_empty_annotation_table_round_trips(tmp_path: Path, tables: Tables) -> None:
    images, _, cats = tables
    store = DatasetStore.create(tmp_path / "e.dcp", images, [], cats, name="d", task="detection")
    ds = store.read()
    assert ds.annotations == () and len(ds.images) == 3


def test_a_symlinked_dataset_directory_is_refused(tmp_path: Path) -> None:
    import pytest as _pytest

    from datacopilot.core.errors import StoreError
    from datacopilot.store import DatasetStore

    real = tmp_path / "real.dcp"
    real.mkdir()
    link = tmp_path / "link.dcp"
    link.symlink_to(real, target_is_directory=True)
    with _pytest.raises(StoreError, match="symlink"):
        DatasetStore.open(link)
