"""``split`` end to end: a new version with lineage, groups that never cross, and a leakage check
on the result that finds nothing. The ``hash`` embedder (no weights) finds near copies of bytes."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import pytest
from fixtures.auditdata import Img, build
from fixtures.synth import png_bytes

import datacopilot as dc
from datacopilot.core.errors import StoreError, ValidationError
from datacopilot.settings import load_settings
from datacopilot.splits import usecase
from datacopilot.store import DatasetStore


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("CI", "DATACOPILOT_CACHE_KEY", "DATACOPILOT_ALLOWED_LICENSES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))


def specs(n: int = 30, *, copies: int = 4) -> list[Img]:
    """``n`` distinct images in train and test, plus ``copies`` near copies of image 0 (the same
    bytes with a few appended), half of them already in the test split: a planted leak."""
    out = [
        Img(
            f"img{k}.png",
            split="test" if k % 5 == 0 else "train",
            boxes=[(k % 3, 2, 2, 30, 30)],
            seed=k,  # the pixels follow the name, not the position in the list
        )
        for k in range(n)
    ]
    base = png_bytes(64, 48, 0)
    out += [
        Img(
            f"copy{k}.png",
            split="test" if k % 2 else "train",
            data=base + bytes([k + 1]) * 4,
            boxes=[(0, 2, 2, 30, 30)],
        )
        for k in range(copies)
    ]
    return out


def store(root: Path, items: list[Img], name: str = "ds.dcp") -> Path:
    ds = build(root, items)
    DatasetStore.create(
        root / name, ds.images, ds.annotations, ds.categories, name="demo", task="detection"
    )
    return root / name


def split(source: Path, out: Path, **kw: Any) -> Any:
    kw.setdefault("ratios", (0.8, 0.1, 0.1))
    return usecase.split_dataset(source, out, settings=load_settings(), **kw)


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def cross_split_leaks(path: Path) -> list[Any]:
    run = dc.audit(path, ["leakage"], embedder="hash", allow_test_embedder=True, fail_on=())
    return [f for f in run.findings if f.payload["rule"] == "cross_split_near_duplicate"]


def test_after_a_near_duplicate_split_the_leakage_check_finds_no_cross_split_pair(
    tmp_path: Path,
) -> None:
    source = store(tmp_path, specs())
    assert cross_split_leaks(source)  # the planted leak is real before the split
    out = tmp_path / "split.dcp"
    report = split(source, out, near_duplicates=True, embedder="hash")
    assert cross_split_leaks(out) == []
    assert report.near_duplicate_pairs >= 6  # 5 near copies of one picture: at least 10 pairs
    assert report.groups_across_splits == 0 and report.largest_group >= 5
    assert report.embedder == "hash@1-test-only" and report.near_duplicate_threshold == 0.95
    images = DatasetStore.open(out).read().images
    copies = {i.split for i in images if Path(i.file_path).name.startswith("copy")}
    assert len(copies) == 1


def test_the_new_version_has_lineage_and_the_source_is_not_touched(tmp_path: Path) -> None:
    source = store(tmp_path, specs())
    parent = DatasetStore.open(source).manifest()
    before = tree_digest(source)
    out = tmp_path / "split.dcp"
    report = split(source, out)
    after = DatasetStore.open(out).manifest()
    assert after.lineage == (*parent.lineage, parent.version)
    assert report.parent_version == parent.version and report.dataset_version == after.version
    assert tree_digest(source) == before
    ds = DatasetStore.open(out).read()
    assert len(ds.annotations) == parent.annotation_count  # labels are carried over unchanged
    assert {i.split for i in ds.images} <= {"train", "val", "test"}
    assert sum(s.images for s in report.splits.values()) == report.image_count == len(ds.images)


def test_groups_from_a_file_name_pattern_never_cross_and_become_the_group_id(
    tmp_path: Path,
) -> None:
    items = [
        Img(f"cam{k % 7}_{k:03d}.png", boxes=[(k % 3, 2, 2, 30, 30)], seed=500 + k)
        for k in range(42)
    ]
    source = store(tmp_path, items)
    out = tmp_path / "split.dcp"
    report = split(source, out, group_key=r"re:^(?P<g>[^_]+)_")
    ds = DatasetStore.open(out).read()
    by_group: dict[str, set[str]] = {}
    for image in ds.images:
        assert image.group_id == Path(image.file_path).name.split("_")[0]
        by_group.setdefault(image.group_id or "", set()).add(image.split)
    assert len(by_group) == 7 and all(len(s) == 1 for s in by_group.values())
    assert report.key_groups == 7 and report.images_without_key == 0
    assert report.group_key == r"re:^(?P<g>[^_]+)_"
    run = dc.audit(out, ["leakage"], embedder="hash", allow_test_embedder=True, fail_on=())
    assert [f for f in run.findings if f.payload["rule"] == "group_overlap"] == []


def test_the_assignment_does_not_depend_on_row_order_and_repeats(tmp_path: Path) -> None:
    items = specs()
    first = store(tmp_path, items, "a.dcp")
    shuffled = list(items)
    random.Random(5).shuffle(shuffled)
    (tmp_path / "other").mkdir()
    second = store(tmp_path / "other", shuffled, "b.dcp")
    kw: dict[str, Any] = {"near_duplicates": True, "embedder": "hash", "stratify": True, "seed": 9}
    split(first, tmp_path / "a1.dcp", **kw)
    split(first, tmp_path / "a2.dcp", **kw)
    split(second, tmp_path / "b1.dcp", **kw)

    def assignment(path: Path) -> list[tuple[str, str]]:
        return sorted((i.image_id, i.split) for i in DatasetStore.open(path).read().images)

    assert assignment(tmp_path / "a1.dcp") == assignment(tmp_path / "a2.dcp")
    assert assignment(tmp_path / "a1.dcp") == assignment(tmp_path / "b1.dcp")
    v1 = DatasetStore.open(tmp_path / "a1.dcp").manifest().version
    assert v1 == DatasetStore.open(tmp_path / "a2.dcp").manifest().version


def test_stratification_is_reported_per_split_and_class(tmp_path: Path) -> None:
    source = store(tmp_path, specs(60, copies=0))
    report = split(source, tmp_path / "s.dcp", stratify=True, ratios=(0.6, 0.2, 0.2))
    assert report.stratify == "category"
    assert set(report.classes) == {"train", "val", "test"}
    assert report.max_class_deviation is not None and report.max_class_deviation <= 0.1
    assert report.moved_images > 0


def test_refusals_write_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = store(tmp_path, specs())
    with pytest.raises(ValidationError, match="outside"):
        split(source, source / "inner.dcp")
    (tmp_path / "exists.dcp").mkdir()
    with pytest.raises(StoreError):
        split(source, tmp_path / "exists.dcp")
    with pytest.raises(ValueError, match="embedder"):
        split(source, tmp_path / "x.dcp", near_duplicates=True)
    with pytest.raises(ValueError, match="threshold"):
        split(source, tmp_path / "x.dcp", similarity_threshold=0.9)
    with pytest.raises(ValueError, match="ratios"):
        split(source, tmp_path / "x.dcp", ratios=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="by-group"):
        split(source, tmp_path / "x.dcp", group_key="re:(a+)+")
    monkeypatch.setattr(usecase, "MAX_PAIRS", 3)
    with pytest.raises(ValidationError, match="pairs"):
        split(source, tmp_path / "x.dcp", near_duplicates=True, embedder="hash")
    assert not (tmp_path / "x.dcp").exists()


def test_an_image_without_an_embedding_refuses_a_near_duplicate_split(tmp_path: Path) -> None:
    source = store(tmp_path, specs())
    (tmp_path / "images" / "img3.png").unlink()
    with pytest.raises(ValidationError, match="file_missing=1"):
        split(source, tmp_path / "x.dcp", near_duplicates=True, embedder="hash")
    assert not (tmp_path / "x.dcp").exists()
    report = split(source, tmp_path / "y.dcp")  # without near duplicates no image is read
    assert report.near_duplicate_pairs == 0 and report.embedder is None


def test_a_split_that_changes_nothing_is_refused(tmp_path: Path) -> None:
    items = [Img(f"i{k}.png", boxes=[(0, 2, 2, 30, 30)]) for k in range(5)]
    source = store(tmp_path, items)
    with pytest.raises(ValidationError, match="nothing"):
        split(source, tmp_path / "s.dcp", ratios=(1.0, 0.0, 0.0))
