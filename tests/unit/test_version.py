import hashlib
import random
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from datacopilot.core.errors import ValidationError
from datacopilot.core.schema import Annotation, Category, DatasetManifest, Image
from datacopilot.core.version import (
    canonical_json,
    dataset_version,
    extend_lineage,
    models_version,
    table_hash,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_canonical_json_sorts_keys_and_has_no_whitespace() -> None:
    assert canonical_json({"b": 1, "a": [1, 2.5, None, True], "c": {"z": "é", "y": 0}}) == (
        '{"a":[1,2.5,null,true],"b":1,"c":{"y":0,"z":"é"}}'
    )


def test_canonical_json_distinguishes_int_and_float() -> None:
    assert canonical_json({"v": 1}) != canonical_json({"v": 1.0})
    assert canonical_json({"v": 1.0}) == '{"v":1.0}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_a_validation_error(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        canonical_json({"v": [bad]})


def test_table_hash_matches_the_documented_procedure() -> None:
    rows = [{"k": 2}, {"k": 1}]
    expected = sha("".join(sorted([sha('{"k":1}'), sha('{"k":2}')])))
    assert table_hash(rows) == expected


def test_table_hash_of_empty_table_is_hash_of_empty_string() -> None:
    assert table_hash([]) == sha("")


def test_table_hash_ignores_excluded_columns() -> None:
    a = [{"id": 1, "file_path": "/a"}]
    b = [{"id": 1, "file_path": "/elsewhere"}]
    assert table_hash(a, exclude=frozenset({"file_path"})) == table_hash(
        b, exclude=frozenset({"file_path"})
    )
    assert table_hash(a) != table_hash(b)


def test_dataset_version_follows_the_documented_procedure() -> None:
    images = [{"image_id": "i", "file_path": "/a"}]
    anns = [{"ann_id": "a"}]
    cats = [{"id": 1}]
    expected = sha(
        "dcp:1\n"
        + table_hash(images, exclude=frozenset({"file_path"}))
        + "\n"
        + table_hash(anns)
        + "\n"
        + table_hash(cats)
    )[:32]
    assert dataset_version(images, anns, cats, schema_version=1) == expected
    assert len(expected) == 32


def test_schema_version_changes_the_hash() -> None:
    args: tuple[Any, ...] = ([{"image_id": "i"}], [], [])
    assert dataset_version(*args, schema_version=1) != dataset_version(*args, schema_version=2)


@given(
    st.lists(st.dictionaries(st.text(max_size=3), st.integers(), max_size=3), max_size=8),
    st.randoms(),
)
def test_row_order_never_changes_the_hash(rows: list[dict[str, int]], rnd: random.Random) -> None:
    shuffled = list(rows)
    rnd.shuffle(shuffled)
    assert table_hash(shuffled) == table_hash(rows)


def test_duplicate_rows_are_counted_not_collapsed() -> None:
    assert table_hash([{"a": 1}, {"a": 1}]) != table_hash([{"a": 1}])


def _tables() -> tuple[list[Image], list[Annotation], list[Category]]:
    images = [
        Image(image_id=f"{i:016x}", file_path=f"/p/{i}.png", width=10, height=10) for i in range(5)
    ]
    anns = [
        Annotation(
            ann_id=str(i),
            image_id=images[i].image_id,
            category_id=1,
            kind="bbox",
            bbox_xyxy_abs=(1.0, 1.0, 5.5, 5.0),
            source={"id": i, "area": 4.5},
        )
        for i in range(5)
    ]
    return images, anns, [Category(id=1, name="a"), Category(id=2, name="b")]


def test_shuffled_models_give_the_same_version() -> None:
    images, anns, cats = _tables()
    base = models_version(images, anns, cats, schema_version=1)
    rnd = random.Random(3)
    for _ in range(10):
        rnd.shuffle(images), rnd.shuffle(anns), rnd.shuffle(cats)
        assert models_version(images, anns, cats, schema_version=1) == base


def test_file_location_does_not_change_the_version_but_content_does() -> None:
    images, anns, cats = _tables()
    base = models_version(images, anns, cats, schema_version=1)
    moved = [i.model_copy(update={"file_path": "/other/" + i.file_path}) for i in images]
    assert models_version(moved, anns, cats, schema_version=1) == base
    split = [images[0].model_copy(update={"split": "train"}), *images[1:]]
    assert models_version(split, anns, cats, schema_version=1) != base
    renamed = [Category(id=1, name="renamed"), cats[1]]
    assert models_version(images, anns, renamed, schema_version=1) != base


def _manifest(version: str, lineage: tuple[str, ...]) -> DatasetManifest:
    return DatasetManifest(
        version=version,
        schema_version=1,
        name="d",
        task="detection",
        categories=(),
        lineage=lineage,
        image_count=0,
        annotation_count=0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_extend_lineage_appends_the_parent_version_last() -> None:
    parent = _manifest("b" * 32, ("a" * 32,))
    assert extend_lineage(parent, "c" * 32) == ("a" * 32, "b" * 32)


def test_extend_lineage_rejects_a_child_equal_to_its_parent() -> None:
    with pytest.raises(ValidationError, match="unchanged"):
        extend_lineage(_manifest("b" * 32, ()), "b" * 32)
