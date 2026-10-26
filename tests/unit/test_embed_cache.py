"""Embedding cache: atomic shards, digest and MAC verification, hostile and partial files.

The cache lives inside a dataset directory a pull request can modify, so every test of a bad
shard asserts the same thing: it is rejected (or ignored) and counted, never silently trusted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datacopilot.embed import cache as ec

VERSION = "a" * 32
OTHER_VERSION = "b" * 32
SECRET = b"s" * 32
DIMS = 8


def ident(**kw: object) -> ec.CacheIdentity:
    base: dict[str, object] = dict(
        model_id="m", revision="1", dims=DIMS, dtype="float32", dataset_version=VERSION, weights={}
    )
    base.update(kw)
    return ec.CacheIdentity(**base)  # type: ignore[arg-type]


def unit(n: int, seed: int = 0, dims: int = DIMS, dtype: str = "float32") -> np.ndarray:  # type: ignore[type-arg]
    """Canonical vectors, as the pipeline hands them to the cache."""
    v = np.random.default_rng(seed).standard_normal((n, dims)).astype(np.float32)
    return ec.canonical_vectors(v, dtype)  # type: ignore[arg-type]


def ids(n: int, start: int = 0) -> list[str]:
    return [f"{start + i:016x}" for i in range(n)]


def dg(n: int, start: int = 0) -> list[str]:
    """Full digests of the image bytes, one per row."""
    return [f"{start + i:064x}" for i in range(n)]


def open_cache(
    root: Path, identity: ec.CacheIdentity | None = None, secret: bytes | None = SECRET
) -> ec.EmbeddingCache:
    return ec.EmbeddingCache.open(root / "embeddings", identity or ident(), secret)


def cache_dir(root: Path) -> Path:
    return root / "embeddings" / "m@1"


def shard_files(root: Path) -> list[Path]:
    return sorted(cache_dir(root).glob("shard-*.parquet"))


def manifest(root: Path) -> dict[str, object]:
    return json.loads((cache_dir(root) / "manifest.json").read_text())  # type: ignore[no-any-return]


def write_manifest(root: Path, doc: dict[str, object], secret: bytes | None = SECRET) -> None:
    """Rewrite the manifest the way an attacker who can edit the dataset directory would."""
    doc = {k: v for k, v in doc.items() if k != "mac"}
    if secret is not None:
        doc["mac"] = ec.mac_for(doc, secret, ident())
    (cache_dir(root) / "manifest.json").write_text(json.dumps(doc))


def fill(root: Path, n: int = 5, **kw: object) -> ec.EmbeddingCache:
    c = open_cache(root, **kw)  # type: ignore[arg-type]
    c.load()
    c.append(ids(n), unit(n), dg(n))
    return c


# ------------------------------------------------------------ round trip
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_roundtrip_returns_exactly_the_canonical_vectors(tmp_path: Path, dtype: str) -> None:
    identity = ident(dtype=dtype)
    c = ec.EmbeddingCache.open(tmp_path / "embeddings", identity, SECRET)
    assert c.load().state == "absent"
    raw = np.random.default_rng(0).standard_normal((20, DIMS)).astype(np.float32) * 7.0
    canonical = ec.canonical_vectors(raw, dtype)  # type: ignore[arg-type]
    c.append(ids(20), canonical, dg(20))
    got = ec.EmbeddingCache.open(tmp_path / "embeddings", identity, SECRET).load()
    assert got.state == "ok" and got.rejected == {}
    assert list(got.ids) == sorted(ids(20))
    np.testing.assert_array_equal(got.matrix, canonical)  # bit-identical: fresh == cached
    assert got.matrix.dtype == np.float32 and got.matrix.shape == (20, DIMS)


def test_float16_rounding_is_bounded_and_documented(tmp_path: Path) -> None:
    v = unit(200, seed=3, dims=DIMS)
    c16 = ec.canonical_vectors(v, "float16")
    assert np.max(np.abs(c16 - v)) < 1e-3  # IEEE half: 11 significant bits, round to nearest even
    np.testing.assert_allclose(np.linalg.norm(c16, axis=1), 1.0, atol=1e-3)
    np.testing.assert_allclose(ec.canonical_vectors(v, "float32"), v, atol=1e-6)  # only normalises
    np.testing.assert_array_equal(c16.astype(np.float16).astype(np.float32), c16)  # representable


def test_canonical_vectors_reject_nan_inf_zero_and_wrong_shapes() -> None:
    for bad in (np.full((2, DIMS), np.nan), np.full((2, DIMS), np.inf), np.zeros((2, DIMS))):
        with pytest.raises(ValueError):
            ec.canonical_vectors(bad.astype(np.float32), "float32")
    with pytest.raises(ValueError):
        ec.canonical_vectors(np.ones(DIMS, dtype=np.float32), "float32")  # 1-D


def test_resume_appends_shards_and_keeps_earlier_ones(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(4), unit(4), dg(4))
    c2 = open_cache(tmp_path)
    assert len(c2.load().ids) == 4
    c2.append(ids(3, start=4), unit(3, seed=1), dg(3, start=4))
    got = open_cache(tmp_path).load()
    assert list(got.ids) == sorted(ids(7)) and len(shard_files(tmp_path)) == 2


def test_the_manifest_records_identity_dataset_version_weights_and_digests(
    tmp_path: Path,
) -> None:
    weights = {"model.safetensors": "c" * 64}
    c = ec.EmbeddingCache.open(tmp_path / "embeddings", ident(weights=weights), SECRET)
    c.load()
    c.append(ids(3), unit(3), dg(3))
    doc = manifest(tmp_path)
    assert (doc["model_id"], doc["revision"], doc["dims"], doc["dtype"]) == (
        "m",
        "1",
        DIMS,
        "float32",
    )
    assert doc["dataset_version"] == VERSION and doc["weights"] == weights
    (entry,) = doc["shards"]  # type: ignore[misc]
    shard = cache_dir(tmp_path) / entry["name"]
    assert entry["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert entry["rows"] == 3 and entry["bytes"] == shard.stat().st_size
    assert isinstance(doc["mac"], str) and len(doc["mac"]) == 64


# ------------------------------------------------------------ binding and trust
def test_a_different_dataset_version_is_stale_not_an_attack(tmp_path: Path) -> None:
    fill(tmp_path)
    got = open_cache(tmp_path, ident(dataset_version=OTHER_VERSION)).load()
    assert got.state == "stale" and len(got.ids) == 0 and got.rejected == {}


@pytest.mark.parametrize(
    "change",
    [dict(dims=16), dict(dtype="float16"), dict(weights={"model.safetensors": "d" * 64})],
)
def test_a_different_model_identity_is_never_reused(
    tmp_path: Path, change: dict[str, object]
) -> None:
    fill(tmp_path)
    got = open_cache(tmp_path, ident(**change)).load()
    assert len(got.ids) == 0 and got.state in ("stale", "rejected")


def test_a_manifest_edited_without_the_key_is_rejected_whole(tmp_path: Path) -> None:
    fill(tmp_path)
    doc = manifest(tmp_path)
    write_manifest(tmp_path, doc, secret=b"x" * 32)  # forged: a valid-looking MAC, wrong key
    got = open_cache(tmp_path).load()
    assert got.state == "rejected" and got.rejected == {"bad_mac": 1} and len(got.ids) == 0


def test_a_forged_shard_with_a_matching_digest_but_no_key_is_rejected(tmp_path: Path) -> None:
    """An attacker replaces vectors and fixes the digest; without the MAC key the manifest fails."""
    fill(tmp_path, 5)
    doc = manifest(tmp_path)
    (entry,) = doc["shards"]  # type: ignore[misc]
    forged = ec.encode_shard(ident(), ids(5), unit(5, seed=99), dg(5))
    (cache_dir(tmp_path) / entry["name"]).write_bytes(forged)
    entry.update(sha256=hashlib.sha256(forged).hexdigest(), bytes=len(forged))
    (cache_dir(tmp_path) / "manifest.json").write_text(json.dumps(doc))  # old MAC kept
    got = open_cache(tmp_path).load()
    assert got.state == "rejected" and len(got.ids) == 0 and "bad_mac" in got.rejected


def test_without_a_secret_the_cache_is_disabled_and_writes_nothing(tmp_path: Path) -> None:
    c = ec.EmbeddingCache.open(tmp_path / "embeddings", ident(), None)
    got = c.load()
    assert got.state == "disabled" and len(got.ids) == 0
    c.append(ids(2), unit(2), dg(2))  # a no-op, not an error: the caller computes in memory
    assert not (tmp_path / "embeddings").exists()


def test_disabled_cache_reports_the_reason_it_was_given(tmp_path: Path) -> None:
    c = ec.EmbeddingCache.disabled("ci_without_key")
    assert c.load().state == "disabled" and c.disabled_reason == "ci_without_key"


# ------------------------------------------------------------ partial and damaged shards
def test_a_truncated_shard_is_rejected_and_the_others_survive(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(3), unit(3), dg(3))
    c.append(ids(3, 3), unit(3, 1), dg(3, 3))
    first = shard_files(tmp_path)[0]
    first.write_bytes(first.read_bytes()[:-20])
    got = open_cache(tmp_path).load()
    assert got.state == "ok" and got.rejected in ({"bad_size": 1}, {"bad_digest": 1})
    assert len(got.ids) == 3  # the intact shard only


def test_a_flipped_bit_fails_the_digest(tmp_path: Path) -> None:
    fill(tmp_path)
    (shard,) = shard_files(tmp_path)
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0x01
    shard.write_bytes(bytes(data))
    got = open_cache(tmp_path).load()
    assert got.rejected == {"bad_digest": 1} and len(got.ids) == 0


def test_a_missing_listed_shard_is_counted(tmp_path: Path) -> None:
    fill(tmp_path)
    shard_files(tmp_path)[0].unlink()
    got = open_cache(tmp_path).load()
    assert got.rejected == {"shard_missing": 1} and len(got.ids) == 0


def test_unlisted_shards_and_temporary_files_are_ignored_and_cleaned_on_write(
    tmp_path: Path,
) -> None:
    fill(tmp_path)
    orphan = cache_dir(tmp_path) / ("shard-" + "e" * 32 + ".parquet")
    orphan.write_bytes(b"half written")
    partial = cache_dir(tmp_path) / "shard-x.parquet.tmp-1234"
    partial.write_bytes(b"x")
    got = open_cache(tmp_path).load()
    assert got.state == "ok" and got.rejected == {} and len(got.ids) == 5
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(1, 9), unit(1, 7), dg(1, 9))
    assert not orphan.exists() and not partial.exists()
    assert len(open_cache(tmp_path).load().ids) == 6


def test_a_crash_between_the_shard_and_the_manifest_leaves_the_old_state_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill(tmp_path, 4)
    real_replace = os.replace

    def failing(src: object, dst: object) -> None:
        if str(dst).endswith("manifest.json"):
            raise OSError("simulated crash")
        real_replace(src, dst)  # type: ignore[arg-type]

    c = open_cache(tmp_path)
    c.load()
    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError):
        c.append(ids(3, 4), unit(3, 5), dg(3, 4))
    monkeypatch.undo()
    got = open_cache(tmp_path).load()
    assert got.state == "ok" and got.rejected == {} and list(got.ids) == sorted(ids(4))


def test_a_rejected_manifest_means_the_next_write_starts_a_fresh_cache(tmp_path: Path) -> None:
    fill(tmp_path)
    (cache_dir(tmp_path) / "manifest.json").write_text("{not json")
    c = open_cache(tmp_path)
    assert c.load().state == "rejected"
    c.append(ids(2, 20), unit(2, 2), dg(2, 20))
    got = open_cache(tmp_path).load()
    assert got.state == "ok" and list(got.ids) == sorted(ids(2, 20))
    assert len(shard_files(tmp_path)) == 1  # the old shard was an orphan and was removed


# ------------------------------------------------------------ hostile manifests
def hostile_manifest(tmp_path: Path, text: bytes | str) -> ec.LoadResult:
    fill(tmp_path)
    path = cache_dir(tmp_path) / "manifest.json"
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return open_cache(tmp_path).load()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "null",
        "{",
        '{"format": 2}',
        "[" * 10_000 + "]" * 10_000,  # nesting bomb
        '{"a":' * 5_000 + "1" + "}" * 5_000,
        "\x00" * 100,
        '{"format": 2, "extra": 1}',
    ],
)
def test_malformed_or_hostile_manifests_are_rejected_not_crashed_on(
    tmp_path: Path, text: str
) -> None:
    got = hostile_manifest(tmp_path, text)
    assert got.state == "rejected" and len(got.ids) == 0
    assert set(got.rejected) <= {"manifest_invalid", "manifest_unreadable"}


def test_an_oversized_manifest_is_rejected_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ec, "MAX_MANIFEST_BYTES", 64)
    got = hostile_manifest(tmp_path, json.dumps({"pad": "x" * 500}))
    assert got.rejected == {"manifest_unreadable": 1}


def test_a_symlinked_manifest_is_refused(tmp_path: Path) -> None:
    fill(tmp_path)
    real = cache_dir(tmp_path) / "manifest.json"
    moved = tmp_path / "elsewhere.json"
    moved.write_text(real.read_text())
    real.unlink()
    real.symlink_to(moved)
    got = open_cache(tmp_path).load()
    assert got.state == "rejected" and len(got.ids) == 0


def test_a_symlinked_cache_directory_is_refused_and_never_written_through(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    (tmp_path / "embeddings").symlink_to(target)
    c = open_cache(tmp_path)
    assert c.load().rejected == {"not_a_directory": 1}
    with pytest.raises(OSError):
        c.append(ids(1), unit(1), dg(1))
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "name", ["../x.parquet", "a/b.parquet", "shard-1.parquet", "..", "/etc/passwd"]
)
def test_shard_names_in_the_manifest_are_validated(tmp_path: Path, name: str) -> None:
    fill(tmp_path)
    doc = manifest(tmp_path)
    doc["shards"][0]["name"] = name  # type: ignore[index]
    write_manifest(tmp_path, doc)
    got = open_cache(tmp_path).load()
    assert got.rejected == {"manifest_invalid": 1} and len(got.ids) == 0


def test_a_shard_that_is_a_symlink_or_a_directory_is_rejected(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(3), unit(3), dg(3))
    c.append(ids(3, 3), unit(3, 1), dg(3, 3))
    a, b = shard_files(tmp_path)
    data = a.read_bytes()
    a.unlink()
    outside = tmp_path / "copy.parquet"
    outside.write_bytes(data)
    a.symlink_to(outside)
    b.unlink()
    b.mkdir()
    got = open_cache(tmp_path).load()
    assert len(got.ids) == 0
    assert sum(got.rejected.values()) == 2 and set(got.rejected) <= {
        "shard_unreadable",
        "shard_missing",
    }


# ------------------------------------------------------------ hostile shards (digest-valid)
def plant(
    tmp_path: Path, data: bytes, rows: int = 2, then: Callable[[], None] | None = None
) -> ec.LoadResult:
    """Install ``data`` as the only shard with a correct digest *and* a correct MAC.

    Models a shard produced by a writer that holds the key (or a bug): the content checks, not the
    digest, must still refuse it.
    """
    fill(tmp_path, 2)
    doc = manifest(tmp_path)
    entry = doc["shards"][0]  # type: ignore[index]
    (cache_dir(tmp_path) / entry["name"]).write_bytes(data)
    entry.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=rows)
    write_manifest(tmp_path, doc)
    if then is not None:
        then()
    return open_cache(tmp_path).load()


def parquet_bytes(table: pa.Table, **meta: str) -> bytes:
    schema_meta = {
        b"dcp_model_id": b"m",
        b"dcp_revision": b"1",
        b"dcp_dims": str(DIMS).encode(),
        b"dcp_dtype": b"float32",
        b"dcp_dataset_version": VERSION.encode(),
    }
    schema_meta.update({k.encode(): v.encode() for k, v in meta.items()})
    buf = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(schema_meta), buf, compression="zstd")
    return buf.getvalue()


def table_of(
    id_list: list[str | None],
    matrix: np.ndarray,
    dtype: object = pa.float32(),  # type: ignore[type-arg]
) -> pa.Table:
    n, d = matrix.shape
    flat = pa.array(matrix.reshape(-1).astype(np.float32), type=pa.float32()).cast(dtype)  # type: ignore[arg-type]
    vec = pa.ListArray.from_arrays(pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32)), flat)
    return pa.table(
        {
            "image_id": pa.array(id_list, type=pa.string()),
            "vector": vec,
            "content_sha256": pa.array(dg(n), type=pa.string()),
        }
    )


def test_the_planting_helper_accepts_a_valid_shard(tmp_path: Path) -> None:
    good = parquet_bytes(table_of(ids(2), unit(2)))
    got = plant(tmp_path, good)
    assert got.rejected == {} and len(got.ids) == 2


BAD_SHARDS: list[tuple[str, Callable[[], bytes], str]] = [
    ("not parquet", lambda: b"PAR1" + b"\x00" * 64, "bad_parquet"),
    ("garbage", lambda: os.urandom(300), "bad_parquet"),
    (
        "nan vector",
        lambda: parquet_bytes(table_of(ids(2), np.full((2, DIMS), np.nan, dtype=np.float32))),
        "bad_values",
    ),
    (
        "inf vector",
        lambda: parquet_bytes(table_of(ids(2), np.full((2, DIMS), np.inf, dtype=np.float32))),
        "bad_values",
    ),
    (
        "unnormalised vector",
        lambda: parquet_bytes(table_of(ids(2), unit(2) * 5.0)),
        "bad_values",
    ),
    (
        "zero vector",
        lambda: parquet_bytes(table_of(ids(2), np.zeros((2, DIMS), dtype=np.float32))),
        "bad_values",
    ),
    ("wrong dims", lambda: parquet_bytes(table_of(ids(2), unit(2, dims=DIMS * 2))), "bad_shape"),
    (
        "wrong declared dims in metadata",
        lambda: parquet_bytes(table_of(ids(2), unit(2)), dcp_dims="16"),
        "bad_shape",
    ),
    (
        "wrong declared model",
        lambda: parquet_bytes(table_of(ids(2), unit(2)), dcp_model_id="other"),
        "bad_shape",
    ),
    (
        "wrong declared revision",
        lambda: parquet_bytes(table_of(ids(2), unit(2)), dcp_revision="2"),
        "bad_shape",
    ),
    (
        "wrong declared dtype",
        lambda: parquet_bytes(table_of(ids(2), unit(2)), dcp_dtype="float16"),
        "bad_shape",
    ),
    (
        "wrong declared dataset version",
        lambda: parquet_bytes(table_of(ids(2), unit(2)), dcp_dataset_version=OTHER_VERSION),
        "bad_shape",
    ),
    (
        "wrong dtype",
        lambda: parquet_bytes(table_of(ids(2), unit(2), pa.float64())),
        "bad_shape",
    ),
    (
        "float16 in a float32 cache",
        lambda: parquet_bytes(table_of(ids(2), unit(2), pa.float16())),
        "bad_shape",
    ),
    (
        "null id",
        lambda: parquet_bytes(table_of([ids(1)[0], None], unit(2))),
        "bad_values",
    ),
    (
        "non-hex id",
        lambda: parquet_bytes(table_of(["../../etc/passwd", "x" * 16], unit(2))),
        "bad_values",
    ),
    (
        "duplicate ids in one shard",
        lambda: parquet_bytes(table_of(["0" * 16, "0" * 16], unit(2))),
        "bad_values",
    ),
    (
        "malformed content digest",
        lambda: parquet_bytes(
            table_of(ids(2), unit(2)).set_column(2, "content_sha256", pa.array(["zz", "yy"]))
        ),
        "bad_values",
    ),
    (
        "null content digest",
        lambda: parquet_bytes(
            table_of(ids(2), unit(2)).set_column(
                2, "content_sha256", pa.array([dg(1)[0], None], type=pa.string())
            )
        ),
        "bad_values",
    ),
    (
        "missing content digest column",
        lambda: parquet_bytes(table_of(ids(2), unit(2)).drop_columns(["content_sha256"])),
        "bad_shape",
    ),
    (
        "extra column",
        lambda: parquet_bytes(table_of(ids(2), unit(2)).append_column("x", pa.array([1, 2]))),
        "bad_shape",
    ),
    (
        "ragged list",
        lambda: parquet_bytes(
            pa.table(
                {
                    "image_id": pa.array(ids(2)),
                    "vector": pa.array(
                        [[0.1] * DIMS, [0.1] * (DIMS - 1)], type=pa.list_(pa.float32())
                    ),
                    "content_sha256": pa.array(dg(2)),
                }
            )
        ),
        "bad_shape",
    ),
    (
        "null vector",
        lambda: parquet_bytes(
            pa.table(
                {
                    "image_id": pa.array(ids(2)),
                    "vector": pa.array([None, [0.1] * DIMS], type=pa.list_(pa.float32())),
                    "content_sha256": pa.array(dg(2)),
                }
            )
        ),
        "bad_shape",
    ),
]


@pytest.mark.parametrize(("label", "make", "reason"), BAD_SHARDS, ids=[b[0] for b in BAD_SHARDS])
def test_content_checks_reject_a_shard_even_when_its_digest_is_correct(
    tmp_path: Path, label: str, make: Callable[[], bytes], reason: str
) -> None:
    got = plant(tmp_path, make())
    assert len(got.ids) == 0, label
    assert got.rejected == {reason: 1}, (label, got.rejected)


def test_declared_rows_must_match_the_parquet_footer(tmp_path: Path) -> None:
    good = parquet_bytes(table_of(ids(2), unit(2)))
    got = plant(tmp_path, good, rows=3)
    assert got.rejected == {"bad_shape": 1} and len(got.ids) == 0


def test_row_and_decoded_size_limits_apply_before_any_row_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = parquet_bytes(table_of(ids(2), unit(2)))
    # the manifest model also validates ``rows``, so a shard that claims more is refused there
    got = plant(tmp_path, good, then=lambda: monkeypatch.setattr(ec, "MAX_SHARD_ROWS", 1))
    assert len(got.ids) == 0 and set(got.rejected) <= {"manifest_invalid", "too_many_rows"}


def test_decoded_size_is_bounded_from_the_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = parquet_bytes(table_of(ids(2), unit(2)))
    got = plant(tmp_path, good, then=lambda: monkeypatch.setattr(ec, "MAX_SHARD_DECODED_BYTES", 8))
    assert got.rejected == {"too_many_rows": 1}


def test_a_shard_over_the_byte_cap_is_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill(tmp_path)
    monkeypatch.setattr(ec, "MAX_SHARD_BYTES", 16)
    got = open_cache(tmp_path).load()
    assert len(got.ids) == 0 and sum(got.rejected.values()) >= 1


def test_the_same_id_in_two_shards_is_refused_not_picked_arbitrarily(tmp_path: Path) -> None:
    fill(tmp_path, 3)
    doc = manifest(tmp_path)
    second = ec.encode_shard(ident(), ids(3, 2), unit(3, 1), dg(3, 2))  # id 2 occurs in both shards
    name = "shard-" + "f" * 32 + ".parquet"
    (cache_dir(tmp_path) / name).write_bytes(second)
    doc["shards"].append(  # type: ignore[attr-defined]
        dict(name=name, rows=3, bytes=len(second), sha256=hashlib.sha256(second).hexdigest())
    )
    write_manifest(tmp_path, doc)  # a writer that holds the key still must not duplicate ids
    got = open_cache(tmp_path).load()
    assert got.rejected == {"duplicate_ids": 1} and len(got.ids) == 3


# ------------------------------------------------------------ writer contract
def test_append_rejects_duplicates_bad_ids_and_wrong_shapes(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    with pytest.raises(ValueError):
        c.append(["0" * 16, "0" * 16], unit(2), dg(2))
    with pytest.raises(ValueError):
        c.append(["zz"], unit(1), dg(1))
    with pytest.raises(ValueError):
        c.append(ids(2), unit(3), dg(2))
    with pytest.raises(ValueError):
        c.append(ids(1), unit(1, dims=DIMS + 1), dg(1))
    with pytest.raises(ValueError):
        c.append([], np.zeros((0, DIMS), dtype=np.float32), [])
    assert not cache_dir(tmp_path).exists() or shard_files(tmp_path) == []


def test_append_refuses_vectors_that_are_not_canonical(tmp_path: Path) -> None:
    c = open_cache(tmp_path, ident(dtype="float16"))
    c.load()
    raw = np.random.default_rng(0).standard_normal((2, DIMS)).astype(np.float32)
    with pytest.raises(ValueError, match="canonical"):
        c.append(
            ids(2), raw / np.linalg.norm(raw, axis=1, keepdims=True), dg(2)
        )  # not f16-representable
    with pytest.raises(ValueError):
        c.append(ids(2), unit(2, dtype="float16") * 3.0, dg(2))  # not unit norm
    with pytest.raises(ValueError):
        c.append(ids(1), unit(1).astype(np.float64), dg(1))  # wrong dtype
    assert not cache_dir(tmp_path).exists() or shard_files(tmp_path) == []


def test_append_refuses_ids_already_in_the_cache(tmp_path: Path) -> None:
    c = fill(tmp_path, 3)
    with pytest.raises(ValueError):
        c.append(ids(1), unit(1), dg(1))


def test_the_total_row_cap_stops_the_cache_with_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ec, "MAX_CACHE_ROWS", 5)
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(4), unit(4), dg(4))
    with pytest.raises(ec.CacheFullError):
        c.append(ids(2, 4), unit(2, 1), dg(2, 4))
    assert len(open_cache(tmp_path).load().ids) == 4


def test_the_shard_count_cap_stops_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ec, "MAX_SHARDS", 2)
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(1), unit(1), dg(1))
    c.append(ids(1, 1), unit(1, 1), dg(1, 1))
    with pytest.raises(ec.CacheFullError):
        c.append(ids(1, 2), unit(1, 2), dg(1, 2))


def test_float16_shards_are_smaller_than_float32(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for root, dtype in ((a, "float32"), (b, "float16")):
        root.mkdir()
        c = ec.EmbeddingCache.open(root / "embeddings", ident(dtype=dtype), SECRET)
        c.load()
        c.append(ids(500), unit(500, seed=1, dtype=dtype), dg(500))
    size = lambda root: sum(p.stat().st_size for p in (root / "embeddings" / "m@1").glob("shard-*"))  # noqa: E731
    assert size(b) < size(a)


def test_identity_validates_its_fields() -> None:
    for bad in (
        dict(model_id="../x"),
        dict(model_id=""),
        dict(revision="a/b"),
        dict(dims=0),
        dict(dims=ec.MAX_DIMS + 1),
        dict(dtype="float64"),
        dict(dataset_version="xyz"),
    ):
        with pytest.raises(ValueError):
            ident(**bad)


def test_reset_drops_the_old_shards_on_the_next_write(tmp_path: Path) -> None:
    c = fill(tmp_path, 4)
    c.reset()
    c.append(ids(2, 50), unit(2, 4), dg(2, 50))
    got = open_cache(tmp_path).load()
    assert list(got.ids) == sorted(ids(2, 50)) and len(shard_files(tmp_path)) == 1


def test_digests_come_back_with_their_vectors_in_id_order(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(3, 5), unit(3), dg(3, 100))
    c.append(ids(3), unit(3, 1), dg(3, 200))
    got = open_cache(tmp_path).load()
    assert list(got.ids) == sorted(ids(3) + ids(3, 5))
    assert list(got.digests) == dg(3, 200) + dg(3, 100)


def test_append_requires_one_well_formed_digest_per_row(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    for bad in ([], ["zz", "yy"], dg(3), ["0" * 63 + "g", "0" * 64]):
        with pytest.raises(ValueError, match="digests"):
            c.append(ids(2), unit(2), bad)


def test_a_manifest_with_a_duplicated_key_is_rejected_before_the_mac_is_checked(
    tmp_path: Path,
) -> None:
    fill(tmp_path)
    path = cache_dir(tmp_path) / "manifest.json"
    text = path.read_text()
    path.write_text(text.replace('"format": 2', '"format": 2, "format": 2', 1))
    got = open_cache(tmp_path).load()
    assert got.state == "rejected" and got.rejected == {"manifest_invalid": 1}
    path.write_text(text.replace('"shards": [', '"shards": [], "shards": [', 1))
    assert open_cache(tmp_path).load().rejected == {"manifest_invalid": 1}


def test_the_mac_key_is_separated_from_the_secrets_other_uses(tmp_path: Path) -> None:
    import hashlib as h

    identity = ident()
    plain = h.sha256(SECRET + b"\x00" + identity.binding()).digest()  # the old derivation
    assert ec._mac_key(SECRET, identity) != plain
    assert ec._mac_key(SECRET, identity) != ec._mac_key(SECRET, ident(preprocess=2))


def test_a_cache_for_another_preprocessing_version_is_stale(tmp_path: Path) -> None:
    fill(tmp_path)
    got = open_cache(tmp_path, ident(preprocess=2)).load()
    assert got.state == "stale" and len(got.ids) == 0
    with pytest.raises(ValueError):
        ident(preprocess=0)


# ------------------------------------------------------------ N-relative bounds, wanted ids
def test_only_the_wanted_ids_are_returned_but_the_whole_cache_stays_known(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(6), unit(6), dg(6))
    c2 = open_cache(tmp_path)
    got = c2.load(wanted=frozenset(ids(6)[1:3]))
    assert list(got.ids) == ids(6)[1:3] and got.matrix.shape == (2, DIMS)
    assert list(got.digests) == dg(6)[1:3]
    with pytest.raises(ValueError, match="already cached"):  # unwanted ids are still cached ids
        c2.append(ids(1, 5), unit(1), dg(1))


def test_the_row_cap_follows_the_dataset_not_a_global_constant(tmp_path: Path) -> None:
    c = open_cache(tmp_path)
    c.load()
    c.append(ids(20), unit(20), dg(20))
    small = ec.EmbeddingCache.open(tmp_path / "embeddings", ident(), SECRET, max_rows=10)
    got = small.load()
    assert got.rejected == {"too_many_rows": 1} and len(got.ids) == 0
    with pytest.raises(ec.CacheFullError):
        ec.EmbeddingCache.open(tmp_path / "x", ident(), SECRET, max_rows=3).append(
            ids(4), unit(4), dg(4)
        )


def test_a_cache_in_an_older_format_is_stale_not_rejected_and_is_replaced(tmp_path: Path) -> None:
    fill(tmp_path)
    path = cache_dir(tmp_path) / "manifest.json"
    legacy = json.loads(path.read_text())
    legacy["format"] = 1
    del legacy["preprocess"]  # the shape of the previous format
    path.write_text(json.dumps(legacy))
    c = open_cache(tmp_path)
    got = c.load()
    assert got.state == "stale" and got.rejected == {} and len(got.ids) == 0
    c.append(ids(2, 30), unit(2), dg(2))  # the next write replaces it
    again = open_cache(tmp_path).load()
    assert again.state == "ok" and list(again.ids) == ids(2, 30)
    assert len(shard_files(tmp_path)) == 1


# ------------------------------------------------------------ variants (box crops)
def test_a_variant_is_a_separate_authenticated_cache_beside_the_image_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "embeddings"
    plain = ec.EmbeddingCache.open(root, ident(), SECRET)
    plain.append(ids(2), unit(2), dg(2))
    crops = ec.EmbeddingCache.open(root, ident(variant="crop-v1-p100-m32"), SECRET)
    assert crops.load().state == "absent"
    crops.append(ids(3, 10), unit(3, seed=1), dg(3, 10))
    assert {p.name for p in root.iterdir()} == {"m@1", "m@1+crop-v1-p100-m32"}
    again = ec.EmbeddingCache.open(root, ident(variant="crop-v1-p100-m32"), SECRET).load()
    assert again.state == "ok" and again.ids == tuple(ids(3, 10))
    assert ec.EmbeddingCache.open(root, ident(), SECRET).load().ids == tuple(ids(2))


def test_another_variant_or_a_renamed_directory_is_never_trusted(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    ec.EmbeddingCache.open(root, ident(variant="crop-v1-p100-m32"), SECRET).append(
        ids(2), unit(2), dg(2)
    )
    (root / "m@1+crop-v1-p100-m32").rename(root / "m@1+crop-v1-p200-m32")
    other = ec.EmbeddingCache.open(root, ident(variant="crop-v1-p200-m32"), SECRET).load()
    assert other.ids == () and other.state in ("stale", "rejected")
    (root / "m@1+crop-v1-p200-m32").rename(root / "m@1")
    assert ec.EmbeddingCache.open(root, ident(), SECRET).load().ids == ()


@pytest.mark.parametrize("bad", ["Crop", "crop/../x", "a" * 65, "-x"])
def test_a_variant_is_a_plain_name(bad: str) -> None:
    with pytest.raises(ValueError, match="variant"):
        ident(variant=bad)
