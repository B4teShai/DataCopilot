"""The embedding pipeline: cache use, per-image failures, budgets and the fail-closed bookkeeping.

Every way an image can end up without a vector is a named reason in ``EmbeddingSet.missing``; the
checks turn those into gating findings (tests/unit/test_audit_embedding_checks.py).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.auditdata import Img, build, image_id_of
from fixtures.embedfakes import DIMS, FakeEmbedder, random_unit
from fixtures.synth import png_bytes
from pydantic import SecretStr

from datacopilot.audit import embedding
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.core.schema import Dataset
from datacopilot.embed import cache as ec
from datacopilot.embed.protocol import ImageDecodeError

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
KEY = "k" * 32


def make_ctx(root: Path, provider: Any, **kw: Any) -> CheckContext:
    (root / "ds.dcp").mkdir(exist_ok=True)
    base: dict[str, Any] = dict(
        run_id=RUN,
        root=root,
        cache_dir=root / "cache",
        dataset_dir=root / "ds.dcp",
        workers=2,
        embedder="fake",
        embeddings=provider,
        cache_key=SecretStr(KEY),
    )
    base.update(kw)
    return CheckContext(**base)


def dataset(root: Path, n: int = 12, **kw: Any) -> Dataset:
    return build(root, [Img(f"i{k}.png", seed=k, **kw) for k in range(n)])


def provider(emb: FakeEmbedder, **kw: Any) -> embedding.PipelineProvider:
    return embedding.PipelineProvider(emb, **kw)


def identity(ds: Dataset, dtype: str = "float16") -> ec.CacheIdentity:
    return ec.CacheIdentity("fake", "1", DIMS, dtype, ds.manifest.version, {})  # type: ignore[arg-type]


# ------------------------------------------------------------ the happy path
def test_every_image_is_embedded_in_sorted_order_with_canonical_vectors(tmp_path: Path) -> None:
    ds, emb = dataset(tmp_path), FakeEmbedder()
    es = provider(emb, batch_size=5).get(ds, make_ctx(tmp_path, None))
    assert es.ids == tuple(sorted(i.image_id for i in ds.images)) and es.total == 12
    assert es.matrix.shape == (12, DIMS) and es.matrix.dtype == np.float32
    assert es.missing == {} and es.error is None and es.incomplete is None
    assert np.allclose(np.linalg.norm(es.matrix, axis=1), 1.0, atol=1e-3)
    assert emb.calls == [5, 5, 2]  # batches of the requested size


def test_a_fresh_vector_equals_the_cached_one_bit_for_bit(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    first = provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    second = provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    assert second.cache_hits == 12 and second.cache_state == "ok"
    np.testing.assert_array_equal(first.matrix, second.matrix)


def test_a_warm_cache_re_embeds_only_the_spot_check_sample(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    emb = FakeEmbedder()
    es = provider(emb).get(ds, make_ctx(tmp_path, None))
    assert es.spot_checked == embedding.sample_size(12) == 4 and es.spot_mismatched == 0
    assert sum(emb.calls) == 4


def test_the_provider_computes_once_per_dataset(tmp_path: Path) -> None:
    ds, emb = dataset(tmp_path), FakeEmbedder()
    prov = provider(emb)
    ctx = make_ctx(tmp_path, prov)
    assert prov.get(ds, ctx) is prov.get(ds, ctx)
    assert sum(emb.calls) == 12


def test_images_are_only_read_when_they_are_not_cached(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    for path in (tmp_path / "images").iterdir():
        path.unlink()  # a CI run with precomputed embeddings and no images on the runner
    es = provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    assert es.cache_hits == 12 and len(es.ids) == 12  # the sample cannot be re-read: skipped
    assert es.spot_checked == 0


def test_resume_embeds_only_what_is_missing(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    stopper = FakeEmbedder()
    stopper.fail_after = 1
    es = provider(stopper, batch_size=5).get(ds, make_ctx(tmp_path, None))
    assert es.error == "RuntimeError" and len(es.ids) == 5  # the first batch survived
    emb = FakeEmbedder()
    es2 = provider(emb, batch_size=5).get(ds, make_ctx(tmp_path, None))
    assert es2.error is None and len(es2.ids) == 12 and es2.cache_hits == 5
    assert sum(emb.calls) == 7 + es2.spot_checked


# ------------------------------------------------------------ cache trust
def test_a_cache_edited_without_the_key_is_rejected_and_recomputed(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    good = provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    shard = next((tmp_path / "ds.dcp" / "embeddings").rglob("shard-*.parquet"))
    forged = ec.encode_shard(
        identity(ds),
        sorted(i.image_id for i in ds.images),
        ec.canonical_vectors(np.ones((12, DIMS), np.float32), "float16"),
    )
    shard.write_bytes(forged)
    emb = FakeEmbedder()
    es = provider(emb).get(ds, make_ctx(tmp_path, None))
    assert es.cache_state in ("ok", "rejected") and es.cache_rejected  # counted, never silent
    assert es.cache_hits == 0 and sum(emb.calls) == 12
    np.testing.assert_array_equal(es.matrix, good.matrix)


def test_a_poisoned_cache_written_with_the_key_is_caught_by_the_spot_check(
    tmp_path: Path,
) -> None:
    ds = dataset(tmp_path)
    ids = sorted(i.image_id for i in ds.images)
    (tmp_path / "ds.dcp").mkdir()
    cache = ec.EmbeddingCache.open(tmp_path / "ds.dcp" / "embeddings", identity(ds), KEY.encode())
    cache.load()
    poison = ec.canonical_vectors(np.tile(random_unit(b"decoy"), (12, 1)), "float16")
    cache.append(
        ids, poison
    )  # every image claims the same vector: it would "hide" nothing, but lie
    emb = FakeEmbedder()
    es = provider(emb).get(ds, make_ctx(tmp_path, None))
    assert es.spot_mismatched > 0 and es.cache_hits == 0
    honest = provider(FakeEmbedder(), use_cache=False).get(ds, make_ctx(tmp_path, None))
    np.testing.assert_array_equal(es.matrix, honest.matrix)  # the poison never reached the checks
    again = provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    assert again.spot_mismatched == 0 and again.cache_hits == 12  # rewritten with real vectors


def test_in_ci_without_a_key_the_cache_is_neither_read_nor_written(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    ctx = make_ctx(tmp_path, None, in_ci=True, cache_key=None)
    es = provider(FakeEmbedder()).get(ds, ctx)
    assert es.cache_state == "disabled" and es.cache_disabled_reason == "ci_without_key"
    assert not (tmp_path / "ds.dcp" / "embeddings").exists() and not (tmp_path / "cache").exists()
    assert len(es.ids) == 12


def test_in_ci_with_a_key_the_cache_works(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    ctx = make_ctx(tmp_path, None, in_ci=True)
    provider(FakeEmbedder()).get(ds, ctx)
    assert provider(FakeEmbedder()).get(ds, ctx).cache_hits == 12


def test_locally_without_a_key_a_private_secret_beside_the_cache_is_used(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    ctx = make_ctx(tmp_path, None, cache_key=None)
    provider(FakeEmbedder()).get(ds, ctx)
    assert (tmp_path / "cache" / "secret.key").is_file()
    assert provider(FakeEmbedder()).get(ds, ctx).cache_hits == 12


def test_a_cache_directory_other_users_can_write_disables_the_cache(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir(mode=0o777)  # noqa: S103
    os.chmod(tmp_path / "cache", 0o777)  # noqa: S103
    es = provider(FakeEmbedder()).get(dataset(tmp_path), make_ctx(tmp_path, None, cache_key=None))
    assert es.cache_state == "disabled" and es.cache_disabled_reason == "directory_not_private"


def test_a_cache_for_another_dataset_version_is_not_reused(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    provider(FakeEmbedder()).get(ds, make_ctx(tmp_path, None))
    other = build(
        tmp_path, [Img(f"i{k}.png", seed=k) for k in range(12)] + [Img("extra.png", seed=99)]
    )
    es = provider(FakeEmbedder()).get(other, make_ctx(tmp_path, None))
    assert es.cache_state == "stale" and es.cache_hits == 0 and len(es.ids) == 13


def test_use_cache_false_never_touches_the_dataset_directory(tmp_path: Path) -> None:
    es = provider(FakeEmbedder(), use_cache=False).get(dataset(tmp_path), make_ctx(tmp_path, None))
    assert es.cache_state == "disabled" and es.cache_disabled_reason == "disabled_by_option"
    assert not (tmp_path / "ds.dcp" / "embeddings").exists()


def test_without_a_dataset_directory_there_is_no_cache(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, None).model_copy(update={"dataset_dir": None})
    es = provider(FakeEmbedder()).get(dataset(tmp_path), ctx)
    assert es.cache_state == "disabled" and es.cache_disabled_reason == "no_dataset_directory"


def test_a_read_only_dataset_still_embeds_and_says_the_cache_was_not_written(
    tmp_path: Path,
) -> None:
    ds = dataset(tmp_path)
    ctx = make_ctx(tmp_path, None)
    os.chmod(tmp_path / "ds.dcp", 0o555)  # noqa: S103
    try:
        es = provider(FakeEmbedder()).get(ds, ctx)
    finally:
        os.chmod(tmp_path / "ds.dcp", 0o755)  # noqa: S103
    assert len(es.ids) == 12 and es.cache_unwritable is True


def test_the_cache_full_condition_does_not_stop_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ec, "MAX_CACHE_ROWS", 5)
    es = provider(FakeEmbedder(), batch_size=4).get(dataset(tmp_path), make_ctx(tmp_path, None))
    assert len(es.ids) == 12 and es.cache_unwritable is True


# ------------------------------------------------------------ images that cannot be embedded
def test_each_unreadable_class_of_image_has_its_own_reason(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")
    ds = build(
        tmp_path,
        [
            Img("ok.png", seed=1),
            Img("gone.png", seed=2, write=False),
            Img("remote.png", seed=3, write=False, path="s3://bucket/x.png"),
            Img("evil.png", seed=4, write=False, path="../x.png"),
            Img("dev.png", seed=5, write=False, path=str(tmp_path / "pipe")),
            Img("outside.png", seed=6, write=False, path=str(tmp_path.parent / "elsewhere.png")),
            Img("big.png", 16, 16, seed=7),
            Img("stale.png", 16, 16, seed=8),
        ],
    )
    (tmp_path / "images" / "big.png").write_bytes(b"\x00" * 5000)
    (tmp_path / "images" / "stale.png").write_bytes(png_bytes(16, 16, 123))  # edited after ingest
    ctx = make_ctx(tmp_path, None, config=CheckConfig(max_embed_image_bytes=4096))
    es = provider(FakeEmbedder()).get(ds, ctx)
    by_name = {
        r: sorted(
            next(i.file_path.rsplit("/", 1)[-1] for i in ds.images if i.image_id == x) for x in ids
        )
        for r, ids in es.missing.items()
    }
    assert by_name == {  # names are the base names of ``file_path`` (overridden for some rows)
        "file_missing": ["gone.png"],
        "remote": ["x.png"],
        "unsafe_path": ["x.png"],
        "not_a_regular_file": ["pipe"],
        "outside_root": ["elsewhere.png"],
        "file_too_large": ["big.png"],
        "content_changed": ["stale.png"],
    }
    assert len(es.ids) == 1 and es.total == 8


def test_a_changed_file_is_never_embedded_under_its_old_id(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png", seed=1), Img("b.png", seed=2)])
    (tmp_path / "images" / "b.png").write_bytes(png_bytes(64, 48, 200))
    emb = FakeEmbedder()
    es = provider(emb).get(ds, make_ctx(tmp_path, None))
    assert png_bytes(64, 48, 200) not in emb.seen and len(es.ids) == 1
    cached = ec.EmbeddingCache.open(
        tmp_path / "ds.dcp" / "embeddings", identity(ds), KEY.encode()
    ).load()
    assert len(cached.ids) == 1


def test_an_undecodable_image_is_isolated_and_the_rest_of_the_batch_survives(
    tmp_path: Path,
) -> None:
    ds = dataset(tmp_path, 6)
    bad = png_bytes(64, 48, 2)
    emb = FakeEmbedder()
    emb.decode_errors[bad] = "decode_failed"
    es = provider(emb, batch_size=6).get(ds, make_ctx(tmp_path, None))
    assert len(es.ids) == 5 and es.error is None
    assert es.missing == {"decode_failed": (image_id_of(bad),)}


def test_several_bad_images_in_one_batch_are_all_isolated(tmp_path: Path) -> None:
    ds = dataset(tmp_path, 8)
    emb = FakeEmbedder()
    emb.decode_errors[png_bytes(64, 48, 1)] = "decode_failed"
    emb.decode_errors[png_bytes(64, 48, 6)] = "too_many_pixels"
    es = provider(emb, batch_size=8).get(ds, make_ctx(tmp_path, None))
    assert len(es.ids) == 6
    assert set(es.missing) == {"decode_failed", "too_many_pixels"}


def test_an_embedder_exception_stops_the_run_and_marks_the_rest(tmp_path: Path) -> None:
    ds = dataset(tmp_path)
    emb = FakeEmbedder()
    emb.fail_after = 1
    emb.error = ValueError("secret text that must not travel: hf_TOKEN123")
    es = provider(emb, batch_size=4).get(ds, make_ctx(tmp_path, None))
    assert es.error == "ValueError" and "hf_TOKEN123" not in repr(es)
    assert len(es.ids) == 4 and len(es.missing["embedder_error"]) == 8


@pytest.mark.parametrize(
    "mangle",
    [
        lambda v: v[:, :-1],  # wrong width
        lambda v: v[:-1],  # wrong row count
        lambda v: np.full_like(v, np.nan),
        lambda v: np.full_like(v, np.inf),
        lambda v: v.astype(np.float64).astype(np.float64)[..., None],  # wrong rank
    ],
    ids=["width", "rows", "nan", "inf", "rank"],
)
def test_malformed_embedder_output_is_an_embedder_error_not_a_vector(
    tmp_path: Path, mangle: Any
) -> None:
    emb = FakeEmbedder()
    emb.mangle = mangle
    es = provider(emb).get(dataset(tmp_path, 4), make_ctx(tmp_path, None))
    assert es.error == "InvalidEmbedderOutput" and len(es.ids) == 0
    assert len(es.missing["embedder_error"]) == 4


def test_a_zero_vector_marks_only_that_image(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    emb.plant(png_bytes(64, 48, 3), np.zeros(DIMS, dtype=np.float32))
    es = provider(emb).get(dataset(tmp_path, 6), make_ctx(tmp_path, None))
    assert len(es.ids) == 5 and es.error is None
    assert list(es.missing) == ["zero_vector"] and len(es.missing["zero_vector"]) == 1


def test_system_exit_from_an_embedder_is_not_swallowed(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    emb.fail_after = 0
    emb.error = SystemExit(0)
    with pytest.raises(SystemExit):
        provider(emb).get(dataset(tmp_path, 3), make_ctx(tmp_path, None))


def test_keyboard_interrupt_propagates(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    emb.fail_after = 0
    emb.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        provider(emb).get(dataset(tmp_path, 3), make_ctx(tmp_path, None))


# ------------------------------------------------------------ budgets
def test_the_time_budget_ends_the_run_with_an_explicit_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = dataset(tmp_path, 20)
    emb = FakeEmbedder()
    real_embed = emb.embed

    def slow(images: Any) -> Any:
        out = real_embed(images)
        time.sleep(0.06)
        return out

    emb.embed = slow  # type: ignore[method-assign]
    ctx = make_ctx(tmp_path, None, config=CheckConfig(max_check_seconds=0.1), workers=1)
    es = provider(emb, batch_size=2).get(ds, ctx)
    assert es.incomplete is not None and es.incomplete[0] == "time_budget"
    assert 0 < len(es.ids) < 20
    assert es.incomplete[1] == len(es.ids) and es.incomplete[2] == 20
    assert len(es.missing["time_budget"]) == 20 - len(es.ids)


def test_a_batch_never_holds_more_bytes_than_the_batch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embedding, "MAX_BATCH_BYTES", 3 * len(png_bytes(64, 48, 0)) + 10)
    emb = FakeEmbedder()
    provider(emb, batch_size=100).get(dataset(tmp_path, 10), make_ctx(tmp_path, None))
    assert max(emb.calls) <= 3 and sum(emb.calls) == 10


def test_a_file_that_grows_while_it_is_read_is_bounded(tmp_path: Path) -> None:
    ds = dataset(tmp_path, 2)
    victim = tmp_path / "images" / "i1.png"
    victim.write_bytes(victim.read_bytes() + b"\x00" * 10_000)  # now larger than the cap below
    ctx = make_ctx(tmp_path, None, config=CheckConfig(max_embed_image_bytes=5_000))
    es = provider(FakeEmbedder()).get(ds, ctx)
    assert list(es.missing) == ["file_too_large"]


def test_batch_size_is_validated() -> None:
    for bad in (0, -1, embedding.MAX_EMBED_BATCH + 1):
        with pytest.raises(ValueError):
            embedding.PipelineProvider(FakeEmbedder(), batch_size=bad)
    with pytest.raises(ValueError):
        embedding.PipelineProvider(FakeEmbedder(), dtype="float64")  # type: ignore[arg-type]


def test_an_embedder_that_declares_a_bad_width_is_refused_up_front(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    emb.dims = 0
    with pytest.raises(ValueError):
        embedding.PipelineProvider(emb)


def test_preflight_reports_why_the_cache_cannot_be_used(tmp_path: Path) -> None:
    assert provider(FakeEmbedder()).preflight(make_ctx(tmp_path, None)) is None
    assert (
        provider(FakeEmbedder()).preflight(make_ctx(tmp_path, None, in_ci=True, cache_key=None))
        == "ci_without_key"
    )
    assert (
        provider(FakeEmbedder(), use_cache=False).preflight(make_ctx(tmp_path, None))
        == "disabled_by_option"
    )


def test_the_spot_check_sample_is_deterministic_per_run_and_bounded() -> None:
    keys = [f"{i:016x}" for i in range(10_000)]
    a = embedding.pick_sample(keys, RUN)
    assert a == embedding.pick_sample(keys, RUN) and len(a) == embedding.SPOT_MAX
    assert a != embedding.pick_sample(keys, "2222aaaa-2222-4333-8444-5555bbbb5555")
    assert embedding.sample_size(0) == 0 and embedding.sample_size(3) == 3
    assert embedding.sample_size(10_000) == embedding.SPOT_MAX


def test_embeddings_computed_is_logged_without_paths(tmp_path: Path) -> None:
    import structlog

    with structlog.testing.capture_logs() as logs:
        provider(FakeEmbedder()).get(dataset(tmp_path, 4), make_ctx(tmp_path, None))
    (event,) = [e for e in logs if e["event"] == "embeddings_computed"]
    assert event["model_id"] == "fake" and event["image_count"] == 4 and event["cache_hits"] == 0
    assert str(tmp_path) not in repr(logs)


def test_decode_error_type_is_importable_for_embedders() -> None:
    assert ImageDecodeError(0, "decode_failed").reason == "decode_failed"
