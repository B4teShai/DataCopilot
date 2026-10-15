"""The optional perceptual-hash prefilter of ``near_duplicates`` (needs Pillow)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fixtures.auditdata import Img, build
from fixtures.embedfakes import FakeEmbedder, near, random_unit
from pydantic import SecretStr

from datacopilot.audit import embedding
from datacopilot.audit.checks import near_duplicates
from datacopilot.audit.context import CheckConfig, CheckContext

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
BASE = random_unit(b"phash")


def png(seed: int, brighten: int = 0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 200, (48, 64, 3), dtype=np.uint8) + brighten
    out = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


def setup(tmp_path: Path, same_look: bool, max_distance: int | None) -> tuple[list, CheckContext]:  # type: ignore[type-arg]
    a, b = png(1), (png(1, brighten=15) if same_look else png(2))
    others = [png(10 + k) for k in range(6)]
    emb = FakeEmbedder()
    emb.plant(a, BASE)
    emb.plant(b, near(BASE, 0.1))  # the model says they are near-duplicates either way
    ds = build(
        tmp_path,
        [
            Img("a.png", data=a),
            Img("b.png", data=b),
            *[Img(f"o{k}.png", data=o) for k, o in enumerate(others)],
        ],
    )
    (tmp_path / "ds.dcp").mkdir()
    ctx = CheckContext(
        run_id=RUN,
        root=tmp_path,
        cache_dir=tmp_path / "cache",
        dataset_dir=tmp_path / "ds.dcp",
        workers=2,
        embedder="fake",
        embeddings=embedding.PipelineProvider(emb),
        cache_key=SecretStr("k" * 32),
        config=CheckConfig(near_duplicate_phash_max_distance=max_distance),
    )
    return near_duplicates.plugin.run(ds, ctx), ctx  # type: ignore[return-value]


def test_without_the_prefilter_the_model_alone_decides(tmp_path: Path) -> None:
    findings, _ = setup(tmp_path, same_look=False, max_distance=None)
    assert [f.payload["rule"] for f in findings] == ["near_duplicate"]


def test_a_pair_the_hash_confirms_is_kept(tmp_path: Path) -> None:
    findings, _ = setup(tmp_path, same_look=True, max_distance=8)
    assert [f.payload["rule"] for f in findings] == ["near_duplicate"]


def test_a_pair_the_hash_contradicts_is_dropped(tmp_path: Path) -> None:
    findings, _ = setup(tmp_path, same_look=False, max_distance=8)
    assert findings == []


def test_images_that_cannot_be_hashed_keep_their_pairs_and_say_so(tmp_path: Path) -> None:
    findings, ctx = setup(tmp_path, same_look=False, max_distance=8)
    assert findings == []
    (tmp_path / "ds2").mkdir()
    a, b = png(1), png(2)
    emb = FakeEmbedder()
    emb.plant(a, BASE)
    emb.plant(b, near(BASE, 0.1))
    ds = build(
        tmp_path / "ds2", [Img("a.png", data=a), Img("b.png", data=b), Img("c.png", data=png(5))]
    )
    ctx2 = ctx.model_copy(
        update={
            "root": tmp_path / "ds2",
            "embeddings": embedding.PipelineProvider(emb),
            "dataset_dir": None,
        }
    )
    # make the second read of a.png fail after it was embedded: hashing happens after embedding
    real_read = embedding.read_image_bytes
    calls: list[str] = []

    def flaky(image, context):  # type: ignore[no-untyped-def]
        calls.append(image.file_path)
        if len(calls) > 3 and image.file_path.endswith("a.png"):
            return None, "file_missing"
        return real_read(image, context)

    import unittest.mock as mock

    with mock.patch.object(embedding, "read_image_bytes", flaky):
        out = near_duplicates.plugin.run(ds, ctx2)
    rules = [f.payload["rule"] for f in out]
    assert "near_duplicate" in rules and "phash_not_applied" in rules  # kept: cannot refute
    (note,) = [f for f in out if f.payload["rule"] == "phash_not_applied"]
    assert note.severity == "WARN" and note.payload["count"] == 1
