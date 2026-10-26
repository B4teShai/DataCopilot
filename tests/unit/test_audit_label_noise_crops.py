"""label_noise with box-crop embeddings: one vector per box, cached by crop geometry, and a reported
fallback to the whole-image rule when crops cannot be cut.

The scenes are photos with coloured boxes; ``ColourEmbedder`` maps a crop to the mean colour of
its centre, so boxes of one colour are neighbours and a box labelled with the other colour's class
is the planted mistake. No weights, no network.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.auditdata import Img, build
from PIL import Image as PILImage
from pydantic import SecretStr

from datacopilot.audit import crops, embedding
from datacopilot.audit.checks import label_noise
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.core.errors import ModelError
from datacopilot.core.schema import Dataset
from datacopilot.embed import imaging

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
RED, BLUE = (210, 40, 40), (40, 40, 210)


class ColourEmbedder:
    """Mean colour of the centre half of the picture, centred on mid-grey."""

    model_id, revision, dims = "colour", "1", 3
    weights: dict[str, str] = {}  # noqa: RUF012

    def __init__(self, shift: float = 0.0) -> None:
        self.calls = 0
        self.shift = shift

    def embed(self, images: list[bytes]) -> np.ndarray:  # type: ignore[type-arg]
        self.calls += 1
        out = []
        for data in images:
            rgb = np.asarray(PILImage.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
            h, w = rgb.shape[:2]
            centre = rgb[h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)]
            vec = centre.reshape(-1, 3).mean(axis=0) / 255.0 - 0.5
            out.append(vec + np.array([self.shift, -self.shift, 0.0], dtype=np.float32))
        return np.stack(out).astype(np.float32)


def photo(boxes: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]], seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(110, 146, size=(96, 128, 3), dtype=np.uint8)  # grey noise background
    for colour, (x1, y1, x2, y2) in boxes:
        jitter = rng.integers(-6, 7, size=3)
        pixels[y1:y2, x1:x2] = np.clip(np.array(colour) + jitter, 0, 255)
    out = io.BytesIO()
    PILImage.fromarray(pixels, "RGB").save(out, format="PNG")
    return out.getvalue()


def specs(n: int = 20, flip: int | None = 3) -> list[Img]:
    """``n`` photos, each with a red box (class 0) and a blue box (class 1); photo ``flip`` has its
    red box labelled class 1."""
    out = []
    for k in range(n):
        red, blue = (8 + k % 5, 10, 48 + k % 5, 50), (70, 30 + k % 4, 118, 80 + k % 4)
        data = photo([(RED, red), (BLUE, blue)], seed=k)
        red_class = 1 if k == flip else 0
        out.append(
            Img(
                f"p{k:02d}.png",
                width=128,
                height=96,
                data=data,
                boxes=[(red_class, *map(float, red)), (1, *map(float, blue))],
            )  # fmt: skip
        )
    return out


def context(root: Path, emb: Any = None, **config: Any) -> CheckContext:
    (root / "ds.dcp").mkdir(exist_ok=True)
    return CheckContext(
        run_id=RUN,
        root=root,
        cache_dir=root / "cache",
        dataset_dir=root / "ds.dcp",
        workers=2,
        embedder="colour",
        embeddings=embedding.PipelineProvider(emb or ColourEmbedder(), batch_size=7),
        cache_key=SecretStr("k" * 32),
        config=CheckConfig(**{"suspect_min_images": 10, "suspect_k": 6, **config}),
    )


def of(findings: list[Any], rule: str) -> list[Any]:
    return [f for f in findings if f.payload["rule"] == rule]


def test_a_box_labelled_with_the_other_colour_s_class_is_the_one_crop_suspect(
    tmp_path: Path,
) -> None:
    ds = build(tmp_path, specs())
    found = label_noise.plugin.run(ds, context(tmp_path))
    (f,) = of(found, "suspect_class")
    assert f.payload["basis"] == "crop" and f.payload["file_name"] == "p03.png"
    assert f.payload["category"] == "dog" and f.payload["support"] == 0.0
    assert f.ann_id == "3-0" and f.payload["neighbours"] == 6
    assert f.payload["neighbour_classes"] == [{"category_id": 1, "category": "cat", "boxes": 6}]
    assert of(found, "crop_rule_unavailable") == [] and of(found, "annotations_not_scored") == []


def test_the_whole_image_rule_cannot_see_the_same_mistake(tmp_path: Path) -> None:
    ds = build(tmp_path, specs())
    found = label_noise.plugin.run(ds, context(tmp_path, suspect_basis="image"))
    assert of(found, "suspect_class") == []  # every photo holds both classes: no lead


def test_crops_are_cached_by_geometry_and_reused(tmp_path: Path) -> None:
    ds = build(tmp_path, specs())
    first = label_noise.plugin.run(ds, context(tmp_path))
    names = sorted(p.name for p in (tmp_path / "ds.dcp" / "embeddings").iterdir())
    assert names == ["colour@1+crop-v1-p100-m32-c50-s512"]
    emb = ColourEmbedder()
    again = label_noise.plugin.run(ds, context(tmp_path, emb))
    assert again == first
    provider_hits = emb.calls  # only the spot check re-embeds a sample
    assert provider_hits <= 2
    label_noise.plugin.run(ds, context(tmp_path, crop_padding=0.2))
    names = sorted(p.name for p in (tmp_path / "ds.dcp" / "embeddings").iterdir())
    assert "colour@1+crop-v1-p200-m32-c50-s512" in names


def test_a_cache_the_model_no_longer_agrees_with_is_discarded_and_reported(tmp_path: Path) -> None:
    ds = build(tmp_path, specs())
    label_noise.plugin.run(ds, context(tmp_path))
    found = label_noise.plugin.run(ds, context(tmp_path, ColourEmbedder(shift=0.3)))
    (m,) = of(found, "embedding_cache_mismatch")
    assert m.severity == "WARN" and m.payload["mismatched"] >= 1


def test_small_and_capped_boxes_are_counted_not_dropped(tmp_path: Path) -> None:
    items = specs()
    # an 8-pixel box, and a third large box that is the smallest of three: over a cap of two
    items[0].boxes = [*items[0].boxes, (2, 1.0, 1.0, 9.0, 9.0), (2, 2.0, 55.0, 36.0, 89.0)]
    ds = build(tmp_path, items)
    found = label_noise.plugin.run(ds, context(tmp_path, max_crops_per_image=2))
    (n,) = of(found, "annotations_not_scored")
    assert n.payload["reasons"] == {"over_image_cap": 1, "too_small": 1}
    assert n.payload["count"] == 2 and n.payload["total"] == 42
    assert n.severity == "WARN"


def test_an_image_whose_pixels_do_not_match_its_row_size_is_not_cropped(tmp_path: Path) -> None:
    items = specs()
    items[5].declared = (130, 96)  # the row claims a size the file does not have
    ds = build(tmp_path, items, validate=False)
    found = label_noise.plugin.run(ds, context(tmp_path))
    (n,) = of(found, "annotations_not_scored")
    assert n.payload["reasons"] == {"size_mismatch": 2}


def test_without_pillow_the_image_rule_runs_and_the_fallback_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, specs())

    def no_pillow() -> Any:
        raise ModelError("image decoding needs Pillow")

    monkeypatch.setattr(imaging, "_pil", no_pillow)
    found = label_noise.plugin.run(ds, context(tmp_path))
    (fb,) = of(found, "crop_rule_unavailable")
    assert fb.severity == "WARN" and fb.payload == {
        "rule": "crop_rule_unavailable",
        "reason": "pillow_missing",
        "basis": "image",
    }


def test_a_provider_without_crop_support_falls_back_and_says_so(tmp_path: Path) -> None:
    ds = build(tmp_path, specs())
    real = embedding.PipelineProvider(ColourEmbedder())

    class ImagesOnly:
        def get(self, d: Dataset, c: CheckContext) -> Any:
            return real.get(d, c)

    ctx = context(tmp_path).model_copy(update={"embeddings": ImagesOnly()})
    found = label_noise.plugin.run(ds, ctx)
    assert of(found, "crop_rule_unavailable")[0].payload["reason"] == "provider_without_crops"


def test_an_embedder_error_is_critical_and_skips(tmp_path: Path) -> None:
    ds = build(tmp_path, specs())

    class Broken(ColourEmbedder):
        def embed(self, images: list[bytes]) -> np.ndarray:  # type: ignore[type-arg]
            raise RuntimeError("token=hf_SECRET")

    found = label_noise.plugin.run(ds, context(tmp_path, Broken()))
    assert of(found, "embedder_error")[0].severity == "CRITICAL"
    assert of(found, "skipped")[0].payload["reason"] == "no_embeddings"
    assert of(found, "annotations_not_scored")[0].payload["reasons"] == {"embedder_error": 40}
    assert "hf_SECRET" not in repr(found)


def test_too_few_crops_is_a_skip_with_the_reason(tmp_path: Path) -> None:
    ds = build(tmp_path, specs(4))
    (skip,) = label_noise.plugin.run(ds, context(tmp_path))
    assert skip.payload == {
        "rule": "skipped",
        "reason": "too_few_boxes",
        "boxes": 8,
        "min_boxes": 10,
        "not_cropped": {},
    }


def test_findings_do_not_depend_on_row_order(tmp_path: Path) -> None:
    ds = build(tmp_path, specs())
    shuffled = list(ds.images)
    random.Random(3).shuffle(shuffled)
    anns = list(ds.annotations)
    random.Random(4).shuffle(anns)
    other = Dataset(ds.manifest, tuple(shuffled), tuple(anns), ds.root)
    assert label_noise.plugin.run(other, context(tmp_path)) == label_noise.plugin.run(
        ds, context(tmp_path)
    )


def test_crop_rectangles_are_padded_clamped_and_rounded_outwards() -> None:
    assert crops.crop_rect((10.0, 10.0, 30.0, 20.0), 100, 100, 0.1) == (8, 9, 32, 21)
    assert crops.crop_rect((0.5, 0.5, 99.5, 49.2), 100, 50, 0.5) == (0, 0, 100, 50)
    plan = crops.CropPlan(min_side=32, padding=0.1, per_image=50)
    assert plan.variant == "crop-v1-p100-m32-c50-s512"


def test_a_large_crop_is_downscaled_before_it_is_encoded(tmp_path: Path) -> None:
    rgb = np.zeros((1200, 2000, 3), dtype=np.uint8)
    png = crops._encode(rgb, (0, 0, 2000, 1200))
    assert PILImage.open(io.BytesIO(png)).size == (512, 307)
