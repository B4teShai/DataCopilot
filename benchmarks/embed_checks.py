"""Benchmark of the embedding checks on synthetic embeddings (see 01-overview.md).

R50k embeddings: 50,000 vectors of 768 dimensions (the width of DINOv2-B and SigLIP2-B), 200
classes, 80/10/10 split, with planted near-duplicate pairs (half of them across splits). No image
files and no model are involved: the benchmark measures what happens *after* embedding, which is
what a CI run with a precomputed cache pays (plus the sample re-embedding, which needs the images
and the model and is not measured here). Phases: exact kNN, cache write and verified read, and the
three checks.

Usage::

    uv run python benchmarks/embed_checks.py                 # full 50,000 x 768
    uv run python benchmarks/embed_checks.py --images 5000   # smoke test

Prints one JSON document; ``peak_rss_gb`` covers the whole process.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from datacopilot.audit.checks import leakage, near_duplicates, outliers
from datacopilot.audit.context import CheckContext
from datacopilot.audit.embedding import EmbeddingSet
from datacopilot.core.schema import (
    SCHEMA_VERSION,
    Annotation,
    Category,
    Dataset,
    DatasetManifest,
    Image,
)
from datacopilot.core.version import models_version
from datacopilot.embed import cache, knn

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"


def build(n: int, dims: int, classes: int, pairs: int) -> tuple[Dataset, np.ndarray]:  # type: ignore[type-arg]
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((classes, dims)).astype(np.float32)
    labels = rng.integers(0, classes, n)
    vectors = centres[labels] * 0.6 + rng.standard_normal((n, dims)).astype(np.float32)
    splits = np.where(np.arange(n) % 10 < 8, 0, np.where(np.arange(n) % 10 == 8, 1, 2))
    # plant near-duplicate pairs: row 2k+1 becomes a jittered copy of row 2k
    for k in range(pairs):
        a, b = 2 * k, 2 * k + 1
        if k % 2 == 0:
            splits[b] = (splits[a] + 1) % 3  # a cross-split pair
        vectors[b] = vectors[a] + 0.05 * rng.standard_normal(dims).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    names = ("train", "val", "test")
    images = [
        Image(
            image_id=f"{i:016x}",
            file_path=f"images/img{i:06d}.jpg",
            width=1920,
            height=1080,
            split=names[int(splits[i])],  # type: ignore[arg-type]
        )
        for i in range(n)
    ]
    annotations = [
        Annotation(
            ann_id=f"a{i}",
            image_id=images[i].image_id,
            category_id=int(labels[i]) + 1,
            kind="classification",
        )
        for i in range(n)
    ]
    cats = [Category(id=c + 1, name=f"class{c}") for c in range(classes)]
    manifest = DatasetManifest(
        version=models_version(images, annotations, cats, schema_version=SCHEMA_VERSION),
        schema_version=SCHEMA_VERSION,
        name="bench",
        task="classification",
        categories=tuple(cats),
        image_count=n,
        annotation_count=n,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return Dataset(manifest, tuple(images), tuple(annotations)), vectors


class Stub:
    def __init__(self, es: EmbeddingSet) -> None:
        self.es = es

    def get(self, ds: Dataset, ctx: CheckContext) -> EmbeddingSet:
        return self.es


def timed(fn):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    result = fn()
    return result, round(time.perf_counter() - start, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=50_000)
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--classes", type=int, default=200)
    args = ap.parse_args()
    out: dict[str, object] = {
        "images": args.images,
        "dims": args.dims,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": __import__("os").cpu_count(),
    }
    ds, raw = build(args.images, args.dims, args.classes, pairs=min(400, args.images // 4))
    canonical = cache.canonical_vectors(raw, "float16")
    ids = tuple(sorted(i.image_id for i in ds.images))
    (_, out["knn_k10_s"]) = timed(lambda: knn.knn(canonical, canonical, 10, same_set=True))
    with tempfile.TemporaryDirectory() as tmp:
        ident = cache.CacheIdentity("bench", "1", args.dims, "float16", ds.manifest.version, {})
        root = Path(tmp) / "embeddings"
        root.mkdir()
        store = cache.EmbeddingCache.open(root, ident, b"k" * 32)
        store.load()

        def write() -> None:
            for start in range(0, len(ids), cache.WRITE_SHARD_ROWS):
                store.append(
                    list(ids[start : start + cache.WRITE_SHARD_ROWS]),
                    canonical[start : start + cache.WRITE_SHARD_ROWS],
                )

        _, out["cache_write_s"] = timed(write)
        out["cache_mb"] = round(sum(p.stat().st_size for p in root.rglob("shard-*")) / 1e6, 1)
        loaded, out["cache_verified_read_s"] = timed(
            lambda: cache.EmbeddingCache.open(root, ident, b"k" * 32).load()
        )
        if not (loaded.state == "ok" and len(loaded.ids) == len(ids) and not loaded.rejected):
            raise SystemExit("the cache did not verify")
    # the stub holds the vectors in the order of sorted ids; image i has id i, so rows == ids
    es = EmbeddingSet(
        ids=ids, matrix=canonical, total=len(ids), model_id="bench", revision="1", dims=args.dims
    )
    ctx = CheckContext(run_id=RUN, root=Path("."), cache_dir=Path("."), embeddings=Stub(es))
    for name, check in (
        ("near_duplicates", near_duplicates),
        ("leakage", leakage),
        ("outliers", outliers),
    ):
        findings, seconds = timed(lambda c=check: c.plugin.run(ds, ctx))
        out[f"{name}_s"] = seconds
        out[f"{name}_findings"] = len(findings)
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
