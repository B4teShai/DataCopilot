"""Benchmark of the embedding checks on synthetic embeddings (see 01-overview.md).

R50k embeddings: 50,000 vectors of 768 dimensions (the width of DINOv2-B and SigLIP2-B), 200
classes, 80/10/10 split, with planted near-duplicate pairs (half of them across splits). No image
files and no model are involved: the benchmark measures what happens *after* embedding, which is
what a CI run with a precomputed cache pays (plus the sample re-embedding, which needs the images
and the model and is not measured here). Phases: exact kNN, cache write and verified read, and the
three checks (``label_noise`` is timed on its own).

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
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from datacopilot.audit.checks import label_noise, leakage, near_duplicates, outliers
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
    ap.add_argument(
        "--backend",
        choices=("numpy", "auto"),
        default="numpy",
        help="numpy pins the exact NumPy index for every phase; auto lets FAISS take over above "
        f"{knn.FAISS_MIN_ROWS} rows when it is installed",
    )
    ap.add_argument("--no-assert", action="store_true", help="report only; skip recall and budgets")
    args = ap.parse_args()
    if args.backend == "numpy":
        knn.FAISS_MIN_ROWS = 10**12  # the checks call the index with backend="auto"
    faiss_used = args.backend == "auto" and knn._import_faiss() is not None  # noqa: SLF001
    out: dict[str, object] = {
        "images": args.images,
        "dims": args.dims,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": __import__("os").cpu_count(),
        "platform": sys.platform,
        "knn_backend": "faiss" if faiss_used and args.images >= knn.FAISS_MIN_ROWS else "numpy",
    }
    ds, raw = build(args.images, args.dims, args.classes, pairs=min(400, args.images // 4))
    canonical = cache.canonical_vectors(raw, "float16")
    ids = tuple(sorted(i.image_id for i in ds.images))
    (_, out["knn_k10_s"]) = timed(
        lambda: knn.knn(canonical, canonical, 10, same_set=True, backend=args.backend)
    )
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
                    [
                        f"{n:064x}"
                        for n in range(start, min(start + cache.WRITE_SHARD_ROWS, len(ids)))
                    ],
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
    # reported apart from ``check_total_s``, which keeps the three checks of the documented budget
    findings, out["label_noise_s"] = timed(lambda: label_noise.plugin.run(ds, ctx))
    out["label_noise_findings"] = len(findings)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    out["peak_rss_gb"] = round(peak / (1e9 if sys.platform == "darwin" else 1024**2), 2)
    out["check_total_s"] = round(
        float(out["near_duplicates_s"]) + float(out["leakage_s"]) + float(out["outliers_s"]),  # type: ignore[arg-type]
        2,
    )
    planted = min(400, args.images // 4)
    out["planted_pairs"] = planted
    out["near_duplicate_recall"] = round(
        min(1.0, float(out["near_duplicates_findings"]) / planted), 3
    )  # type: ignore[arg-type]
    out["leakage_recall"] = round(
        min(1.0, float(out["leakage_findings"]) / max(planted // 2, 1)), 3
    )  # type: ignore[arg-type]
    print(json.dumps(out, indent=2))
    if not args.no_assert:
        failures = []
        if out["near_duplicate_recall"] < 0.95:  # type: ignore[operator]
            failures.append("near-duplicate recall below 0.95 of the planted pairs")
        if out["leakage_recall"] < 0.95:  # type: ignore[operator]
            failures.append("leakage recall below 0.95 of the planted cross-split pairs")
        if args.images >= 50_000 and out["check_total_s"] > 600:  # type: ignore[operator]
            failures.append("the three checks took more than the 10 minute budget")
        if args.images >= 50_000 and out["peak_rss_gb"] > 4:  # type: ignore[operator]
            failures.append("peak RSS above the 4 GB budget")
        if failures:
            raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
