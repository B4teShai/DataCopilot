"""Benchmark of `datacopilot audit` on a synthetic R50k-like dataset (see 01-overview.md).

R50k is 50,000 JPEG images (1920x1080) with 500,000 boxes over 200 classes. The files written here
are header-valid JPEGs (declared 1920x1080) followed by random bytes up to ``--image-kb``; the
default (64 KiB) keeps the benchmark at about 3 GB of disk, so hashing time is reported both as
measured and as MB/s so it can be scaled to real 1080p JPEGs (about 250 KiB each).

Usage::

    uv run python benchmarks/audit_cpu.py --workdir /path/with/10GB            # full R50k
    uv run python benchmarks/audit_cpu.py --workdir /tmp/b --images 2000       # smoke test

Each audit runs in a fresh subprocess so peak RSS (``ru_maxrss``) covers only the audit: reading
the tables, running the checks, writing the run and the three reports. Three configurations:
cold (empty hash cache), warm (cache present) and ``--no-hash-cache``. Results are printed as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import struct
import subprocess  # noqa: S404  # runs this same script as a clean-RSS child
import sys
import time
from pathlib import Path


def _jpeg(width: int, height: int, filler: bytes) -> bytes:
    sof = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9" + filler


def generate(workdir: Path, images: int, boxes_per_image: int, classes: int, image_kb: int) -> Path:
    import numpy as np

    from datacopilot.core.schema import Annotation, Category, Image
    from datacopilot.store import DatasetStore

    rng = np.random.default_rng(0)
    folder = workdir / "images"
    folder.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    image_rows: list[Image] = []
    annotations: list[Annotation] = []
    weights = 1.0 / np.arange(1, classes + 1)
    weights /= weights.sum()
    t0 = time.perf_counter()
    for n in range(images):
        data = _jpeg(width, height, rng.bytes(image_kb * 1024))
        path = folder / f"img{n:06d}.jpg"
        path.write_bytes(data)
        split = "train" if n % 10 < 8 else ("val" if n % 10 == 8 else "test")
        image_id = hashlib.sha256(data).hexdigest()[:16]
        image_rows.append(
            Image(
                image_id=image_id,
                file_path=str(path),
                width=width,
                height=height,
                split=split,  # type: ignore[arg-type]
                group_id=f"cam{n % 40}",
            )
        )
        xy = rng.uniform(0, [width - 60, height - 60], size=(boxes_per_image, 2))
        wh = rng.uniform(20, 400, size=(boxes_per_image, 2))
        x2 = np.minimum(xy[:, 0] + wh[:, 0], width)
        y2 = np.minimum(xy[:, 1] + wh[:, 1], height)
        cats = rng.choice(classes, size=boxes_per_image, p=weights) + 1
        for k in range(boxes_per_image):
            annotations.append(
                Annotation(
                    ann_id=f"{n}-{k}",
                    image_id=image_id,
                    category_id=int(cats[k]),
                    kind="bbox",
                    bbox_xyxy_abs=(float(xy[k, 0]), float(xy[k, 1]), float(x2[k]), float(y2[k])),
                )
            )
    categories = [Category(id=i + 1, name=f"class_{i + 1}") for i in range(classes)]
    DatasetStore.create(
        workdir / "r50k.dcp", image_rows, annotations, categories, name="r50k", task="detection"
    )
    print(
        f"generated {images} images, {len(annotations)} boxes in {time.perf_counter() - t0:.0f}s",
        file=sys.stderr,
    )
    return workdir / "r50k.dcp"


def audit_child(dcp: Path, workers: int, hash_cache: bool, cache_dir: Path) -> dict[str, object]:
    """Run the audit in this process and report timings and peak RSS."""
    from datacopilot.audit import engine
    from datacopilot.audit.context import CheckContext
    from datacopilot.report import render
    from datacopilot.store import DatasetStore

    t0 = time.perf_counter()
    store = DatasetStore.open(dcp)
    ds = store.read()
    t_read = time.perf_counter() - t0
    ctx = CheckContext(
        run_id="00000000-0000-4000-8000-000000000000",
        root=dcp.resolve().parent,
        cache_dir=cache_dir,
        workers=workers,
        use_hash_cache=hash_cache,
    )
    t1 = time.perf_counter()
    run = engine.run_audit(ds, "all", ctx, ["critical"])
    t_audit = time.perf_counter() - t1
    t2 = time.perf_counter()
    render(
        run,
        html=dcp.parent / "bench-report.html",
        json=dcp.parent / "bench-report.json",
        sarif=dcp.parent / "bench-report.sarif",
    )
    t_render = time.perf_counter() - t2
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1024 / 1024 if platform.system() == "Darwin" else rss / 1024
    return {
        "read_tables_s": round(t_read, 2),
        "audit_s": round(t_audit, 2),
        "render_s": round(t_render, 2),
        "total_s": round(time.perf_counter() - t0, 2),
        "per_check_ms": run.timings_ms,
        "findings": run.counts.model_dump(),
        "peak_rss_mb": round(rss_mb),
        "images": len(ds.images),
        "annotations": len(ds.annotations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=50_000)
    parser.add_argument("--boxes-per-image", type=int, default=10)
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--image-kb", type=int, default=64)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--child", type=Path, help="internal: audit this dataset and print JSON")
    parser.add_argument("--no-cache", action="store_true", help="internal (child)")
    parser.add_argument("--cache-dir", type=Path, help="internal (child)")
    args = parser.parse_args()
    if args.child:
        result = audit_child(args.child, args.workers, not args.no_cache, args.cache_dir)
        print(json.dumps(result))
        return
    args.workdir.mkdir(parents=True, exist_ok=True)
    dcp = args.workdir / "r50k.dcp"
    if not dcp.exists():
        generate(args.workdir, args.images, args.boxes_per_image, args.classes, args.image_kb)
    cache = args.workdir / "cache"
    out: dict[str, object] = {
        "machine": f"{platform.machine()} {platform.system()} cpus={os.cpu_count()}",
        "python": platform.python_version(),
        "image_kb": args.image_kb,
    }
    for label, extra in (("cold", []), ("warm", []), ("no_cache", ["--no-cache"])):
        if label == "cold" and cache.exists():
            for p in cache.iterdir():
                p.unlink()
        proc = subprocess.run(  # noqa: S603
            [
                sys.executable,
                __file__,
                "--workdir",
                str(args.workdir),
                "--child",
                str(dcp),
                "--workers",
                str(args.workers),
                "--cache-dir",
                str(cache),
                *extra,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        out[label] = json.loads(proc.stdout.strip().splitlines()[-1])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
