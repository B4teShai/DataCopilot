# Tutorial: audit, gate and convert a dataset

This tutorial builds a small synthetic dataset with a few planted problems, then walks the whole
loop: ingest it, audit it, turn the audit into a CI gate, find a leaked image with embeddings and
convert the dataset to other formats while counting what each conversion loses. Nothing is
downloaded and no real data is used.

Every command and script on this page is executed by the test suite
(`tests/docs/test_tutorial.py`) against the code in this repository, and its output is checked
against what is shown. Values that differ from run to run are written as `<uuid>`, `<hex32>` or
`...`.

You need DataCopilot installed (see [Install](guide/install.md)) and a scratch directory.

## 1. Generate the dataset

The script writes twelve tiny PNG pictures and a COCO annotation file per split into `shop/`. The
problems are planted on purpose:

- `v0.png` (validation) is `t0.png` (training) with one pixel changed: the same picture, but not the
  same bytes, so only an embedding check can see it.
- `t1.png` has a one-pixel box.
- 12 of the 15 boxes are `cup`, only one is `bowl`, and the validation split has no `bowl` at all.

<!-- run -->
```python
import json
import struct
import zlib
from pathlib import Path


def png(width, height, seed, poke=None):
    """A small grayscale PNG. Stored (uncompressed) blocks make a one-pixel change a local one."""

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        rows.extend((seed * 53 + x * 7 + y * 11) % 256 for x in range(width))
    if poke is not None:
        rows[poke] ^= 0x40
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 0))
        + chunk(b"IEND", b"")
    )


WIDTH, HEIGHT = 64, 48
CLASSES = ["cup", "plate", "bowl"]
# file name, split, picture seed, boxes as (class index, x, y, width, height)
IMAGES = [(f"t{n}.png", "train", n, [(0, 8, 6, 24, 20)]) for n in range(10)]
IMAGES[0][3].append((1, 30, 20, 30, 22))
IMAGES[1][3].append((0, 40, 4, 1, 1))  # a one-pixel box
IMAGES[2][3].append((2, 10, 24, 20, 18))
IMAGES += [("v0.png", "val", 0, [(0, 8, 6, 24, 20)]), ("v1.png", "val", 11, [(1, 4, 4, 40, 30)])]

root = Path("shop")
(root / "images").mkdir(parents=True)
(root / "annotations").mkdir()
documents = {"train": ([], []), "val": ([], [])}
ann_id = 0
for number, (name, split, seed, boxes) in enumerate(IMAGES, start=1):
    # v0.png is t0.png with one pixel changed: the same picture, but not the same bytes
    poke = 40 if name == "v0.png" else None
    (root / "images" / name).write_bytes(png(WIDTH, HEIGHT, seed, poke))
    images, annotations = documents[split]
    images.append({"id": number, "file_name": name, "width": WIDTH, "height": HEIGHT})
    for cls, x, y, w, h in boxes:
        ann_id += 1
        annotations.append(
            {
                "id": ann_id,
                "image_id": number,
                "category_id": cls + 1,
                "bbox": [x, y, w, h],
                "iscrowd": 0,
            }
        )
categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
for split, (images, annotations) in documents.items():
    document = {"images": images, "annotations": annotations, "categories": categories}
    (root / "annotations" / f"{split}.json").write_text(json.dumps(document))
```

## 2. Ingest

`ingest` reads a dataset in any supported format, checks its invariants and writes a versioned
`shop.dcp/` directory. It prints the content version: the same data always gives the same version,
whatever the file order or format it came from.

<!-- run -->
```console
$ datacopilot ingest shop --format coco -o shop.dcp
<hex32>
```

## 3. Audit

`audit` runs the CPU checks (file integrity, exact duplicates, class balance, box sanity), writes
`report.html`, `report.json` and `report.sarif` next to each other and prints one line. The default
gate is `--fail-on critical`.

<!-- run -->
```console
$ datacopilot audit shop.dcp --no-hash-cache
passed critical=0 warn=3 info=0 run_id=<uuid>
```

Three warnings (the one-pixel box and two imbalance findings), nothing critical, so the command
exits 0. Make warnings fail the run, as a stricter CI job would:

<!-- run -->
```console
$ datacopilot audit shop.dcp --no-hash-cache --fail-on warn  # exit 1
failed critical=0 warn=3 info=0 run_id=<uuid>
```

`--no-hash-cache` is what a CI gate should use: the file-hash cache is an optimisation that a gate
does not need (see [Audit in CI](guide/audit-ci.md)).

## 4. Find the leaked image with embeddings

The exact-duplicate check compares file bytes, so `v0.png` slips through. The embedding checks
compare what the pictures look like. This tutorial uses the built-in `hash` embedder, a
dependency-free test double that compares file bytes in a fuzzy way: it finds this planted copy,
but it is not a model, so it is refused for a gate unless you say `--allow-test-embedder`. For real
datasets use `--embedder dinov2_base` (extra `[embed]`, see [Embeddings](guide/embeddings.md)).

<!-- run -->
```console
$ datacopilot audit shop.dcp --no-hash-cache --embedder hash --allow-test-embedder --fail-on leakage  # exit 1
failed critical=2 warn=3 info=1 run_id=<uuid>
warning: embedder 'hash' is a test double that compares file bytes, not pictures; its verdicts say nothing about look-alike images
```

The warning goes to standard error. Read the machine-readable report to see which checks fired:

<!-- run -->
```python
import json

report = json.load(open("report.json"))
for check, severity in sorted({(f["check"], f["severity"]) for f in report["findings"]}):
    print(check, severity)
```

<!-- output -->
```text
boxes WARN
imbalance WARN
leakage CRITICAL
near_duplicates CRITICAL
outliers INFO
```

`leakage` is CRITICAL because the near copy sits in a different split than the picture it copies:
the validation score would be inflated.

## 5. Convert and count the loss

Export the dataset as YOLO. Every field YOLO cannot hold is counted and logged, never dropped
silently; `--strict` turns any loss into a failure and writes nothing:

<!-- run -->
```console
$ datacopilot export shop.dcp --format yolo -o shop_yolo
shop_yolo
```

<!-- run -->
```console
$ datacopilot export shop.dcp --format yolo -o shop_strict --strict  # exit 1
error: conversion dcp -> yolo is lossy: annotations.source.id=15, categories.id=3, images.source.id=12
```

The source COCO ids are not part of YOLO, so a strict export refuses. CVAT and Label Studio formats
are written and read the same way. The CVAT file we just wrote reads back with `--strict`, which
also fails if the reader had to skip anything (a polygon, a track):

<!-- run -->
```console
$ datacopilot export shop.dcp --format cvat_xml -o shop_cvat
shop_cvat
$ datacopilot ingest shop_cvat --format cvat_xml -o shop_cvat.dcp --strict
<hex32>
```

`datacopilot formats` lists every format and what it can do:

<!-- run -->
```console
$ datacopilot formats
coco read,write Apache-2.0 -
cvat_xml read,write Apache-2.0 -
imagefolder read,write Apache-2.0 -
label_studio read,write Apache-2.0 -
voc read,write Apache-2.0 -
yolo read,write Apache-2.0 -
```

[Formats](guide/formats.md) lists, for each pair, exactly what is lost.

## 6. The same thing from Python

The Python API has three functions, `load`, `audit` and `export`, over the same code as the commands
(see [Python API](guide/python-api.md)). Logs are JSON on standard error, as on the command line.

<!-- run -->
```python
import datacopilot as dc

ds = dc.load("shop", format="coco")
print(ds.manifest.image_count, ds.manifest.annotation_count)

run = dc.audit(ds, fail_on=("warn",))
print(run.passed, run.counts.warn)

report = dc.export(ds, "label_studio", "shop_ls")
print(sorted(report.dropped))
```

<!-- output -->
```text
12 15
False 3
['annotations.source.id', 'categories.id', 'images.source.id']
```

## 7. Check the installation

`doctor` reports versions, which extras are installed, the validated settings, the cache directory,
FAISS and GPU visibility, and never prints a secret. `--json` gives the same report as one JSON
object.

<!-- run -->
```console
$ datacopilot doctor
ok
package: datacopilot ... (dcp schema 1)
python: ... on ...
dependencies: ok (...)
extras: ...
settings: ok (...)
cache: ...
faiss: ...
gpu: ...
plugins: formats=6, checks=7, embedders=4
```

## Where to go next

- [Audit in CI](guide/audit-ci.md): exit codes, reports, SARIF and a workflow example.
- [Trust model](guide/trust-model.md): what DataCopilot does with files you do not control.
- [CLI reference](reference/cli.md): every command and option.
