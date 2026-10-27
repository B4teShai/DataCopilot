# Tutorial

In ten minutes you will build a small demo dataset with planted mistakes, audit it, make a CI build
fail on it, find a copied picture, and convert it to other formats. Nothing is downloaded. You need
DataCopilot installed (see [Install](guide/install.md)) and an empty scratch directory.

Every command on this page is run by the test suite, and its output is checked against what is
shown. Values that change from run to run are written `<uuid>`, `<hex32>` or `...`. Unfamiliar word?
See [Concepts](guide/concepts.md).

Results go to standard output. DataCopilot also writes JSON log lines to standard error, one per
event at level INFO by default. Add `2>/dev/null` to hide them, or set `DATACOPILOT_LOG_LEVEL=WARNING`.
A failed command, such as the `--strict` export in step 5, logs its error there too, with a traceback.
That is the normal error log, not a crash.

## 1. Generate the dataset

The script writes twelve tiny pictures and a COCO annotation file per split into `shop/`. The
mistakes are planted on purpose:

- `v0.png` (validation) is `t0.png` (training) with one pixel changed. It is the same picture but
  not the same bytes, so only an image model can see it.
- `t1.png` has a one-pixel box.
- 12 of the 15 boxes are `cup`, one is `bowl`, and validation has no `bowl` at all.

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

`ingest` reads a dataset, checks it and writes a `shop.dcp/` folder. It prints the dataset's
version: the same content always gives the same version.

<!-- run -->
```console
$ datacopilot ingest shop --format coco -o shop.dcp
<hex32>
```

## 3. Audit

`audit` runs the CPU checks, writes `report.html`, `report.json` and `report.sarif`, and prints one
line: pass or fail, then the number of findings per severity. `--no-hash-cache` skips the cache of
file hashes that speeds up repeated audits. A clean run, and CI, should not use it.

<!-- run -->
```console
$ datacopilot audit shop.dcp --no-hash-cache
passed critical=0 warn=3 info=0 run_id=<uuid>
```

There are three warnings (the one-pixel box and two imbalance findings) and nothing critical, so
the exit code is 0. Make warnings fail the run, as a strict CI job would:

<!-- run -->
```console
$ datacopilot audit shop.dcp --no-hash-cache --fail-on warn  # exit 1
failed critical=0 warn=3 info=0 run_id=<uuid>
```

See [Audit in CI](guide/audit-ci.md) for the CI settings.

## 4. Find the copied picture

The exact-duplicate check compares bytes, so `v0.png` slips through. The image-model checks compare
what pictures look like. This tutorial uses `hash`, a built-in stand-in that needs no download. It
finds this planted copy but is not a real model, so a gate refuses it unless you add
`--allow-test-embedder`. For real data use `--embedder dinov2_base` (see
[Embeddings](guide/embeddings.md)). `--fail-on leakage` names a check, not a severity: the run fails
if that check reports anything.

<!-- run -->
```console
$ datacopilot audit shop.dcp --no-hash-cache --embedder hash --allow-test-embedder --fail-on leakage  # exit 1
failed critical=2 warn=4 info=1 run_id=<uuid>
warning: embedder 'hash' is a test double that compares file bytes, not pictures; its verdicts say nothing about look-alike images
```

The warning goes to standard error. Read `report.json` to see which checks fired:

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
label_noise WARN
leakage CRITICAL
near_duplicates CRITICAL
outliers INFO
```

`leakage` is CRITICAL because the copy sits in a different split from its original. `label_noise` is
a WARN that says it did not run: the shop is too small to compare a box with its neighbours. A
skipped check is reported, never counted as a pass.

## 5. Convert and count the loss

Export as YOLO. Every value YOLO cannot hold is counted and logged. `--strict` turns any loss into a
failure and writes nothing:

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

YOLO has no place for the original COCO ids, so a strict export refuses. Reading a file back with
`--strict` also fails if the reader had to skip anything:

<!-- run -->
```console
$ datacopilot export shop.dcp --format cvat_xml -o shop_cvat
shop_cvat
$ datacopilot ingest shop_cvat --format cvat_xml -o shop_cvat.dcp --strict
<hex32>
```

`formats` lists every format and what it can do. [Formats](guide/formats.md) says what each pair
loses.

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

## 6. The same from Python

`load`, `audit` and `export` run the same code as the commands. See [Python API](guide/python-api.md).

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

Run `datacopilot doctor` any time to check the installation.

## Next

- [Audit in CI](guide/audit-ci.md): exit codes, reports and a workflow example.
- [Trust model](guide/trust-model.md): what happens with files you do not control.
- [CLI reference](reference/cli.md): every command and option.
