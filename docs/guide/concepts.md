# Concepts

Every term DataCopilot uses is defined once, here. Other pages link to this table.

## Glossary

| Term | Meaning |
|---|---|
| dataset | Images plus labels (boxes with a class, or one class per image) in one of six formats. |
| `dataset.dcp` | The folder `ingest` writes: tables of images, boxes and classes, plus `manifest.json`. Images stay where they are. |
| version | A 32-character fingerprint of a dataset's content (`manifest.version`). Same content, same version, in any file order. |
| `image_id` | The first 16 hex digits of the SHA-256 of the image file. It follows the bytes, not the file name. |
| split | The side of an image: `train`, `val`, `test` or `unassigned`. |
| `group_id` | An optional label for images from one source, such as a camera or session. Split and leakage checks keep a group on one side. |
| leakage | The same picture, or a copy, in both train and test. The test score then looks better than it is. |
| near-duplicate | An image that looks almost the same as another, although the files differ. |
| embedder | An image model that turns a picture into a list of numbers (an embedding). Similar pictures get similar numbers. Needs the `embed` extra. |
| extra | An optional install group: `embed` (image models) or `vlm` (the box-proposing model). |
| nearest-neighbour search | Finding the images with the closest embeddings (kNN). Done exactly in NumPy; the FAISS library helps on large sets. |
| audit | A run of checks over a dataset. Each check reports findings. |
| finding | One problem a check found, with a severity: `CRITICAL`, `WARN` or `INFO`. |
| gate | The rule that turns findings into exit code 1, set with `--fail-on`. Use it to fail a CI build. |
| SARIF | A standard JSON report format that code-scanning services read. `audit` writes `report.sarif`. |
| conversion loss | A value a target format cannot hold. It is counted as `dropped` (kept row, lost value) or `skipped` (row left out). |
| lineage | The list of earlier versions a dataset came from, oldest first. `split` and `apply` write it. |
| label-noise lead | A box whose class its look-alike neighbours do not share. A weak hint, not proof of a wrong label. |
| proposal | A box a model suggests your labels missed. Nothing changes until a person accepts it. |
| review | A person marks each proposal `accepted` or `rejected` in CVAT, a labeling tool. |
| apply | Writes the accepted proposals as a new dataset version. The old one is untouched. |
| CV-Pilot | A separate tool that tracks training runs. It can record the dataset version a run used. |

## Identity and version

- `image_id` follows the bytes. Two files with identical bytes cannot be ingested together; a
  near copy is found by an embedder check instead (see the [tutorial](../tutorial.md)).
- The version hashes the three tables, not the file paths. It also covers where each row came from,
  so a COCO copy of a YOLO dataset has another version although the boxes are the same. Record it
  next to a training run to reproduce the run.

## Conversion reports

Every read and write returns a report of what was lost. `dropped` counts values with no place in
the target, for example `annotations.is_crowd`. `skipped` counts rows left out, for example
`images.file_missing`. Losses are logged as `conversion_loss`; `--strict` turns any loss into exit
code 1 before anything is written. See [Formats](formats.md).

## Record the dataset version in CV-Pilot

Give [CV-Pilot](https://github.com/B4teShai/CV-Pilot) the `dataset.dcp` folder as `dataset_uri` and
its run record holds `dataset_hash = "dcp:<version>"`:

<!-- skip: needs CV-Pilot, a separate package installed from its own clone -->
```python
import cvpilot

run = cvpilot.init("shop", task="detection", dataset_uri="shop.dcp")
# `cvpilot runs show <run_id> --project shop` then prints
# dataset_hash: dcp:9e96d9fb67b1d64fafe14c25ccdb43f0   (what `datacopilot ingest` printed)
```

CV-Pilot reads only the top-level `version` field of `shop.dcp/manifest.json`. For any other
folder it records `merkle:<hex>` over file names, sizes and modification times. Neither package
imports the other. A new version (`apply`, or another `ingest`) is a new folder or a new
`manifest.json`: point the next run at it.

## Exit codes

`0` success, `1` the command ran and failed, `2` usage or configuration error, `130` interrupted.
Details are in [Settings and exit codes](../reference/settings.md#exit-codes).
