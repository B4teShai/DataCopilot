# Concepts

## The dataset directory

`datacopilot ingest` converts a dataset from any supported format into a `dataset.dcp/` directory:
Parquet tables for images, annotations and categories, plus a `manifest.json`. Images are referenced
by path and never copied. The store is append-only: it refuses to overwrite, verifies the content
hash on every read and bounds table size and row count before decoding anything.

## Identity and version

- `image_id` is the first 16 hex digits of the SHA-256 of the image file, so it follows the bytes,
  not the file name. Two files with identical bytes cannot be ingested together (the tutorial shows
  how a near copy is found by an embedding check instead).
- `manifest.version` is an order-independent content hash of the three tables (file paths excluded).
  The same data gives the same version whatever the file order or source format; record it next to
  a training run to reproduce it.

## Rows

| Table | Main fields |
|---|---|
| Image | `image_id`, `file_path`, `width`, `height`, `split` (`train`, `val`, `test`, `unassigned`), `group_id`, `source` |
| Annotation | `ann_id`, `image_id`, `category_id`, `kind` (`bbox` or `classification`), `bbox_xyxy_abs`, `score`, `is_crowd`, `occluded`, `truncated`, `attributes`, `provenance` (`human`, `model`, `proposal_accepted`), `source` |
| Category | `id`, `name`, `aliases`, `attributes` |

Boxes are absolute pixel corners and must lie inside their image. `source` keeps everything a reader
could not map into the model, verbatim, so a conversion back to the same format loses nothing.
`group_id` (set with `ingest --group-by REGEX`) names the capture session or camera; splitting and
the leakage check use it so frames of one session do not end up on both sides of a split.

## Conversion reports

Every read and write returns a `ConversionReport`: `dropped` counts values present but not
representable in the target (for example `annotations.is_crowd`), `skipped` counts rows left out
entirely (for example `images.file_missing`, or `annotations.unsupported.polygon` when reading CVAT).
Losses are logged as `conversion_loss` and `--strict` turns any of them into exit code 1 before
anything is written, on `export` and on `ingest`.

## Exit codes

`0` success; `1` the command ran and failed (invalid input, a matched gate, a lossy `--strict`);
`2` usage or configuration error; `130` interrupt. Expected failures print one `error:` line on
standard error and one `command_failed` log record.
