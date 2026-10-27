# 03 Data model

![The stored entities and how they relate](../diagrams/png/data-model.png)

*A manifest owns its images and categories; audits, proposals and embeddings point back at them.*

Models are Pydantic v2 classes in `core/schema.py` and `core/proposals.py`. Tables are Parquet files
with explicit Arrow schemas ([ADR-0006](../adr/0006-dataset-format.md)). A `?` after a type means the
value may be null. Terms are in the [glossary](../guide/concepts.md).

## Entities

### Category

Table `categories.parquet`. Key `id` (source ids are kept where the source has them). Fields: `name`
(unique), `aliases` (synonyms, also used as text prompts for proposers) and free `attributes`.

### Image

Table `images.parquet`. Key: `image_id`.

| Field | Type | Meaning |
|---|---|---|
| `image_id` | str (16 hex) | `sha256(bytes)[:16]`. Follows the content, not the location. |
| `file_path` | str | Local path or URI. Never a copy. |
| `width`, `height` | int | Pixels, after EXIF orientation. |
| `split` | `train`, `val`, `test`, `unassigned` | Default `unassigned`. |
| `group_id` | str? | Session, device or video group used by leakage checks and re-split. |
| `source` | dict | Original format and ids, plus unmapped fields, verbatim. |

### Annotation

Table `annotations.parquet`. Key: `ann_id`. Foreign keys: `image_id`, `category_id`.

| Field | Type | Meaning |
|---|---|---|
| `ann_id` | str | Unique; from the source id or generated. |
| `kind` | `bbox`, `classification` | Polygons and masks are reserved. |
| `bbox_xyxy_abs` | float[4]? | Pixel `x_min, y_min, x_max, y_max`. Null for classification. |
| `score` | float? | Model confidence. Null for human labels. |
| `is_crowd`, `occluded`, `truncated` | bool | Flags from COCO, VOC and CVAT. |
| `attributes` | dict | Format attributes, such as CVAT attributes. |
| `provenance` | `human`, `model`, `proposal_accepted` | Who produced the row. |
| `source` | dict | Unmapped source fields, verbatim. |

### DatasetManifest

File `manifest.json`, with an immutable copy under `versions/<version>/`. Key `version`.

- `version` (32 hex) is the content hash of the three tables. CV-Pilot records it.
- `schema_version` is the table schema generation, starting at 1.
- `name` and `task` (`detection` or `classification`; `task` decides which checks apply).
- `categories` is a copy for readers without Parquet.
- `lineage` lists parent versions, oldest first.
- `image_count` and `annotation_count` must equal the table rows; they are checked on read.
- `created_at` is not part of the hash.

### ProposalSet and Proposal

File `proposals/<set_id>.json`, read with a bound of 128 MiB and 200,000 proposals. A set is
content-addressed: `set_id` is the first 32 hex digits of the SHA-256 of every other field, recomputed
on build and on read, so an edited file no longer matches its name.

| ProposalSet field | Meaning |
|---|---|
| `set_id`, `dataset_version` | The digest, and the version the run read. `apply` refuses another version. |
| `proposer`, `model_id`, `model_revision`, `weights`, `prompt_hash` | Which pinned model and prompts made it. |
| `min_score`, `class_min_score`, `overlap_iou` | Thresholds used. |
| `images_total`, `images_done`, `incomplete_reason` | Set when a run did not reach every image. |
| `rejected` | Dropped candidates, counted by reason. |
| `parent_set_id`, `test_only` | The set a review came from (`apply` requires it), and the test-double flag. |
| `proposals` | The proposals, in canonical order. |

A `Proposal` has `proposal_id`, `kind` (always `add_box`), `image_id`, `category_id`, a pixel box
`bbox_xyxy_abs`, `score` (above 0, at most 1, 4 decimals), the `prompt` it answers, a `status`
(`pending`, `accepted`, `rejected`; `review import` writes a new set) and `edited` (the reviewer moved
the box).

### AuditRun and Finding

File `audits/<run_id>.json`. `run_id` equals the `request_id` of the command, and findings are embedded.

- **AuditRun:** `run_id`, `dataset_version`, `started_at`, `checks`, `embedder`, `counts` (findings per
  severity), `passed` and `fail_on` (the verdict and its gate), `timings_ms` and `summary`.
- **Finding:** `finding_id` (a hash of check, image, annotation and payload, so reruns dedupe), `check`,
  `severity` (`CRITICAL`, `WARN`, `INFO`), optional `image_id` and `ann_id`, and a `payload` bounded to
  64 KiB of JSON, depth 6, finite numbers.

### Embedding

Cache folder `embeddings/<model_id>@<model_revision>/` with a `manifest.json` and
`shard-<32 hex>.parquet` files ([ADR-0007](../adr/0007-embedding-cache-and-index.md)). A shard row has
`image_id`, an L2-normalised `vector` and `content_sha256`. The manifest names the model, preprocessing
and dataset version, lists shard digests and carries an HMAC. Box-crop vectors use the same format in
`...@<revision>+<variant>/`, where the variant names the crop geometry.

## Relations

| From | To | Cardinality | Via |
|---|---|---|---|
| DatasetManifest | Category, Image | 1 to many | its tables |
| DatasetManifest | DatasetManifest | 1 to 0 or more parents | `lineage` |
| Image | Annotation | 1 to 0 or more | `Annotation.image_id` |
| Category | Annotation | 1 to 0 or more | `Annotation.category_id` |
| Image | Embedding | 1 to 0 or more, one per model revision | `image_id` |
| AuditRun | DatasetManifest | many to 1 | `dataset_version` |
| AuditRun | Finding | 1 to 0 or more | `Finding.run_id` |
| Finding | Image, Annotation | 0 or 1 each | `image_id`, `ann_id` |
| ProposalSet | DatasetManifest | many to 1 | `dataset_version` |
| ProposalSet | ProposalSet | 0 or 1 parent | `parent_set_id` |
| Proposal | Image | many to 1 | `image_id` |

## Invariants

Tested in `tests/unit/test_schema_invariants.py`, `test_embed_cache.py`, `test_store.py`,
`test_audit_schema.py`, `test_cli_audit.py`, `test_split_usecase.py` and `tests/golden/`.

1. `Image.image_id` is `sha256(file bytes)[:16]` for every row. Moving the file does not change it.
2. `Image.width` and `Image.height` are positive.
3. Every `Annotation.image_id` and `category_id` exists.
4. A `bbox` has `0 <= x_min < x_max <= width` and `0 <= y_min < y_max <= height`. A `classification`
   has no box.
5. `Annotation.score` is null when `provenance` is `human`.
6. `Category.name` and `Category.id` are unique.
7. Only `add_box` proposals exist, so a proposal never targets an existing annotation.
8. A proposal set records its model id, revision, weight hashes and prompt hash. `apply` accepts a set
   for a registered model only if they equal the pinned values.
9. `manifest.version` equals the version recomputed from the tables. Row order does not matter.
10. `apply`, `split` and `ingest` never modify an existing version. They create exactly one new dataset
    folder whose `lineage` is the parent's lineage plus the parent's version (empty for `ingest`).
11. After `apply`, each new annotation has `provenance == "proposal_accepted"`, the id
    `proposal:<proposal_id>`, and the `proposal_id` and `set_id` in `source`.
12. After `split`, no `group_id` and no near-copy pair found by `--near-dup` occurs in more than one
    split. `groups_across_splits` is 0.
13. `Finding.severity` is `CRITICAL`, `WARN` or `INFO`, and `AuditRun.counts` equals the findings per
    severity.
14. An embedding has `dims` finite values and L2 norm 1. A shard that breaks this, whose digest differs
    from the manifest, or whose manifest fails its HMAC is ignored and counted, never used.
15. `AuditRun.run_id` equals the `request_id` of the log records of that run.

## On-disk layout

```text
dataset.dcp/
  manifest.json                  # DatasetManifest of the current version
  images.parquet                 # current tables
  annotations.parquet
  categories.parquet
  versions/<version>/            # immutable copy of manifest and tables
  embeddings/<model>@<rev>/      # embedding cache (+<variant> for box crops)
  audits/<run_id>.json           # one AuditRun with its findings
  proposals/<set_id>.json        # one ProposalSet
```

- A new version is written to `versions/<version>.tmp-<uuid>/`, fsynced, renamed, and only then are
  the top-level tables and `manifest.json` replaced by rename. Readers ignore `*.tmp-*` folders.
- `source` and `attributes` are `map<string,string>` holding canonical JSON text, so numbers and
  booleans survive. Rows are sorted by key, so output is byte-deterministic.
- Images are never inside `dataset.dcp/`. `file_path` may be relative to the folder's parent, absolute
  or a URI.
- Reads are bounded: a table over 1 GiB, over the row or byte limits in
  [Settings](../reference/settings.md), or with another schema is rejected before decoding.
- Embeddings, audits and proposals are not part of the version and are not copied into `versions/`.
- Two audits of the same dataset and options give byte-identical JSON, SARIF and HTML after the
  run-specific content is replaced: `started_at`, `timings_ms` and the run id.
  `tests/golden/test_audit_golden.py` checks this with different run ids and worker counts.

## Versioning and compatibility

**Content hash** (`core/version.py`):

1. For each of the three tables, take every row as a mapping (drop `file_path` for images) and
   serialise it as canonical JSON: sorted keys, no whitespace, UTF-8, shortest round-trip floats, no
   `NaN` or infinity.
2. `table_hash = sha256("".join(sorted(sha256(row_json) for each row)))`.
3. `version = sha256("dcp:<schema_version>\n<images_hash>\n<annotations_hash>\n<categories_hash>")[:32]`.

What changes the version: any field value in the three tables except `file_path`, a split change, a
category rename, `schema_version`. What does not: row order, file locations, `created_at`, `name`,
embeddings, audits, proposals, lineage.

**Schema version.** It starts at 1. Adding a nullable column does not bump it; removing or retyping a
column, or changing the hash procedure, does. A store with a higher `schema_version` than the installed
package is refused (exit 1). There is no migration yet because 1 is the only generation.

**Promise.** For a given input, `manifest.version` is stable within a `schema_version`.

## Format mapping and loss matrix

Rules shared by every reader and writer:

- Nothing a reader cannot map is discarded. It is kept verbatim in `source`, with `source.format`
  naming the producing format. A writer re-emits the `source` keys of its own format. For rows from
  another format it can hold only what its format has a place for, and counts every other key in
  `dropped`.
- Absent files are never silent. A missing image skips the image and its annotations (`skipped`:
  `images.file_missing`, `annotations.of_missing_image`). A declared size that differs from the file
  header is counted and the header wins.
- Detection formats (COCO, YOLO, VOC, CVAT XML) hold detection only. ImageFolder holds classification
  only. Label Studio holds either. Converting between the two groups is a `FormatError`.
- Two images with identical bytes share one `image_id` and are rejected at ingest, naming both files.
- COCO `bbox` and VOC `bndbox` are taken as given. YOLO coordinates are normalised and written with 7
  decimals. Boxes outside the image are an error, except YOLO overshoot of up to 0.001 pixel, which is
  snapped.
- Only COCO holds `group_id`; other formats count it as dropped.
- YOLO, VOC and ImageFolder re-number category ids; a changed id is counted in `categories.id`.
- `ingest --strict` fails if a reader dropped or skipped anything, before writing anything.

How CVAT and Label Studio are read is in [Formats](../guide/formats.md).

Dropped-value counts on the synthetic fixtures (`tests/fixtures/synth.py`), asserted by
`tests/golden/test_roundtrip_matrix.py`. Rows are the source format, columns the target. Same-format
conversions drop nothing, and ingest, export, ingest is a fixed point.

| Source \ target | COCO | YOLO | VOC | CVAT XML | Label Studio |
|---|---|---|---|---|---|
| COCO | 0 | 31 | 31 | 31 | 29 |
| YOLO | 0 | 0 | 0 | 0 | 3 |
| VOC | 24 | 24 | 0 | 24 | 27 |
| CVAT XML | 16 | 16 | 12 | 0 | 19 |
| Label Studio | 20 | 20 | 20 | 20 | 0 |

The counts come from these columns:

- **COCO:** `annotations.is_crowd` and `categories.attributes`/`id`, plus the source keys `area`,
  `id`, `segmentation`, `date_captured` and `license`. Into Label Studio, `categories.id` is 1 not 3.
- **VOC:** `annotations.attributes`, `truncated` and the source keys `pose`, `depth`, `folder`,
  `segmented`. Into CVAT XML, `attributes_stringified` replaces `attributes`.
- **CVAT XML:** `annotations.attributes`, `occluded` (kept by VOC) and `z_order`,
  `categories.attributes`, and the source `id`.
- **Label Studio:** the source keys `from_name`, `id`, `to_name` and `annotation`.
- **YOLO:** nothing, except `categories.id` into Label Studio.
- Label Studio as a target adds `categories.id` 3 for VOC and CVAT XML sources.

ImageFolder to ImageFolder drops nothing. ImageFolder to or from COCO, YOLO, VOC and CVAT XML is
refused; Label Studio holds both tasks, so ImageFolder to Label Studio converts. YOLO into COCO, VOC or
CVAT XML loses nothing, because a YOLO row has no fields beyond box, class and split.
