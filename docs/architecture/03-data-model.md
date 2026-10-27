# 03 Data model

![The stored entities and how they relate](../diagrams/png/data-model.png)

Models are Pydantic v2 classes in `core/schema.py` and `core/proposals.py`. Tables are Parquet files
with explicit Arrow schemas ([ADR-0006](../adr/0006-dataset-format.md)). A `?` after a type means the
value may be null. Terms are in the [glossary](../guide/concepts.md).

## Entities

### Category

Table `categories.parquet`. Key: `id`. `name` is unique.

| Field | Type | Meaning |
|---|---|---|
| `id` | int | Category id; source ids are kept where the source has them. |
| `name` | str | Display name. |
| `aliases` | list[str] | Synonyms; also used as text prompts for proposers. |
| `attributes` | dict[str, str] | Free metadata. |

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

File `manifest.json`, with an immutable copy under `versions/<version>/`. Key: `version`.

| Field | Type | Meaning |
|---|---|---|
| `version` | str (32 hex) | Content hash of the three tables. CV-Pilot records it. |
| `schema_version` | int | Table schema generation, starting at 1. |
| `name`, `task` | str, `detection` or `classification` | `task` decides which checks apply. |
| `categories` | list[Category] | Copy for readers without Parquet. |
| `lineage` | list[str] | Parent versions, oldest first. |
| `image_count`, `annotation_count` | int | Must equal the table rows; checked on read. |
| `created_at` | datetime (UTC) | Not part of the hash. |

### ProposalSet and Proposal

File `proposals/<set_id>.json`. A set is content-addressed: `set_id` is the first 32 hex digits of the
SHA-256 of the canonical JSON of every other field, recomputed on build and on read, so an edited file
no longer matches its name. Reads are bounded at 128 MiB and 200,000 proposals.

| ProposalSet field | Meaning |
|---|---|
| `set_id` | Digest of the rest of the document; also the file name. |
| `dataset_version` | The version the run read. `apply` refuses another. |
| `proposer`, `model_id`, `model_revision`, `weights` | Which model, pinned revision and file hashes made it. |
| `prompt_hash` | SHA-256 of the sorted phrases the model was asked about. |
| `min_score`, `class_min_score`, `overlap_iou` | Thresholds used. |
| `images_total`, `images_done`, `incomplete_reason` | Set when a run did not reach every image. |
| `rejected` | Dropped candidates, counted by reason. |
| `parent_set_id` | The set a review was derived from. `apply` requires it. |
| `test_only` | The proposer is a test double. `apply` refuses it unless asked. |
| `proposals` | The proposals, in canonical order. |

| Proposal field | Meaning |
|---|---|
| `proposal_id` | Unique in its set. |
| `kind` | Always `add_box`. |
| `image_id`, `category_id` | Where the box was found and its class. |
| `bbox_xyxy_abs` | Pixel box. After a review, the reviewer's box. |
| `score` | Model confidence, above 0 and at most 1, rounded to 4 decimals. |
| `prompt` | The class name or alias the model was asked about. |
| `status` | `pending`, `accepted` or `rejected`. `review import` writes a new set. |
| `edited` | The reviewer moved the box before accepting it. |

### AuditRun and Finding

File `audits/<run_id>.json`. `run_id` equals the `request_id` of the command. Findings are embedded.

| AuditRun field | Meaning |
|---|---|
| `run_id`, `dataset_version`, `started_at` | Identity of the run and the version audited. |
| `checks`, `embedder` | What ran. |
| `counts` | Findings per severity. |
| `passed`, `fail_on` | The verdict and the gate that produced it. |
| `timings_ms`, `summary` | Wall time per check; totals and per-split class counts. |

| Finding field | Meaning |
|---|---|
| `finding_id` | `sha256` of check, image, annotation and payload, cut to 16 hex. Reruns dedupe. |
| `check`, `severity` | Check name and `CRITICAL`, `WARN` or `INFO`. |
| `image_id`, `ann_id` | The row concerned, if any. |
| `payload` | Check-specific data, bounded to 64 KiB of JSON, depth 6, finite numbers. |

### SplitReport

Returned by `dc.split` and written by `split --json`. It is not stored in the dataset. It records the
parent and new version, seed, ratios, group counts, `groups_across_splits` (0 when written), and the
deviation from the target per split and per class.

### Embedding

Cache folder `embeddings/<model_id>@<model_revision>/` with a `manifest.json` and `shard-<32 hex>.parquet`
files ([ADR-0007](../adr/0007-embedding-cache-and-index.md)). A shard row has `image_id`, `vector`
(L2-normalised, float16 or float32) and `content_sha256` (the digest of the image bytes it came from).
The manifest names the model, dimensions, preprocessing version and dataset version, lists the shards
with their digests, and carries an HMAC over everything. Box-crop vectors for `label_noise` use the same
format in `...@<revision>+<variant>/`, where the variant names the crop geometry.

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
- `bbox_xyxy_abs` is `list<float64>` of length 4. `source` and `attributes` are `map<string,string>`
  holding canonical JSON text, so numbers and booleans survive. Rows are sorted by key, so output is
  byte-deterministic.
- Images are never inside `dataset.dcp/`. `file_path` may be relative to the folder's parent, absolute
  or a URI.
- Reads are bounded. A table is rejected before decoding if its file exceeds 1 GiB, its rows exceed
  `DATACOPILOT_MAX_TABLE_ROWS` (5,000,000), its uncompressed size exceeds `DATACOPILOT_MAX_TABLE_BYTES`
  (4 GiB), or its schema differs from the expected one.
- Embeddings, audits and proposals are not part of the version and are not copied into `versions/`.
  Embeddings are still bound to the dataset version they were computed for.
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

**Schema version.** It starts at 1. Adding a nullable column with a default does not bump it.
Removing or retyping a column, or changing the hash procedure, does. A store with a higher
`schema_version` than the installed package is refused with exit code 1 and a message naming the
package version needed. `schema_version` 1 is the only generation so far, so there is no migration.

**Promise.** For a given input, `manifest.version` is stable within a `schema_version`, so CV-Pilot
and users can compare versions across releases.

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
- Splits come from COCO file names, YOLO `images/<split>/`, VOC `ImageSets/Main/<split>.txt` and
  ImageFolder top-level folders. Only COCO holds `group_id`; other formats count it as dropped.
- YOLO, VOC and ImageFolder re-number category ids; a changed id is counted in `categories.id`.
- `ingest --strict` fails if a reader dropped or skipped anything, before writing anything. Report
  columns are bounded names, and a report holds at most 200 distinct columns.

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

- **From COCO:** `annotations.is_crowd` 1; `annotations.source.{area,id,segmentation}` 4 each;
  `categories.attributes` 3; `categories.id` 3; `images.source.{date_captured,id,license}` 4 each.
  Into Label Studio, `categories.id` is 1 instead of 3.
- **From YOLO:** only `categories.id` 3, and only into Label Studio.
- **From VOC:** `annotations.attributes` 4; `annotations.source.pose` 4; `annotations.truncated` 4;
  `images.source.{depth,folder,segmented}` 4 each. Into CVAT XML, `annotations.attributes_stringified`
  4 replaces `annotations.attributes`. Into Label Studio, add `categories.id` 3.
- **From CVAT XML:** `annotations.attributes` 1; `annotations.occluded` 4; `annotations.source.z_order`
  4; `categories.attributes` 3; `images.source.id` 4. Into VOC, `annotations.occluded` is kept (12 in all). Into
  Label Studio, add `categories.id` 3.
- **From Label Studio:** `annotations.source.{from_name,id,to_name}` 4 each;
  `images.source.{annotation,id}` 4 each.

ImageFolder to ImageFolder drops nothing. ImageFolder to or from COCO, YOLO, VOC and CVAT XML is
refused; Label Studio holds both tasks, so ImageFolder to Label Studio converts. YOLO into COCO, VOC or
CVAT XML loses nothing, because a YOLO row has no fields beyond box, class and split.
