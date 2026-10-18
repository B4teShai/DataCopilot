# 04 Flows

Flow A is built, including the embedding checks (`--embedder`); `export` and `embed` are built;
Flow B is built for missing boxes (`add_box`) reviewed through CVAT. Exit
codes follow [errors-and-logging.md](../engineering/errors-and-logging.md): `0` success,
`1` the command ran and failed (including a gate or check that was asked to fail), `2`
usage or configuration error, `130` interrupted. Every failure is logged once, at the CLI
boundary, as `command_failed`.

## Flow A: ingest → audit → report → exit code

**Purpose:** turn a raw export into a `dataset.dcp/` version and decide, in CI or on a
laptop, whether the dataset is fit to train on.

**Preconditions:** the export directory exists and matches `--format`; images are readable
at their paths or URIs; for `--embedder`, the `[embed]` extra is installed and the model
license is allowed; the output directory is writable.

![Flow A](../diagrams/png/flow-a.png)

Steps:

1. `datacopilot ingest ./export --format coco -o ds.dcp [--group-by REGEX]` validates options
   (output must end in `.dcp`, the regex must compile) and loads settings (exit 2 on a bad
   option or `DATACOPILOT_*` value).
2. `convert.ingest` resolves the reader by name from `io/registry.py`; the reader streams
   `(Image, [Annotation])` pairs, keeps unmapped fields in `source`, and `image_id` is
   computed from file bytes.
3. Rows are validated against `core/schema.py` (invariants 1 to 6); the store writes
   `versions/<version>/`, then the top-level tables and `manifest.json`.
4. `manifest.version` is printed on `stdout`. (Steps 1 to 4 are built.)

`datacopilot export ds.dcp --format yolo -o ./out [--strict]` (built) reads the dataset
through the bounded, hash-verified store, resolves the writer by name, writes into a sibling
temporary directory, logs one `conversion_loss` per dropped column, fails with exit 1 and
writes nothing if `--strict` and anything was dropped, and otherwise renames the directory
into place and prints its path. The output directory must be absent or empty.
5. `datacopilot audit ds.dcp --checks all --report report.html` (the default gate is
   `--fail-on critical`; `--fail-on none` applies no gate) validates the
   options first (colliding report paths, `--allow-external-images` in CI without
   `--allow-external-images-in-ci`, unknown check, unknown or never-firing `--fail-on` token, a `--fail-on` check that
   is not part of the run, an embedding check without `--embedder`, `--embedder` with no check that
   uses it, a bad `--embed-batch` or `--similarity-threshold`: exit 2), creates the embedder (if `--embedder`) **before** the dataset is read, which runs the
   license guard against `DATACOPILOT_ALLOWED_LICENSES` (no weights are touched yet), reads the tables
   through the bounded, hash-verified store reader, and runs each check in `audit/registry.py` order.
   The embedding checks (`near_duplicates`, `leakage`, `label_noise`, `outliers`) share one embedding pass
   (`audit/embedding.py`): verified cache vectors are taken from `embeddings/<model>@<rev>/`, a
   sample is re-embedded to authenticate them, and the remaining images are read (contained in the
   dataset root, hashed against their id), batched and embedded, one shard per 4,096 vectors.
   `--checks all` includes these four only when `--embedder` is given.
   The `duplicates` check uses the authenticated hash cache unless `--no-hash-cache`
   (or `DATACOPILOT_HASH_CACHE=false`) is given.
6. Findings are collected into an `AuditRun` (`run_id` is the invocation's `request_id`) and written
   to `audits/<run_id>.json`.
7. `report/` renders HTML, JSON and SARIF: `--report` (default `./report.html`), `--json` and
   `--sarif` (default: next to the HTML file, same name, `.json` / `.sarif`).
8. One result line goes to `stdout` (`passed|failed critical=N warn=N info=N run_id=ID`). The gate
   evaluates `--fail-on`: a match sets exit code 1; the reports are still written.

Failure paths:

| Condition | Behaviour | Exit code |
|---|---|---|
| Unknown command or option; invalid `DATACOPILOT_*` value | one line on `stderr`, no log record | 2 |
| `--format` not registered, or its extra is missing | `FormatError` names the format and the extra | 1 |
| Export directory missing or unreadable; malformed COCO/YOLO/VOC/CVAT/Label Studio/ImageFolder content (invalid JSON, unsafe path, bad label line, XML with a DOCTYPE, an out-of-range coordinate) | one `FormatError` naming the file and the problem | 1 |
| Two images with identical content; no images found | `ValidationError` / `FormatError` naming the files; nothing is written | 1 |
| `export` output not empty, or `--strict` with dropped fields | `FormatError` / `ConversionLossError` listing columns and counts; nothing is written | 1 |
| Table over `DATACOPILOT_MAX_TABLE_ROWS` / `_BYTES`, wrong Parquet schema, manifest counts that do not match | `StoreError` naming the file and the limit | 1 |
| A row violates an invariant (box out of bounds, unknown category) | `ValidationError` names `ann_id`/`image_id`; nothing is written | 1 |
| An image file is missing, unreadable, not a regular file, has an invalid header or a size that differs from its row during `integrity` | `Finding(check="integrity", severity=CRITICAL)`; the run continues | 0 unless `--fail-on` matches |
| `--checks` names an unknown check; `--fail-on` is unknown, empty, a check that is not built (`sku_confusion`) or a check that is not in `--checks`; `--workers` out of range | one usage line on `stderr`; nothing is read or written | 2 |
| Images that were not examined: remote URIs, real path outside the dataset's parent directory (`--allow-external-images` opts in; refused in CI without `--allow-external-images-in-ci`), unhashable files | counted in a WARN finding, CRITICAL when more than 50 percent of the images are affected or with `--require-local-images`, so a dataset cannot pass by pointing its images elsewhere | 1 with the default gate once CRITICAL |
| Hash cache in CI (`CI` set) without `DATACOPILOT_CACHE_KEY`; directory, secret or file not private to this user; or a cached digest that fails the spot check | cache disabled (`audit_cache_untrusted`, reason `ci_without_key` and others), ignored or discarded, WARN event (and a `hash_cache_mismatch` WARN finding for a failed spot check); the audit continues | 0 unless `--fail-on` matches |
| Embedder requested without `[embed]` | the first use fails: an `embedder_error` finding (CRITICAL) with the message that names `pip install "datacopilot[embed]"`, an `embedder_failed` log event, and every image counted as not embedded | 1 with the default gate |
| Model license not in `DATACOPILOT_ALLOWED_LICENSES` | `LicenseError` names model, license and the variable; raised before the dataset is read and before any download or load | 1 |
| Unknown or malformed `--embedder`; embedding check without `--embedder`; `--embedder` without an embedding check; a test-only embedder (`hash`) with a gate and no `--allow-test-embedder`; `--require-cache-key` without `DATACOPILOT_CACHE_KEY` | one usage line on `stderr`; nothing is read or written | 2 |
| The dataset path is a symlink | `StoreError`; nothing is read | 1 |
| More than 500,000 images with an embedder | `audit_incomplete` (`too_many_images`) from each embedding check before the cache or the model is touched | 1 with the default gate |
| Images without an embedding (remote, outside the root, missing, not regular, too large, changed since ingest, undecodable, animated or multi-frame, over the pixel limit, zero vector, embedder error, time budget) | `images_not_embedded` finding of each embedding check with counts by reason. `leakage`: CRITICAL under any gate when even one image is not embedded for a reason other than the time budget (an attacker needs only one undecodable file in the test split), WARN with `--fail-on none`. `near_duplicates` and `outliers`: WARN, CRITICAL above 50 percent or with `--require-local-images`. Any finding of the check fails `--fail-on <check>` | 1 when the gate matches |
| Embedder raises, returns a wrong shape, NaN or infinity | `embedder_error` (CRITICAL; exception type only), the remaining images counted as not embedded | 1 with the default gate |
| Embedding time budget or neighbour-search budget reached, or more than 500,000 images | `audit_incomplete` (CRITICAL when a gate is applied, else WARN) | 1 with the default gate |
| Embedding cache shard or manifest fails verification, sample disagrees, or too few sampled hits are verifiable | cache ignored or discarded and recomputed; `embedding_cache_rejected`, `embedding_cache_mismatch` or `embedding_cache_unverified` (WARN) | by the gate |
| `embed`: any image without an embedding, embedder failure, or cache not writable | `error:` lines on `stderr` | 1 |
| `embed`: cache unusable (`CI` without key, `--no-embedding-cache`) | usage error, nothing is stored | 2 |
| `--fail-on` matches at least one finding (default gate: CRITICAL) | report written; `audit_completed` has `passed=false` | 1 |
| Dataset with no images | `ValidationError`: `dataset has no images`; nothing is written | 1 |
| Plugin code raises `SystemExit` or another `BaseException` (not `KeyboardInterrupt`) | the audit fails with `check '<name>' aborted the audit`; the exit status can never be 0 | 1 |
| `--report`, `--json` and `--sarif` resolve to the same file (case-insensitively on macOS and Windows) | usage error before anything runs | 2 |
| A check hits `max_check_seconds` or `max_hash_bytes` | `audit_incomplete` finding (CRITICAL when a gate is applied, else WARN) | 1 with the default gate |
| The dataset directory is read-only | reports are written first, then `audit_not_stored` (WARN) is logged; `--no-store-audit` skips storing | by the gate |
| Interrupted write of a version | temporary directory left behind and ignored by readers | 130 |

Logged events (all `INFO` unless stated): `dataset_ingested` (`format`, `image_count`,
`annotation_count`, `dataset_version`, `duration_ms`), `conversion_loss` (`WARN`;
`source_format`, `target_format`, `column`, `count`, `kind` = `dropped` or `skipped`), `dataset_exported`
(`format`, `image_count`, `annotation_count`, `dataset_version`, `dropped_count`, `skipped_count`,
`strict`, `duration_ms`), `audit_check_completed` (`check_name`, `findings_count`,
`duration_ms`), `audit_cache_unavailable`, `audit_cache_untrusted` and `audit_cache_discarded` (WARN, see
[06-cross-cutting.md](06-cross-cutting.md)), `embeddings_computed` (`model_id`, `model_revision`, `image_count`, `embedded_count`,
`cache_hits`, `missing_count`, `duration_ms`), `embedder_failed` (WARN; `error_type`, `detail`
for package errors only), `embedding_cache_disabled`, `embedding_cache_rejected`,
`embedding_cache_unverified` and `embedding_cache_discarded` (WARN, see
[06-cross-cutting.md](06-cross-cutting.md)), `model_loaded`, `model_license_shown`, `audit_completed` (`run_id`, `dataset_version`, `critical`, `warn`, `info`,
`passed`), `report_written` (`format`, `path` = the file name only, never the directory), `command_failed` (`ERROR`; `command`,
`exc_info`). Expected failures also print one `error: ...` line on `stderr`.

Status: `ingest`, `export`, `audit` (all checks but `sku_confusion`), `embed`, the
gate and the reports built.

## Flow B: propose → review export → review import → apply

**Purpose:** let a model suggest missing boxes, route them to a reviewer, and apply only accepted
ones as a new dataset version with provenance and lineage.

**Preconditions:** a `dataset.dcp/` of task `detection` exists; the `[vlm]` extra is installed on the
machine that runs `propose`; the model's license is on `DATACOPILOT_ALLOWED_LICENSES`; the reviewer can
import a CVAT for images 1.1 export.

![Flow B](../diagrams/png/flow-b.png)

The picture shows the full design. Built today: `add_box` proposals from Grounding DINO (tiny), CVAT
review, and `apply`. Consensus between models, SAM 3, Qwen3-VL, FiftyOne and the other proposal kinds
are planned ([ADR-0008](../adr/0008-propose-review-apply.md)).

Steps (built):

1. `datacopilot propose ds.dcp --proposer grounding_dino` (options: `--classes`, `--min-score`,
   `--min-score-for CLASS=SCORE`, `--time-budget`).
2. The proposer and its license are resolved before the dataset is read. Prompts are built from class
   names and aliases after normalisation (lower-case words only; a name that could rewrite the prompt is
   skipped and counted).
3. For each image (read through the audit's contained, size-capped, hash-verified reader) the model returns
   candidates. Each is validated: finite numbers, a score in (0, 1], a label that was asked for, inside
   the image (one pixel of overshoot is clipped), at least 2 pixels wide and high. A candidate below the
   class threshold, or whose class already has a box at IoU 0.5 or more in the image, or that a better
   candidate of the same class overlaps at IoU 0.5 or more, is dropped; every drop is counted by reason.
4. The set is stored as `proposals/<set_id>.json` (a content-addressed file, see
   [03-data-model.md](03-data-model.md)) bound to the dataset version and the model identity. A run that
   did not reach every image (time budget, unreadable image, proposal cap) is marked incomplete and exits 1,
   but the partial set is kept.
5. `datacopilot review export ds.dcp --proposals ID --to cvat -o DIR` writes the dataset's own boxes and
   the proposals as boxes labelled `<class>__proposed` with the attributes `proposal_id`, `score` and
   `decision`, through the CVAT writer, with the images.
6. The reviewer types `accepted` or `rejected` in `decision` (and may move a box first: that makes an
   edited proposal). A deleted box decides nothing.
7. `datacopilot review import ds.dcp DIR --proposals ID --from cvat` reads the file through the CVAT
   reader and stores a NEW set (`parent_set_id` is the reviewed set). The dataset's own layer is ignored
   and counted. Unknown, repeated or foreign ids, changed classes, invalid decisions, boxes edited outside
   the image and a file without decisions are refused as a whole.
8. `datacopilot apply ds.dcp --proposals NEWID -o NEW.dcp` writes a new dataset: the old rows plus one row
   per accepted proposal (`provenance=proposal_accepted`, id `proposal:<proposal_id>`, `source` has
   `proposal_id`, `set_id`, `model_id`, `model_revision` and `edited`; an edited box has no score), and copies
   the reviewed set and its parent into `NEW.dcp/proposals/`; `lineage` = the parent's lineage plus the parent's version. The new
   `manifest.version` is printed on `stdout`. The source directory is never written.

Failure paths:

| Condition | Behaviour | Exit code |
|---|---|---|
| Proposer unknown, option invalid, class unknown, no usable class name | usage error | 2 |
| Extra missing | error names the extra (`datacopilot[vlm]`) | 1 |
| Model license not allowed | `LicenseError` before the dataset is read and before any download or load | 1 |
| Model fails or its output is unusable | `ModelError`; nothing is stored | 1 |
| Model candidate fails validation | dropped, counted in the set and in `proposal_candidates_rejected` (WARN) | 0 |
| Run did not reach every image | set stored and marked incomplete; `apply` refuses it without `--allow-incomplete` | 1 |
| Set file tampered, unknown or not a regular file | `StoreError` | 1 |
| Set made for another dataset version, or by a model that is not the pinned revision and files | `ValidationError`; nothing written | 1 |
| Decisions file refused (see step 7; also images missing from the folder and rotated proposal boxes) | `ValidationError` or `FormatError`; no set stored | 1 |
| `apply` of a set that did not come through `review import`, is not a review of its parent, or comes from a test double (without `--allow-test-proposer`) | `ValidationError`; nothing written | 1 |
| `apply` with nothing accepted, output existing or inside the source | `ValidationError` / `StoreError`; nothing written | 1 |

Logged events: `proposer_started`, `proposer_image_skipped`, `propose_progress`,
`proposal_candidates_rejected`, `proposal_set_written`, `review_exported`, `review_imported`,
`proposals_applied`, `command_failed`; fields in [06-cross-cutting.md](06-cross-cutting.md). No class
name, prompt text, path or reviewer name is logged.

Status: built for `add_box` proposals with Grounding DINO and CVAT review; the rest of the picture is
planned.

