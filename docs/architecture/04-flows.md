# 04 Flows

Two flows cover the product: Flow A checks a dataset, Flow B proposes missing boxes and applies the
accepted ones. Exit codes are `0` success, `1` the command ran and failed (including a gate that
matched), `2` usage or configuration error, `130` interrupted
([errors and logging](../engineering/errors-and-logging.md)). Every failure is logged once, at the
command-line boundary, as `command_failed`.

## Flow A: ingest, audit, report, exit code

**Purpose:** turn a raw export into a `dataset.dcp/` version and decide, in CI or on a laptop, whether
the dataset is fit to train on.

**Needs:** the export exists and matches `--format`; its images are readable; for `--embedder`, the
`embed` extra is installed and the model's license is allowed.

![Flow A: ingest, audit, report, exit code](../diagrams/png/flow-a.png)

*A dataset is ingested once, then exported or audited.*

1. `datacopilot ingest ./export --format coco -o ds.dcp` checks options (the output must end in
   `.dcp`) and loads settings. A bad option or `DATACOPILOT_*` value is exit 2.
2. `convert.ingest` finds the reader by name in the `io/` registry. The reader yields rows, keeps
   unmapped fields in `source` and computes `image_id` from file bytes.
3. Rows are validated (invariants 1 to 6). The store writes `versions/<version>/`, then the top-level
   tables and `manifest.json`. The version is printed on `stdout`.
4. `datacopilot export ds.dcp --format yolo -o out [--strict]` reads the dataset through the bounded,
   hash-verified store, writes into a temporary folder, and logs one `conversion_loss` per dropped
   column. With `--strict` and any loss it fails and writes nothing. Otherwise it renames the folder
   into place and prints its path.
5. `datacopilot audit ds.dcp --report report.html` checks every option first (exit 2 on a problem). It
   creates the embedder, if any, before reading the dataset, which runs the license check. It reads the
   tables and runs each check in registry order. The embedding checks share one embedding pass.
6. Findings become an `AuditRun`, written to `audits/<run_id>.json`.
7. `report/` writes HTML, JSON and SARIF.
8. One line goes to `stdout`: `passed|failed critical=N warn=N info=N run_id=ID`. The gate evaluates
   `--fail-on`. A match is exit 1, and the reports are still written.

Main failure paths:

| Condition | Behaviour | Exit |
|---|---|---|
| Unknown option, invalid `DATACOPILOT_*` value | One line on `stderr`. | 2 |
| Format not registered, missing input, malformed file, unsafe path | One error naming the file or format. Nothing written. | 1 |
| Two images with identical bytes, no images, a row that breaks an invariant | An error naming the rows. Nothing written. | 1 |
| `export` output not empty, or `--strict` with losses | An error listing columns and counts. Nothing written. | 1 |
| Table over a limit, wrong schema, counts that do not match | `StoreError` naming the file and limit. | 1 |
| An image missing, unreadable or with a bad header | A CRITICAL `integrity` finding. The run continues. | 0 unless the gate matches |
| Images not examined (remote, outside the folder, unhashable) | A WARN finding, CRITICAL above half the images. | 1 once CRITICAL |
| Cache untrusted or failing its spot check | Cache ignored or discarded, WARN. The run continues. | by the gate |
| Embedder missing its extra, failing, or a license not allowed | A CRITICAL `embedder_error`, or `LicenseError` before any download. | 1 |
| Images that could not be embedded | `images_not_embedded`. For `leakage` CRITICAL under any gate. | by the gate |
| Time or byte budget reached | `audit_incomplete`, CRITICAL when a gate applies. | 1 with the default gate |
| Plugin raises `SystemExit` or other `BaseException` | The audit fails. The status is never 0. | 1 |
| Dataset folder read-only | Reports written first, then `audit_not_stored` WARN. | by the gate |
| Interrupted version write | Temporary folder left behind and ignored. | 130 |

Details of the audit rules are in the [Audit in CI](../guide/audit-ci.md) and
[Embeddings](../guide/embeddings.md) guides.

## Flow B: propose, review, apply

**Purpose:** let a model suggest missing boxes, send them to a reviewer, and apply only accepted ones
as a new dataset version with provenance and lineage.

**Needs:** a `dataset.dcp/` of task `detection`; the `vlm` extra; an allowed model license; a reviewer
who can open CVAT for images 1.1 files.

![Flow B: propose, review, apply](../diagrams/png/flow-b.png)

*A person decides in CVAT, and apply writes a new dataset.*

1. `datacopilot propose ds.dcp` (options `--classes`, `--min-score`, `--min-score-for`,
   `--time-budget`). The proposer and its license are resolved before the dataset is read. Prompts come
   from class names and aliases after normalisation.
2. For each image, read through the audit's contained, hash-verified reader, the model returns
   candidates. Each is validated and dropped, with a counted reason, if it is invalid, below its
   threshold, or already covered by an existing or better box.
3. The set is stored as `proposals/<set_id>.json`, bound to the dataset version and the model. A run
   that did not reach every image is marked incomplete and exits 1, but the partial set is kept.
4. `datacopilot review export ds.dcp --proposals ID --to cvat -o DIR` writes your boxes and the
   proposals (class `<class>__proposed`, with `proposal_id`, `score` and `decision`) as CVAT files with
   the images.
5. The reviewer types `accepted` or `rejected` in `decision`. A moved box becomes an edited proposal. A
   deleted box decides nothing.
6. `datacopilot review import ds.dcp DIR --proposals ID --from cvat` reads the file through the CVAT
   reader and stores a NEW set whose parent is the reviewed one. Your own layer is ignored and counted.
   A file with anything unexpected is refused whole.
7. `datacopilot apply ds.dcp --proposals NEWID -o NEW.dcp` writes a new dataset: the old rows plus one
   row per accepted proposal (`provenance=proposal_accepted`). The reviewed set and its parent are
   copied in. The lineage is the parent's plus the parent's version. The new version is printed. The
   source is never written.

Main failure paths:

| Condition | Behaviour | Exit |
|---|---|---|
| Unknown proposer, bad option, unknown class | Usage error. | 2 |
| Extra missing | Error naming `datacopilot[vlm]`. | 1 |
| License not allowed | `LicenseError` before the dataset is read or anything is downloaded. | 1 |
| Model fails or returns unusable output | `ModelError`. Nothing stored. | 1 |
| A candidate fails validation | Dropped and counted. | 0 |
| Run did not reach every image | Set stored, marked incomplete. `apply` refuses it by default. | 1 |
| Set tampered with, or made for another version or model | `StoreError` or `ValidationError`. Nothing written. | 1 |
| Decisions file refused | `ValidationError` or `FormatError`. No set stored. | 1 |
| `apply` of a set that skipped `review import`, a test double, or nothing accepted | `ValidationError`. Nothing written. | 1 |
| `apply` output exists or is inside the source | `StoreError`. Nothing written. | 1 |

Events are listed in [06 Cross-cutting](06-cross-cutting.md#logging).
