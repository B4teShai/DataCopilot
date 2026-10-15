# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning
once a release is tagged.

## Unreleased

### Added

- `datacopilot audit`: runs the CPU checks `integrity` (existence, readable header, declared size,
  EXIF orientation; no full decode), `duplicates` (streamed sha256, CRITICAL across splits),
  `imbalance` (per-split long-tail ratio, effective class number, missing and unused classes) and
  `boxes` (out of bounds, zero area, tiny, extreme aspect, duplicate and conflicting boxes), stores
  the run as `audits/<run_id>.json` and returns the `--fail-on` verdict as the exit code (0, 1, 2).
  Options: `--checks`, `--fail-on`, `--report`, `--json`, `--sarif`, `--workers`,
  `--hash-cache/--no-hash-cache`, `--allow-external-images`.
- JSON, static HTML (score card, class histogram, split-by-class table, findings) and SARIF 2.1.0
  reports; golden tests pin all three formats.
- `Finding`, `AuditRun` and `Summary` models, the `datacopilot.checks` plugin group, `iou_xyxy`.
- Events `audit_check_completed`, `audit_completed`, `report_written`, `audit_cache_unavailable`,
  `audit_cache_untrusted`, `audit_cache_discarded`.
- Settings `DATACOPILOT_WORKERS`, `DATACOPILOT_CACHE_DIR`, `DATACOPILOT_HASH_CACHE`.
- `benchmarks/audit_cpu.py`; measured on a 50,000-image, 500,000-box dataset: 37.6 s, 1.35 GB RSS.
- `datacopilot ingest`: read a COCO, YOLO, Pascal VOC or ImageFolder dataset into a versioned
  `dataset.dcp/` directory (Parquet tables plus `manifest.json`) and print `manifest.version`.
  Options: `--format`, `-o`, `--group-by REGEX`, `--name`.
- `datacopilot export`: write a `dataset.dcp/` back out as COCO, YOLO, VOC or ImageFolder, with
  `--strict` to fail when any field cannot be carried over. `datacopilot formats` lists the
  registered formats.
- Domain schema (`Category`, `Image`, `Annotation`, `DatasetManifest`) with the invariants of
  the data model, box geometry conversions, and an order-independent content hash with lineage.
- Atomic, append-only dataset store: refuses to overwrite, snapshots each version, verifies the
  content hash on read, and bounds table size, row count and schema before reading.
- Format registry on the `datacopilot.formats` entry-point group, and a `ConversionReport` that
  counts every dropped or skipped value; the cross-format loss matrix is documented and tested.
- Events `dataset_ingested`, `dataset_exported` and `conversion_loss`.
- Settings `DATACOPILOT_MAX_TABLE_ROWS` and `DATACOPILOT_MAX_TABLE_BYTES`.

- Model registry (`models/registry.py`): cards pinned to a full commit hash and to the sha256 of
  each safetensors file for `dinov2_small`, `dinov2_base` and `siglip2_base`; license guard
  `get_model` / `require_license` against `DATACOPILOT_ALLOWED_LICENSES` (default
  `Apache-2.0,MIT,BSD-3-Clause`; an empty or malformed list is a configuration error, exit 2);
  `weight_hash`. `LicenseError` and `ModelError`.
- Embedder protocol and registry (`embed/`, entry-point group `datacopilot.embedders`): embedders
  receive encoded bytes, never paths; a first-party model is license-checked from its card before its
  module is imported, any plugin before `create`.
- `[embed]` extra declared (`pillow`, `safetensors`, `huggingface-hub`, `transformers`, `faiss-cpu`
  and the tensor library); the core install is unchanged. `THIRD_PARTY_MODELS.md` lists the three models.

### Changed

- Core dependencies `pyarrow`, `numpy` and `jinja2` added (development: `hypothesis`); see ADR-0003.
- Expected failures print one `error:` line on `stderr` in addition to the `command_failed`
  log record.
- `audit` now defaults to `--fail-on critical` (`--fail-on none` applies no gate) and refuses an empty
  dataset. Images that could not be examined (remote, outside the dataset's parent directory,
  unhashable) are a WARN finding and CRITICAL above 50 percent or with `--require-local-images`.
  `content_changed`, a duplicate group between `unassigned` and a real split and an unrecognised
  image format are CRITICAL; `pairwise_skipped` is a WARN. `--report`, `--json` and `--sarif` must be
  different files; `--allow-external-images` is refused in CI without `--allow-external-images-in-ci`.
- Reports: HTML and SARIF sort by severity before their caps and count what they omit; SARIF
  `automationDetails.id` is the constant `datacopilot/audit/` with the run id as `correlationGuid`
  and file locations use the `DATASET_IMAGES` base id; the HTML headline says "no gate applied"
  without a gate; reports are written before the run is stored (`--no-store-audit`, read-only
  datasets tolerated).
- New setting `DATACOPILOT_CACHE_KEY`, the `CI` environment variable, and events `audit_not_stored`
  and `audit_nofollow_unavailable`.

### Security

- Audit hardening: the hash cache is authenticated (HMAC), ownership-checked and spot-verified;
  image reads are confined to the dataset root (symlinks resolved, `O_NOFOLLOW`, regular files
  only, size-capped); reports escape every dataset string, carry a strict Content-Security-Policy
  and contain no absolute paths or full file digests.
- Untrusted dataset files are bounded and contained: paths cannot leave the dataset root, XML
  documents with a DOCTYPE or entities are rejected, and JSON, XML and `data.yaml` input is
  size- and depth-limited.
- Gate fail-open fixes: plugin `SystemExit`/`BaseException` fails the audit, `main()` defaults to
  exit 1, error text is cleaned (no `::` workflow commands, no ANSI).
- Hash cache: out-of-band MAC key, disabled in CI without one, file identity (ctime, inode, device)
  in the key, 64 MiB cap that stops appending, streamed bounded load.
- Bounded JPEG fill and header walks, BMP header sizes, descriptor-based regular-file reads for
  manifests, tables, audit files and images, a bounded future window with time and byte budgets,
  string length limits on paths, names and ids.

### Known limitations

- Two images with identical bytes cannot be ingested together (`image_id` is the primary key).
- Embedding-based audit checks, CVAT XML and Label Studio formats, `split`, `propose`, `review`, `apply` and the
  Python API are not implemented yet.
