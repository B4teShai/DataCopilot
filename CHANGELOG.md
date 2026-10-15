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

- Embedding cache (`embed/cache.py`): per-model directory of Parquet shards plus a manifest
  (identity, dataset version, weight hashes, shard sha256 and an HMAC); bounded from the Parquet
  footer before any row is decoded, digest verified on the bytes that are parsed, atomic and
  resumable, `float16` or `float32` storage with a documented rounding. A forged, truncated, stale or
  malformed shard is ignored and counted by reason, never trusted.

- Embedding backends (`embed/backends.py`): `dinov2_small`, `dinov2_base` and `siglip2_base` through a
  lazy `transformers` import. Revision pinned to a full commit, only the card's safetensors files and
  `config.json` downloaded, sha256 of every weight file verified before it is loaded, weight hashes
  logged (`model_loaded`), `trust_remote_code=False`, `use_safetensors=True`, bounded download
  timeouts, CPU fallback, deterministic Pillow preprocessing. Decoding is bounded: only JPEG, PNG, GIF
  and BMP reach a decoder, the declared pixel count is checked against 64,000,000 before any pixel is
  decoded, truncated files and decompression bombs are errors.
- `hash`: a dependency-free test embedder (SimHash of byte shingles) for tests and benchmarks.
- A `real_model` pytest marker (off by default) for smoke tests with real weights and real FAISS.

- Exact cosine kNN (`embed/knn.py`): blocked NumPy matrix multiply with memory bounded by a block
  budget (256 MiB), deterministic order (similarity descending, lower index first), caps of 500,000
  rows and 100 neighbours, a deadline between blocks, and an exact FAISS `IndexFlatIP` adapter
  (`[embed]`) used above 20,000 rows when installed and equal within 1e-4 in similarity.
  Measured: 50,000 x 768, k = 10: 15.9 s, 0.83 GB peak RSS (NumPy, Apple M4 Pro).

- Embedding checks `near_duplicates` (WARN clusters), `leakage` (CRITICAL cross-split near-duplicates,
  searched only across splits, and `group_id` overlap) and `outliers` (INFO, per-class robust
  z-score of the mean neighbour distance), fed by `audit/embedding.py`: authenticated cache with a
  sampled re-embedding check, reads confined to the dataset root, files must hash to their image id,
  batch and byte budgets, per-image decode isolation. They fail closed: a missing embedder, an
  embedder error, images without embeddings, an incomplete run and a distrusted cache are findings
  of every embedding check. Optional perceptual-hash prefilter (`near_duplicate_phash_max_distance`).
  `--checks all` includes the embedding checks only with an embedder; `registry.EMBEDDING_CHECKS`.

- Cache verification rule: at least half of the sampled cached vectors must be re-verifiable
  (sample 1 percent, 10 to 64, cosine at least 0.999) or no cached vector is trusted; the new
  `embedding_cache_unverified` finding and event report it.

- `datacopilot audit --embedder NAME [--embed-batch N] [--similarity-threshold X]
  [--no-embedding-cache]` and `datacopilot embed DATASET --embedder NAME`. `--fail-on leakage` is now
  valid (with `--embedder`). The license guard (`DATACOPILOT_ALLOWED_LICENSES`) runs before the
  dataset is read and before any download or load. Events `embeddings_computed`, `model_loaded`,
  `model_license_shown`, `embedder_failed` and the `embedding_cache_*` family.

- `benchmarks/embed_checks.py`; measured on synthetic 50,000 x 768 float16 embeddings (Apple M4 Pro,
  Python 3.10): `near_duplicates` 16.2 s, `leakage` 5.9 s, `outliers` 0.4 s, cache write 1.3 s
  (72.8 MB), verified read 0.7 s, peak RSS 1.27 GB. GPU embedding time, the sample re-embedding of a
  warm cache and FAISS were not measured.
- ADR-0007 (embedding cache and index) and updated diagrams.

- Embedding cache format 2: each shard row carries the full sha256 of the image bytes; the audit
  hashes the file of every cache hit and recomputes vectors whose file changed (the cache is bound
  to the dataset rows, which do not see the files), treats unreadable cached images as unchecked,
  and keeps `image_id` out of the comparison (it is a 64-bit label). The manifest is parsed once
  with duplicate keys refused, the MAC key has a domain label and the identity names the
  preprocessing version. New finding `embedding_cache_content_changed`.
- `leakage`: under a gate any image not embedded (other than by time budget) is CRITICAL, so one
  unembeddable test image cannot pass the default gate; `images_without_group` (INFO) shows images
  that cannot take part in `group_overlap`.

- Decoding: 16-bit and float images are scaled instead of clipped, alpha is composited on white,
  unknown pixel modes and animations are refused (`multi_frame`), missing Pillow is a `ModelError`
  with the install hint.
- `near_duplicates`: a cluster across splits is CRITICAL (like exact duplicates); the perceptual-hash
  prefilter never drops a cross-split pair; pairs that decide a verdict (reported and borderline,
  at most 50 each) are re-embedded fresh when a cached vector is involved
  (`embedding_pair_contradicted`); a pair cap (2,000,000) ends a hostile search with a CRITICAL
  `audit_incomplete` (`pair_cap`).
- The index re-normalises its working copies and callers compare with an epsilon, so a threshold of
  1.0 or 0.9995 holds for identical embeddings.

- Model loading pins `config.json` (sha256, part of the cache identity) and the size of each weight
  file (checked before hashing); `HF_HUB_DISABLE_TELEMETRY=1` by default; a custom `HF_ENDPOINT` is
  refused in CI unless `DATACOPILOT_ALLOW_HF_ENDPOINT=1`. The `[embed]` floors are the versions the
  suite and the real-model runs used.
- Embedding pipeline: a dataset over the neighbour-search cap is refused before the cache or the
  model is touched; the cache keeps only the rows of the dataset's ids and its row cap follows the
  dataset size; every undecodable image of a batch is dropped in one pass; plugin `model_id` and
  `revision` are validated and `AuditRun.embedder` is bounded.
- `leakage`: group labels are compared after NFKC, case folding and trimming; findings show a hashed
  `group_ref` and the label only with `--show-group-labels`.
- `--allow-test-embedder` (the `hash` embedder is test-only: warned about, labelled
  `hash@1-test-only` in reports and refused for any gate without it), `--require-cache-key`, and a
  symlinked dataset directory is refused. A cache written in an older format is stale, not rejected.
- On macOS the exact NumPy index is used when the tensor library is already loaded, because importing
  `faiss` after it aborts the interpreter (two OpenMP runtimes).

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
- The embedding cache is bound to the dataset version, so a new version recomputes every embedding;
  `label_noise`, CVAT XML and Label Studio formats, `split`, `propose`, `review`, `apply` and the
  Python API are not implemented yet.
