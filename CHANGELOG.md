# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning
once a release is tagged.

## Unreleased

### Added (split)

- `datacopilot split DS.dcp -o NEW.dcp --ratios 0.8,0.1,0.1 [--by-group KEY] [--near-dup --embedder NAME]
  [--stratify category] [--seed N] [--json PATH]` and `dc.split(...)`, returning the new `SplitReport` public
  model. Images with the same key (`group_id`, `source.<name>` or a `re:` file-name pattern that is refused
  unless it cannot backtrack badly) and, with `--near-dup`, every pair of near copies (an exact search of all
  pairs, `knn.pairs_at_least`) stay in one split. Whole groups are assigned largest first by a deterministic,
  seeded greedy rule, optionally balanced per class; the deviation per split and class is reported. The result
  is a new dataset with lineage; nothing existing is written. An image without an embedding, a search ended by the
  budget, over 2,000,000 pairs or a split equal to the current one write nothing (exit 1). The leakage check on the
  result finds no cross-split pair (tested, and in notebook 05). See [ADR-0010](docs/adr/0010-split-and-box-crops.md).

### Changed (label noise from box crops)

- `label_noise` now embeds **box crops** by default (boxes of at least `crop_min_side_px` = 32 pixels on the
  shorter side, `crop_padding` = 0.1, at most `max_crops_per_image` = 50 per image, downscaled to 512 pixels),
  cached in the authenticated embedding cache under a variant named by the crop geometry. The whole-photo rule is
  `suspect_basis = "image"`, used for classification datasets and as a fallback that is reported
  (`crop_rule_unavailable`, WARN). Findings carry `basis`; `annotations_not_scored` carries `reasons`; the new
  `skipped` reason `too_few_boxes`. The `embedder_error` and cache findings apply to crops as to photos. Finding
  ids of `suspect_class` change because the payload gained `basis`.
- Measured in `notebooks/05_split_and_crops.ipynb` on COCO128 with DINOv2-small (five seeds, 5 percent planted
  flips): precision 0.250 against 0.108 for the whole-photo rule, recall 0.470 against 0.226, false-positive rate
  0.073 against 0.097, 455 boxes scored against 296. Better, and still weak; the notebook says so.
- Embedding cache identities have an optional `variant` (its own directory, bound into the MAC); existing caches
  keep their MAC.

### Changed (install, Python 3.14)

- Nothing is on PyPI: the README, the install page and the CI example install from a clone (`uv sync`, or a
  virtual environment and `pip install -e .`) or from GitHub (`uv add git+...`); missing-extra messages name the
  extra and `uv sync --extra NAME`. `scripts/check_clone.sh` (and `pytest -m clone`) installs a fresh clone and
  runs the README quick start; it passed on Python 3.10 and 3.14.
- Python 3.14 is supported: classifier, CI test matrix, audit matrices.
- The concepts page shows how CV-Pilot records `dcp:<manifest.version>` for a `dataset.dcp` directory.

### Added (label noise)

- Audit check `label_noise` (rule `suspect_class`): with `--embedder`, a box is reported as an INFO lead when
  at most `suspect_max_support` of the nearest neighbouring photos (exact, chunked cosine search over the existing
  embeddings; no new model, no new dependency) carry its class. It only reports and never edits or proposes. It
  is deterministic, capped per photo and per class with the excess counted (`suspects_truncated`), counts boxes
  it could not score (`annotations_not_scored`) and says when it did not run (`skipped` WARN with a reason:
  `embedder_missing`, `no_embeddings`, `no_annotations`, `too_few_images`, `single_class`). It joins
  `--checks all` with an embedder and `--fail-on label_noise` is now valid. Unscored boxes carry the severity
  of images a check could not examine (WARN, CRITICAL above half of the boxes), so a check that scored under
  half of the boxes fails the default gate. Thresholds are new `CheckConfig`
  fields (`suspect_*`, `max_suspects_per_image`, `max_suspects_per_class`). The Cleanlab backend remains planned.
  See [ADR-0009](docs/adr/0009-label-noise-check.md) for what was left out on purpose.
- Notebook `04_label_noise.ipynb`: COCO128 with seeded planted class mistakes and the real DINOv2-small model.
  Measured, and weak: with default thresholds the check scores 296 of 929 boxes; recall is 0.23 over all boxes
  and 0.83 over the scorable ones, the false-positive rate 0.097 and 0.30, precision 0.11 (2.2 times the random
  base rate, per seed 1.5 to 2.9); confusions with another class of the same photo reach recall 0.09 and 0.28.
  The notebook says so in its computed verdict.
- `benchmarks/embed_checks.py` times `label_noise` on its own (16.5 s at 50,000 x 768 on 12 cores).

### Added (propose, review, apply)

- `datacopilot propose`, `review export`, `review import` and `apply`, and `dc.propose` and `dc.apply` with
  `ProposalSet`: a detection model finds boxes a dataset is probably missing, they are reviewed as a separate
  CVAT layer, and the accepted ones are written as a NEW dataset version with lineage; nothing is applied
  without a person's decision and no existing version is touched. Only `add_box` proposals exist. See
  [ADR-0008](docs/adr/0008-propose-review-apply.md) for what was left out on purpose.
- Proposal sets are one JSON file under `dataset.dcp/proposals/`, named by the digest of their content and bound
  to the dataset version and the model (id, revision, weight hashes); every candidate from a model is validated
  and counted by reason; an incomplete run is marked and refused by `apply` without `--allow-incomplete`.
- Proposer registry (`datacopilot.proposers`), the `grounding_dino` proposer (Grounding DINO tiny, Apache-2.0,
  pinned weights, configuration and tokenizer files, safetensors, no remote code) and the test-only `test_boxes`.
- New extra `vlm` (`pillow`, `safetensors`, `huggingface-hub`, `transformers`, torch); `ModelCard` gained
  `aux_sha256` and `pinned_files()`, and `dims` is optional for a detector.
- `DatasetStore.create(..., lineage=...)`, `write_proposals` and `read_proposals`.
- Public API: `propose`, `apply` and `ProposalSet` join `__all__` (the API snapshot is updated; before 1.0 a minor
  version may break the API).
- Notebook `03_propose_review_apply.ipynb`: the real model on COCO128 with a seeded held-out answer key.

### Fixed

- Reports no longer show the query string or fragment of a remote image path (a presigned URL carries a
  credential and a signature there) or its host: `display_name` keeps the last path segment only.
- The exact kNN no longer sorts every column that ties at the k-th place; a crowd of identical images costs no
  sort (the lowest indices win ties, as before).

- Security: every exporter and `review export` copied whatever file `Image.file_path` named, so a hostile `.dcp`
  could copy `~/.ssh/id_rsa` into the output. Exports now verify each image (inside the dataset's directory, no
  link to elsewhere, size cap, content equal to `image_id`) and refuse the whole export otherwise;
  `export --allow-external-images` lifts only the directory rule. `DatasetStore.read` now sets `Dataset.root`.
- Propose, review and apply (fix wave after independent reviews): `apply` needs a reviewed set (and, when
  present, its parent), refuses test doubles without `--allow-test-proposer`, pins the identity a set claims,
  keeps the model score only for unedited boxes and copies the sets into the new dataset; `review import`
  refuses missing images and rotated proposal boxes and caps the file; `propose` rejects bad threshold keys
  before the model runs, drops a phrase two classes share for both, no longer loses a photo without
  detections (the real model returns one empty label there), lowers the word threshold with `--min-score`,
  skips an image whose decoded size differs, counts merged and fragmentary labels (`ambiguous_label`) instead of
  guessing, stores scores with 4 decimals, writes non-ASCII class names safely and no longer reports a late
  `ValueError` as a bad option. Real numbers of the notebook did not change at the default threshold
  (215 proposals, 84 right, precision 0.39, recall 0.47); the 39 dropped labels are now 37 `ambiguous_label` and
  2 `unknown_label`; the sweep rows below 0.35 changed (0.20: 824 proposals, precision 0.14, recall 0.63).
- The command line prints one `error:` line (no directory, no temporary-file suffix) when the output
  folder cannot be written, instead of only a JSON log with a traceback.
- YOLO and VOC writers kept annotation order by comparing ids as text, so an image with more than ten
  boxes came back reordered and with another `manifest.version`; they now use a natural order.
- YOLO: ingest, export, ingest is now a fixed point. Boxes snapped onto an image edge drifted by up to
  0.00003 pixel on every pass until the second round trip; the reader now snaps boxes to the 7-decimal
  grid the writer prints (a documented rounding, not a loss). `manifest.version` of YOLO datasets that
  have boxes overshooting the edge therefore changes once (COCO128: `0bce2115...` to `d9105ffb...`).
- Every reader now refuses a picture in an unsupported format (`.webp`, `.tif`, `.tiff`) with an error
  naming the file; YOLO and ImageFolder used to skip and count it (`images.unsupported_format`), so
  neither the audit nor a gate saw the missing photo.

### Added

- `integrity` rule `truncated_file` (WARN): a JPEG, PNG or GIF whose last 4 KiB lack the end marker. It
  adds one small read per image (about 0.07 ms per photo on COCO128, 9 ms for all 128).
- Docs: `concepts.md` and `status.md` no longer say that the version is the same across source formats.

## 0.1.0 - 2026-10-15

First release candidate. Before 1.0 a minor version may break the public API.

### Added

- CVAT for images 1.1 XML (`cvat_xml`) and Label Studio JSON (`label_studio`) readers and writers.
  CVAT: boxes, labels with colour and type, attributes, `occluded`, `source` as provenance, task subset as
  split; polygons, polylines, points, ellipses, masks, skeletons, cuboids, tags, rotated boxes and tracks
  are counted as `skipped` (`annotations.unsupported.<shape>`), never dropped silently. Label Studio:
  `rectanglelabels` (percent of the original size, converted to pixels, rounded to 1e-6) and `choices`
  (classification), annotations as human rows and the first prediction as model rows with scores, one
  annotation per task (ground truth first, else the lowest id) with the rest counted, unsupported result
  types, rotated rectangles and remote images counted; only an allow-list of task, annotation, prediction and
  result fields is stored and re-emitted (everything else, including `completed_by`, usernames, comments and
  presigned URLs, is dropped and counted by name; task `data` fields only with `--keep-task-data`).
  Both are in the loss matrix and the round-trip goldens.
- `datacopilot ingest --strict`: fail, writing nothing, when a reader skipped or dropped anything.
- `datacopilot doctor` is a real diagnosis: versions of the package and its core dependencies, importable
  extras, validated settings (the cache key only as set or unset), the cache directory (existence,
  writability, bounded size, symlinks not followed), FAISS, GPU visibility and the registered plugin names;
  `--json` prints schema 1; exit 0 (ok or warn), 1 (a core dependency is missing), 2 (invalid settings,
  reported by `doctor` itself).
- Python API: `datacopilot.load`, `audit` and `export` over the same code as the commands, `Dataset.root`,
  `__all__` pinned by a snapshot test, and `datacopilot.deprecation.deprecated` for the one-minor-version
  deprecation policy. `python -m datacopilot` runs the command line.
- Documentation site (MkDocs Material, pinned in a `docs` dependency group, strict build, no remote assets):
  user guide, a tutorial on a generated synthetic dataset whose commands are executed by a test, a CLI
  reference generated from the Typer application (a drift test fails when it is stale) and the design
  documents. Every command block of a user page is executed by a test or marked non-executable.
- Release pipeline: `release.yml` (build job with read-only permissions and no secrets; publish job with
  trusted publishing, a protected `pypi` environment, version tags only, the exact build artifact, tag
  equal to `__version__`), pinned actions in every workflow, CI jobs for the lock file, the docs, the
  package (artifact contents, `twine check --strict`, clean-environment install on 3.10 and 3.13) and
  `pip-audit`; tests that lint the workflows and assert the wheel and sdist contents; the runbook
  `docs/engineering/release.md`.
- Package metadata: classifiers, project URLs, authors and keywords; the sdist is an allow-list.
- Element, attribute and depth limits in the XML parser (1,000,000 elements per document, 256 attributes per
  element, a per-call depth limit), `parse_json(unique_keys=True)`.

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

- Release readiness: the empty extras (`vlm`, `sam3`, `cleanlab`, `fiftyone`, `cvat`) are no longer published
  (`embed` is the only extra); the README is correct on the package index (absolute links, `pip install
  datacopilot`, one marked `Release status` line that the release commit removes and that
  `scripts/check_release_tag.py` refuses to tag with), and tests keep it that way; the tag checker rejects
  leading zeros; the runbook verifies the published files with `pip download` and the recorded digests; docs
  drift fixed (seven checks, the core packages, `uv sync`, the settings error behaviour) and the shared
  system-context diagram is generated from a source in this repository.
- `audit` leaves rows whose provenance is `model` out of every check unless `--include-model-annotations` (API:
  `include_model_annotations=True`) is given, and reports the number in the summary, the HTML report and a
  note on standard error. `AuditRun.raise_for_gate()` and `GateFailedError` let API callers turn a failed
  gate into an exception; the docs no longer claim the API raises on its own. The callable
  `datacopilot.audit` forwards its signature and pickles; the API surface snapshot (`api_surface.json`) now
  pins field types, defaults and exception hierarchies, and the audit golden files carry a `{{VERSION}}`
  placeholder (`tests/golden/expected/`) with a separate test that the reports name the running version.
- Label Studio reader: a skipped entry with no results is no longer counted as zero (a normal export with a
  cancelled annotation or a second annotator without results made `ingest` exit 1 without a message and made
  `load` leak a validation error); `ConversionReport` drops zero counts and a reader's row-level validation
  error is a `FormatError`; a null prediction score is no score; the task type follows the human rows;
  predictions of an unlabeled task are counted instead of read as rows; the writer keeps empty predictions,
  the original document files (`annotations/<name>.json`) and the form of the image reference, and counts
  `categories.unused` and renamed images. CVAT: one document per original file, image `<attribute>` children
  counted (`images.attributes`), labels derived from boxes bounded like the others.
- `doctor` prints several lines instead of `ok`; the first line is still the status. `Dataset` gained
  an optional `root` (not part of equality or of the content version).
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

- Label Studio and CVAT data handling: allow-listed fields only (personal data and presigned URLs never reach
  the store, the reports or re-exported files), bounded report column names (`safe_key`) and at most 200
  distinct columns, every illegal-character replacement in the CVAT writer counted and the written image name
  equal to the file name, reserved `format` and `file_name` attributes can no longer replace the metadata of
  a row (CVAT, COCO), images are opened once, without following a symlink, and their size and digest come
  from the same descriptor, a single huge start tag is refused after about 1 MiB instead of being built in
  memory.
- Release pipeline hardening: the build refuses a commit that is not an ancestor of `origin/main`; the build
  backend and its dependencies are exact pins; the publish job is download, verify (exact file set and the
  digest the build job exported) and publish, nothing else; the audit tool is the locked one and runs for
  Python 3.10 and 3.13 in CI, in the release build and nightly (`audit.yml`); the runbook makes the
  repository-settings evidence (environment reviewers, tag ruleset, deployment tag policy, trusted
  publisher) a mandatory one-time step; workflow lint rules run against a hostile fixture; the wheel is
  compared with a committed manifest and both artifacts are scanned for more secret formats.
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
  `label_noise`, `split`, `propose`, `review` and `apply` are not implemented yet.
- A Label Studio export does not carry the labelling configuration, so classes that no row uses are not
  carried (the writer counts them as `categories.unused`); CVAT label attribute definitions beyond a name are counted, not kept.
- Before 1.0 a minor version may break the public API (`datacopilot.__all__`); the changelog says so.
