# 06 Cross-cutting concerns

Contracts live in [../engineering/](../engineering/README.md). This document lists only
what is specific to DataCopilot.

## Logging

Contract: [errors-and-logging.md](../engineering/errors-and-logging.md) and
[ADR-0002](../adr/0002-logging.md). `service` is `datacopilot`; `request_id` is one UUID
per CLI invocation and doubles as `AuditRun.run_id` (built, tested through the CLI).

| Event (`message`) | Level | Fields | Status |
|---|---|---|---|
| `doctor_completed` | INFO | `status`, `package_version`, `python_version` | built |
| `command_failed` | ERROR | `command`, `exception` (from `exc_info`) | built |
| `dataset_ingested` | INFO | `format`, `image_count`, `annotation_count`, `dataset_version`, `duration_ms` | built |
| `conversion_loss` | WARN | `source_format`, `target_format`, `column`, `count`, `kind` (`dropped` or `skipped`) | built |
| `dataset_exported` | INFO | `format`, `image_count`, `annotation_count`, `dataset_version`, `dropped_count`, `skipped_count`, `strict`, `duration_ms` | built |
| `audit_check_completed` | INFO | `check_name`, `findings_count`, `duration_ms` | built |
| `audit_check_skipped` | WARN | `check_name`, `reason` (for example `extra_missing`) | planned |
| `audit_cache_unavailable` | WARN | `reason` (exception class name; the hash cache directory cannot be created) | built |
| `audit_cache_untrusted` | WARN | `reason` (`ci_without_key`, `directory_not_private`, `secret_not_private`, `file_not_private`, `file_too_large`) | built |
| `audit_not_stored` | WARN | `reason` (exception class; the dataset is read-only, the reports were written) | built |
| `audit_nofollow_unavailable` | WARN | none (the platform lacks `O_NOFOLLOW`) | built |
| `audit_cache_discarded` | WARN | `sampled`, `mismatched` (the spot check found a wrong cached digest) | built |
| `embeddings_computed` | INFO | `model_id`, `model_revision`, `image_count`, `embedded_count`, `cache_hits`, `missing_count`, `duration_ms` | built |
| `embedding_cache_disabled` | WARN | `reason` (`ci_without_key`, `directory_not_private`, `secret_not_private`, `cache_directory_unavailable`) | built |
| `embedding_cache_rejected` | WARN | `state`, `reasons` (rejected shards or manifest by reason code, for example `bad_digest`, `bad_mac`) | built |
| `embedding_cache_unverified` | WARN | `reason` (`too_few_verifiable`), `sampled`, `verified` (fewer than half of the sampled cached vectors could be re-read and re-embedded; none were trusted) | built |
| `embedding_cache_discarded` | WARN | `sampled`, `mismatched` (a re-embedded vector disagreed with the cache) | built |
| `model_loaded` | INFO | `model_id`, `model_revision`, `device`, `weights_sha256`, `duration_ms` | built |
| `embedder_device_fallback` | WARN | `model_id`, `reason` (exception class name) | built |
| `audit_completed` | INFO | `run_id`, `dataset_version`, `critical`, `warn`, `info`, `passed` | built |
| `report_written` | INFO | `format`, `path` (file name only, never the directory) | built |
| `split_completed` | INFO | `parent_version`, `dataset_version`, `train`, `val`, `test`, `group_count` | planned |
| `proposer_started` | INFO | `proposer`, `model_id`, `model_revision`, `image_count` | planned |
| `proposal_set_written` | INFO | `set_id`, `proposal_count`, `high_confidence_count` | planned |
| `proposal_candidates_rejected` | WARN | `reason`, `count` | planned |
| `review_exported` / `review_imported` | INFO | `target`, `set_id`, counts by status | planned |
| `proposals_applied` | INFO | `parent_version`, `dataset_version`, `applied_count` | planned |
| `model_license_shown` | INFO | `model_id`, `license`, `gated` | built |
| `plugin_rejected` | WARN | `group`, `name`, `license`, `reason` | planned |

Rules specific to this project: no image paths that contain user names in log fields
(log `image_id`, not `file_path`); no prompt text in logs (log `prompt_hash`); no
reviewer names (log `reviewer_id` only when it is an opaque id).

## Configuration

Keys and precedence are in [05-runtime.md](05-runtime.md#configuration). Built:
`DATACOPILOT_ENV`, `DATACOPILOT_LOG_LEVEL`, `DATACOPILOT_MAX_TABLE_ROWS`, `DATACOPILOT_MAX_TABLE_BYTES`,
`DATACOPILOT_WORKERS`, `DATACOPILOT_CACHE_DIR`, `DATACOPILOT_HASH_CACHE`, `DATACOPILOT_ALLOWED_LICENSES` (each
reaches its code path in a CLI test). Planned: `DATACOPILOT_VLLM_URL`, and the secret
variables `DATACOPILOT_HF_TOKEN`, `DATACOPILOT_CVAT_TOKEN`, `DATACOPILOT_VLLM_API_KEY`.
Validation happens in `settings.py` before logging is configured; an invalid value is exit
code 2.

## Errors and exit codes per command family

| Command family | 0 | 1 | 2 | 130 |
|---|---|---|---|---|
| `--version`, `doctor` (built) | printed | internal error (`command_failed`) | bad `DATACOPILOT_*` value, unknown option | Ctrl-C |
| `ingest`, `export` (built) | converted; `export --strict` with zero dropped fields | unreadable or malformed input, invariant violation, identical image content, non-empty output, over-limit table, `--strict` with dropped fields, unknown format | bad option (`--format` missing, output not `*.dcp`, invalid `--group-by`), bad env | Ctrl-C; no partial version left visible |
| `audit` (built) | run completed and gate passed (or no `--fail-on`) | `--fail-on` matched; dataset unreadable; report not writable; embedder license refused; embedder missing its extra or failing (a CRITICAL finding) | bad option (unknown check, bad or never-firing `--fail-on`, `--workers` out of range, embedding check without `--embedder`, bad `--embedder`, `--embed-batch` or `--similarity-threshold`), bad env | Ctrl-C |
| `embed` (built) | every image embedded and stored | an image without an embedding, embedder failure, license refused, cache not writable, unreadable dataset | bad option, cache unusable (`CI` without key) | Ctrl-C |
| `split` (planned) | new version written | ratios do not sum to 1, impossible stratification, store error | bad option | Ctrl-C |
| `propose`, `review`, `apply` (planned) | set written / exported / applied | model or integration error, license refused, stale `target_ann_id`, GPU OOM | bad option | Ctrl-C |

Exception classes (all subclasses of `DataCopilotError` except `SettingsError`, which is a
`ValueError` handled before logging starts): `SettingsError`, `FormatError`, `ValidationError`,
`StoreError`, `ConversionLossError`, `LicenseError`, `ModelError` (built); `IntegrationError`
(planned). Each message names the offending input (format, path, id, variable).

## Security and threat model

Assets: human labels (the only irreplaceable data), image files (often customer data),
credentials (Hub, CVAT, cloud, vLLM), the integrity of `manifest.version` as a
reproducibility record, the CI gate's verdict.

Trust boundaries: (1) files read from disk or object storage, including dataset exports
and `dataset.dcp/` directories produced elsewhere; (2) third-party plugin packages; (3)
model weights and model output; (4) network services (Hub, vLLM, CVAT); (5) the CI
environment, where logs are public artifacts.

| Threat | Mitigation | Status |
|---|---|---|
| Malicious export file (XML entity expansion, path traversal in file names, oversized or deeply nested JSON/XML, regex backtracking) | XML is parsed from `expat` events with every DOCTYPE and entity declaration rejected and nesting limited to 32 levels; every path resolved and kept inside the dataset root (symlinks included) before reading or writing; size limits per file (JSON 256 MiB, XML 8 MiB, text 8 MiB, `data.yaml` 1 MiB and 4096 characters per line, no regular expressions in its parser); untrusted `source` content written to XML is limited to valid element names, legal characters and bounded depth and size, and everything skipped is counted | built |
| Pickled or executable payloads in weights | Cards pin a full commit and the sha256 of each `.safetensors` file and cannot name a pickle checkpoint; only those files and `config.json` are downloaded (`allow_patterns`); each file is hashed and compared before the deserialiser sees it; `trust_remote_code=False`, `use_safetensors=True`, load from the verified local directory with `local_files_only`; hashes recorded in `model_loaded` and the cache manifest; hub error text is never copied into messages; download timeouts bounded | built (fake-library tests; real DINOv2 and SigLIP2 smoke runs behind the `real_model` marker) |
| Forged or poisoned embedding cache hides leakage from the gate | `embed/cache.py` and `audit/embedding.py` treat `dataset.dcp/embeddings/` as untrusted: bounded from metadata before decoding, per-shard sha256 on the parsed bytes, HMAC over the manifest bound to model, revision, dimensions, dtype, weight hashes and dataset version (key: `DATACOPILOT_CACHE_KEY`, or a private local secret; in CI without a key the cache is neither read nor written). Every run re-embeds a sample of cache hits chosen from `sha256(run_id : image_id)` with a per-run random `run_id` (1 percent, 10 to 64): one vector with cosine below 0.999 to the fresh one discards the whole cache and recomputes. **Verified-fraction rule:** if fewer than half of the sampled hits can be verified (image unreadable, embedder failing, time budget), no cached vector is trusted for this run: they are dropped, everything is recomputed, and images that cannot be read are reported as not embedded (WARN, CRITICAL above `max_unchecked_fraction` or with `--require-local-images`) next to an `embedding_cache_unverified` WARN. A rejected shard or manifest is an `embedding_cache_rejected` WARN. Residual risk: a holder of the MAC key can forge the vectors outside the sample; the key is the trust anchor | built |
| Model hallucinations written into labels | proposals only; nothing auto-applied; consensus gating; review in an external tool; `provenance=proposal_accepted` on every applied row; lineage allows rollback | planned |
| Prompt injection through category names or aliases used as prompts | prompts are built from an allow-listed character set; prompt hash recorded; model output validated as boxes and class ids only, never executed | planned |
| License contamination | allow-list in the model registry; copyleft libraries only behind an extra with lazy import and a user-facing notice; copyleft-derived code in separate packages; CI dependency check that the core lock has no copyleft package | planned (policy fixed by ADR-0001) |
| Secrets and markup in logs or reports | key-based redaction (built); secrets only via environment; no free-text logging of headers; reports contain no environment values; every dataset string in a report is cleaned (control, bidirectional, zero-width characters shown as escapes, bounded length) and then escaped by the serialiser: Jinja2 autoescaping (sandboxed), ASCII JSON for JSON and SARIF; HTML has no script, no inline event or `style` attributes, no links or remote resources and a CSP meta tag that allows only the one inline stylesheet by hash; SARIF uses relative, percent-encoded locations and `[a-z0-9_]` rule ids | built |
| Personal data in logs | log `image_id` not `file_path`; no reviewer names; `report_written` logs the file name only; audit events carry counts and ids only | built for audit, planned for the rest |
| Plugin shadowing or malicious entry point | duplicate names fail; `PluginSpec` validated (name must match the entry point); plugin output validated against the schema | built for formats |
| Resource exhaustion through a crafted `dataset.dcp` table (a small, highly compressed Parquet file that expands to gigabytes, or a manifest that lies about its size) | Before any row is decoded, the Parquet footer is checked: file size at most 1 GiB, `num_rows` at most `DATACOPILOT_MAX_TABLE_ROWS` (default 5,000,000), summed uncompressed row-group size at most `DATACOPILOT_MAX_TABLE_BYTES` (default 4 GiB), and the Arrow schema must equal the expected one. Rows are converted in batches of 10,000. `manifest.image_count` and `annotation_count` must equal the actual row counts. Violations are a `StoreError` naming the file and the limit (exit code 1); an invalid limit variable is exit code 2 | built |
| Poisoned hash cache hides duplicates, leakage or changed files from the gate | The cache (`audit/hashcache.py`) is untrusted input. Directory, secret and file must be regular entries owned by the current user, opened with `O_NOFOLLOW`, without group or world write (the secret without any group or world access), else the cache is ignored with one `audit_cache_untrusted` WARN (constant message, `reason` code, no paths). Each line carries an HMAC-SHA256 tag; the MAC key comes from `DATACOPILOT_CACHE_KEY` (out of band, never stored next to the cache) or, for local interactive use only, from a random `secret.key` beside the cache. That in-directory secret does not protect against a restored cache (it travels with it), so when `CI` is set and no key is given the cache is disabled (`ci_without_key`). Keys cover path, size, mtime, ctime, inode and device taken from `fstat` of the descriptor that is hashed, which `hash_file` re-checks (`changed_during_audit` otherwise). A deterministic-random sample of hits (1 percent, 10 to 1,000) is re-hashed every run; one mismatch discards the cache and adds a `hash_cache_mismatch` WARN finding. A cache over 64 MiB is untrusted and never appended to; lines are read 256 bytes at a time; a zero-byte or wrong-sized secret is recreated. CI gates should still prefer `--no-hash-cache` | built |
| Audit reads or discloses files outside the dataset, or passes without examining them | Image paths are classified in `audit/paths.py`: remote URIs are never fetched; relative paths must be free of `..`, backslashes and NUL and resolve inside the dataset root; absolute paths and symlinks are resolved and, unless `--allow-external-images` is given (refused in CI without `--allow-external-images-in-ci`, and always warned about), a real path outside the root is neither stat-ed, opened nor hashed. Files are opened with `O_NOFOLLOW` and non-blocking, only regular files are read. Images that were not examined are a gating outcome: WARN, and CRITICAL above 50 percent of the images (`max_unchecked_fraction`) or with `--require-local-images`, so pointing every image at `/nonexistent` or `http://...` fails the default gate. Findings carry base names only, a 12-hex `group_id`, never a directory or a full digest | built |
| Resource exhaustion through audit inputs (huge or hostile images, boxes, findings, cache, strings, special files) | Explicit constants, each with a boundary test: files over `max_image_bytes` (1 GiB) are not read; hashing reads 1 MiB chunks; a JPEG header walk allows 64 fill bytes in a row, 4,096 segments and 1 MiB in total, BMP headers are size-checked; FIFOs, devices and symlinks named by `manifest.json`, tables, audit files or images are rejected on the opened descriptor (`O_NOFOLLOW`, non-blocking, `fstat`) and every read is bounded; file-reading checks keep at most `4 x workers` futures alive (`audit/pool.py`) and stop at `max_check_seconds` (30 minutes) or `max_hash_bytes` (200 GiB) with an `audit_incomplete` finding that is CRITICAL whenever a gate is applied; the box pair search runs in 256-column blocks and an image with more than 20,000 boxes raises a `pairwise_skipped` WARN; at most 10,000 findings per box rule and 50,000 per check with a truncation finding that keeps the gate's verdict; payload strings are cut to 512 characters and payloads limited to 64 KiB, depth 6; `Image.file_path` 4,096, `Category.name` 65,536, `ann_id` 1,024 characters (names are shown cut to 100 or 200); the hash cache file 64 MiB; HTML 2,000 rows and SARIF 5,000 results, both most severe first with omitted counts; `--workers` 256 | built |
| Gate fail-open (a hostile dataset or plugin makes a failing audit exit 0) | The CLI default is `--fail-on critical` (`none` is explicit); an empty dataset is refused (exit 1); `main()` starts with exit code 1 and sets 0 only after the command returned, and a stray `SystemExit` becomes exit 1; plugin loading and `check.run` run under a guard that turns `SystemExit` and other `BaseException` (not `KeyboardInterrupt`) into a failed audit; `content_changed` and a duplicate group between `unassigned` and a real split are CRITICAL; `unsupported` image magic is `invalid_header` (CRITICAL) because ingest accepted every image; `--report`, `--json` and `--sarif` may not name the same file | built |
| Terminal and CI-log injection through error text | The error boundary prints `clean_text(str(exc))` (control, format and surrogate characters, including U+061C, U+180E and tag characters, shown as `\uXXXX`) with leading `:` removed, so a dataset cannot emit `::error::` workflow commands or ANSI sequences | built |
| Tampering with `manifest.version` | version recomputed from the tables on `read`; mismatch is a `StoreError` | built |
| Supply chain | committed lockfile, `uv sync --locked` in CI, pinned Actions | built (per [dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md)) |
| Network calls from a CI gate | `audit` makes no network calls (remote image URIs are counted, never fetched); no telemetry anywhere | built for the CPU checks |

## Licensing and extras policy

Allow-list and procedure: [dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md).
Project-specific decisions ([ADR-0001](../adr/0001-scope-and-license.md),
[ADR-0005](../adr/0005-plugin-system.md)):

| Component | License | Ships as |
|---|---|---|
| DINOv2, SigLIP2, Grounding DINO, Qwen3-VL | Apache-2.0 | extras `[embed]`, `[vlm]`; weights downloaded by the user |
| Florence-2 | MIT | `[vlm]` (captions, OCR) |
| DINOv3 | DINOv3 License (gated; "Built with DINOv3" notice required for redistribution) | `[embed]`, allowed only when the user adds it to `DATACOPILOT_ALLOWED_LICENSES` |
| SAM 3 / SAM 3.1 | SAM License (custom, gated) | `[sam3]`, same opt-in |
| FiftyOne, cvat-sdk, supervision, Datumaro | Apache-2.0 / MIT | extras or core |
| Cleanlab (CleanVision if adopted) | AGPL-3.0 | optional extra `[cleanlab]`, lazy import inside the `label_noise` adapter; `--help` and the runtime docs carry the notice that using the extra puts the user's combined work under AGPL-3.0 |
| YOLOE proposer (derives from Ultralytics code) | AGPL-3.0 | separate package `datacopilot-yoloe`; never part of this distribution |
| fastdup | CC BY-NC-ND 4.0 | forbidden |

Every model used appears in `THIRD_PARTY_MODELS.md` with source, license and commercial
status; the registry logs `model_license_shown` on first use.

## Observability

What a user can inspect today:

- `datacopilot doctor` prints `ok` and emits one `doctor_completed` record with
  `package_version` and `python_version`; exit code 0.
- Every record on `stderr` carries `request_id`; `DATACOPILOT_ENV=dev` with
  `DATACOPILOT_LOG_LEVEL=DEBUG` enables debug output.

Planned additions to `doctor`: installed extras and whether their imports succeed; GPU
presence and free memory; `DATACOPILOT_CACHE_DIR` writability; the allowed-license list;
registered plugins per group with license and extra, without loading them; the
`schema_version` the package writes. Planned per-run artefacts: `audits/<run_id>.json`
with timings per check, `ConversionReport` on every conversion, and `duration_ms` on every
completed-event record.
