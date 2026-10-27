# 06 Cross-cutting concerns

The contracts live in [engineering](../engineering/README.md). This page lists only what is specific
to DataCopilot.

## Logging

Contract: [errors and logging](../engineering/errors-and-logging.md) and
[ADR-0002](../adr/0002-logging.md). `service` is `datacopilot`. `request_id` is one UUID per command
and is also the `AuditRun.run_id`.

Events, by area. All are INFO unless marked WARN or ERROR.

| Area | Events |
|---|---|
| Commands | `doctor_completed`, `command_failed` (ERROR), `report_written` (file name only), `model_license_shown` |
| Datasets | `dataset_loaded`, `dataset_ingested`, `dataset_exported`, `dataset_split`, `conversion_loss` (WARN) |
| Audit | `audit_check_completed`, `audit_completed`, `audit_not_stored` (WARN), `audit_nofollow_unavailable` (WARN) |
| Hash cache | `audit_cache_unavailable`, `audit_cache_untrusted`, `audit_cache_discarded` (all WARN) |
| Embeddings | `embeddings_computed`, `crop_embeddings_computed`, `model_loaded`, `embedder_device_fallback` (WARN) |
| Embedding cache | `embedding_cache_disabled`, `_rejected`, `_unverified`, `_content_changed`, `_discarded` (all WARN) |
| Proposals | `proposer_started`, `propose_progress`, `proposal_set_written`, `review_exported`, `review_imported`, `proposals_applied`; WARN: `proposer_image_skipped`, `proposer_device_fallback`, `proposal_candidates_rejected` |

Each event's fields are in the code that emits it. Rules for this project: log `image_id`, not
`file_path`, so no user name leaks. Log `prompt_hash`, not prompt text. Never log reviewer names.

## Configuration

Keys and precedence are in [05 Runtime](05-runtime.md#configuration) and
[Settings](../reference/settings.md). Validation happens in `settings.py` before logging starts, and
an invalid value is exit code 2.

## Errors and exit codes

| Command family | 0 | 1 | 2 |
|---|---|---|---|
| `ingest`, `export` | Converted. | Bad input, invariant violation, identical images, non-empty output, over-limit table, `--strict` with losses. | Bad option or environment. |
| `audit` | Run completed and gate passed. | Gate matched, unreadable dataset, license refused, embedder failing. | Bad option, bad `--fail-on`, embedding check without `--embedder`. |
| `embed` | Every image embedded. | An image without an embedding, embedder failure, cache not writable. | Bad option, cache unusable in CI. |
| `split` | New version written. | Image not embedded, budget hit, split unchanged, output refused. | Bad ratios, unsafe `--by-group`, `--near-dup` without `--embedder`. |
| `propose` | Set stored, every image reached. | Incomplete run, model error, license refused, missing extra. | Bad option, unknown proposer or class. |
| `review`, `apply` | Folder or dataset written. | Tampered or mismatching set, refused decisions, nothing accepted, output exists. | Bad option or id. |

`130` is Ctrl-C everywhere, and no partial version is left visible. Exception classes are
`FormatError`, `ValidationError`, `StoreError`, `ConversionLossError`, `LicenseError` and `ModelError`,
all subclasses of `DataCopilotError`, plus `SettingsError`, a `ValueError` handled before logging
starts. Each message names the offending input.

## Security and threat model

Assets: the human labels (the only irreplaceable data), the image files (often customer data),
credentials, the integrity of `manifest.version` as a reproducibility record, and the CI gate's
verdict.

Trust boundaries: files read from disk (including `dataset.dcp/` folders made elsewhere), third-party
plugins, model weights and output, network services, and the CI environment where logs are public.

| Threat | Mitigation |
|---|---|
| Malicious export file: XML entities, path traversal, huge or deeply nested JSON or XML | Expat events with DOCTYPE rejected, nesting, size and count caps, paths kept inside the dataset root, every number validated, unsupported shapes counted. |
| Pickled or executable weights | Cards pin a commit and the SHA-256 of each safetensors file. Only those files are downloaded and verified before loading. No `trust_remote_code`. |
| Model output or review files used to plant labels | Candidates validated, prompts normalised, sets content-addressed and bound to model and version, decisions read by a hardened reader, `apply` writes only a new dataset. |
| Test double mistaken for a real embedder | `hash` is marked test-only, warns, and a gate refuses it without `--allow-test-embedder`. |
| Evading the embedding checks through image tricks | Only JPEG, PNG, GIF and BMP are decoded. Alpha is composited on white, animations are refused and counted as not embedded. |
| Forged or poisoned embedding cache hides leakage | Shards are bounded, digest-checked and covered by an HMAC. Each vector carries its image's SHA-256. A sample is re-embedded each run; one mismatch discards the cache. |
| Poisoned hash cache hides duplicates or changes | Owner and mode checks, `O_NOFOLLOW`, a tag per line, a re-hashed sample each run. In CI without a key the cache is off. |
| Crafted table that expands to gigabytes | The Parquet footer is checked before any row is decoded: file size, rows, uncompressed size and schema. |
| Audit reads files outside the dataset, or passes without examining images | Paths are classified. Outside paths are not opened without `--allow-external-images`. Unexamined images are findings, CRITICAL above half. |
| Gate fails open | Default `--fail-on critical`. Empty dataset refused. `main()` starts at exit 1 and sets 0 only on success. Plugins that raise `SystemExit` fail the audit. |
| Terminal and CI-log injection through error text | Error text is cleaned of control characters, so a dataset cannot emit `::error::` commands or ANSI sequences. |
| Secrets and markup in logs or reports | Key-based redaction. Dataset strings cleaned, then escaped. HTML has no script, no remote resource and a content-security tag. |
| Personal data in leakage findings | Group ids come from paths, so only a hashed reference is shown unless `--show-group-labels`. |
| Supply chain | Committed lockfile, `uv sync --locked` in CI, pinned actions. |
| Network calls from a CI gate | None. Remote image URIs are counted, never fetched. No telemetry. |

Known residual risks: a holder of the cache key can forge cache contents outside the re-checked
sample, and `image_id` is a 64-bit label, so two files can be built with one id. The embedding checks
compare full digests; the exact-duplicate check and the version use the short id.

## Licensing and extras

The allow-list and procedure are in
[dependencies and licensing](../engineering/dependencies-and-licensing.md). Project decisions are in
[ADR-0001](../adr/0001-scope-and-license.md) and [ADR-0005](../adr/0005-plugin-system.md).

| Component | License | Ships as |
|---|---|---|
| DINOv2, SigLIP2, Grounding DINO | Apache-2.0 | Extras `embed` and `vlm`. You download the weights. |
| DINOv3 | Gated DINOv3 license | Allowed only if you add it to `DATACOPILOT_ALLOWED_LICENSES`. |
| SAM 3 | Gated SAM license | Same opt-in. |
| Cleanlab | AGPL-3.0 | Would be an optional extra, imported lazily, with a notice. |
| YOLOE (derived from Ultralytics code) | AGPL-3.0 | A separate package, never part of this one. |
| fastdup | CC BY-NC-ND 4.0 | Forbidden. |

Every model appears in `THIRD_PARTY_MODELS.md`. The registry logs `model_license_shown` on first use.
Which of these exist today is in [Status](../status.md).

## Observability

`datacopilot doctor` prints a status (`ok`, `warn`, `fail`) and one line per section, or one JSON
object with `--json`: package and dependency versions, importable extras (looked up, not imported),
validated settings (the cache key only as set or unset), the cache folder, FAISS (never imported), GPU
visibility, allowed licenses and registered plugin names. It exits 0 (ok or warn), 1 (a core
dependency is missing) or 2 (an invalid setting). Every record on `stderr` carries `request_id`.
`DATACOPILOT_ENV=dev` with `DATACOPILOT_LOG_LEVEL=DEBUG` turns on debug output.
