# 05 Runtime

![Runtime and deployment](../diagrams/png/runtime.png)

## Process model

- One Python process per command. No daemon, no server, no background workers between
  commands. `datacopilot --version` and `doctor` (built) run and exit in one thread.
- Built: the `integrity` (header reads) and `duplicates` (streamed sha256) checks run on a
  `ThreadPoolExecutor` with `--workers` threads (default `DATACOPILOT_WORKERS`, else
  `os.cpu_count()`, at most 256). Threads, not processes: both are I/O bound and `hashlib`
  releases the GIL on large chunks, so there is no pickling or spawn cost; the `boxes` and
  `imbalance` checks are single-threaded NumPy and counting passes. No decoding of pixels happens
  anywhere in the CPU audit.
- Built (differs from the first design): reading image bytes for embedding runs on the bounded
  thread pool (at most four files ahead of the model, `audit/pool.py`), and pixel decoding and
  preprocessing happen in the embedder, in the calling thread, one batch at a time under a
  64-megapixel limit per image. Memory of one embedding batch is the encoded bytes of the batch
  (at most 256 MiB) plus one decoded image at a time (at most about 200 MB of pixels at the limit,
  freed once the 224-pixel tensor exists) plus the model; images ahead of the model are at most four
  files of at most 32 MiB. Every undecodable image of a batch is reported by one failing call and the
  rest is retried once. A process pool for decoding is planned; it is not needed until decoding,
  not the model, is the bottleneck.
- Not planned any more for Phase 2: DuckDB table scans. The audit reads the tables through the
  bounded store reader and works on NumPy arrays ([ADR-0003](../adr/0003-repo-and-tooling.md)).
- Built for embedders, planned for the rest: model inference runs in-process on the local GPU
  (CUDA when available) or CPU (`[embed]`, `[vlm]`, `[sam3]`); Qwen3-VL may instead be called over HTTP on a vLLM endpoint, in which case no
  model is loaded locally.
- No subprocesses are spawned for external tools; CVAT and FiftyOne are reached through
  their Python SDKs in-process.
- `request_id` (a UUID) is bound at startup and cleared at exit; the planned `AuditRun.run_id`
  reuses it.

## Deployment nodes

| Node | What runs there | Needs |
|---|---|---|
| Developer laptop (CPU) | the CLI, core library, format adapters, CPU checks, report generator, a local `dataset.dcp/` | Python ≥ 3.10, core wheel |
| GPU machine | the same plus `[embed]`, `[vlm]`, `[sam3]`; embedders, proposers; model cache | 8 GB VRAM for ViT-B embeddings and Grounding DINO base; 16–24 GB for SAM 3; Qwen3-VL-8B ~16 GB F16 or ~5 GB Q4; `[sam3]` needs Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6 |
| CI runner | `datacopilot audit --fail-on …` as a gate, `report.sarif` uploaded to code scanning; the embedding checks need `[embed]` and the model even with a precomputed cache, because every run re-embeds a sample to verify the cache (without the images or the model the cache is not trusted and the gate fails closed) | core wheel, read access to `dataset.dcp/` and images; `DATACOPILOT_CACHE_KEY` for the cache |
| Object storage (S3, GCS, Azure) | image files referenced by URI; optionally the `dataset.dcp/` directory itself | `fsspec` with the matching filesystem package |
| vLLM endpoint (optional) | Qwen3-VL served by vLLM; receives image crops, returns text | network reachability; an API key if the server requires one |
| Hugging Face Hub | source of weights; gated repos require the user's token | outbound HTTPS |
| CV-Pilot (sibling) | reads `manifest.json` from the dataset directory to record `manifest.version` | file or object-storage read access |

## Storage locations

| Data | Location | Default |
|---|---|---|
| Dataset | `dataset.dcp/` given on the command line | required argument; may be an `fsspec` URI |
| Images | wherever `file_path` points | never copied |
| Embedding cache | inside the dataset: `embeddings/<model_id>@<revision>/` (`manifest.json` plus `shard-*.parquet`, [ADR-0007](../adr/0007-embedding-cache-and-index.md)); its MAC secret is `DATACOPILOT_CACHE_KEY` or `secret.key` in the cache directory | on unless `--no-embedding-cache`; in CI only with `DATACOPILOT_CACHE_KEY` |
| Model weights | Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`) | upstream default |
| DataCopilot cache (thumbnails, evidence crops, FAISS indexes) | `DATACOPILOT_CACHE_DIR` | `$XDG_CACHE_HOME/datacopilot`, else `~/.cache/datacopilot` |
| Reports | `--report <path>`; `.json` and `.sarif` next to it unless `--json` / `--sarif` are given | `./report.html` |
| Audit runs | `audits/<run_id>.json` inside the dataset | always |
| Hash cache | `DATACOPILOT_CACHE_DIR`: `sha256-v2.cache` and `secret.key`, directory mode 0700, files 0600 | on unless `--no-hash-cache` |
| Logs | `stderr`, JSON lines | always; nothing written to disk by the logger |

## Configuration

Environment variables with the `DATACOPILOT_` prefix. Precedence, highest first:
command-line option, environment variable, built-in default. There is no configuration
file.

| Variable | Values | Default | Status |
|---|---|---|---|
| `DATACOPILOT_ENV` | `dev`, `staging`, `prod` | `prod` | built |
| `DATACOPILOT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`; `DEBUG` honoured only when `ENV=dev` | `INFO` | built |
| `DATACOPILOT_MAX_TABLE_ROWS` | integer ≥ 1; row limit per Parquet table on read | `5000000` | built |
| `DATACOPILOT_MAX_TABLE_BYTES` | integer ≥ 1; uncompressed-size limit per Parquet table on read | `4294967296` | built |
| `DATACOPILOT_ALLOWED_LICENSES` | comma-separated SPDX ids and custom names, e.g. `Apache-2.0,MIT,SAM-License` | `Apache-2.0,MIT,BSD-3-Clause` | built (case-insensitive; empty or malformed is exit 2; reaches `audit` and `embed` through the real CLI path, tested) |
| `DATACOPILOT_CACHE_DIR` | directory path (not empty, not `.`) | see storage table | built (hash cache) |
| `DATACOPILOT_WORKERS` | integer 1 to 256 | `os.cpu_count()` | built (`audit`) |
| `DATACOPILOT_CACHE_KEY` | secret, 16 to 1024 characters; out-of-band MAC key of the hash cache, never stored next to it; required to use the cache when `CI` is set | unset | built |
| `CI` | any value except empty, `0`, `false`, `no`: marks a CI environment (cache needs the key; `--allow-external-images` needs `--allow-external-images-in-ci`). Only this variable is looked at: a runner that does not set it (a self-hosted job, a container without the variable) is treated as a laptop, where a private local secret beside the cache directory is used. Use `--require-cache-key` where the cache directory is shared or restored from elsewhere | unset | built (read, not prefixed) |
| `DATACOPILOT_HASH_CACHE` | `true` or `false` | `true`; CI gates should set `false` | built (`audit`; `--hash-cache/--no-hash-cache` override it) |
| `HF_ENDPOINT` (read by the Hub library), `DATACOPILOT_ALLOW_HF_ENDPOINT` | a model hub other than `https://huggingface.co` is refused while `CI` is set unless `DATACOPILOT_ALLOW_HF_ENDPOINT=1` confirms it (the Hub token travels to that host); `HF_HUB_DISABLE_TELEMETRY=1` is set by default | unset | built |
| `DATACOPILOT_VLLM_URL` | `http(s)://host:port/v1` | unset (local inference) | planned |

An invalid value of any `DATACOPILOT_*` variable exits with code 2 and one line on `stderr` before logging
is configured, for every command except `doctor`, which reports the problem in its own output (exit 2).

## Secrets

| Secret | Where it may live | Where it must never appear |
|---|---|---|
| Hugging Face token (gated weights) | `HF_TOKEN` in the environment, read by the Hub library itself (datacopilot never reads or logs it); `DATACOPILOT_HF_TOKEN` is planned | command-line arguments, `manifest.json`, reports, logs, proposals, `source` dicts |
| CVAT credentials | `DATACOPILOT_CVAT_TOKEN` (or the SDK's own auth) | same |
| vLLM API key | `DATACOPILOT_VLLM_API_KEY` | same |
| Object storage credentials | the cloud SDK's standard environment or instance role | same |

The logging contract redacts any key named `token`, `password`, `api_key`, `jwt` or
`authorization`, or ending in `_token`, `_api_key`, and so on (built). The variable names
above are chosen so the suffix rule covers them. Redaction is key-based: a secret must
never be placed in a free-text field or an exception message.

## Optional extras

| Extra | Unlocks | Dependencies | Hardware |
|---|---|---|---|
| `embed` | built: embedders `dinov2_small`, `dinov2_base`, `siglip2_base` and the near-duplicate, leakage and outlier checks; planned: `dinov3-*` (gated; DINOv3 License) and the label-noise check | `pillow`, `safetensors`, `huggingface-hub`, `transformers`, `faiss-cpu`, `torch` (declared in `pyproject.toml`; nothing is imported by the core) | CPU works (slow); GPU with 8 GB VRAM recommended |
| `vlm` | built: proposer `grounding_dino` (Grounding DINO tiny, Apache-2.0); planned: `class_verify` (Qwen3-VL locally or via `DATACOPILOT_VLLM_URL`) | `pillow`, `safetensors`, `huggingface-hub`, `transformers`, `torch` (declared in `pyproject.toml` with the floors the real run used) | about 2 GB of memory on the CPU (the measured run used the CPU); a GPU is used when available |
| `sam3` | `box_refine` and SAM 3 agreement in `missing_boxes` | the SAM 3 package; user accepts the SAM License on the Hub | 16–24 GB VRAM; Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6 |
| `fiftyone` | `review --to fiftyone`, `ProposalSet.to_fiftyone`, `from_fiftyone` | `fiftyone` (Apache-2.0; runs its own MongoDB) | CPU |
| `cvat` | planned: `review --to cvat` against a CVAT server (today review works through exported files, no extra needed) | `cvat-sdk` (MIT) | network to a CVAT server |
| `cleanlab` | Cleanlab confident learning as the backend of the `label_noise` check (classification and detection); without it the check is skipped with a `WARN` | `cleanlab` (AGPL-3.0; imported lazily inside the adapter). Notice: installing and using this extra makes the user's combined work subject to AGPL-3.0 ([ADR-0001](../adr/0001-scope-and-license.md)) | CPU |

All extras are imported inside the function that needs them; a missing extra produces one
error line naming the `pip install "datacopilot[<extra>]"` command and exit code 1.
