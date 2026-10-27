# 05 Runtime

How DataCopilot runs, where it keeps files and how it is configured. There is no diagram: everything
runs in one process on one machine.

## Process model

- One Python process per command. No daemon, no server, no background worker between commands.
- The `integrity` (header reads) and `duplicates` (streamed SHA-256) checks run on a thread pool of
  `--workers` threads (default `DATACOPILOT_WORKERS`, else the CPU count, at most 256). Threads, not
  processes: the work is input and output bound and `hashlib` releases the GIL on large chunks. The
  `boxes` and `imbalance` checks are single-threaded NumPy and counting passes. The CPU audit decodes
  no pixels.
- For embedding, image bytes are read on a bounded thread pool, at most four files ahead of the model.
  Pixel decoding and preprocessing happen in the embedder, one batch at a time, under a 64-megapixel
  limit per image. Memory for one batch is the encoded bytes (at most 256 MiB), one decoded image at a
  time and the model.
- Model inference runs in the same process on a GPU if CUDA is available, else on the CPU.
- No subprocess is started for an outside tool.
- A `request_id` (UUID) is bound at startup and cleared at exit. It is also the `AuditRun.run_id`.

## Where it runs

| Place | What runs there | Needs |
|---|---|---|
| Laptop (CPU) | The command line, core library, format readers, CPU checks, reports. | Python 3.10 or newer, the core install. |
| GPU machine | The same, plus the `embed` and `vlm` extras. | About 8 GB of GPU memory is recommended for ViT-B embeddings. The CPU works, slowly. |
| CI runner | `datacopilot audit --fail-on ...` and an upload of `report.sarif`. | Read access to the dataset and images. Embedding checks need the `embed` extra even with a cache, which is verified by re-embedding a sample. |

## Where files live

| Data | Location | Default |
|---|---|---|
| Dataset | The `dataset.dcp/` folder you name. | Required argument. |
| Images | Wherever `file_path` points. | Never copied. |
| Embedding cache | Inside the dataset: `embeddings/<model>@<revision>/`. | On, unless `--no-embedding-cache`. In CI only with `DATACOPILOT_CACHE_KEY`. |
| Model weights | The Hugging Face cache (`HF_HOME`). | `~/.cache/huggingface`. |
| Hash cache | `DATACOPILOT_CACHE_DIR` (`sha256-v2.cache`, `secret.key`; folder mode 0700, files 0600). | On, unless `--no-hash-cache`. |
| Reports | `--report PATH`; JSON and SARIF next to it. | `./report.html`. |
| Audit runs | `audits/<run_id>.json` inside the dataset. | Always. |
| Logs | `stderr`, JSON lines. | Nothing is written to disk by the logger. |

## Configuration

Settings are environment variables with the `DATACOPILOT_` prefix. The command line beats the
environment, and the environment beats the default. There is no configuration file. The variables are
listed in [Settings and exit codes](../reference/settings.md). An invalid value exits with code 2 and
one line on `stderr` before logging starts. `doctor` is the exception: it reports the problem in its
own output, also with exit 2.

Two more variables are read without the prefix:

| Variable | Effect |
|---|---|
| `CI` | Any value except empty, `0`, `false` or `no` marks a CI run. The cache then needs a key, and external images need `--allow-external-images-in-ci`. |
| `HF_ENDPOINT` | A hub other than `https://huggingface.co` is refused while `CI` is set unless `DATACOPILOT_ALLOW_HF_ENDPOINT=1`, because the Hub token goes to that host. |

## Secrets

| Secret | Where it may live | Where it must never appear |
|---|---|---|
| Hugging Face token (gated weights) | `HF_TOKEN` in the environment. The Hub library reads it; DataCopilot never reads or logs it. | Command-line arguments, `manifest.json`, reports, logs, proposals. |
| Cache key | `DATACOPILOT_CACHE_KEY`, out of band. | The same places. |

The logging contract redacts any key named `token`, `password`, `api_key`, `jwt` or `authorization`, or
ending in `_token` or `_api_key`. Redaction is by key name, so never put a secret in free text or an
exception message.

## Extras

| Extra | Unlocks | Adds |
|---|---|---|
| `embed` | The embedders and the near-duplicate, leakage, label-noise and outlier checks. | `pillow`, `safetensors`, `huggingface-hub`, `transformers`, `faiss-cpu`, `torch` |
| `vlm` | The `grounding_dino` proposer. | The same, without `faiss-cpu`. |

Extras are imported inside the function that needs them. A missing extra gives one error line naming
the extra (`datacopilot[embed]`, or `uv sync --extra embed` in a clone) and exit code 1. Further extras
are listed in [Status](../status.md).
