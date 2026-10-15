# Install

DataCopilot needs Python 3.10 or newer. The core install is CPU-only and pulls no deep-learning
framework: `click`, `jinja2`, `numpy`, `pyarrow`, `pydantic`, `structlog` and `typer`.

<!-- skip: network install of a package that is published by the maintainer, not part of the tested tree -->
```bash
pip install datacopilot
```

> **Release status:** not yet published to PyPI. Until it is, install a wheel you built yourself (see the
> [release runbook](../engineering/release.md)) or work from a checkout with
> [`uv`](https://docs.astral.sh/uv/):

<!-- skip: needs a checkout and network access -->
```bash
git clone https://github.com/B4teShai/DataCopilot.git
cd DataCopilot
uv sync
uv run datacopilot --version
```

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| `embed` | `pillow`, `safetensors`, `huggingface-hub`, `transformers`, `faiss-cpu`, and the tensor library | `--embedder dinov2_small`, `dinov2_base`, `siglip2_base` |

<!-- skip: network install -->
```bash
pip install "datacopilot[embed]"
```

`embed` is the only extra. Extras for the proposal workflow (vision-language models, SAM 3, the Cleanlab
backend) will be published when those features exist.

## Check the installation

`datacopilot doctor` reports the versions of the package and its dependencies, which extras are
importable, the validated settings, the cache directory, FAISS and GPU visibility. It exits 0 when
the installation works (extras may be missing), 1 when a core dependency is missing and 2 when a
`DATACOPILOT_*` variable is invalid. `datacopilot doctor --json` prints one JSON object. The report is safe to paste into a bug report: it never
holds a secret (the cache key shows only as set or unset) and writes your home directory as `~`, but it does
include versions, the operating system name and the cache directory (in full when it is outside your home
directory), so read it before you share it.

## Settings

Settings come from environment variables; an invalid value is a usage error (exit 2) before any
work starts. Secrets are only ever read from the environment and never printed or logged.

| Variable | Meaning | Default |
|---|---|---|
| `DATACOPILOT_ENV` | `dev`, `staging` or `prod`; `DEBUG` logging is honoured only in `dev` | `prod` |
| `DATACOPILOT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `DATACOPILOT_MAX_TABLE_ROWS`, `DATACOPILOT_MAX_TABLE_BYTES` | limits applied to a `dataset.dcp` before it is decoded | 5,000,000 rows, 4 GiB |
| `DATACOPILOT_WORKERS` | threads for file checks (1 to 256) | CPU count |
| `DATACOPILOT_CACHE_DIR` | where caches live | `~/.cache/datacopilot` |
| `DATACOPILOT_HASH_CACHE` | `false` disables the file-hash cache | `true` |
| `DATACOPILOT_CACHE_KEY` | out-of-band key that authenticates caches (16 to 1024 characters) | unset |
| `DATACOPILOT_ALLOWED_LICENSES` | licenses a model may carry | `Apache-2.0,MIT,BSD-3-Clause` |

Logs are JSON records on standard error with `@timestamp`, `level`, `message`, `service`, `env`,
`request_id` and `logger`; one `request_id` per command.
