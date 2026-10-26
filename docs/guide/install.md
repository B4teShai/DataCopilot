# Install

DataCopilot needs Python 3.10 or newer. The core install is CPU-only and pulls no deep-learning
framework: `click`, `jinja2`, `numpy`, `pyarrow`, `pydantic`, `structlog` and `typer`.

DataCopilot is not on the Python Package Index: install it from a clone of the repository.

> **Release status:** not yet published to PyPI, so `pip install datacopilot` does not work; install
> from a clone as below.

## From a clone, with uv

<!-- skip: needs network access; scripts/check_clone.sh (pytest -m clone) runs the same steps on a fresh clone -->
```bash
git clone https://github.com/B4teShai/DataCopilot.git
cd DataCopilot
uv sync
uv run datacopilot --version
uv run datacopilot doctor
```

`uv sync` creates `.venv/` with the locked versions; `source .venv/bin/activate` puts `datacopilot` on
your `PATH`, or prefix every command with `uv run`.

## From a clone, with pip

<!-- skip: needs network access; scripts/check_clone.sh (pytest -m clone) runs the same steps on a fresh clone -->
```bash
git clone https://github.com/B4teShai/DataCopilot.git
cd DataCopilot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
datacopilot --version
```

`pip install .` (without `-e`) installs a copy instead of linking the clone. pip resolves the
dependencies itself, so it may pick newer versions than the lock file; `uv sync` uses the locked ones.

## Use it in your own project

Add it as a dependency straight from GitHub (pin a commit you have reviewed with `@<commit>`), or
point at a local clone:

<!-- skip: needs network access -->
```bash
uv add git+https://github.com/B4teShai/DataCopilot
pip install "datacopilot @ git+https://github.com/B4teShai/DataCopilot"
pip install -e ../DataCopilot
```

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| `embed` | `pillow`, `safetensors`, `huggingface-hub`, `transformers`, `faiss-cpu`, and the tensor library | `--embedder dinov2_small`, `dinov2_base`, `siglip2_base` |
| `vlm` | `pillow`, `safetensors`, `huggingface-hub`, `transformers`, and the tensor library | `propose --proposer grounding_dino` ([Propose, review, apply](proposals.md)) |

<!-- skip: needs network access -->
```bash
uv sync --extra embed                # in a clone, with uv
pip install -e ".[embed]"            # in a clone, with pip
uv add "datacopilot[vlm] @ git+https://github.com/B4teShai/DataCopilot"   # in your own project
```

`embed` and `vlm` are the extras that exist. Extras for SAM 3 and the Cleanlab backend will be published
when those features exist. `datacopilot doctor` lists which extras are importable.

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
