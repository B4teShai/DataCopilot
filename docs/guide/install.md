# Install

DataCopilot needs Python 3.10 or newer. The core install is CPU-only and pulls no deep-learning
framework: `click`, `jinja2`, `numpy`, `pyarrow`, `pydantic`, `structlog` and `typer`. Install it
from a clone of the repository; it is not on the Python Package Index.

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

`uv sync` creates `.venv/` with the locked versions. Run `source .venv/bin/activate` to put
`datacopilot` on your `PATH`, or prefix every command with `uv run`.

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

pip resolves versions itself, so it may pick newer ones than the lock file. `uv sync` uses the
locked ones.

## Use it in your own project

Add it straight from GitHub, pinned to a full commit you have reviewed (a URL without
`@<commit>` follows whatever the default branch holds), or point at a local clone:

<!-- skip: needs network access -->
```bash
uv add "git+https://github.com/B4teShai/DataCopilot@<full-commit-sha>"
pip install "datacopilot @ git+https://github.com/B4teShai/DataCopilot@<full-commit-sha>"
pip install -e ../DataCopilot
```

## Extras

An extra is an optional install group, listed in the [glossary](concepts.md).

| Extra | Adds | Needed for |
|---|---|---|
| `embed` | image-model libraries and `faiss-cpu` | `--embedder dinov2_small`, `dinov2_base`, `siglip2_base` |
| `vlm` | the same model libraries, without FAISS | `propose` ([Propose, review, apply](proposals.md)) |

<!-- skip: needs network access -->
```bash
uv sync --extra embed                # in a clone, with uv
pip install -e ".[embed]"            # in a clone, with pip
uv add "datacopilot[vlm] @ git+https://github.com/B4teShai/DataCopilot@<full-commit-sha>"   # in your own project
```

## Check the installation

`datacopilot doctor` reports package and dependency versions, importable extras, settings, the
cache folder, FAISS and GPU. It exits 0 when the install works (extras may be missing), 1 when a
core dependency is missing and 2 when a `DATACOPILOT_*` variable is invalid. `--json` prints one
JSON object.

The report never holds a secret and writes your home folder as `~`, but it does include versions,
the operating system name and the cache folder. Read it before you paste it into a bug report.

Settings are environment variables; see [Settings and exit codes](../reference/settings.md).
