# AGENTS.md

Instructions for coding agents and other automated contributors. Humans: start at
[CONTRIBUTING.md](CONTRIBUTING.md). The binding rules are in [docs/engineering/](docs/engineering/README.md);
this file points at them and does not replace them.

## What this is

DataCopilot is a CLI-first, headless "dataset doctor" for detection and classification datasets: it
converts formats losslessly, audits, and proposes fixes that a human reviews before they are applied.
Public, Apache-2.0, Python 3.10 to 3.14, managed with `uv`. What works today: [docs/status.md](docs/status.md).

## Quick commands (run from the repository root)

```bash
uv sync                                   # dev environment from uv.lock (CI: uv sync --locked)
uv sync --extra embed                     # heavy extras (torch, transformers, faiss); also: --extra vlm
uv run ruff check . && uv run ruff format --check .
uv run mypy                               # strict, src/ only
uv run pytest --cov=datacopilot --cov-fail-under=85     # the CI gate
uv run pytest tests/unit/test_logging_contract.py -q    # one file
uv run --group docs --group release pytest              # everything, incl. site build and wheel checks
uv run --group docs mkdocs build --strict -d "$TMPDIR/site"   # docs site, strict links and anchors
uv run python scripts/gen_cli_reference.py --check      # docs/reference/cli.md matches the app
uv lock --check                                         # lock matches pyproject.toml
uv run datacopilot --help                               # also: --version, doctor, formats
```

If the checkout was moved, `.venv` shebangs and `__pycache__` keep the old path (a tool reports a
version that differs from `uv.lock`, or a test sees an old file path): delete both and `uv sync` again.

## Module map (`src/datacopilot/`, layers per [ADR-0004](docs/adr/0004-layering-and-dependency-rule.md))

- `cli/` (interface): Typer app, one thin function per command; `main()` is the only error boundary.
- `api.py` (interface): public Python API (`load`, `audit`, `export`, `split`, `propose`, `apply`); surface frozen.
- `convert.py` (application): `ingest` and `export` use cases.
- `audit/` (application): check registry, engine, findings, `--fail-on` gate; checks in `audit/checks/`.
- `report/` (application): JSON, static HTML (Jinja2, autoescaped) and SARIF from an `AuditRun`.
- `proposals/` (application): propose, review export/import and apply use cases.
- `splits/` (application): group-aware re-split written as a new dataset version.
- `doctor.py` (application): bounded, read-only diagnosis of the install.
- `core/` (domain): Pydantic schemas, `PluginSpec`, version hashing, geometry, errors; stdlib, pydantic, numpy only.
- `io/` (infrastructure): one module per format plus `registry.py`; safe XML (`xmlsafe.py`), YAML subset (`miniyaml.py`), bounded reads (`files.py`).
- `store/` (infrastructure): the `dataset.dcp/` directory: Parquet tables, manifest, atomic versioned writes.
- `embed/` (infrastructure): embedders, embedding cache, exact kNN.
- `models/` (infrastructure): model registry, license guard, weight hashing.
- `proposers/` (infrastructure): proposer registry, Grounding DINO backend, dependency-free `test_boxes`.
- `logging.py`, `settings.py`, `deprecation.py`: cross-cutting, beside the layers, import no layer.

Architecture: [docs/architecture/](docs/architecture/README.md). Decisions: [docs/adr/](docs/adr/README.md).

## Invariants (never break these)

- **Core dependencies are an allow-list:** exactly `click`, `jinja2`, `numpy`, `pyarrow`, `pydantic`,
  `structlog`, `typer` ([ADR-0003](docs/adr/0003-repo-and-tooling.md)). Changing it is an ADR change and
  a workspace QA-gate change. `tests/release/test_core_dependencies.py` fails if the locked core closure
  pulls a heavy package.
- **Heavy dependencies are extras only** (`embed`, `vlm`): imported inside the function that needs them,
  never at module top level; the import error names the extra. `tests/unit/test_layering.py` enforces it.
- **Apache-only core:** core dependencies are permissive only (license table in
  [dependencies-and-licensing.md](docs/engineering/dependencies-and-licensing.md)). No copyleft code in
  this package, nothing vendored, no model weights committed or redistributed.
- **Imports point down** (interface, application, domain, infrastructure). Application code reaches
  formats, checks, embedders and proposers through registries, never by importing an adapter module.
- **Lossless store** ([ADR-0006](docs/adr/0006-dataset-format.md), [03-data-model.md](docs/architecture/03-data-model.md)):
  readers keep unmapped fields verbatim in `source`; every conversion returns a `ConversionReport`
  counting drops (`--strict` turns any drop into exit 1). Versions are content hashes and append-only:
  `ingest`, `split` and `apply` create a new version and never modify or delete an existing one.
- **Proposals are never auto-applied** ([ADR-0008](docs/adr/0008-propose-review-apply.md)): `propose`
  only stores a set; a human decides via `review export`/`review import`; only `apply` changes data,
  writing a NEW dataset, and it refuses unreviewed, incomplete, foreign or test-proposer sets.
- **Untrusted input** (dataset files, plugin output, model output) is validated with the Pydantic models
  in `core/schema.py`; paths are resolved and contained before reading or writing.
- **Logging contract** ([errors-and-logging.md](docs/engineering/errors-and-logging.md),
  [ADR-0002](docs/adr/0002-logging.md)): JSON lines on stderr via `datacopilot.logging.get_logger`;
  every record has `@timestamp`, `level`, `message`, `service`, `env`, `request_id`, `logger`.
  `message` is a constant `snake_case` event name; variable data goes in separate `snake_case` fields.
  Never log secrets, credentials or personal data (redaction is key-based only). Log an error once, at
  the boundary in `cli/app.py`; never log and re-raise. Contract test: `tests/unit/test_logging_contract.py`.
- **stdout is output only;** logs and usage errors go to stderr. Exit codes: 0 ok, 1 failed or gate
  tripped, 2 usage or settings error, 130 interrupted.
- **Settings** come only from `DATACOPILOT_*` environment variables ([docs/reference/settings.md](docs/reference/settings.md)).
- **Public API is frozen** by `tests/unit/test_api_surface.py`; a deliberate change regenerates the
  snapshot and adds a CHANGELOG entry ([workflow.md](docs/engineering/workflow.md), "Releases").

## Adding a format, check, embedder or proposer

Defined by [ADR-0005](docs/adr/0005-plugin-system.md); the registries are listed in
[02-components.md](docs/architecture/02-components.md#registries).

1. Write one module exposing a module-level `plugin` object with `spec: PluginSpec` (`core/schema.py`:
   `name`, `kind`, `license` as SPDX, `extra`, `gpu_memory_gb`, `version`) and the group's protocol:
   formats `Reader`/`Writer` (`io/registry.py`), checks `Check` (`audit/registry.py`), embedders
   (`embed/protocol.py`), proposers (`proposers/protocol.py`). Copy the closest existing plugin.
2. Register one line in `pyproject.toml` under `[project.entry-points."datacopilot.<group>"]`
   (`formats`, `checks`, `embedders`, `proposers`), then `uv sync` so the entry point is installed.
3. Heavy imports go inside methods; a heavy dependency goes into an extra, never into core.
4. Tests: unit tests in `tests/unit/` (patch the registry's `_entry_points` for registry behaviour, as
   `tests/unit/test_format_registry.py` does); a format also gets a round-trip in `tests/golden/`.
5. Update [docs/status.md](docs/status.md), the matching user guide page and, for a new CLI option,
   regenerate `docs/reference/cli.md` with `uv run python scripts/gen_cli_reference.py`.

## Testing layers ([testing.md](docs/engineering/testing.md))

| Layer | Where | Runs |
|---|---|---|
| Unit and contract | `tests/unit/` | default run |
| Golden | `tests/golden/` (byte-identical reports, round-trips) | default run |
| Chaos | `tests/chaos/` (marker `chaos`: interrupted writes, recovery) | default run; `-m chaos` alone |
| Docs | `tests/docs/` (executes tutorial and API page, CLI reference drift, strict site build) | default run; site build needs `--group docs` |
| Release | `tests/release/` (workflow lint, build backend pins, wheel/sdist contents) | default run; builds need `--group release`; `-m clone` opt-in |
| Notebook | `tests/notebooks/` (static checks of the saved outputs in `notebooks/`) | default run |
| Real model | `tests/real/` (marker `real_model`, downloads weights, needs `[embed]`) | opt-in: `-m real_model` |

- Regenerate goldens deliberately and review the diff:
  `UPDATE_GOLDEN=1 uv run pytest tests/golden` (audit reports) and
  `UPDATE_API_SURFACE=1 uv run pytest tests/unit/test_api_surface.py` (public API snapshot).
- Notebooks are re-executed by hand ([notebooks/README.md](notebooks/README.md)); never hand-edit outputs.
- Unit tests: no network, no files outside `tmp_path`, no wall clock or ambient env vars. Write the
  failing test first; every bug fix adds the test that would have caught it.

## Definition of done

All of these pass (full list: [workflow.md](docs/engineering/workflow.md#definition-of-done)):

1. `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov=datacopilot --cov-fail-under=85`
2. `uv lock --check` after any dependency change.
3. `uv run python scripts/gen_cli_reference.py --check` and the strict docs build after any CLI or docs change.
4. From the workspace root (the directory that contains this repository):
   `python3 tools/qa_check.py datacopilot` prints `ok   datacopilot`. It checks the core-dependency
   allow-list, the license, required docs and ADR headings, relative links, diagrams, secret-looking
   strings and attribution text.
5. Docs, architecture pages and diagrams changed in the same change as the code; every documented
   command was actually run.

## Don'ts

- Don't add a core dependency, or import torch, transformers, faiss, pillow or similar at module top level.
- Don't add or vendor copyleft code or model weights; a copyleft-derived plugin is a separate package.
- Don't modify an existing dataset version, apply proposals without review, or weaken a refusal in `apply`.
- Don't catch broad exceptions to keep going, print tracebacks by default, or write logs to stdout.
- Don't put secrets, tokens, personal data or real customer images in code, logs, fixtures or notebooks.
- Don't edit `docs/reference/cli.md`, golden files or `tests/unit/api_surface.json` by hand.
- Don't add tool, assistant or generator attribution to commits, PRs or source; no agent-tool config
  files are committed (they are gitignored).
- Don't publish, tag or push a release: that needs the maintainer ([release.md](docs/engineering/release.md)).
