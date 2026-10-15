# ADR-0005: Plugin system

**Status:** accepted
**Date:** 2026-10-14

## Context

DataCopilot has four extension points that will grow faster than the core team can
maintain them: dataset formats, audit checks, proposers (models that generate fix
proposals) and embedders (models that produce image vectors). Some extensions carry
dependencies the core must not import (PyTorch, CUDA) and some carry copyleft licenses
(Cleanlab, Ultralytics; AGPL-3.0) that may only be reached lazily through an optional
extra, or derive from copyleft code (a YOLOE proposer) and must live outside this
distribution. [ADR-0004](0004-layering-and-dependency-rule.md) requires application code
to reach infrastructure through registries keyed by name, with lazy resolution so unused
adapters are never imported. [ADR-0001](0001-scope-and-license.md) fixes the license
rules. The model registry must know each model's license and GPU need before loading it.

## Options considered

### Option A: entry-point registries

Each plugin is a Python package that declares entry points in named groups. The registry
lists entry points lazily, loads one by name on first use, and reads a `PluginSpec` from
it before importing the implementation.

Trade-offs: standard packaging mechanism (`importlib.metadata`), third-party plugins are
indistinguishable from first-party ones, discovery costs one metadata scan per process; a
broken plugin package can shadow a name, so names must be unique and validated.

### Option B: subclass discovery

Plugins subclass an abstract base; the registry walks `__subclasses__()` at call time.

Trade-offs: no packaging metadata; but discovery only sees classes already imported, so
either the core imports every adapter module (breaking the lazy-import rule and pulling
torch into the core) or users import plugin modules by hand.

### Option C: config-file registration

A `datacopilot.toml` maps names to dotted import paths.

Trade-offs: explicit and auditable; but every user must edit a file to enable a plugin
they installed, paths break on refactors, and the file becomes a second source of truth
beside the installed packages.

## Decision

Option A. Plugins register through Python entry points in exactly four groups:

| Group | Registry | Interface the plugin provides |
|---|---|---|
| `datacopilot.formats` | `io/registry.py` | `Reader` and/or `Writer` for one format name |
| `datacopilot.checks` | `audit/registry.py` | `Check.run(dataset, ctx) -> list[Finding]` |
| `datacopilot.proposers` | `propose/registry.py` | `Proposer.run(dataset, ctx) -> list[Proposal]` |
| `datacopilot.embedders` | `embed/registry.py` | `Embedder.embed(images) -> ndarray[float32]` |

Rules that reviewers check:

1. Every entry point resolves to an object exposing `spec: PluginSpec` with fields
   `name`, `kind`, `license` (SPDX identifier), `extra` (the pip extra that provides its
   dependencies, or `None`), `gpu_memory_gb` (`None` for CPU) and `version`. The registry
   reads `spec` before importing anything heavier; the heavy import happens inside the
   plugin's first method call.
2. Names are unique per group. A duplicate name from two packages is an error at
   resolution time, not a silent override. First-party plugins are registered under the
   same groups by `pyproject.toml`; nothing is hard-coded in the registries.
3. A plugin whose `license` is not in `DATACOPILOT_ALLOWED_LICENSES` is listed but refuses
   to load, with a message naming the license and the variable.
4. A first-party plugin may depend on a copyleft library through an optional extra
   (`label_noise` check → `[cleanlab]`), importing it inside the method body and stating
   the license consequence in its `spec.license` and `--help`. A plugin that includes or
   derives from copyleft code ships as a separate package (`datacopilot-yoloe`). Such
   packages depend on `datacopilot`; `datacopilot` never depends on them.
5. Plugin output is untrusted input: findings, proposals and decoded tables are validated
   with the Pydantic models in `core/schema.py` before they reach the store.

## Consequences

- A new format, check, proposer or embedder is one module plus one entry-point line; the
  core does not change.
- `datacopilot doctor` can list installed plugins, their licenses and which extra each
  needs, without loading them.
- Harder: plugin authors must declare a `PluginSpec` and keep heavy imports inside method
  bodies; the test suite needs a fixture package that installs a fake plugin.
- Follow-up: the four registry modules, the `PluginSpec` model, a fixture plugin package
  under `tests/fixtures/`, and a documented template repository for third-party plugins.

## Links

- [../architecture/02-components.md](../architecture/02-components.md) (registries and
  plugin contract)
- [../engineering/dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md)
