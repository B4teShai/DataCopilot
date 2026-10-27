# 02 Components

![The four layers of DataCopilot and the dependency rule](../diagrams/png/components.png)

*Each layer may import only the layers below it.*

The import package is `datacopilot`. It has four layers, defined in
[ADR-0004](../adr/0004-layering-and-dependency-rule.md).

Imports point down only. Every layer may import the domain, and the domain imports none of them.

| Layer | Packages |
|---|---|
| Interface | `cli/`, `api.py` |
| Application | `convert.py`, `audit/`, `proposals/`, `splits/`, `report/`, `doctor.py` |
| Infrastructure | `io/`, `store/`, `embed/`, `models/`, `proposers/` |
| Domain | `core/` (`schema.py`, `version.py`, `geometry.py`, `proposals.py`, `patterns.py`, `errors.py`) |
| Beside the layers | `logging.py`, `settings.py`, `deprecation.py` |

## Interface

**`cli/`** is a Typer app with one thin function per command: `doctor`, `ingest`, `export`,
`formats`, `audit`, `embed`, `split`, `propose`, `review export`, `review import`, `apply`. It is the
only error boundary: `main()` turns exceptions into exit codes, writes results to `stdout` and logs to
`stderr`. It imports use cases lazily so `--version` does not load `pyarrow`.

**`api.py`** is the public Python API: `load`, `audit`, `export`, `split`, `propose`, `apply`. It wraps
the same use cases with the same validation. A failed gate is recorded in `AuditRun.passed`, not
raised. `datacopilot.__all__` is the frozen surface, pinned by `tests/unit/test_api_surface.py`.

## Application

**`convert.py`** runs `ingest` and `export`: it picks a reader or writer by name from the `io/`
registry, validates rows against `core/schema.py`, writes or reads the store and returns a
`ConversionReport`. `export` writes to a temporary folder next to the output and renames it into place
only after `--strict` is satisfied.

```python
def ingest(source, format, out, *, group_by, name, limits) -> IngestResult
def export(ds: Dataset, format: str, out, *, strict: bool) -> ConversionReport
class ConversionReport(BaseModel):
    dropped: dict[str, int]   # column -> values the target could not hold
    skipped: dict[str, int]   # reason -> rows left out, such as images.file_missing
```

**`audit/`** runs checks, collects `Finding` rows into an `AuditRun` and evaluates the gate. Each
check is an object with a `spec` and a `run` method, registered through the entry-point group
`datacopilot.checks` (lazy, duplicate names rejected).

```python
class Check(Protocol):
    spec: PluginSpec
    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]
def run_audit(ds, checks, ctx, fail_on=()) -> AuditRun
def gate_failed(run: AuditRun, fail_on) -> bool   # True means exit code 1
```

| Check | Severity | What it looks at |
|---|---|---|
| `integrity` | CRITICAL, WARN, INFO | Image headers only, never a full decode. |
| `duplicates` | CRITICAL across splits, WARN within | SHA-256 of file bytes. |
| `imbalance` | WARN, INFO | Class counts per split. |
| `boxes` | CRITICAL, WARN | Box bounds, size, aspect, duplicates and class conflicts. |
| `near_duplicates` | CRITICAL across splits, WARN within | Clusters of embeddings at cosine 0.95 or above. |
| `leakage` | CRITICAL, plus an INFO note (`images_without_group`) | Near copies across splits and `group_id` shared across splits. |
| `outliers` | INFO | Images far from their class. |
| `label_noise` | INFO, plus WARN or CRITICAL for unscored boxes | Box crops whose neighbours disagree on the class. |

The four embedding checks share one pipeline, `audit/embedding.py` (`audit/crops.py` for box crops):

- It uses verified cached vectors, re-embeds a sample to authenticate them, and embeds the rest in batches.
- It fails closed: an image that cannot be embedded is a finding, not a pass.
- Naming an embedding check without `--embedder` is a usage error (exit 2).

Rules a contributor must keep:

- A check may only refer to rows of the audited dataset. Its output is re-validated.
- A plugin that raises anything but `KeyboardInterrupt` fails the audit; it never ends the process.
- Open image paths only through `audit/paths.py` (inside the dataset root, `O_NOFOLLOW`).
- Each check is capped at 50,000 findings, with a truncation finding that keeps the gate's verdict.

**`splits/`** re-splits a dataset by group and writes a new version. Groups are the connected
components of "same key" and, with near-duplicates, every pair found by `embed/knn.pairs_at_least`.
The package is named `splits` so it does not shadow the `datacopilot.split` function.

**`proposals/`** holds the use cases of the propose, review, apply workflow, and `proposers/` (in
infrastructure) holds the models behind a registry.

```python
class Proposer(Protocol):
    model_id: str; revision: str; weights: dict[str, str]
    def detect(self, image: bytes, phrases, size, *, min_score) -> list[Candidate]
def propose(ds, proposer, *, proposer_name, root, ...) -> ProposalSet
def export_review(ds, ps, out) -> ConversionReport
def import_review(ds, ps, directory) -> tuple[ProposalSet, ReviewSummary]
def apply_proposals(ds, ps, out, *, allowed_licenses, allow_incomplete=False, ...) -> DatasetManifest
```

The proposers are `grounding_dino` (extra `vlm`) and the test-only `test_boxes`. `apply` creates a new
dataset with `DatasetStore.create` and never touches an existing version.

**`report/`** renders an `AuditRun` to JSON, a static HTML page and SARIF 2.1.0. The HTML has no
script, no remote resource and a content-security meta tag. Every dataset-derived string is cleaned of
control and bidirectional characters, and files are written to a temporary name and renamed.

## Domain

**`core/schema.py`** holds the Pydantic models (`Category`, `Image`, `Annotation`, `DatasetManifest`,
`AuditRun`, `Finding`, `PluginSpec`) and the invariants in [03 Data model](03-data-model.md).
`core/proposals.py` holds `Proposal` and `ProposalSet`. The domain imports only the standard library,
`pydantic` and `numpy`.

**`core/version.py`** computes the content hash and extends the lineage. **`core/geometry.py`** holds
box math (conversions, clipping, IoU). **`core/errors.py`** holds `DataCopilotError` and its subclasses.

## Infrastructure

**`io/`** has one module per format and a registry. Readers keep unmapped fields in `source`; writers
re-emit them where the target allows.

```python
class Reader(Protocol):
    spec: PluginSpec
    def read(self, path: Path) -> ReadResult
class Writer(Protocol):
    spec: PluginSpec
    def write(self, ds: Dataset, out: Path) -> ConversionReport
```

Format modules import only the standard library and `core/`. `xmlsafe.py` parses XML from `expat`
events with DOCTYPE rejected, `miniyaml.py` reads the `data.yaml` subset, and `files.py` holds bounded
reads, path containment and image-header parsing.

**`store/`** owns the `dataset.dcp/` layout: Parquet tables with explicit Arrow schemas, atomic version
writes, audit runs and proposal sets. Reads are bounded and hash-verified.

```python
class DatasetStore:
    def open(cls, path, *, limits=None) -> DatasetStore
    def create(cls, path, images, annotations, categories, *, name, task, lineage=()) -> DatasetStore
    def read(self, *, verify=True) -> Dataset
    def write_audit(self, run: AuditRun) -> Path
    def write_proposals(self, ps: ProposalSet) -> Path
```

**`embed/`** holds the embedder backends (`dinov2_small`, `dinov2_base`, `siglip2_base` and the
test-only `hash`), the bounded image decoder, the authenticated cache and exact nearest-neighbour
search. An embedder takes encoded image bytes, not file paths, so path containment lives in one
reviewed place.

**`models/`** is the model registry. Each card pins a full commit and the SHA-256 of each weight file.
`get_model` raises `LicenseError` before any download if the license is not allowed.

## Registries

Four entry-point groups, one registry each ([ADR-0005](../adr/0005-plugin-system.md)):
`datacopilot.formats` (`io/registry.py`), `datacopilot.checks` (`audit/registry.py`),
`datacopilot.proposers` (`proposers/registry.py`) and `datacopilot.embedders` (`embed/registry.py`).
Every plugin has a `PluginSpec` with `name`, `kind`, `license`, `extra`, `gpu_memory_gb` and `version`.
A registry reads the spec before importing the code, rejects duplicate names, refuses a license outside
`DATACOPILOT_ALLOWED_LICENSES` and re-validates plugin output.

## Dependency rule

Imports point down. `tests/unit/test_layering.py` reads every module's imports and fails on:

- `core/` importing anything outside `core/`, the standard library, `pydantic` or `numpy`.
- `io/`, `store/`, `embed/`, `models/` or `proposers/` importing the interface or application layers.
- `io/` importing `store/`. `embed/`, `models/` and `proposers/` importing `store/` or a format module.
  `models/` importing `embed/`.
- Application or interface code importing a format module or a model backend directly. It goes through
  the registry, for example `io.registry.get_reader("coco")`.
- A module importing, at top level, anything but the standard library and the seven core dependencies.
  `torch`, `transformers` and `faiss` are imported inside the function that needs them.
- `logging.py` or `settings.py` importing any layer.
