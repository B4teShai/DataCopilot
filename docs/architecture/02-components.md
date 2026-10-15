# 02 Components

![Components by layer](../diagrams/png/components.png)

Layers follow [ADR-0004](../adr/0004-layering-and-dependency-rule.md). Import package:
`datacopilot`. Built: `cli/` (`doctor`, `ingest`, `export`, `formats`, `audit`), `convert.py`, `audit/` (registry, runner, gate and the checks `integrity`, `duplicates`, `imbalance`, `boxes`), `report/` (JSON, HTML, SARIF), `core/`, `io/` (COCO, YOLO, VOC, ImageFolder), `store/`, `logging.py`, `settings.py`. Everything else is planned.

| Layer | Packages |
|---|---|
| Interface | `cli/`, `api.py` |
| Application | `convert.py`, `audit/`, `split/`, `propose/`, `patch/`, `report/` |
| Domain | `core/schema.py`, `core/version.py`, `core/geometry.py` |
| Infrastructure | `io/`, `store/`, `embed/`, `models/`, `integrations/` |
| Cross-cutting (beside the layers) | `logging.py`, `settings.py` |

## Interface layer

### `cli/`

Responsibility: Typer application; one command per use case; the single error boundary
(`main()`); maps exceptions to exit codes; `stdout` for results, `stderr` for logs.

Public interface (built):

```python
app: typer.Typer
def main() -> NoReturn                       # entry point `datacopilot`
def doctor() -> None                         # prints "ok", logs doctor_completed
# `datacopilot --version` prints "datacopilot <__version__>"
def ingest(path, format, output, group_by, name) -> None    # prints manifest.version
def export(dataset, format, output, strict) -> None         # prints the output directory
def formats() -> None                                       # name, read/write, license, extra
def audit(dataset, checks, fail_on, report, json, sarif, workers, hash_cache,
          allow_external_images) -> None                    # prints one result line; exit 0 / 1 / 2
```

Built commands: `ingest`, `export`, `formats`, `audit` (exit codes in their `--help` and in
[04-flows.md](04-flows.md)). Planned commands: `split`, `propose`, `review`,
`apply`, `plugins`. Each is a thin function that validates options and calls one use case.
Until `api.py` exists the built commands call `convert.py` directly (imported lazily so
`--version` and `doctor` do not load `pyarrow`).

Depends on: `convert.py` (planned: `api.py`), `logging.py`, `settings.py`. Status:
**built** for `doctor`, `ingest`, `export`, `formats`, `audit`; the rest planned.

### `api.py`

Responsibility: the public Python API; stable names under semver.

```python
def load(path: str, *, format: str | None = None, group_by: str | None = None,
         hash_images: bool = True) -> Dataset
def audit(ds: Dataset, *, checks: Sequence[str] | Literal["all"] = "all",
          embedder: str | None = None, fail_on: Sequence[str] = ()) -> AuditRun
def split(ds: Dataset, *, by_group: bool = True, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
          stratify: Literal["category"] | None = "category", seed: int = 0) -> Dataset
def propose(ds: Dataset, proposer: str, *, models: Sequence[str], prompts_from: str | None = None,
            consensus: int = 1) -> ProposalSet
class ProposalSet:
    def to_cvat(self, project_id: int) -> ReviewHandle
    def to_fiftyone(self, dataset_name: str) -> ReviewHandle
    def import_decisions(self, handle: ReviewHandle) -> "ProposalSet"
    def apply(self, ds: Dataset, *, status: Literal["accepted"] = "accepted") -> Dataset
class Dataset:
    manifest: DatasetManifest
    def export(self, format: str, out: str, *, strict: bool = False) -> ConversionReport
```

Depends on: application packages, `core/`. Status: planned.

## Application layer

### `convert.py`

Responsibility: `ingest` and `export` use cases: pick a reader or writer by name from the
`io/` registry, run it, validate rows against `core/schema.py`, write or read the store,
produce a `ConversionReport`.

```python
def ingest(source, format, out, *, group_by: str | None, name: str | None, limits) -> IngestResult
def export(ds: Dataset, format: str, out, *, strict: bool) -> ConversionReport
def parse_group_by(pattern: str) -> re.Pattern[str]
# ConversionReport lives in io/report.py (infrastructure, imported by name, not an adapter):
class ConversionReport(BaseModel):
    source_format: str; target_format: str
    rows_read: int; rows_written: int
    dropped: dict[str, int]            # column -> number of values the target could not hold
    skipped: dict[str, int]            # reason -> number of rows left out (e.g. images.file_missing)
    def is_lossless(self) -> bool      # nothing dropped and nothing skipped
```

`ingest` always hashes image bytes (`image_id`), so the planned `hash_images` switch is not
exposed. `export` writes into a sibling temporary directory and renames it into place only
after `--strict` is satisfied; the output directory must be absent or empty. Events:
`dataset_ingested`, `dataset_exported`, and one `conversion_loss` (WARN) per dropped or
skipped column.

Depends on: `core/`, `io.registry` and `io.report`, `store/`. Status: **built**.

### `audit/`

Responsibility: run checks, collect `Finding` rows into an `AuditRun`, evaluate
`--fail-on`. Checks live in `audit/checks/` and register through the entry-point group
`datacopilot.checks` (`audit/registry.py`, lazy, duplicate names rejected); each is an object with
`spec: PluginSpec` and `run`. Persistence is the store's job (`DatasetStore.write_audit`).

```python
class Check(Protocol):
    spec: PluginSpec
    def run(self, ds: Dataset, ctx: CheckContext) -> list[Finding]
class CheckContext(BaseModel, frozen=True):
    run_id: str; root: Path; cache_dir: Path; workers: int; embedder: str | None
    use_hash_cache: bool; cache_key: SecretStr | None; in_ci: bool
    allow_external_images: bool; require_local_images: bool; incomplete_is_critical: bool
    config: CheckConfig   # thresholds and budgets (max_unchecked_fraction, max_check_seconds, max_hash_bytes)
def run_audit(ds: Dataset, checks: Sequence[str] | Literal["all"], ctx: CheckContext,
              fail_on: Sequence[str] = ()) -> AuditRun
def gate_failed(run: AuditRun, fail_on: Sequence[str]) -> bool   # True -> exit code 1
```

Built checks and findings (`payload["rule"]`):

| Check | Severity | Rules |
|---|---|---|
| `integrity` | CRITICAL / WARN / INFO | `file_missing`, `unsafe_path`, `not_a_regular_file`, `unreadable`, `empty_file`, `invalid_header` (also an unrecognised format: ingest accepted every image, so a file that no longer parses was damaged or replaced), `size_mismatch`, `audit_incomplete` (time budget reached; WARN when no gate is applied) (CRITICAL); `file_too_large` (WARN); `remote_not_checked`, `external_not_checked` (WARN, CRITICAL above `max_unchecked_fraction` = 50 percent of the images or with `--require-local-images`); `exif_orientation` (INFO). Header only (PNG, JPEG, GIF, BMP with DIB header size checked), never a full decode |
| `duplicates` | CRITICAL across train/val/test or between `unassigned` and a real split, WARN within one split | `exact_duplicate` (one finding per group, `group_id` = 12-hex digest prefix), `content_changed` (CRITICAL: the file no longer hashes to `image_id`), `audit_incomplete` (CRITICAL; time or byte budget reached), `hash_cache_mismatch` (WARN), `images_not_hashed` (WARN, CRITICAL like `remote_not_checked`; counts by reason) |
| `imbalance` | WARN / INFO | `long_tail` (per split: ratio `max/min` over present classes, effective number of classes `exp(H)`), `class_missing_in_split`, `class_unused` |
| `boxes` | CRITICAL / WARN | `invalid_box`, `out_of_bounds`, `zero_area` (CRITICAL); `tiny_box`, `extreme_aspect`, `duplicate_box` (earliest same-class match), `conflicting_class` (earliest different-class match; one box can be both), `pairwise_skipped` (WARN: an image with more than 20,000 boxes) (WARN); `boxes_truncated` (INFO) |

`near_duplicates`, `leakage`, `label_noise`, `outliers` and `sku_confusion` are planned;
`--fail-on` rejects their names as "not available in this version" instead of accepting a gate
that can never fire.

`--fail-on` defaults to `critical` at the CLI; `none` (alone) applies no gate. It accepts severity names and check names. A severity name is a threshold: `warn` matches
WARN and CRITICAL, `info` matches everything. A check name matches every finding of that check.
Unknown tokens, empty tokens and check names that are not part of the run are usage errors (exit
code 2).

Run rules: an empty dataset (no images) is refused; plugin code that raises `SystemExit` or any `BaseException` other than `KeyboardInterrupt` fails the audit (exit 1) instead of ending the process; file-reading checks use a bounded window of futures (`audit/pool.py`) and a per-check time budget; checks run in registry order (`integrity`, `duplicates`, ... then third-party names
alphabetically); each check's output is re-validated (Pydantic), must carry this run's id and the
check's name and may reference only images and annotations of the audited dataset; identical keys
collapse to the most severe finding; each check is capped at `MAX_FINDINGS_PER_CHECK` (50,000)
with a `findings_truncated` finding that carries the highest dropped severity, so a gate never
changes because of the cap; findings are ordered by severity, image, annotation, id. Image
paths are classified in `audit/paths.py` (remote, unsafe, external, local) and opened with
`O_NOFOLLOW` after being resolved inside the dataset root ([06-cross-cutting.md](06-cross-cutting.md)).
`audit/hashcache.py` is the authenticated hash cache.

Depends on: `core/`, `io.files` and `io.registry` (header reader, path helpers; never an
adapter module), `logging.py`. The tables arrive as a `Dataset` read through the bounded store
reader; DuckDB is not used (see [ADR-0003](../adr/0003-repo-and-tooling.md)). Hashing and header
reads run on a thread pool (`hashlib` releases the GIL on large chunks), not a process pool.
Status: **built** for the registry, runner, gate and the four CPU checks above; embedding checks
planned.

### `split/`

Responsibility: group-aware, stratified re-split that writes a new dataset version.

```python
def resplit(ds: Dataset, *, by_group: bool, ratios: tuple[float, float, float],
            stratify: Literal["category"] | None, seed: int) -> Dataset
```

Invariant: with `by_group=True` no `group_id` appears in two splits. Depends on: `core/`,
`store/`. Status: planned.

### `propose/`

Responsibility: run proposers, gate by consensus, write a `ProposalSet`.

```python
class Proposer(Protocol):
    spec: PluginSpec
    def run(self, ds: Dataset, ctx: ProposeContext) -> list[Proposal]
def run_proposers(ds: Dataset, proposer: str, models: Sequence[str], ctx: ProposeContext) -> ProposalSet
def consensus(candidates: list[Proposal], *, min_agreement: int, iou: float = 0.7) -> list[Proposal]
```

Built-in proposers: `missing_boxes` (Grounding DINO, SAM 3), `class_verify` (Qwen3-VL),
`box_refine` (SAM 3). Depends on: `core/`, `embed/` and `models/` via registry, `store/`.
Status: planned.

### `patch/`

Responsibility: review export and import, and `apply`, which writes a new version and
appends lineage; never mutates existing rows.

```python
def export_review(ps: ProposalSet, target: Literal["cvat", "fiftyone", "xanylabeling"], **opts) -> ReviewHandle
def import_decisions(ps: ProposalSet, handle: ReviewHandle) -> ProposalSet
def apply(ds: Dataset, ps: ProposalSet, *, status: Literal["accepted"]) -> Dataset
```

Depends on: `core/`, `store/`, `integrations/` via registry. Status: planned.

### `report/`

Responsibility: render an `AuditRun` to JSON, a static HTML page and SARIF 2.1.0.

```python
def render(run: AuditRun, *, html: Path | None, json: Path | None, sarif: Path | None) -> list[Path]
```

The HTML page (Jinja2, autoescaping always on, sandboxed environment) has a score card, the checks
with timings, a class histogram (inline SVG), a split-by-class table and the findings table
(sorted by severity across all checks, then capped at 2,000 rows with the omitted count per severity; thumbnails omitted; the headline says "no gate applied" when `--fail-on none` was used); the data comes from the run itself
(`AuditRun.summary` holds the per-split class counts). It has no script, no event-handler or
`style` attributes, no links or remote resources, and a `Content-Security-Policy` meta tag that
allows only the one inline stylesheet by its SHA-256 hash. SARIF results carry relative
base names only (percent-encoded, relative to the symbolic base `DATASET_IMAGES`; no repository path is invented) plus logical locations, are sorted by severity and capped at 5,000 with a summary result, `automationDetails.id` is the stable `datacopilot/audit/` with the run id as `correlationGuid`, rule ids
`<check>/<rule>` restricted to `[a-z0-9_]`, and `partialFingerprints.findingId/v1`. Every
dataset-derived string is cleaned by `report/text.py` (control, bidirectional and zero-width
characters shown as `\uXXXX`, bounded length); JSON and SARIF are ASCII. Files are written to a
temporary sibling and renamed into place. Not yet in the report: split-by-class heatmap colours,
leakage graph, near-duplicate clusters, box histograms, SKU confusion, proposer precision (their
checks are planned).

Depends on: `core/`, `logging.py`. Status: **built** (JSON, HTML, SARIF).

## Domain layer

### `core/schema.py`

Responsibility: Pydantic models for `Category`, `Image`, `Annotation`, `DatasetManifest`,
`Proposal`, `AuditRun`, `Finding`, `Embedding`, `PluginSpec`, and the invariants in
[03-data-model.md](03-data-model.md). Imports only the standard library, `pydantic`,
`numpy`. Status: **built** for `Category`, `Image`, `Annotation`, `DatasetManifest`,
`PluginSpec`, `Dataset`, `validate_tables` (cross-row invariants), `Finding`, `AuditRun`, `Counts`
and `Summary`; `Proposal` and `Embedding` planned. Exceptions live in `core/errors.py`
(`DataCopilotError`, `FormatError`, `ValidationError`, `StoreError`, `ConversionLossError`).

### `core/version.py`

```python
def table_hash(rows: Iterable[Mapping[str, object]], *, exclude: frozenset[str] = frozenset()) -> str
def dataset_version(images, annotations, categories, *, schema_version: int) -> str
def extend_lineage(parent: DatasetManifest, child_version: str) -> tuple[str, ...]
def models_version(images, annotations, categories, *, schema_version: int) -> str   # over model rows
```

Status: **built** (golden test with an independent oracle in `tests/golden/`).

### `core/geometry.py`

```python
def iou_xyxy(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]   # (n,4),(m,4) -> (n,m)
def clip_to_image(boxes: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]
def to_cxcywh_norm(boxes_xyxy_abs, width, height) -> NDArray[np.float64]
def from_cxcywh_norm(boxes, width, height) -> NDArray[np.float64]
def rescale_grid(boxes_0_1000: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]  # Qwen3-VL output
```

Built today, on plain 4-tuples (no NumPy dependency): `xywh_to_xyxy`, `xyxy_to_xywh` (exact
inverse for boxes produced by `xywh_to_xyxy`), `xyxy_to_cxcywh_norm`, `cxcywh_norm_to_xyxy`,
`xyxy_to_norm`, `norm_to_xyxy`, `clip_to_image`. `iou_xyxy` (NumPy, imported inside the function so the tuple helpers stay import-light) is
**built** and used by the `boxes` check; `rescale_grid` and the other array variants are planned.

## Infrastructure layer

### `io/`

Responsibility: one module per format (`coco`, `yolo`, `voc`, `cvat_xml`, `labelstudio`,
`imagefolder`) and `registry.py`. Readers yield rows and keep unmapped fields in
`source`; writers re-emit them where the target allows.

```python
class ReadResult:                           # rows plus what the reader could not carry
    name; task; images; annotations; categories; dropped: dict[str, int]; skipped: dict[str, int]
class Reader(Protocol):
    spec: PluginSpec
    def read(self, path: Path) -> ReadResult
class Writer(Protocol):
    spec: PluginSpec
    def write(self, ds: Dataset, out: Path) -> ConversionReport   # `out` is an existing empty directory
def get_reader(name: str) -> Reader      # resolves lazily; raises FormatError naming the extra if missing
def get_writer(name: str) -> Writer
def list_formats() -> list[FormatInfo]   # sorted; name, read/write capability, license, extra
```

Built formats and their rules are in [03-data-model.md](03-data-model.md#format-mapping-and-loss-matrix).
Format modules import only the standard library and `core/`; `xmlsafe.py` parses XML from
`expat` events with DOCTYPE and entity declarations rejected; `miniyaml.py` parses the
`data.yaml` subset; `files.py` holds bounded reads, path containment, a regular-file-only non-blocking open and
image-header parsing (`image_header`: format, stored size, EXIF orientation; reused by the
`integrity` check). Depends on: `core/`. Status: **built** for `coco`, `yolo`, `voc`, `imagefolder` and
`registry`; `cvat_xml` and `labelstudio` planned.

### `store/`

Responsibility: `dataset.dcp/` layout, Arrow schemas, DuckDB queries, atomic snapshot
writes, embedding cache keyed by (`image_id`, `model_id`, `model_revision`).

```python
class DatasetStore:
    @classmethod
    def open(cls, path, *, limits: StoreLimits | None = None) -> DatasetStore
    @classmethod
    def create(cls, path, images, annotations, categories, *, name, task, limits=None) -> DatasetStore
    def manifest(self) -> DatasetManifest
    def read(self, *, verify: bool = True) -> Dataset    # bounded, batched, hash-verified
    def write_version(self, images, annotations, categories) -> DatasetManifest   # parent = current
    def query(self, sql: str) -> pyarrow.Table           # planned: read-only DuckDB over the tables
    def write_audit(self, run: AuditRun) -> Path
    def write_proposals(self, ps: ProposalSet) -> Path
    def embeddings(self, model_id: str, revision: str) -> pyarrow.Table | None
```

Depends on: `core/`, `pyarrow` (`duckdb`, `fsspec` arrive with audit). Status: **built**:
`create`, `open`, `read`, `write_version`, atomic writes, refuse-to-overwrite, resource limits
([06-cross-cutting.md](06-cross-cutting.md)). `write_audit` and `read_audit` (`audits/<run_id>.json`, atomic, UUID-named, symlink-safe, bounded
read) are **built**; `query`, embeddings and proposals planned.

### `embed/`

Responsibility: embedder backends (`dinov2`, `dinov3`, `siglip2`), batch decode, cache,
FAISS index for kNN.

```python
class Embedder(Protocol):
    spec: PluginSpec
    def embed(self, images: Iterable[Image], *, batch_size: int) -> NDArray[np.float32]   # (n, dims)
def knn(vectors: NDArray[np.float32], k: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]
```

Depends on: `core/`, `models/`, `[embed]` extra inside functions. Status: planned.

### `models/`

Responsibility: model registry: id, Hugging Face repo, revision, license (SPDX or custom
name), gated flag, GPU memory hint; the license guard against
`DATACOPILOT_ALLOWED_LICENSES`; weight hash recording.

```python
class ModelCard(BaseModel, frozen=True):
    model_id: str; repo: str; revision: str; license: str; gated: bool; gpu_memory_gb: float | None
def get_model(model_id: str, allowed_licenses: frozenset[str]) -> ModelCard   # raises LicenseError
def weight_hash(local_path: Path) -> str
```

Status: planned.

### `integrations/`

Responsibility: adapters to external tools: `fiftyone` (to/from FiftyOne, tags for
findings, a separate `Detections` field for proposals), `cvat` (pre-annotated tasks via
`cvat-sdk`, attributes preserved), `cleanlab` (confident-learning backend for the
`label_noise` check; `[cleanlab]` extra, AGPL-3.0, imported lazily inside the adapter
function; the check's `--help` states that using the extra makes the combined work
subject to AGPL-3.0), `lightly` (COCO-plus-folder export), `supervision` (fallback
reader). Each adapter imports its dependency inside the function and names the extra on
failure. Status: planned.

## Cross-cutting

- `logging.py` (**built**): structlog processor chain per
  [errors-and-logging.md](../engineering/errors-and-logging.md).
- `settings.py` (**built**): `Settings` from `DATACOPILOT_ENV`, `DATACOPILOT_LOG_LEVEL`;
  planned fields in [05-runtime.md](05-runtime.md).

## Registries and the plugin contract

Four entry-point groups, one registry each ([ADR-0005](../adr/0005-plugin-system.md)):

| Entry-point group | Registry module | Provides |
|---|---|---|
| `datacopilot.formats` | `io/registry.py` | `Reader` and/or `Writer` |
| `datacopilot.checks` | `audit/registry.py` | `Check` |
| `datacopilot.proposers` | `propose/registry.py` | `Proposer` |
| `datacopilot.embedders` | `embed/registry.py` | `Embedder` |

Every plugin object exposes:

```python
class PluginSpec(BaseModel, frozen=True):
    name: str  # unique within its group, snake_case
    kind: Literal["format", "check", "proposer", "embedder"]
    license: str  # SPDX id, e.g. "Apache-2.0"; custom names allowed for weights
    extra: str | None  # pip extra that provides the dependency, e.g. "embed"
    gpu_memory_gb: float | None  # None = CPU
    version: str
```

Registries resolve by name on first use, read `spec` before importing the implementation,
reject duplicate names, and refuse to load a plugin whose `license` is outside
`DATACOPILOT_ALLOWED_LICENSES`. Plugin output is validated against `core/schema.py`.

First-party registration (`pyproject.toml`, built for the four format plugins):

```toml
[project.entry-points."datacopilot.formats"]
coco = "datacopilot.io.coco:plugin"
```

## Dependency rule in this project

Imports point downward only; reviewers reject the following:

- `core/*` importing anything from `datacopilot` outside `core/`, or importing `torch`,
  `duckdb`, `pyarrow`, `fsspec`, `PIL`.
- `io/`, `store/`, `embed/`, `models/`, `integrations/` importing `cli/`, `api.py`,
  `audit/`, `split/`, `propose/`, `patch/`, `report/`, `convert.py`.
- `audit/`, `propose/`, `patch/`, `split/`, `report/`, `convert.py` importing
  `datacopilot.io.coco` (or any adapter module) directly; they call
  `io.registry.get_reader("coco")`.
- `cleanlab`, `ultralytics`, `cleanvision` or another copyleft package imported anywhere
  except inside the adapter function in `integrations/` that its extra exists for; never
  at module top level, never from `core/` or the application layer.
- `torch`, `transformers`, `faiss`, `fiftyone`, `cvat_sdk` at module top level anywhere;
  they are imported inside the function that needs them.
- `logging.py` or `settings.py` importing any layer.

The graph is enforced by `tests/unit/test_layering.py`, which reads every module's imports
(no `import-linter` dependency yet; ADR-0004 allows adding it later).
