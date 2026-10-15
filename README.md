# DataCopilot

Headless dataset doctor for detection and classification datasets: lossless format
conversion, an audit that can gate CI, and model-generated fix proposals that are reviewed
by a person before they touch a label.

## Status

Built and tested: `ingest` and `export` between COCO, YOLO, Pascal VOC, CVAT for images 1.1 XML, Label
Studio JSON and ImageFolder, with a
versioned `dataset.dcp/` store, an order-independent `manifest.version`, and a count of every
field a conversion cannot carry (`--strict` turns any loss into a failure, on `export` and on `ingest`).
`datacopilot --version`, `formats` and `doctor` (versions, extras, settings, cache directory, FAISS and
GPU; `--json` for a machine-readable report) also run, and the Python API has three functions: `load`,
`audit` and `export`. Settings come from `DATACOPILOT_ENV`,
`DATACOPILOT_LOG_LEVEL`, `DATACOPILOT_MAX_TABLE_ROWS`, `DATACOPILOT_MAX_TABLE_BYTES`,
`DATACOPILOT_WORKERS`, `DATACOPILOT_CACHE_DIR`, `DATACOPILOT_CACHE_KEY` and `DATACOPILOT_HASH_CACHE`; logging is JSON on `stderr`.

`datacopilot audit` runs four CPU checks (integrity, exact duplicates, class imbalance, box sanity),
stores the run as `audits/<run_id>.json`, writes JSON, static HTML and SARIF 2.1.0 reports, and
exits 1 when `--fail-on` matches, so it can gate CI. With `--embedder` it adds three embedding
checks: near-duplicates, split leakage (`--fail-on leakage`) and per-class outliers. `datacopilot
embed` fills the embedding cache ahead of time.

Everything under "Planned usage" below is **not implemented**: label noise, split, propose, review, apply, the CVAT review integration and all proposal models. The
architecture documents mark every element as built or planned.

## Install

Requires Python 3.10 or newer.

```bash
pip install datacopilot
datacopilot --version
datacopilot doctor
```

> **Release status:** not yet published to PyPI. Until it is, build a wheel from a checkout and install that
> (see the [release runbook](https://github.com/B4teShai/DataCopilot/blob/main/docs/engineering/release.md)), or work from source as below.

From a checkout, with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/B4teShai/DataCopilot.git
cd DataCopilot
uv sync
uv run datacopilot formats
uv run datacopilot doctor
```

The package itself depends on `click`, `jinja2`, `numpy`, `pyarrow`, `pydantic`, `structlog` and
`typer`; `uv sync` also installs the development group (test, lint and type-check tools). No PyTorch, no
CUDA, no database server; the embedding extra (`pip install "datacopilot[embed]"`) adds the models.

## Usage (built)

```bash
uv run datacopilot ingest ./export --format coco -o ds.dcp          # prints manifest.version
uv run datacopilot ingest ./export --format coco -o ds.dcp --group-by "re:^images/(?P<g>[^/_]+)"
uv run datacopilot export ds.dcp --format yolo -o ./yolo_ready      # logs a conversion_loss per dropped column
uv run datacopilot export ds.dcp --format coco -o ./coco_again --strict
```

Formats: `coco` (annotations JSON per split), `yolo` (`data.yaml`, `images/`, `labels/`), `voc`
(`Annotations/`, `JPEGImages/`, `ImageSets/Main/`), `cvat_xml` (CVAT for images 1.1: `annotations.xml` or
`annotations/<split>.xml`), `label_studio` (the JSON export; boxes and choices) and `imagefolder` (class
directories, classification). Polygons, tracks, masks and other shapes that the data model cannot hold
are counted when a CVAT or Label Studio export is read (`ingest --strict` fails on them). Input is
validated: paths cannot leave the dataset directory, XML with a
DOCTYPE is rejected, boxes must lie inside their image, and identical image content is
reported by file name. What each conversion loses is listed in
[the loss matrix](https://github.com/B4teShai/DataCopilot/blob/main/docs/architecture/03-data-model.md#format-mapping-and-loss-matrix).

Exit codes: `0` success, `1` the command ran and failed (unreadable or invalid input, unknown
format, output exists, `--strict` with dropped fields), `2` usage or configuration error.

### Python API

```python
import datacopilot as dc

ds = dc.load("shop", format="coco")  # in memory; nothing is written
run = dc.audit(ds, fail_on=("warn",))  # run.passed is the gate's verdict
report = dc.export(ds, "label_studio", "shop_ls")
print(ds.manifest.image_count, run.passed, sorted(report.dropped))
```

The [tutorial](https://github.com/B4teShai/DataCopilot/blob/main/docs/tutorial.md) runs every command and script above against a generated dataset (it is
executed by `tests/docs/test_tutorial.py`); the documentation site (`uv run --group docs mkdocs build
--strict`) has a user guide, the generated CLI reference and the design documents.

### Audit as a CI gate

```bash
uv run datacopilot audit ds.dcp --fail-on critical --no-hash-cache
# passed critical=0 warn=2 info=1 run_id=31f7d518-11be-4bdf-af1e-e5c03024d5c9     (exit 0)
uv run datacopilot audit ds.dcp --fail-on warn --no-hash-cache
# failed critical=0 warn=2 info=1 run_id=efce5e5f-a037-4be5-b4ac-2934c2572f1b     (exit 1)
uv run datacopilot audit ds.dcp --fail-on leakage
# Error: Invalid value for --fail-on/--checks: --fail-on 'leakage' is not among the checks being run (it needs --embedder)  (exit 2)
```

### Embeddings, near-duplicates and leakage

```bash
uv run datacopilot audit leaky.dcp --embedder hash --allow-test-embedder --fail-on leakage --report leaky.html
# failed critical=2 warn=0 info=0 run_id=6eb90a83-1b80-450e-8157-3b32a58f0381     (exit 1: a test image is a near copy of a train image; one CRITICAL from `near_duplicates`, one from `leakage`)
uv run datacopilot audit clean.dcp --embedder hash --allow-test-embedder --fail-on leakage --report clean.html
# passed critical=0 warn=0 info=0 run_id=2b5901d5-ada0-44e7-b96f-f8cd27b274eb     (exit 0)
DATACOPILOT_ALLOWED_LICENSES=MIT uv run datacopilot audit clean.dcp --embedder dinov2_base
# error: model 'dinov2_base' is licensed 'Apache-2.0', which is not in DATACOPILOT_ALLOWED_LICENSES; add it there if you have accepted its terms   (exit 1, nothing downloaded or read)
uv run datacopilot embed clean.dcp --embedder hash
# embedded=13 total=13 cached=13 missing=0 model=hash@1 run_id=dae7e437-0bc3-4465-9869-02e23534d11d     (exit 0)
```

`--embedder` names an embedder: `dinov2_small`, `dinov2_base` and `siglip2_base` (extra `[embed]`:
`pip install "datacopilot[embed]"`; weights are downloaded by you, pinned to a commit and verified
against a sha256 before they are loaded) or `hash`, a dependency-free byte-level embedder meant for
tests and demos. `--checks all` adds `near_duplicates`, `leakage` and `outliers` only when an
embedder is given. `--embed-batch N`, `--similarity-threshold X` (default 0.95) and
`--no-embedding-cache` tune the run. Vectors are cached in `dataset.dcp/embeddings/` as verified,
authenticated shards (see [ADR-0007](https://github.com/B4teShai/DataCopilot/blob/main/docs/adr/0007-embedding-cache-and-index.md)); in CI the cache
needs `DATACOPILOT_CACHE_KEY`, and a cache that cannot be checked is never trusted. The check fails
closed: images without an embedding, an embedder error or a budget hit are findings, never silence.
The model license must be in `DATACOPILOT_ALLOWED_LICENSES` (default `Apache-2.0,MIT,BSD-3-Clause`).
The `hash` demo above compares file bytes, so it finds copies of the same file, not look-alike
photographs; it is test-only (reports label it `hash@1-test-only`, it prints a warning and is
refused for any gate without `--allow-test-embedder`). Use a model embedder for real audits. Real
DINOv2-small and SigLIP2 runs and real FAISS are covered by
`uv run --extra embed pytest -m real_model tests/real` (downloads about 1.6 GB; not part of the
default test run). `--show-group-labels` shows group ids in leakage findings (default: a hashed
reference); `--require-cache-key` fails unless `DATACOPILOT_CACHE_KEY` is set.

Options: `--checks all|integrity,duplicates,imbalance,boxes`, `--fail-on` (default `critical`; `warn`,
`info` = that severity and above; check names; `none` = no gate), `--report report.html` (JSON and SARIF are written
next to it unless `--json` / `--sarif` are given), `--workers N`, `--hash-cache/--no-hash-cache`,
`--require-local-images` (any unexamined image is CRITICAL), `--allow-external-images`
(refused when `CI` is set unless `--allow-external-images-in-ci` is added), `--no-store-audit`. Reports are written even when the gate fails. Exit codes: `0` gate
passed (or none given), `1` a `--fail-on` match or an unreadable dataset, `2` usage or
configuration error.

Security notes for CI: the default gate is `--fail-on critical`; an empty dataset and a dataset whose
images cannot be examined (more than half remote or outside the dataset's parent directory) fail it.
Pass `--no-hash-cache` (or set `DATACOPILOT_HASH_CACHE=false`) and never restore the hash cache from
an untrusted ref; when `CI` is set the cache is disabled unless an out-of-band `DATACOPILOT_CACHE_KEY`
is provided. Settings: `DATACOPILOT_CACHE_KEY`, `DATACOPILOT_WORKERS`, `DATACOPILOT_CACHE_DIR`,
`DATACOPILOT_HASH_CACHE`. Images whose real path is outside the dataset's parent directory are
not opened unless `--allow-external-images` is given. The HTML report has no script and no remote
resources; see [the security table](https://github.com/B4teShai/DataCopilot/blob/main/docs/architecture/06-cross-cutting.md#security-and-threat-model).

Benchmark (`benchmarks/audit_cpu.py`, 50,000 images and 500,000 boxes, 64 KiB files, 12 cores): 37.6 s
end to end, 1.35 GB peak RSS; details in [the overview](https://github.com/B4teShai/DataCopilot/blob/main/docs/architecture/01-overview.md).

## Planned usage

Nothing in this section runs today. It shows the intended command line and Python API so
that the design can be reviewed against concrete calls.

### Planned: more extras

Extras for the proposal workflow (vision-language models, SAM 3, the Cleanlab label-noise backend) will be
published when those features exist; today the only extra is `embed`.

### Planned: command line

```bash
datacopilot ingest ./fridge_export --format cvat_xml --group-by "re:^(?P<g>cam\d+_\d{8})" -o fridge.dcp  # built
datacopilot audit fridge.dcp --checks all --embedder dinov2_base --report report.html --fail-on leakage,critical
datacopilot split fridge.dcp --by-group --ratios 0.8,0.1,0.1 --stratify category
datacopilot propose fridge.dcp --proposer missing_boxes --models grounding-dino-base,sam3 --consensus 2
datacopilot review fridge.dcp --proposals latest --to cvat --project 12
datacopilot apply fridge.dcp --proposals latest --accepted-only
```

### Planned: Python API

`load`, `audit` and `export` are built (above). The proposal workflow is not:

```python
props = dc.propose(
    ds, "missing_boxes", models=["grounding-dino-base"], prompts_from="category.aliases"
)
props.to_fiftyone("fridge_review")  # human review in FiftyOne
ds2 = props.apply(ds, status="accepted")  # a new dataset version with lineage; ds is unchanged
```

## Architecture

DataCopilot is a CLI and a Python library over one on-disk dataset format (`dataset.dcp/`,
Parquet tables and a manifest, images referenced by path or URI and never copied). Model-backed
features live behind optional extras and plugins so the core stays CPU-only and
Apache-2.0 clean.

The diagrams (system context, containers, components, data model, flows) are in
[docs/diagrams/png/](https://github.com/B4teShai/DataCopilot/blob/main/docs/diagrams/png/).

- Architecture documents (C4 views, data model, flows, runtime, cross-cutting concerns):
  [docs/architecture/README.md](https://github.com/B4teShai/DataCopilot/blob/main/docs/architecture/README.md)
- Engineering guide (standards, testing, errors and logging, dependencies and licensing,
  workflow): [docs/engineering/README.md](https://github.com/B4teShai/DataCopilot/blob/main/docs/engineering/README.md)
- Decision records: [docs/adr/](https://github.com/B4teShai/DataCopilot/blob/main/docs/adr/)
- Third-party model licenses: [THIRD_PARTY_MODELS.md](https://github.com/B4teShai/DataCopilot/blob/main/THIRD_PARTY_MODELS.md)

## License

Apache-2.0. See [LICENSE](https://github.com/B4teShai/DataCopilot/blob/main/LICENSE) and [NOTICE](https://github.com/B4teShai/DataCopilot/blob/main/NOTICE). Model weights are never
redistributed; each model's license is listed in `THIRD_PARTY_MODELS.md` and shown by the
model registry before first use.
