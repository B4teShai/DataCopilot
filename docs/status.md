# What is built and what is planned

DataCopilot 0.1 is a dataset doctor: it converts, checks and reports. The model-assisted fix
workflow is designed but not written yet. Anything not listed as built below does not run today.

## Built and tested

| Area | What works |
|---|---|
| Convert | `ingest` and `export` between COCO, YOLO, Pascal VOC, CVAT for images 1.1 XML, Label Studio JSON and ImageFolder, through a versioned `dataset.dcp/` directory. Every field a conversion cannot carry is counted; `--strict` turns any loss into a failure, on `export` and on `ingest`. See [Formats](guide/formats.md). |
| Version | An order-independent `manifest.version`: the same data in the same format gives the same version whatever the file order. See [Concepts](guide/concepts.md). |
| Audit | Four checks on the CPU (integrity, exact duplicates, class imbalance, box sanity) and, with `--embedder`, three more (near-duplicates, split leakage, per-class outliers). Reports in JSON, static HTML and SARIF 2.1.0; the run is stored as `audits/<run_id>.json`. Exit code 1 when `--fail-on` matches. See [Audit in CI](guide/audit-ci.md). |
| Embeddings | `datacopilot embed` fills the embedding cache ahead of time; `dinov2_small`, `dinov2_base` and `siglip2_base` (extra `[embed]`) and the test-only `hash` embedder. See [Embeddings](guide/embeddings.md). |
| Diagnose | `datacopilot --version`, `formats` and `doctor` (versions, extras, settings, cache directory, FAISS and GPU; `--json` for a machine-readable report). |
| Python | `load`, `audit` and `export`. See [Python API](guide/python-api.md). |
| Settings | Environment variables, listed in [Settings](reference/settings.md). Logs are JSON on standard error. |

Benchmark (`benchmarks/audit_cpu.py`, 50,000 images and 500,000 boxes, 64 KiB files, 12 cores): 37.6 s
end to end, 1.35 GB peak memory; details in the [architecture overview](architecture/01-overview.md).

## Planned, not implemented

| Planned | What it will do |
|---|---|
| `split` | Split a dataset by group so that near copies stay on one side, optionally balanced per class. |
| label noise | Find likely wrong labels (an optional Cleanlab backend). |
| `propose` | Suggest missing boxes or fixes with vision-language models and SAM 3. |
| `review` | Send proposals to CVAT or FiftyOne for a person to accept or reject. |
| `apply` | Write the accepted proposals as a new dataset version; the original is unchanged. |
| more extras | Extras for the proposal models; today the only extra is `embed`. |

The intended command line, kept here so the design can be reviewed against concrete calls:

<!-- skip: planned, not implemented -->
```bash
datacopilot split fridge.dcp --by-group --ratios 0.8,0.1,0.1 --stratify category
datacopilot propose fridge.dcp --proposer missing_boxes --models grounding-dino-base,sam3 --consensus 2
datacopilot review fridge.dcp --proposals latest --to cvat --project 12
datacopilot apply fridge.dcp --proposals latest --accepted-only
```

<!-- skip: planned, not implemented -->
```python
props = dc.propose(
    ds, "missing_boxes", models=["grounding-dino-base"], prompts_from="category.aliases"
)
props.to_fiftyone("fridge_review")  # human review in FiftyOne
ds2 = props.apply(ds, status="accepted")  # a new dataset version; ds is unchanged
```

The [architecture documents](architecture/README.md) mark every component as built or planned.
