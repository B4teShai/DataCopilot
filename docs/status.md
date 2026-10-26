# What is built and what is planned

DataCopilot is a dataset doctor (it converts, checks and reports) with a first model-assisted fix
workflow: `propose`, `review` and `apply` for missing boxes. Anything not listed as built below does not
run today.

## Built and tested

| Area | What works |
|---|---|
| Convert | `ingest` and `export` between COCO, YOLO, Pascal VOC, CVAT for images 1.1 XML, Label Studio JSON and ImageFolder, through a versioned `dataset.dcp/` directory. Every field a conversion cannot carry is counted; `--strict` turns any loss into a failure, on `export` and on `ingest`. See [Formats](guide/formats.md). |
| Version | An order-independent `manifest.version`: the same data in the same format gives the same version whatever the file order. See [Concepts](guide/concepts.md). |
| Audit | Four checks on the CPU (integrity, exact duplicates, class imbalance, box sanity) and, with `--embedder`, four more (near-duplicates, split leakage, label-noise leads from box crops, per-class outliers). Reports in JSON, static HTML and SARIF 2.1.0; the run is stored as `audits/<run_id>.json`. Exit code 1 when `--fail-on` matches. See [Audit in CI](guide/audit-ci.md) and [Label noise](guide/label-noise.md). |
| Embeddings | `datacopilot embed` fills the embedding cache ahead of time; `dinov2_small`, `dinov2_base` and `siglip2_base` (extra `[embed]`) and the test-only `hash` embedder. See [Embeddings](guide/embeddings.md). |
| Diagnose | `datacopilot --version`, `formats` and `doctor` (versions, extras, settings, cache directory, FAISS and GPU; `--json` for a machine-readable report). |
| Split | `split` re-splits a dataset into train, val and test as a new version with lineage: images with the same key (`group_id`, a source field or a safe file-name pattern) and, with `--near-dup`, every near-copy pair stay together; whole groups are assigned deterministically, optionally balanced per class, and the deviation from the ratios is reported. See [Split](guide/split.md). |
| Propose, review, apply | `propose` finds boxes a dataset is probably missing with Grounding DINO (tiny, Apache-2.0, extra `[vlm]`) and stores them as a proposal set; `review export` and `review import` round-trip them through CVAT files; `apply` writes the accepted ones as a new dataset version with lineage. Only missing boxes (`add_box`) are proposed. See [Propose, review, apply](guide/proposals.md). |
| Python | `load`, `audit`, `export`, `split`, `propose` and `apply`. See [Python API](guide/python-api.md). |
| Settings | Environment variables, listed in [Settings](reference/settings.md). Logs are JSON on standard error. |

Benchmark (`benchmarks/audit_cpu.py`, 50,000 images and 500,000 boxes, 64 KiB files, 12 cores): 37.6 s
end to end, 1.35 GB peak memory; details in the [architecture overview](architecture/01-overview.md).

## Planned, not implemented

| Planned | What it will do |
|---|---|
| better label noise | An optional Cleanlab backend for `label_noise`; with box crops it is better than before and still weak (see [Label noise](guide/label-noise.md)). |
| more proposals | Proposals that change a class, delete a box or refine a box; SAM 3 and Qwen3-VL proposers; consensus between models; review in FiftyOne. See [ADR-0008](adr/0008-propose-review-apply.md) for what was left out on purpose. |
| more extras | Extras for SAM 3 and the Cleanlab backend; today the extras are `embed` and `vlm`. |

The [architecture documents](architecture/README.md) mark every component as built or planned.
