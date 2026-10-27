# Embeddings

An embedder is an image model that turns each picture into numbers (see [Concepts](concepts.md)).
With `--embedder NAME`, `audit` adds four checks: `near_duplicates`, `leakage`, `label_noise` and
`outliers`.

| Name | Model | License | Extra |
|---|---|---|---|
| `dinov2_small`, `dinov2_base` | DINOv2 | Apache-2.0 | `embed` |
| `siglip2_base` | SigLIP2 | Apache-2.0 | `embed` |
| `hash` | SimHash of file bytes, no model | n/a | none; test only |

`hash` finds copies of the same file with small edits, not look-alike photographs. It prints a
warning, is labelled `hash@1-test-only` in every report, and a gate refuses it unless you pass
`--allow-test-embedder`. Use a model for real audits.

<!-- skip: needs the [embed] extra and a model download -->
```bash
datacopilot audit shop.dcp --embedder dinov2_base --fail-on leakage,critical
```

## How a model is loaded

The registry pins each model to one commit and to the SHA-256 and size of every weight file. Only
those files are downloaded and each is verified before loading. Pickle checkpoints are impossible
and `trust_remote_code` is off. The model's license must be in `DATACOPILOT_ALLOWED_LICENSES`
(default `Apache-2.0,MIT,BSD-3-Clause`); it is checked before the dataset is read and before any
download. Weights are never redistributed.

## The cache

Vectors are cached in `dataset.dcp/embeddings/` as authenticated shards bound to the model, the
dataset version and the bytes of every image. A forged, stale or unverifiable shard is ignored and
counted, never trusted, and a sample of cached vectors is re-embedded on every run. In CI the cache
needs `DATACOPILOT_CACHE_KEY`. `datacopilot embed DATASET --embedder NAME` fills it ahead of time.
The design is in [ADR-0007](../adr/0007-embedding-cache-and-index.md).

## Failing closed

Images that could not be embedded, an embedder error, or a budget that ended the run are findings.
Under a gate, one unembeddable test image fails `leakage`.

`--similarity-threshold X` (above 0, at most 1; default 0.95) sets the cosine similarity at which
images count as near-duplicates and cross-split pairs count as leakage. Group ids come from file
paths, so leakage findings show only a hashed reference unless you pass `--show-group-labels`.
