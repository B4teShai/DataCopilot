# Embeddings

With `--embedder NAME` the audit adds `near_duplicates`, `leakage` and `outliers`. Embedders:

| Name | Model | License | Extra |
|---|---|---|---|
| `dinov2_small`, `dinov2_base` | DINOv2 | Apache-2.0 | `[embed]` |
| `siglip2_base` | SigLIP2 | Apache-2.0 | `[embed]` |
| `hash` | dependency-free SimHash of file bytes | n/a | none; test-only |

`hash` finds copies of the same file with small edits, not look-alike photographs: it prints a
warning, is labelled `hash@1-test-only` in every report and is refused for any gate unless
`--allow-test-embedder` is given. Use a model for real audits.

## How a model is loaded

The registry pins each model to a full commit and the SHA-256 and size of every weight file; only
those files and `config.json` are downloaded, each is verified before it is loaded, pickle
checkpoints are impossible, `trust_remote_code` is off and the weight hashes are logged. The license
must be in `DATACOPILOT_ALLOWED_LICENSES` (default `Apache-2.0,MIT,BSD-3-Clause`), checked before the
dataset is read and before any download.

## The cache

Vectors are cached in `dataset.dcp/embeddings/` as authenticated shards bound to the model, the
dataset version and the bytes of every image. A forged, truncated, stale or unverifiable shard is
ignored and counted, never trusted; a sample of cached vectors is re-embedded on every run. In CI the
cache needs `DATACOPILOT_CACHE_KEY`. `datacopilot embed DATASET --embedder NAME` fills the cache
ahead of time.

## Failing closed

Images that could not be embedded, an embedder error or a budget that ended the run are findings:
under a gate one unembeddable test image fails `leakage`. `--similarity-threshold X` (above 0, at
most 1; default 0.95) sets the cosine at which images are near-duplicates and cross-split pairs are
leakage. Group ids come from file paths, so leakage findings show only a hashed reference unless
`--show-group-labels` is given.

<!-- skip: needs the [embed] extra and a model download -->
```bash
datacopilot audit shop.dcp --embedder dinov2_base --fail-on leakage,critical
```

## Demo and tests

The `hash` embedder is how the tutorial finds its planted leak without a download. Real DINOv2-small
and SigLIP2 runs and real FAISS are covered by `uv run --extra embed pytest -m real_model tests/real`,
which downloads about 1.6 GB and is not part of the default test run. Model weights are never
redistributed: you download them, and each model's license is listed in `THIRD_PARTY_MODELS.md` and
shown by the model registry before first use. `datacopilot embed clean.dcp --embedder hash` prints
`embedded=13 total=13 cached=13 missing=0 model=hash@1 run_id=...`; an embedder whose license is
not in `DATACOPILOT_ALLOWED_LICENSES` exits 1 before anything is downloaded or read.
