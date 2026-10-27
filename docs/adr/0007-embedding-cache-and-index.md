# ADR-0007: Embedding cache and nearest-neighbour index

**Status:** accepted
**Date:** 2026-10-15

## Context

The near-duplicate, leakage and outlier checks need one vector per image and each vector's nearest
neighbours. 50,000 embeddings cost minutes on a GPU and hours on a CPU, so vectors must be cached and
the cache must survive an interrupted run. Constraints:

- The cache lives in the dataset (`dataset.dcp/embeddings/`) so CI can use precomputed vectors, but a
  dataset folder is attacker-controlled input. A forged cache giving a test image a train image's
  vector hides leakage. A digest beside the data detects damage, not forgery.
- Embeddings depend on image bytes and weights, not on the dataset version, yet a cache silently
  reused after the tables changed is what a forger relies on.
- The core install is CPU-only with no PyTorch, FAISS or Pillow ([ADR-0003](0003-repo-and-tooling.md)).
  The index must work without `embed`, in bounded memory, deterministically, for N up to a few hundred
  thousand.

## Options considered

**Cache.** A: one Parquet file per model, rewritten each run (a crash loses everything; no per-shard
digest). B: a folder of immutable Parquet shards plus a manifest listing each shard's SHA-256 and
carrying an HMAC (a crash leaves the previous state valid; a damaged shard costs only its rows). C: a
`.npy` blob with a sidecar digest (fast, but a custom format with its own bounds logic and no schema).

**Index.** 1: exact blocked cosine in NumPy (exact, deterministic, no dependency; O(N^2 d) time, one
block of memory; 14.7 s for 50,000 x 768). 2: FAISS (faster, heavy, and approximate variants miss
neighbours, which means missed leakage). 3: a graph index such as HNSW (approximate, needs tuning).

## Decision

**Cache: option B.** Rules reviewers check:

1. Layout `embeddings/<model_id>@<revision>/{manifest.json, shard-*.parquet}`. A shard has `image_id`,
   `vector` and `content_sha256` (the digest of the image bytes). Identity lives in the manifest.
2. Bounds are checked before any row is decoded: manifest at most 4 MiB, shard 128 MiB, 16,384 rows per
   shard, 2,048 shards, 2,000,000 rows, dimensions 4,096, decoded size 256 MiB, dtype from a fixed set.
3. A shard's SHA-256 is compared with the manifest on the very bytes that are then parsed.
4. The manifest carries an HMAC-SHA256 bound to model, revision, dimensions, dtype, weight hashes,
   preprocessing version and **dataset version**. The key is `DATACOPILOT_CACHE_KEY`, or outside CI a
   private `secret.key` in `DATACOPILOT_CACHE_DIR`. In CI (the `CI` variable only) without a key the
   cache is neither read nor written. `--require-cache-key` makes a missing key an error everywhere. The
   key is derived with a domain label so the hash cache never shares it.
5. Each vector carries the full SHA-256 of its image, because the dataset version cannot see image
   files. The audit hashes the file of every cache hit. Any change discards the cache; an unreadable
   file is dropped and reported as not embedded. `image_id` is a 64-bit label and is never compared.
6. A sample of hits (1 percent, 10 to 64) is re-embedded every run and compared (cosine at least
   0.999). The sample is ordered by `sha256(run_id : image_id)` with a fresh UUID per run, so the cache's
   writer cannot predict it. One disagreement discards the cache for the run. This catches bulk forgery
   only, so the key stays the trust anchor. The strongest reported and borderline pairs (within 0.02
   below the threshold) involving a cached vector are also re-embedded, at most 50 per kind. A
   contradicted pair is dropped, a confirmed borderline pair promoted, with an
   `embedding_pair_contradicted` WARN.
7. If fewer than half the sampled hits can be verified, no cached vector is trusted: all are recomputed
   and an `embedding_cache_unverified` finding says why. A precomputed cache needs the images present.
8. A rejected shard or manifest is ignored, counted by reason and reported. A cache problem never
   silently reduces what is checked.
9. Vectors are L2-normalised and rounded to float16 (default) or float32, round to nearest even, fresh
   vectors too, so a verdict does not depend on a warm cache.

**Index: option 1 by default, FAISS when installed.** `embed/knn.py` computes exact cosine kNN in
blocks under 256 MiB, ordering ties by similarity then lower index. Vectors are re-normalised in float32
and callers compare with an epsilon of 1e-6, so identical embeddings meet even a threshold of 1.0.
50,000 x 768 with k = 10 took 16 s. N is capped at 500,000 and k at 100; beyond that the check reports a
finding. The exact FAISS adapter (`IndexFlatIP`) matches within 1e-4, is used only above 20,000 base rows
when `faiss` imports, and may pick another column at an exact tie. No approximate index is used: a missed
neighbour is a missed leak.

## Consequences

- Easier: a precomputed cache saves the embedding pass in CI, and an interrupted run resumes by shard.
  The model is still needed in CI to re-embed the sample, or the gate fails closed.
- Harder: the cache is bound to the dataset version, so a re-split or applied proposal recomputes every
  embedding. The default index is quadratic: at 500,000 images it needs hours and the time budget ends
  it with `audit_incomplete`.
- Forbidden: reading a shard without its digest, trusting a vector from a manifest whose HMAC failed,
  and gating on an approximate search.
- Follow-up: carry shards over to a new version after a spot check, a GPU index for very large N,
  `doctor` listing cache state.

## Links

- [ADR-0006](0006-dataset-format.md) (dataset layout, derived data)
- [ADR-0005](0005-plugin-system.md) (embedders as plugins, license gate)
- [../architecture/03-data-model.md](../architecture/03-data-model.md) (Embedding entity)
- [../architecture/06-cross-cutting.md](../architecture/06-cross-cutting.md) (threat model)
