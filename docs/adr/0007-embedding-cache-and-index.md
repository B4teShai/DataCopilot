# ADR-0007: Embedding cache and nearest-neighbour index

**Status:** accepted
**Date:** 2026-10-15

## Context

The near-duplicate, leakage and outlier checks need one vector per image and the nearest
neighbours of every vector. Computing 50,000 embeddings costs minutes on a GPU and hours on a CPU,
so the vectors must be cached and the cache must survive an interrupted run. Three constraints
shape the design:

- The cache has to live with the dataset (`dataset.dcp/embeddings/`) so that a CI gate can use
  precomputed vectors without a GPU, but a dataset directory is attacker-controlled input in a pull
  request. A forged cache that gives a test image the vector of a train image would hide leakage
  and make the gate pass. A digest next to the data only detects damage, never forgery.
- Embeddings are a function of the image bytes and the model weights, not of the dataset version,
  yet a cache that is silently reused after the tables changed is exactly what a forger relies on.
- The core install is CPU-only and carries NumPy and PyArrow but no PyTorch, FAISS or Pillow
  ([ADR-0003](0003-repo-and-tooling.md)). The index must work without the `[embed]` extra, bounded
  in memory, deterministic, and for N up to a few hundred thousand.

## Options considered

### Option A: one Parquet file per model, rewritten on every run

Simple, matches the first sketch of the data model. A crash during the rewrite loses the whole
cache, resuming is impossible, and there is no place for a per-shard digest.

### Option B: a directory of immutable Parquet shards plus a manifest

Each shard is written once (temporary file, fsync, rename); the manifest, renamed last, lists the
shards with their sha256 and carries an HMAC. A crash leaves the previous state valid; a damaged
shard costs only its own rows. More files, and a manifest to keep consistent.

### Option C: a binary `.npy` blob with a sidecar digest

Smallest and fastest to read, but a custom format that needs its own bounds logic, no schema, and
no float16 handling beyond what NumPy offers. Parquet already has footer metadata for the bounds.

### Index options

1. **Exact blocked cosine in NumPy.** Multiplies a block of rows by the full matrix and keeps the
   top k per row. Exact, deterministic, no extra dependency; cost is O(N^2 d) and memory is one
   block. About 10 s per 50,000 x 768 on a laptop CPU.
2. **FAISS** (`IndexFlatIP`, or an approximate index). Faster on large N and GPU, but a heavy
   dependency, and approximate indexes miss neighbours, which for a leakage gate means missed leakage.
3. **A graph index (HNSW).** Sub-linear queries, approximate, and needs tuning and storage.

## Decision

**Cache: Option B.** Rules reviewers check:

1. Layout `embeddings/<model_id>@<revision>/{manifest.json, shard-*.parquet}`; a shard has two
   columns (`image_id`, `vector`); identity lives in the manifest and is repeated in each shard's
   metadata.
2. Every bound is checked before any row is decoded: manifest at most 4 MiB, shard at most 128 MiB,
   at most 16,384 rows per shard, at most 2,048 shards, at most 2,000,000 rows, dimensions at most
   4,096, declared decoded size at most 256 MiB, dtype from a fixed set.
3. The sha256 of a shard is compared with the manifest on the very bytes that are then parsed.
4. The manifest carries an HMAC-SHA256 over its canonical JSON, keyed by a secret (the same
   `DATACOPILOT_CACHE_KEY` mechanism as the hash cache) and bound to model, revision, dimensions,
   dtype, weight hashes and **dataset version**. Without a key outside CI the secret is generated
   beside the cache directory; in CI without a key the cache is not read and not written. A forged
   shard cannot verify without the key; a shard from another dataset version fails the binding.
5. As a second layer a sample of cache hits (1 percent, 10 to 64) is re-embedded on every run and
   compared (cosine at least 0.999 between canonical vectors: both sides are rounded identically, so
   only the backend's nondeterminism, about 1e-5, remains). The sample is ordered by
   `sha256(run_id : image_id)` where `run_id` is a fresh UUID per invocation, so the writer of the
   cache cannot predict it. One disagreeing vector discards the whole cache for the run and
   recomputes everything. **Verified-fraction rule:** if fewer than half of the sampled hits can be
   verified (image unreadable, embedder failing, time budget), no cached vector is trusted: they are
   dropped and recomputed, an unreadable image becomes "not embedded" (the existing gating outcome),
   and an `embedding_cache_unverified` finding and event say why. A precomputed cache therefore
   needs the images to be present; without them the gate fails rather than trusting vectors it
   cannot check. This catches bulk forgery by someone who holds the key; it does not prove every
   vector, so the key stays the trust anchor.
6. A rejected shard or manifest is ignored, counted by reason and reported; the vectors are
   recomputed. A cache problem never silently reduces what is checked.
7. Vectors are L2-normalised and rounded to float16 (default) or float32. Rounding is IEEE round to
   nearest even; float16 moves each component of a unit vector by less than 1e-3. The rounding is
   applied to freshly computed vectors as well, so a check's verdict does not depend on whether the
   cache was warm.

**Index: option 1 by default, FAISS when installed.** `embed/knn.py` computes exact cosine kNN in
blocks sized so that one block of similarities plus its sort keys stays under 64 MiB. Ties are broken
by the lower index through a combined sort key, so the result is deterministic for a platform. N is
capped at 500,000 and k at 100; beyond the caps the check reports an explicit finding instead of
running. The FAISS adapter (`IndexFlatIP`, exact) returns the same neighbours within a similarity
tolerance of 1e-4 and is selected automatically when `faiss` imports. No approximate index is used,
because a missed neighbour is a missed leak.

## Consequences

- Easier: a CI gate can run the embedding checks from a precomputed cache and a CI secret, with no
  GPU and no model download; an interrupted embedding run resumes at shard granularity.
- Harder: the cache is bound to the dataset version, so a new version (a re-split, an applied
  proposal) recomputes every embedding. Carrying shards over to a new version after a spot check is
  a follow-up. The default CPU index is quadratic; at 500,000 images it needs hours and the audit
  time budget will end it with an explicit `audit_incomplete` finding.
- Forbidden: reading a shard without its digest, trusting a vector from a manifest whose MAC failed,
  and gating on an approximate neighbour search.
- Follow-up: version carry-over, a GPU-resident index for very large N, `datacopilot doctor` listing
  cache state.

## Links

- [ADR-0006](0006-dataset-format.md) (dataset layout, derived data)
- [ADR-0005](0005-plugin-system.md) (embedders as plugins, license gate)
- [../architecture/03-data-model.md](../architecture/03-data-model.md) (Embedding entity)
- [../architecture/06-cross-cutting.md](../architecture/06-cross-cutting.md) (threat model)
