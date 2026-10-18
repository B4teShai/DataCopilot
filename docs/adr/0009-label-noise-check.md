# ADR-0009: The label_noise check over existing embeddings

**Status:** accepted
**Date:** 2026-10-18

## Context

`label_noise` was reserved as a check name from the start, planned as a Cleanlab backend
([ADR-0001](0001-scope-and-license.md), [ADR-0005](0005-plugin-system.md)): an AGPL-3.0 optional extra
fed with class probabilities, which DataCopilot does not have, because it trains no model. The audit does
have one thing that tells photos apart: the embedding per photo that `near_duplicates`, `leakage` and
`outliers` already share. The smallest useful step is a check that reuses it, with no new model, no new
dependency and no new network surface, and that is measured honestly before anything heavier is added.

## Options considered

### Option A: wait for the Cleanlab backend

Better theory (confident learning), but it needs predicted probabilities, an AGPL extra and a trained model
this tool does not own.

### Option B: neighbour agreement over the existing photo embeddings

For each box, do the neighbouring photos carry its class? Cheap, deterministic, no dependency; weak by
construction because the vector describes the photo, not the box.

### Option C: embed every box crop with the embedder

Closer to the question (is this crop a dog?), but it adds a second embedding pass, a cache keyed by box, a
crop-size policy and a new resource bound (boxes outnumber photos by an order of magnitude).

## Decision

Option B, as the check named `label_noise` with the rule `suspect_class`.

1. **Rule.** Over embedded photos that have annotations: exact cosine k-nearest-neighbours (the shared
   chunked search, `embed/knn.py`), neighbours under a similarity floor dropped, a photo with too few usable
   neighbours not scored, a box suspect when at most `suspect_max_support` of the usable neighbours carry its
   class. Defaults `suspect_k=10`, `suspect_min_similarity=0.3`, `suspect_min_neighbours=3`,
   `suspect_max_support=0.0` were fixed before the first measurement and have not been tuned since.
2. **Deterministic and bounded.** Order is `(support, -mean similarity, ann_id)`; per-image (3) and per-class
   (20) caps keep memory and report size bounded, and the excess is counted (`suspects_truncated`); the search
   respects `max_check_seconds` and `MAX_KNN_ROWS`; per-class candidate lists are trimmed while streaming.
3. **Never a silent pass.** Missing embedder, no embeddings, no annotations, too few photos (30) and a
   single class each produce a WARN `skipped` finding with a reason; unscored boxes are counted
   (`annotations_not_scored`).
4. **INFO leads, report only.** The check cannot edit or propose anything; it is an embedding check, so it
   joins `--checks all` only with `--embedder`.
5. **Thresholds are `CheckConfig` fields** with bounds, not command-line options or environment variables.

## Measured result

On COCO128 with real DINOv2-small embeddings and five seeded 5 percent random class flips
(`notebooks/04_label_noise.ipynb`): recall 0.23, precision 0.11 (2.2 times the base rate of 0.05), false-positive
rate 0.097, 633 of 929 boxes not scored; for confusions with a class of the same photo recall 0.07; the loosest
thresholds tried, tuned on the same data, reach recall 0.84 at precision 0.12 and false-positive rate 0.33. The
check is **weak** on this dataset and the notebook says so. An independent NumPy re-derivation of the rule agrees
with the check on every box of every run.

### What was deliberately NOT built

- **Box-crop embeddings** (Option C). The proper fix for the main weakness; a separate decision with its own
  cache and resource bounds.
- **The Cleanlab backend** and its `[cleanlab]` extra. It remains planned as an optional second method of this
  check; nothing here depends on it.
- **A learned or calibrated score**, class-prior correction, or ranking by rarity. A report is ordered by
  support only.
- **`change_class` or `delete_box` proposals** from a suspect finding. The check reports; changing a human
  label is the question [ADR-0008](0008-propose-review-apply.md) left open.
- **Command-line options or environment variables for the thresholds**, a per-class threshold, a `--fail-on`
  severity of its own (it is INFO; a gate names the check).
- **Approximate nearest neighbours.** The search is exact, as for leakage: a missed neighbour here would be a
  false alarm.
- **Cross-split neighbours only, soft class sets, hierarchical classes** (a dog is also an animal).

### Accepted risks

- Noisy neighbours hide each other: when several boxes of the same kind are wrong in one cluster of photos,
  their neighbours carry the wrong class and support rises. Clustered noise is under-reported.
- Image-level evidence cannot see a confusion with a class that really occurs in the photo (recall 0.07 above).
- The measurement is one dataset of 128 photos; larger single-domain data was not tried.

## Consequences

- Easier: a first, dependency-free lead generator whose limits are measured; the report writers need no new
  code; `--fail-on label_noise` is valid.
- Harder: a user may read `suspect_class` as an error list. The docs, the notebook verdict and the INFO
  severity say otherwise.
- Forbidden: editing a label, proposing a change, or returning no findings when the check did not run.
- Follow-up: box-crop embeddings if the numbers justify the cost; Cleanlab as a second method.

## Links

- [ADR-0001](0001-scope-and-license.md), [ADR-0005](0005-plugin-system.md) (the reserved name and the AGPL extra)
- [ADR-0007](0007-embedding-cache-and-index.md) (the shared embeddings and the exact search)
- [../guide/label-noise.md](../guide/label-noise.md)
