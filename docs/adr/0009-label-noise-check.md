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
2. **Deterministic and bounded.** Suspects rank by `(support, -mean similarity, ann_id)`; the rank decides which
   survive, because the engine stores findings ordered by severity, image and annotation. Per-image (3),
   per-class (20) and global (50,000) caps bound the report, the excess is counted (`suspects_truncated`), and
   only small tuples are held while streaming (payloads are built for the survivors). The deadline is checked
   for every row, not only between kNN blocks. At most 256 classes (the lowest ids) of one image count as
   carried, so a hub image with a million classes cannot make every row cost a million operations
   (`classes_truncated`). The search is pinned to the NumPy index, so FAISS being installed cannot change the
   result, and the floor is compared with `knn.SIM_EPSILON` of slack.
3. **Never a silent pass.** Missing embedder, no embeddings, a size the embedding run refused and too few photos
   (30) are WARN `skipped` findings with a reason; no annotations and a single class are INFO `skipped`
   findings (there is nothing to check, so they must not fail a build, but they are reported). Unscored boxes
   are counted (`annotations_not_scored`) with the severity images that could not be examined get: WARN, and
   CRITICAL above `max_unchecked_fraction` (half) of the boxes or with `--require-local-images`. A check that
   scored under half of the boxes therefore fails the default gate; on COCO128-like sparse data that is the
   honest outcome.
4. **INFO leads, report only.** The check cannot edit or propose anything; it is an embedding check, so it
   joins `--checks all` only with `--embedder`.
5. **Thresholds are `CheckConfig` fields** with bounds, not command-line options or environment variables.

## Measured result

On COCO128 with real DINOv2-small embeddings and five seeded 5 percent random class flips
(`notebooks/04_label_noise.ipynb`), over two denominators because 633 of the 929 boxes are never scored (a box
that was not scored is a coverage limit, not a miss of the rule):

| | All 929 boxes | The 296 scorable boxes |
|---|---|---|
| Recall | 0.23 | 0.83 |
| False-positive rate | 0.097 | 0.30 |
| Precision | 0.11, 2.2 times the base rate of 0.050 (per seed 1.5 to 2.9) | 2.5 times the base rate of 0.042 (2.0 to 2.8) |

For confusions with a class of the same photo (858 eligible boxes) recall is 0.09 over all boxes and 0.28 over
scorable ones. The loosest thresholds tried, tuned on the same data, reach recall 0.84 at precision 0.12 and
false-positive rate 0.33. Five seeds give ranges, not confidence intervals, and the lift assumes planted mistakes
fall uniformly on scorable boxes. The untouched dataset may hold real errors, so precision against the planted key
is a lower bound. The check is **weak** on this dataset, mainly through coverage, and the notebook says so. An
independent NumPy re-derivation of the rule agrees with the check on every box and every unscored count of every run.

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
- **Re-embedding the reported suspects and their neighbours** (what `near_duplicates` does for pairs): a
  possible follow-up if the check is ever used as a gate.
- **Cross-split neighbours only, soft class sets, hierarchical classes** (a dog is also an animal).

### Accepted risks

- Noisy neighbours hide each other: when several boxes of the same kind are wrong in one cluster of photos,
  their neighbours carry the wrong class and support rises. Clustered noise is under-reported.
- Image-level evidence sees few confusions with a class that really occurs in the photo (recall 0.09 above).
- Cached vectors are trusted as the other embedding checks trust them (authenticated, spot-checked); unlike
  `near_duplicates` and `leakage`, `label_noise` does not re-embed the images behind its reports with fresh
  vectors. A cache forged by someone who holds the key could hide or invent a suspect.
- Classes beyond the 256 lowest ids of one image do not count as carried by it.
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
