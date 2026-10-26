# ADR-0010: Group-aware split, and box crops for label_noise

**Status:** accepted
**Date:** 2026-10-26

## Context

Two gaps remained in the core. `split` was the last v0.1 command still planned: the leakage check could
report near copies and shared groups across splits, but nothing could produce a split without them. And
`label_noise` compared a box with neighbouring *photos* ([ADR-0009](0009-label-noise-check.md)); on COCO128
it scored 296 of 929 boxes at precision 0.11. ADR-0009 named box-crop embeddings as the next step. Both
can reuse what exists: the store's append-only versions with lineage, the embedding pipeline and its
authenticated cache, and the exact chunked similarity search.

## Options considered

### Split

- **A: random per image, stratified.** Simple, but leaks every near copy and group the leakage check knows about.
- **B: groups from keys and near copies, assigned whole.** Groups are connected components of "same key" and
  "near copy"; a deterministic greedy assignment of whole groups; the deviation from the ratios is reported.
- **C: an optimal assignment (integer programming).** Closer to the ratios, but a solver dependency and no
  guarantee of a bounded run time.

### Label noise

- **A: keep the image rule** and wait for Cleanlab (AGPL, needs class probabilities this tool does not have).
- **B: embed every box crop with the existing embedder** and apply the same neighbour rule to crops.
- **C: a detector or classifier head** trained on the dataset (a training loop this tool does not own).

## Decision

**Split: option B.**

1. **Groups.** `--by-group` is `group_id` (default), `source.<name>` or `re:<pattern>` on the file name; labels
   use the leakage check's normalisation (one function in `core`), and the new version's `group_id` is the key.
   An existing `group_id` is always a constraint too and is never lost (kept, or moved to
   `source["group_id_before_split"]` when the key replaces it).
   `--near-dup` adds every pair at or above the threshold, found by an exact upper-triangle search
   (`knn.pairs_at_least`), not a top-k search, so a crowd of copies cannot hide an edge; pairs are taken down to
   the threshold less `SIM_EPSILON` and `BACKEND_TOLERANCE`, the slack the leakage check (with FAISS) allows.
2. **Safe patterns.** A pattern is parsed with the standard library's parser and refused unless it is at most
   200 characters, has no flags, back references or look-arounds, quantifies only single characters or classes,
   and fences each variable-length quantifier from the next with a required character it cannot match. Names over
   255 characters (counted after a URL query is removed) are refused. A search is then at most quadratic in a
   name of at most 255 characters. `ingest --group-by` uses the same check (`core/patterns.py`).
3. **Assignment.** Whole groups, largest first, ties broken by `sha256(seed:members)`, each into the split whose
   squared image deviation (plus the mean squared class deviation with `--stratify category`) grows least. The
   result depends on the data, ratios and seed only. Deviations per split and per class are reported.
4. **Append-only.** The output is a new directory written by `DatasetStore.create` with the parent's lineage plus
   the parent's version; the source is never written (`--near-dup` runs without the embedding cache).
   Relative image paths are rewritten for the output's directory, and an output whose directory does not
   hold the images is refused (as for `apply`). A group in two splits after the assignment is an internal
   error, never a written dataset; a result equal to the current or an earlier version is refused.
5. **Fail closed.** An image without an embedding, a search ended by the budget, more than 2,000,000 pairs, or a
   split identical to the current one write nothing.
6. **The package is `splits/`**, not `split/` as the components document planned: importing a sub-package named
   `split` would bind it over the `datacopilot.split` function.

**Label noise: option B, as the default rule.** Boxes of at least `crop_min_side_px` (32) on the shorter side,
padded by `crop_padding` (0.1), at most `max_crops_per_image` (50, largest first) per image, downscaled to at most
512 pixels, embedded through the existing embedder. Crops are read through the audit's containment, decoded once
per image with the bounded decoder, and must match the row's size. Vectors are cached in the existing
authenticated cache under a *variant* named by the geometry (its own directory, bound into the MAC; whole-image
caches keep their MAC), with the per-image digest check and the spot check of the image pipeline. Every box
without a vector is counted by reason in `annotations_not_scored`. The image rule stays as `suspect_basis =
"image"`, for classification datasets, and as a fallback that is reported (`crop_rule_unavailable`, WARN); every
finding names its `basis`. The defaults were fixed before the measurement.

## Measured result

`notebooks/05_split_and_crops.ipynb`, COCO128, real DINOv2-small:

- COCO128 holds no near-copy pair down to cosine 0.8. Five planted re-encoded copies in `val` failed the
  leakage gate (5 clusters); after `split --near-dup` all five sat with their originals and the gate passed
  (0 clusters). The same input and seed reproduced the same version.
- Label noise, five seeds of 5 percent planted random class flips, caps lifted, means: precision 0.250 against
  0.108 for the image rule, recall 0.470 against 0.226, false-positive rate 0.073 against 0.097, 455 boxes
  scored against 296 (416 boxes are under the 32-pixel floor). The image rule re-measured there reproduced the
  Phase 6 numbers exactly. Same-photo confusions: recall 0.252 against 0.091. Verdict: better on all three
  measures, **still weak**.

### What was deliberately NOT built

- **An optimal or solver-based assignment**, and ratio guarantees: the greedy result is reported, not promised.
- **Keeping an existing split fixed** (re-split only `unassigned`, or freeze `test`); every image is re-assigned.
- **Splitting in place** (a new version inside the same `.dcp`) and an implicit output path: `-o` is required.
- **Grouping by more than one key at once**, arbitrary user regular expressions, regexes over the full path.
- **Near-copy grouping by perceptual hash or exact duplicates alone**; the embedder is the one source of pairs.
- **A per-box fallback** to the image rule for boxes too small to crop; they are counted as not scored.
- **The Cleanlab backend** (confident learning, AGPL extra), class-aware calibration, a learned score.
- **Re-embedding reported suspects with fresh vectors**, and command-line options for the crop settings.
- **`change_class` or `delete_box` proposals** from a suspect.

### Accepted risks

- With nothing to group a near-duplicate split is a plain stratified split; class deviations on small classes
  can be large (0.2 in the notebook) and are reported, not fixed.
- A near-duplicate split computes its vectors afresh on every run (no cache), which costs a model pass.
- Boxes under the size floor are invisible to the crop rule; on small-object data most boxes may be unscored and
  `annotations_not_scored` is then CRITICAL under the default gate.
- Crops cost one model call per box (8.5 s for 513 crops on a CPU against 5.5 s for 128 photos).
- One dataset of 128 photos with planted, not real, errors.

## Consequences

- Easier: a split the leakage check verifies; a label-noise lead list more than twice as precise as before (0.250 against 0.108).
- Harder: two rules to explain; the payload says which one ran.
- Forbidden: writing into an existing version; reporting a weaker rule's result as the default one.
- Follow-up: SAM 3 box refinement, or the Cleanlab backend if leads stay this weak on real data.

## Links

- [ADR-0006](0006-dataset-format.md) (append-only versions), [ADR-0007](0007-embedding-cache-and-index.md),
  [ADR-0009](0009-label-noise-check.md)
- [../guide/split.md](../guide/split.md), [../guide/label-noise.md](../guide/label-noise.md)
