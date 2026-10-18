# Label noise

`label_noise` points at boxes whose class might be wrong. It is the weakest check in DataCopilot,
and this page says how weak, with the numbers.

It needs an embedder (`--embedder dinov2_small`, see [Embeddings](embeddings.md)) and no extra
model or dependency: it reuses the one embedding per photo that the other embedding checks share.
It only reports. It never edits a label and never proposes a change.

<!-- skip: needs the [embed] extra and a model download -->
```bash
datacopilot audit shop.dcp --checks label_noise --embedder dinov2_small --fail-on none
```

## The rule

1. The base set is every embedded photo that has at least one annotation. A photo without
   annotations says nothing about classes, so it is never a neighbour.
2. For each photo the `suspect_k` nearest other photos are found by exact cosine similarity
   (the same chunked search as `near_duplicates`: memory stays near one block, the result does not
   depend on the order of the rows). Neighbours below `suspect_min_similarity` are dropped.
3. A photo left with fewer than `suspect_min_neighbours` usable neighbours is **not scored**; its
   boxes are counted in `annotations_not_scored`. At most 256 distinct classes (the lowest ids) of one
   photo count as carried, so one photo with a million classes cannot make every row slow; photos cut this
   way are counted in `classes_truncated`.
4. A neighbour carries class `c` when any of its boxes has that class. The **support** of a box of
   class `c` is the share of usable neighbours that carry `c`.
5. A box is a `suspect_class` finding (INFO) when `support <= suspect_max_support`.

Each finding has the box (`ann_id`, photo), its class, the support, how many neighbours were used,
their mean similarity and the five classes the neighbours carry most (`neighbour_classes`). Suspects
**rank** by support, then mean similarity (higher first), then id; at most `max_suspects_per_image`
per photo, `max_suspects_per_class` per class and 50,000 in all are kept (the best by that rank) and the
rest are counted in `suspects_truncated`. The rank only decides which suspects survive the caps: the
audit stores and shows findings ordered by severity, photo and box, not by rank. The search uses the
exact NumPy index whether or not FAISS is installed, so results do not depend on it.

| Threshold (`CheckConfig`) | Default | Bounds |
|---|---|---|
| `suspect_k` | 10 | 1 to 100 |
| `suspect_min_similarity` | 0.3 | above 0, at most 1 |
| `suspect_min_neighbours` | 3 | 1 to 100, not above `suspect_k` |
| `suspect_max_support` | 0.0 | 0 to below 1 |
| `suspect_min_images` | 30 | 5 and up |
| `max_suspects_per_image` | 3 | 1 to 1,000 |
| `max_suspects_per_class` | 20 | 1 to 10,000 |

The thresholds are fields of `CheckConfig` (`datacopilot.audit.context`); like `outlier_k` they are not
command-line options in this version.

## It never passes silently

A check that did not run says so, as a `skipped` finding with a `reason`. It is a WARN for
`embedder_missing`, `no_embeddings` (every image failed to embed), `too_many_images` (the embedding run
refused the dataset size) and `too_few_images` (fewer annotated, embedded photos than
`suspect_min_images`). It is an INFO for `no_annotations` and `single_class`, where there is nothing to
check. The shared embedding findings (`embedder_error`, `images_not_embedded`, `audit_incomplete`, cache
findings) apply too, and `--fail-on label_noise` fails on any finding of the check, including a skipped
one.

Boxes the check could not score are not hidden either. `annotations_not_scored` is a WARN, and a
CRITICAL when more than `max_unchecked_fraction` (half) of the boxes were not scored or with
`--require-local-images`, exactly like images a check could not examine. So on a sparse dataset the
default gate (`--fail-on critical`) fails with `label_noise` in the run, as COCO128 does below. The
[tutorial](../tutorial.md) dataset is too small for the check and shows the `skipped` WARN.

## What the numbers say

Measured in [`notebooks/04_label_noise.ipynb`](https://github.com/B4teShai/DataCopilot/blob/main/notebooks/04_label_noise.ipynb)
on COCO128 (128 photos, 929 boxes, 71 classes) with the real DINOv2-small embeddings. A seeded 5 percent
of the boxes (46) were given a random other class; five seeds, default thresholds, caps lifted. Boxes the
check cannot score are neither found nor falsely reported, so every measure has two denominators and a
box that was never scored is **not** counted as a miss of the rule.

| | All 929 boxes | The 296 scorable boxes |
|---|---|---|
| Recall (planted mistakes found) | 0.23 (per seed 0.15 to 0.30) | 0.83 (0.64 to 0.93), of about 12 planted mistakes per seed |
| False-positive rate (reports on boxes not planted) | 0.097 | 0.30 |
| Precision | 0.11 (0.08 to 0.14), the same in both | |
| Lift over the random base rate | 2.2 times (per seed 1.5 to 2.9) against 0.050 | 2.5 times (2.0 to 2.8) against 0.042 |

- **Coverage is the limit.** At the default similarity floor only 296 of 929 boxes (32 percent) have enough
  close neighbours to be scored. Of the 32 planted mistakes seed 0 missed, 31 were on boxes that were not scored.
- The untouched dataset gets 90 reports (0.097 of all boxes, 0.30 of the scorable ones). It may hold real
  mistakes, so those are not all false alarms; precision against the planted answer key is a lower bound.
- Mistakes that are a confusion with another class **in the same photo** (858 of the 929 boxes are eligible):
  recall 0.09 over all boxes and 0.28 over scorable boxes, against 0.23 and 0.83 for random classes.
- The lift has no confidence interval (five seeds, a range) and is a fair comparison only if planted mistakes
  fall uniformly on scorable boxes; they were drawn uniformly on all boxes.
- Loosest thresholds tried (tuned on the same data, so an upper bound): recall 0.84, precision 0.12,
  false-positive rate 0.33, with 5 boxes not scored.
- False alarms are not a rare-class effect in this data: the median class of a false alarm has 24 boxes,
  against 22 for all untouched boxes.
- Run time of the check: 6 to 7 ms on 128 photos; 16.5 s on 50,000 synthetic 768-dimensional vectors (12 cores).

The notebook's own verdict is **weak on this dataset** and the numbers support it: a report is mostly not
planted mistakes (89 percent), and the check finds under a quarter of the planted ones, mainly because it
cannot score most boxes in a small, varied sample. Where it can score, it finds most planted random-class
mistakes at the price of a false-positive rate near 0.3. It cannot see a wrong class that fits the scene. The
embedding describes the whole photo, not the box, and a rare but correct object looks exactly like a wrong
label ("similar photos do not contain this class"). Use a report as a short list of leads to look at, not as a
list of errors. The check was not measured on a large single-domain dataset, where neighbours are closer;
better coverage is plausible there but untested.

See [ADR-0009](../adr/0009-label-noise-check.md) for the decision and what was not built.
