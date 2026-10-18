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
   boxes are counted in `annotations_not_scored`.
4. A neighbour carries class `c` when any of its boxes has that class. The **support** of a box of
   class `c` is the share of usable neighbours that carry `c`.
5. A box is a `suspect_class` finding (INFO) when `support <= suspect_max_support`.

Each finding has the box (`ann_id`, photo), its class, the support, how many neighbours were used,
their mean similarity and the five classes the neighbours carry most (`neighbour_classes`). Findings
are ordered by support, then similarity, then id; at most `max_suspects_per_image` per photo and
`max_suspects_per_class` per class are listed and the rest are counted in `suspects_truncated`.

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

A check that did not run says so, as a WARN finding `skipped` with a `reason`: `embedder_missing`,
`no_embeddings` (every image failed to embed), `no_annotations`, `too_few_images` (fewer annotated,
embedded photos than `suspect_min_images`), `single_class`. The shared embedding findings
(`embedder_error`, `images_not_embedded`, `audit_incomplete`, cache findings) apply too, and
`--fail-on label_noise` fails on any finding of the check, including a skipped one. The
[tutorial](../tutorial.md) dataset is too small for the check and shows the `skipped` WARN.

## What the numbers say

Measured in [`notebooks/04_label_noise.ipynb`](https://github.com/B4teShai/DataCopilot/blob/main/notebooks/04_label_noise.ipynb)
on COCO128 (128 photos, 929 boxes, 71 classes) with the real DINOv2-small embeddings. A seeded 5 percent
of the boxes (46) were given a random other class; five seeds, default thresholds, caps lifted:

| | Result |
|---|---|
| Recall (planted mistakes found) | 0.23 on average (0.15 to 0.30) |
| Precision (reported that were planted) | 0.11 on average (0.08 to 0.14), 2.2 times the random base rate of 0.05 |
| False-positive rate on boxes not planted | 0.097; the untouched dataset gives 90 reports of 929 boxes |
| Boxes not scored (too few close neighbours) | 633 of 929 |
| Mistakes that are a confusion with another class in the same photo | recall 0.07 |
| Loosest thresholds tried (tuned on the same data, so an upper bound) | recall 0.84, precision 0.12, false-positive rate 0.33 |
| Run time of the check | 6 to 66 ms on 128 photos; 16.5 s on 50,000 synthetic 768-dimensional vectors (12 cores) |

The notebook's own verdict is **weak on this dataset**: most reported boxes are not mistakes, and most
mistakes are not reported. The reason is structural. The embedding describes the whole photo, not the
box, and in a small, varied sample most photos have no close neighbour; a rare but correct object looks
exactly like a wrong label ("similar photos do not contain this class"), and a wrong class that fits the
scene is invisible. Use a report as a short list of leads to look at, not as a list of errors, and do not
gate a build on it for a dataset like this one. The check was not measured on a large single-domain
dataset, where neighbours are closer; expect better numbers there but treat that as untested.

See [ADR-0009](../adr/0009-label-noise-check.md) for the decision and what was not built.
