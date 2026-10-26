# Label noise

`label_noise` points at boxes whose class might be wrong. It is still the weakest check in DataCopilot,
and this page says how weak, with the numbers.

It needs an embedder (`--embedder dinov2_small`, see [Embeddings](embeddings.md)) and no extra model:
it embeds box crops with the same model the other embedding checks use. It only reports. It never edits
a label and never proposes a change.

<!-- skip: needs the [embed] extra and a model download -->
```bash
datacopilot audit shop.dcp --checks label_noise --embedder dinov2_small --fail-on none
```

## The rule

Every finding says which of two rules produced it (`basis`).

**Box crops (default for detection datasets, `basis: crop`).**

1. Every box whose shorter side is at least `crop_min_side_px` pixels is cropped: the box grown by
   `crop_padding` of its width and height on each side, clamped to the photo, at most
   `max_crops_per_image` boxes per photo (the largest first). A crop longer than 512 pixels is downscaled
   to 512 before it is embedded. Boxes that are too small or over the cap are counted, not dropped.
2. For each crop the `suspect_k` nearest other crops are found by exact cosine similarity (chunked,
   memory near one block, independent of the row order and of whether FAISS is installed). Neighbours
   below `suspect_min_similarity` are dropped; a box left with fewer than `suspect_min_neighbours` is not
   scored.
3. The **support** of a box of class `c` is the share of usable neighbour crops whose box has class `c`.
   A box is a `suspect_class` finding (INFO) when `support <= suspect_max_support`.

**Whole photos (`basis: image`).** The Phase 6 rule, used for classification datasets, when
`suspect_basis = "image"` is set, and as the fallback: neighbours are whole photos, and a photo carries a
class when any of its boxes has it (at most 256 classes per photo count; the photos cut are counted in
`classes_truncated`). When crops were asked for but cannot be cut (Pillow missing, an embedding provider
without crop support, more than 500,000 boxes), the check runs this rule and adds a WARN
`crop_rule_unavailable` with the reason, so a weaker result never looks like the default one.

Each finding has the box (`ann_id`, photo), its class, the support, how many neighbours were used, their
mean similarity and the five classes they carry most (`neighbour_classes`, counted in `boxes` or
`images`). Suspects **rank** by support, then mean similarity, then id; at most `max_suspects_per_image`
per photo, `max_suspects_per_class` per class and 50,000 in all are kept and the rest are counted in
`suspects_truncated`.

| Setting (`CheckConfig`) | Default | Bounds |
|---|---|---|
| `suspect_basis` | `crop` | `crop` or `image` |
| `crop_min_side_px` | 32 | 1 to 4,096 |
| `crop_padding` | 0.1 | 0 to 1 |
| `max_crops_per_image` | 50 | 1 to 200 |
| `suspect_k` | 10 | 1 to 100 |
| `suspect_min_similarity` | 0.3 | above 0, at most 1 |
| `suspect_min_neighbours` | 3 | 1 to 100, not above `suspect_k` |
| `suspect_max_support` | 0.0 | 0 to below 1 |
| `suspect_min_images` | 30 | 5 and up (crops or photos the rule needs) |
| `max_suspects_per_image` | 3 | 1 to 1,000 |
| `max_suspects_per_class` | 20 | 1 to 10,000 |

The settings are fields of `CheckConfig` (`datacopilot.audit.context`); like `outlier_k` they are not
command-line options in this version.

## Crops are read, cut and cached like photos

A photo is read through the same containment as every other check (no path outside the dataset root, no
symlinks, size cap, bytes must still hash to the image id), decoded once with the bounded decoder, and its
decoded size must equal the `width` and `height` of its row; otherwise the box coordinates do not describe
these pixels and its boxes are counted as `size_mismatch`. Crop vectors live in the authenticated embedding
cache under a directory named by the crop geometry (`dinov2_small@<revision>+crop-v1-p100-m32-c50-s512`),
so vectors cut with other settings are never reused; each cached crop is bound to the SHA-256 of its photo,
and a sample is cut and embedded again on every run (see [Trust model](trust-model.md)).

## It never passes silently

A check that did not run says so, as a `skipped` finding with a `reason`: WARN for `embedder_missing`,
`no_embeddings`, `too_many_images`, `too_few_images` and `too_few_boxes` (fewer crops than
`suspect_min_images`; the payload says how many boxes were not cropped and why), INFO for `no_annotations`
and `single_class`. The shared embedding findings (`embedder_error`, `audit_incomplete`, cache findings)
apply too, and `--fail-on label_noise` fails on any finding of the check.

Boxes the check could not score are not hidden either. `annotations_not_scored` counts them, with
`reasons` for the crop rule (`too_small`, `over_image_cap`, `too_few_neighbours`, `size_mismatch`, a
read or decode reason, `time_budget`). It is a WARN, and a CRITICAL when more than `max_unchecked_fraction`
(half) of the boxes were not scored or with `--require-local-images`, like images a check could not
examine. So on a dataset of small objects the default gate (`--fail-on critical`) fails with `label_noise`
in the run, as COCO128 does below.

## What the numbers say

Measured in [`notebooks/05_split_and_crops.ipynb`](https://github.com/B4teShai/DataCopilot/blob/main/notebooks/05_split_and_crops.ipynb)
on COCO128 (128 photos, 929 boxes, 71 classes) with the real DINOv2-small model, with the protocol of
notebook 04: a seeded 5 percent of the boxes (46) get a random other class, five seeds, default settings,
caps lifted. Both rules ran on the same planted mistakes; the whole-image rule gave exactly the numbers
notebook 04 saved in Phase 6 (the notebook checks that). Means of the five seeds:

| | Whole photos (Phase 6) | Box crops (now) |
|---|---|---|
| Precision | 0.108 | 0.250 (per seed 0.210 to 0.287) |
| Recall, all 929 boxes | 0.226 | 0.470 (per seed 0.370 to 0.587) |
| False-positive rate, all boxes | 0.097 | 0.073 |
| Boxes scored | 296 | 455 |
| Recall, scorable boxes | 0.831 | 0.962 |
| False-positive rate, scorable boxes | 0.303 | 0.149 |
| Reports on the untouched dataset (share of boxes) | 0.097 | 0.071 |
| Confusion with another class of the same photo: recall, all boxes | 0.091 | 0.252 |

- **Size is the new limit.** 416 of the 929 boxes are under 32 pixels on their shorter side and are not
  cropped; 58 more have too few close neighbours. With the default caps the command line reported 94 boxes
  on seed 0's data and counted 474 not scored (a CRITICAL `annotations_not_scored`).
- Other settings, tuned on the same data and so an upper bound: a 16-pixel floor scores 659 boxes, recall
  0.665 at precision 0.243; the highest precision in the grid was 0.318 at recall 0.217 (similarity floor 0.5).
- Run time: cutting and embedding 513 crops took 8.5 s on a CPU, against 5.5 s for 128 photos; the check
  itself runs in 11 to 12 ms over cached vectors.

The notebook's computed verdict: the crop rule is better than the whole-photo rule on precision, recall and
false-positive rate, and it is **still weak**: three of four reports are not planted mistakes, and it finds
under half of them. Use a report as a short list of leads to look at, never as a list of errors, and do not
gate on it. It was not measured on a large single-domain dataset or on real (unplanted) label errors.
Confident learning (a Cleanlab backend) is not built; see [ADR-0010](../adr/0010-split-and-box-crops.md).
