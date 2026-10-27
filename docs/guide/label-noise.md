# Label noise

`label_noise` points at boxes whose class might be wrong. It is the weakest check in DataCopilot:
most of its leads are not mistakes. Treat each as a place to look, never as an error, and do not
gate on it.

It needs an embedder (see [Embeddings](embeddings.md)) and no other model. It only reports. It never
edits a label.

<!-- skip: needs the [embed] extra and a model download -->
```bash
datacopilot audit shop.dcp --checks label_noise --embedder dinov2_small --fail-on none
```

## How it decides

1. Every box whose shorter side is at least 32 pixels is cut out of its photo, with 10 percent
   padding, and embedded. At most 50 boxes per photo, the largest first. Boxes that are too small or
   over the cap are counted, not dropped.
2. For each crop, the 10 most similar other crops are found. Neighbours below cosine similarity 0.3
   are ignored, and a box left with fewer than 3 neighbours is not scored.
3. The **support** of a box of class `c` is the share of its neighbours whose box has class `c`.
   A box with support 0 becomes an INFO finding `suspect_class`.

Each finding gives the box, its class, the support, the neighbours used and the five classes the
neighbours carry most. Findings rank by support, then mean similarity. At most 3 per photo, 20 per
class and 50,000 in all are kept; the rest are counted.

For classification datasets, and when crops cannot be cut, the check compares whole photos instead
and says so with a WARN `crop_rule_unavailable`. Every finding names its `basis` (`crop` or
`image`), so a weaker result never looks like the default one.

## It never passes silently

A check that did not run says so, as a `skipped` finding with a reason (no embedder, too few images
or boxes, no annotations, one class). Boxes it could not score are counted in
`annotations_not_scored`. That is a WARN, and a CRITICAL when more than half the boxes were not
scored. On a dataset of small objects, the default gate `--fail-on critical` therefore fails.
`--fail-on label_noise` fails on any finding of this check.

## How weak is it

On COCO128 (929 boxes, 71 classes) with 5 percent of the boxes given a random other class, the box
rule reached precision 0.25 and recall 0.47. About three leads in four were not planted mistakes,
and under half of the planted mistakes were found. Boxes under 32 pixels are never scored, which was
416 of the 929. It was not measured on a large single-domain dataset or on real label errors.

The full table, the comparison with the whole-photo rule and the run times are in the
[changelog](https://github.com/B4teShai/DataCopilot/blob/main/CHANGELOG.md); the design is in
[ADR-0010](../adr/0010-split-and-box-crops.md).

## Settings

These are fields of `CheckConfig` in `datacopilot.audit.context`. They are not command-line options.

| Setting | Default | Meaning |
|---|---|---|
| `suspect_basis` | `crop` | `crop` or `image` |
| `crop_min_side_px` | 32 | Smallest box side that is cropped |
| `crop_padding` | 0.1 | Padding around a box, as a share of its size |
| `max_crops_per_image` | 50 | Boxes cropped per photo |
| `suspect_k` | 10 | Neighbours per box |
| `suspect_min_similarity` | 0.3 | Neighbours below this are ignored |
| `suspect_min_neighbours` | 3 | Fewest neighbours needed to score a box |
| `suspect_max_support` | 0.0 | A box is a lead when its support is at most this |
| `suspect_min_images` | 30 | Fewest crops (or photos) the rule needs |
| `max_suspects_per_image` | 3 | Leads kept per photo |
| `max_suspects_per_class` | 20 | Leads kept per class |
