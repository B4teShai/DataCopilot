# Split

`datacopilot split` assigns every image to `train`, `val` or `test` again so that images that belong
together stay on one side, and writes the result as a **new** dataset. The original is not changed; the
new one names it in its `lineage`.

<!-- skip: needs a dataset; the unit tests and notebook 05 run these commands -->
```bash
datacopilot split shop.dcp -o shop_split.dcp --ratios 0.8,0.1,0.1 --stratify category --seed 0
datacopilot split shop.dcp -o shop_split.dcp --by-group 're:^(?P<g>[^_]+)_'
datacopilot split shop.dcp -o shop_split.dcp --near-dup --embedder dinov2_small --json split.json
```

<!-- skip: illustrative; tests/unit/test_api_split.py runs the same call -->
```python
import datacopilot as dc

report = dc.split("shop.dcp", "shop_split.dcp", near_duplicates=True, embedder="dinov2_small")
print(report.dataset_version, report.groups_across_splits, report.splits["val"].deviation)
```

## Groups

Images form one group when they share a key, and with `--near-dup` when they are near copies; groups
joined by either relation merge.

| `--by-group` | Key of an image |
|---|---|
| `group_id` (default) | its `group_id` (set by `ingest --group-by` or by the source format) |
| `source.<name>` | a string or integer field the reader kept in `source`, for example a COCO `video_id` |
| `re:<pattern>` | a match of the pattern on the **file name** (the named group `g`, else group 1, else the whole match) |

Keys are compared like the leakage check compares them (case-insensitive, NFKC, surrounding space
ignored), and the new version's `group_id` is the key, so the leakage check's `group_overlap` verifies
the result. An image without a key is a group of its own.

File names come from the dataset, so a pattern is refused unless it cannot backtrack badly: at most 200
characters, no flags, back references or look-arounds, every quantifier on a single character or class,
and between two variable-length quantifiers a required character the first one cannot match. Write
`^(?P<g>[^_]+)_`, not `^(?P<g>\w+)_` (`\w` also matches `_`). File names longer than 255 characters are
refused.

`--near-dup` (needs `--embedder`) joins **every** pair of images whose embeddings have a cosine
similarity of at least `--similarity-threshold` (0.95), found exactly, not among a few nearest neighbours,
so a crowd of copies cannot hide one. The pairs are taken down to the threshold less the small slack the
leakage check allows (`1e-6` plus the `1e-4` that FAISS may differ by), so that check cannot find a
cross-split pair the split did not group. It uses and fills the dataset's embedding cache.

## Assignment

Whole groups are placed largest first, in the split whose share of images (and with
`--stratify category` the share of boxes of each class) moves least away from `--ratios`. Groups of equal
size are ordered by a hash of their members and `--seed`, so the same data, ratios and seed give the same
assignment, byte for byte, whatever the order of the rows. A split with ratio 0 receives nothing.

Splitting whole groups cannot always meet the ratios: a group can be larger than a target, and a class
that lives in one group cannot be in every split. The report says how far it ended instead of claiming
balance: per split the images, the target and the deviation, and per split and class the boxes, the share
and the deviation (`--json PATH` writes it all; the command prints the new version, a summary line and one
line per split). `groups_across_splits` is checked after the assignment and is 0 in every written report.

## It fails closed

Nothing is written when:

- an image could not be embedded for `--near-dup` (missing, unreadable, undecodable, changed since ingest
  or outside the root): its copies are unknown, so it could be the leaked one; the error counts the
  images by reason;
- the near-copy search ended at the time budget, or found more than 2,000,000 pairs (raise the threshold
  or group by key);
- the result equals the current dataset, the output exists, or it lies inside the source;
- an option is invalid (exit code 2): ratios that are not three numbers between 0 and 1 summing to 1,
  an unknown or unsafe `--by-group`, `--near-dup` without `--embedder`.

## What was measured

[`notebooks/05_split_and_crops.ipynb`](https://github.com/B4teShai/DataCopilot/blob/main/notebooks/05_split_and_crops.ipynb)
found no near-copy pair in COCO128 down to cosine 0.8 with DINOv2-small, so it planted five re-encoded
copies of train photos in `val`. The leakage gate failed on them (5 cross-split clusters, exit 1);
`split --near-dup --stratify category` kept all five with their originals, and the gate passed on the new
version (0 clusters, exit 0). The same input and seed gave the same version again; with
`--stratify category` the largest class deviation was 0.2, for classes whose boxes all ended in `train`
(the three largest: 5, 21 and 1 boxes).
