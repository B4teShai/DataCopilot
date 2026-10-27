# Split

`datacopilot split` assigns every image to `train`, `val` or `test` again, so that images that
belong together stay on one side. It writes a **new** dataset. The original is unchanged, and the
new one lists it in its [lineage](concepts.md).

<!-- skip: needs a dataset; the unit tests run these commands -->
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

Images form one group when they share a key. With `--near-dup` they also join when they are near
copies. Groups joined by either rule merge.

| `--by-group` | Key of an image |
|---|---|
| `group_id` (default) | Its `group_id`, set by `ingest --group-by` or by the source format. |
| `source.<name>` | A string or integer field the reader kept in `source`, such as a COCO `video_id`. |
| `re:<pattern>` | A match on the **file name**: the named group `g`, else group 1, else the whole match. |

An existing `group_id` always stays together, whatever the key. A `group_id` that the key replaces
is kept in `source["group_id_before_split"]`.

A `re:` pattern must be safe. It has no flags, back references or look-arounds, every quantifier
applies to one character or class, and two variable-length parts are separated by a character the
first cannot match. Write `^(?P<g>[^_]+)_`, not `^(?P<g>\w+)_` (`\w` also matches `_`). The same
rules apply to `ingest --group-by`.

`--near-dup` needs `--embedder`. It joins every pair of images with cosine similarity at or above
`--similarity-threshold` (default 0.95), found exactly rather than among a few nearest neighbours,
so a crowd of copies cannot hide one. The `hash` test embedder is refused unless you pass
`--allow-test-embedder`.

## Assignment

Whole groups are placed largest first, into the split whose share of images moves least away from
`--ratios`. With `--stratify category`, the share of boxes of each class counts too. Ties are broken
by a hash of the members and `--seed`, so the same data, ratios and seed give the same result in any
row order.

Whole groups cannot always meet the ratios: one group can be larger than a target, and a class that
lives in one group cannot be in every split. The report says how far it ended: per split the images,
target and deviation, and per split and class the boxes, share and deviation. `--json PATH` writes it
all. `groups_across_splits` is 0 in every written report.

## It fails closed

Nothing is written when:

- an image could not be embedded for `--near-dup`, because it could be the leaked one;
- the near-copy search hit its time budget or found more than 2,000,000 pairs (raise the threshold
  or group by key);
- the result equals the current version or an earlier one in its lineage;
- the output exists, lies inside the source, or lies in a folder that does not hold the images;
- `--json PATH` lies inside the dataset, exists already or is a symlink;
- an option is invalid (exit 2): ratios not three numbers between 0 and 1 that sum to 1, an unknown
  or unsafe `--by-group`, `--near-dup` without `--embedder`.

The design and the measured result are in [ADR-0010](../adr/0010-split-and-box-crops.md).
