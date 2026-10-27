# Audit in CI

`datacopilot audit DATASET.dcp` runs checks, writes `report.html`, `report.json` and `report.sarif`
(see [SARIF](concepts.md)), stores the run as `audits/<run_id>.json` and prints one line:

```text
passed|failed critical=N warn=N info=N run_id=ID
```

## Checks

| Check | Finds | Needs |
|---|---|---|
| `integrity` | Missing or unreadable images, a header that disagrees with the size, a file cut off after its header. | |
| `duplicates` | Identical files (CRITICAL when they cross splits). | |
| `imbalance` | Long-tail ratio, classes missing from a split, unused classes. | |
| `boxes` | Out of bounds, zero area, tiny, extreme shape, duplicate and conflicting boxes. | |
| `near_duplicates` | Clusters of look-alike images (CRITICAL across splits). | `--embedder` |
| `leakage` | Near copies across splits, and a `group_id` shared between splits. | `--embedder` |
| `label_noise` | Boxes whose class no neighbouring photo shares (INFO, weak). See [Label noise](label-noise.md). | `--embedder` |
| `outliers` | Images far from their class (INFO). | `--embedder` |

## The gate

`--fail-on` takes severities (`critical`, `warn`, `info`: that severity and above) and check names.
The command exits 1 if any finding matches. The default is `critical`; `--fail-on none` applies no
gate. Reports are written even when the gate fails.

A gate that cannot examine the data fails instead of passing: an empty dataset, more than half of
the images unreadable or outside the dataset folder, or a time or byte budget that ended a check
early all count as findings.

Exit codes: `0` passed (or no gate); `1` a match, an empty or unreadable dataset, or an embedder
that is not allowed or fails; `2` a usage or configuration error.

## Options

- `--checks all|integrity,duplicates,imbalance,boxes` picks the checks. `all` adds the four
  embedding checks only when `--embedder` is given.
- `--report PATH`, `--json PATH`, `--sarif PATH` choose where reports go.
- `--workers N` sets threads; `--hash-cache/--no-hash-cache` the file-hash cache.
- `--require-local-images` makes any image that could not be examined CRITICAL.
- `--allow-external-images` allows images outside the dataset's parent folder. In CI it is refused
  unless `--allow-external-images-in-ci` is added.
- `--no-store-audit` skips writing `audits/<run_id>.json`.

The [CLI reference](../reference/cli.md) has every option.

## Model-made annotations

Annotations a model produced (Label Studio predictions, pre-annotations) are left out of every check
unless you pass `--include-model-annotations`: model output is not ground truth. The report says how
many were left out. Accepted proposals are audited like human labels.

## Recommended CI settings

- Keep the default gate, `--fail-on critical`, or add check names such as `leakage`.
- Pass `--no-hash-cache` (or set `DATACOPILOT_HASH_CACHE=false`). Never restore a cache from an
  untrusted branch. When `CI` is set, the cache is off unless `DATACOPILOT_CACHE_KEY` supplies a key.
- Treat the dataset as untrusted input.
- Upload `report.sarif` to your code-scanning service and keep `report.html` as an artifact. The
  HTML has no script and loads nothing remote.

<!-- skip: workflow example for a repository that is not part of this project -->
```yaml
name: dataset-audit
on: [pull_request]
permissions:
  contents: read
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      # pin third-party actions to a full commit SHA
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false
      # not on PyPI: install from GitHub, pinned to a commit you have reviewed
      - run: pip install "datacopilot @ git+https://github.com/B4teShai/DataCopilot@${DATACOPILOT_COMMIT}"
        env:
          DATACOPILOT_COMMIT: "<the full commit SHA you reviewed>"
      - run: datacopilot ingest data --format coco -o data.dcp
      - run: datacopilot audit data.dcp --no-hash-cache --fail-on critical
```
