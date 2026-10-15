# Audit in CI

`datacopilot audit DATASET.dcp` runs the checks, writes `report.html`, `report.json` and
`report.sarif` (next to each other unless `--report`, `--json` and `--sarif` say otherwise), stores
the run as `audits/<run_id>.json` and prints one line:

```text
passed|failed critical=N warn=N info=N run_id=ID
```

## Checks

| Check | Finds | Needs |
|---|---|---|
| `integrity` | missing or unreadable images, a header that disagrees with the declared size, EXIF orientation | |
| `duplicates` | identical files (CRITICAL when they cross splits) | |
| `imbalance` | long-tail ratio, effective class number, classes missing from a split or unused | |
| `boxes` | out of bounds, zero area, tiny, extreme aspect, duplicate and conflicting boxes | |
| `near_duplicates` | clusters of look-alike images (CRITICAL across splits) | `--embedder` |
| `leakage` | cross-split near copies and `group_id` shared between splits | `--embedder` |
| `outliers` | images far from their class (INFO) | `--embedder` |

## The gate

`--fail-on` takes severities (`critical`, `warn`, `info`: that severity and above) and/or check
names; the command exits 1 if any finding matches. The default is `critical`; `--fail-on none`
applies no gate. A gate that cannot examine the data fails closed: an empty dataset, more than half
of the images unreadable, remote or outside the dataset directory, or a time or byte budget that
ended a check early are findings that gate, not silent passes.

Exit codes: `0` gate passed (or no gate); `1` a match, an empty or unreadable dataset, an embedder
that is not allowed or fails; `2` a usage or configuration error. Reports are written even when the
gate fails.

## Recommended CI settings

- Pass `--no-hash-cache` (or set `DATACOPILOT_HASH_CACHE=false`). Never restore the hash cache from
  an untrusted ref. When `CI` is set the cache is disabled unless `DATACOPILOT_CACHE_KEY` supplies an
  out-of-band key.
- Treat the dataset as untrusted input. `--allow-external-images` is refused in CI unless
  `--allow-external-images-in-ci` is added too.
- Upload `report.sarif` to your code-scanning service and keep `report.html` as an artifact; the
  HTML has no script and no remote resources.

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
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
        with:
          python-version: "3.12"
      - run: pip install datacopilot
      - run: datacopilot ingest data --format coco -o data.dcp
      - run: datacopilot audit data.dcp --no-hash-cache --fail-on critical
```
