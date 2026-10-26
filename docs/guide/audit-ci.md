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
| `integrity` | missing or unreadable images, a header that disagrees with the declared size, a JPEG, PNG or GIF cut off after its header (`truncated_file`, WARN), EXIF orientation | |
| `duplicates` | identical files (CRITICAL when they cross splits) | |
| `imbalance` | long-tail ratio, effective class number, classes missing from a split or unused | |
| `boxes` | out of bounds, zero area, tiny, extreme aspect, duplicate and conflicting boxes | |
| `near_duplicates` | clusters of look-alike images (CRITICAL across splits) | `--embedder` |
| `leakage` | cross-split near copies and `group_id` shared between splits | `--embedder` |
| `label_noise` | boxes whose class no neighbouring photo shares (INFO leads, weak; says when it was skipped), see [Label noise](label-noise.md) | `--embedder` |
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

## Options

`--checks all|integrity,duplicates,imbalance,boxes` chooses the checks (`all` adds the four embedding
checks only when an embedder is given). `--fail-on` is described above. `--report report.html` sets
where the HTML goes (JSON and SARIF are written next to it unless `--json` and `--sarif` are given).
`--workers N` sets the threads, `--hash-cache/--no-hash-cache` the file-hash cache,
`--require-local-images` makes any image that could not be examined CRITICAL,
`--allow-external-images` allows images outside the dataset's parent directory (refused when `CI` is
set unless `--allow-external-images-in-ci` is added) and `--no-store-audit` skips writing
`audits/<run_id>.json`. The [CLI reference](../reference/cli.md) has every option and
[Settings](../reference/settings.md) the environment variables.

## Model-generated annotations

Annotations whose provenance is `model` (Label Studio predictions, a model's pre-annotations) are left out of every check unless you pass
`--include-model-annotations`: model output is not ground truth, so it must not hide a class imbalance or
add box findings to the people who labelled the data. The report and a line on standard error say how many
were left out. Accepted proposals are audited like human labels.

## Recommended CI settings

- The default gate is `--fail-on critical`. An empty dataset, or one whose images cannot be examined
  (more than half remote or outside the dataset's parent directory), fails it.
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
