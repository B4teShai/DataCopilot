# Python API

Six functions run the same code as the commands. Import them from the top-level package.

<!-- run -->
```python
import datacopilot as dc

print(sorted(dc.__all__)[:3], callable(dc.load), callable(dc.audit), callable(dc.export))
```

<!-- output -->
```text
['Annotation', 'AuditRun', 'Category'] True True True
```

| Function | Returns | Purpose |
|---|---|---|
| `load(path, format=None, *, group_by=None, strict=False)` | `Dataset` | read a source dataset or a `.dcp` directory into memory; nothing is written |
| `audit(dataset, checks="all", *, fail_on=("critical",), embedder=None, ...)` | `AuditRun` | run the checks; `run.passed` is the gate's verdict |
| `export(dataset, format, out, *, strict=False)` | `ConversionReport` | write a format; every loss is counted |
| `split(dataset, out, *, ratios=(0.8, 0.1, 0.1), by_group="group_id", near_duplicates=False, ...)` | `SplitReport` | re-split by group (and near copies) into a new dataset; see [Split](split.md) |
| `propose(dataset, proposer="grounding_dino", *, classes=None, min_score=0.35, ...)` | `ProposalSet` | find missing boxes and store them in the `.dcp`; see [Propose, review, apply](proposals.md) |
| `apply(dataset, proposals, out, *, allow_incomplete=False)` | `DatasetManifest` | write the accepted proposals of a reviewed set as a new dataset |

<!-- skip: illustrative; the tutorial runs the same calls -->
```python
import datacopilot as dc

ds = dc.load("./export", format="coco", group_by=r"re:^images/(?P<g>[^/]+)")
run = dc.audit(ds, fail_on=("critical", "leakage"), embedder="dinov2_base")
if not run.passed:
    raise SystemExit(f"{run.counts.critical} critical findings")
dc.export(ds, "yolo", "./yolo_ready", strict=True)
```

## Behaviour

- `load` of a source dataset needs `format`; a `.dcp` folder needs none. `strict=True` raises
  `ConversionLossError` if the reader skipped or dropped anything.
- `audit` never uses the hash cache, never reads an image outside the dataset folder and never writes
  into the dataset. A failed gate is **recorded** in `run.passed`, not raised. Call
  `run.raise_for_gate()` (raises `GateFailedError`) where a failure must stop the program. In CI pass
  `require_local_images=True`, so images that could not be examined count as CRITICAL.
- Rows made by a model (for example Label Studio predictions) are left out of every check unless
  `include_model_annotations=True`; `run.summary.excluded_model_annotations` says how many.
- Expected failures are subclasses of `DataCopilotError` (`FormatError`, `ValidationError`,
  `StoreError`, `ConversionLossError`, `LicenseError`, `ModelError`). `SettingsError` is a
  `ValueError`, and usage mistakes raise `ValueError`.
- Nothing is printed to standard output. Logs are JSON records on standard error.
- `datacopilot.audit` is both a module and a callable, so `datacopilot.audit(ds)` works. For
  `help()` or an executor, use `datacopilot.api.audit`, a plain function.

## Stability

The public API is `datacopilot.__all__`; everything else may change in any release. Before 1.0 a
minor version may break it, and the changelog says so. A renamed name keeps working for one more
minor version and emits a `DeprecationWarning` that names the replacement. A test fails on any
change of a public name, signature or field list.
