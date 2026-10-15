# Python API

Three functions over the same code as the commands, importable from the top-level package. The
public API is `datacopilot.__all__`; everything else may change in any release.

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

- `load` of a source dataset needs `format`; a `.dcp` directory needs none. `strict=True` raises
  `ConversionLossError` if the reader skipped or dropped anything.
- `audit` never uses the hash cache, never reads an image outside the dataset's directory
  (`Dataset.root`, or `root=`) and never writes into the dataset. A failed gate is recorded, not
  raised. The license guard, the test-only embedder refusal and the threshold checks are the same as
  on the command line.
- All expected failures are subclasses of `DataCopilotError` (`FormatError`, `ValidationError`,
  `StoreError`, `ConversionLossError`, `LicenseError`, `ModelError`); `SettingsError` is a
  `ValueError`. Usage mistakes raise `ValueError`.
- Nothing is printed to standard output. Logs are the JSON records of the command line, on standard
  error; a call configures logging for its duration unless a request id is already bound.
- `datacopilot.audit` is also the audit sub-package; it is callable, so `datacopilot.audit(ds)` works
  however the package was imported.

## Stability and deprecation

Before 1.0 a minor version may break the API and `CHANGELOG.md` says so. A renamed name keeps working
for one more minor version and emits a `DeprecationWarning` naming the replacement and the removal
version (`datacopilot.deprecation.deprecated`). `tests/unit/test_api_surface.py` fails on any change
of a public name, signature or field list.
