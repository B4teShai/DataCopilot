"""The public API is frozen: any change of a name, signature or field list fails here.

A deliberate change is made by editing SURFACE in the same commit, with a CHANGELOG entry (before
1.0 a minor version may break the API; after, only a major one may).
"""

import __future__

import dataclasses
import inspect
import subprocess
import sys
import types
from typing import Any

import pytest
from pydantic import BaseModel

import datacopilot as dc
from datacopilot import api, deprecation

SURFACE = {
    "Annotation": "model(ann_id,image_id,category_id,kind,bbox_xyxy_abs,score,is_crowd,occluded,truncated,attributes,provenance,source)",  # noqa: E501
    "AuditRun": "model(run_id,dataset_version,started_at,checks,embedder,counts,passed,fail_on,timings_ms,summary,findings)",  # noqa: E501
    "Category": "model(id,name,aliases,attributes)",
    "ConversionLossError": "exception(DataCopilotError)",
    "ConversionReport": "model(source_format,target_format,rows_read,rows_written,dropped,skipped)",
    "DataCopilotError": "exception(Exception)",
    "Dataset": "dataclass(manifest,images,annotations,root)",
    "DatasetManifest": "model(version,schema_version,name,task,categories,lineage,image_count,annotation_count,created_at)",  # noqa: E501
    "Finding": "model(finding_id,run_id,check,severity,image_id,ann_id,payload)",
    "FormatError": "exception(DataCopilotError)",
    "Image": "model(image_id,file_path,width,height,split,group_id,source)",
    "LicenseError": "exception(DataCopilotError)",
    "ModelError": "exception(DataCopilotError)",
    "SettingsError": "exception(ValueError)",
    "StoreError": "exception(DataCopilotError)",
    "ValidationError": "exception(DataCopilotError)",
    "__version__": "str",
    "audit": "(POSITIONAL_OR_KEYWORD:dataset: Dataset | str | Path, POSITIONAL_OR_KEYWORD:checks: Sequence[str] | Literal['all']='all', KEYWORD_ONLY:fail_on: Sequence[str]=('critical',), KEYWORD_ONLY:embedder: str | None=None, KEYWORD_ONLY:embed_batch: int | None=None, KEYWORD_ONLY:similarity_threshold: float | None=None, KEYWORD_ONLY:allow_test_embedder: bool=False, KEYWORD_ONLY:root: str | Path | None=None, KEYWORD_ONLY:workers: int | None=None, KEYWORD_ONLY:require_local_images: bool=False) -> AuditRun",  # noqa: E501
    "export": "(POSITIONAL_OR_KEYWORD:dataset: Dataset | str | Path, POSITIONAL_OR_KEYWORD:format: str, POSITIONAL_OR_KEYWORD:out: str | Path, KEYWORD_ONLY:strict: bool=False) -> ConversionReport",  # noqa: E501
    "load": "(POSITIONAL_OR_KEYWORD:path: str | Path, POSITIONAL_OR_KEYWORD:format: str | None=None, KEYWORD_ONLY:group_by: str | None=None, KEYWORD_ONLY:strict: bool=False) -> Dataset",  # noqa: E501
}
DEPRECATION_SURFACE = {
    "deprecated": "(KEYWORD_ONLY:since: str, KEYWORD_ONLY:removal: str, KEYWORD_ONLY:replacement: str) -> Callable[[Callable[P, R]], Callable[P, R]]"  # noqa: E501
}


def signature(fn: Any) -> str:
    sig = inspect.signature(fn)
    parts = []
    for p in sig.parameters.values():
        default = "" if p.default is inspect.Parameter.empty else f"={p.default!r}"
        note = "" if p.annotation is inspect.Parameter.empty else f": {p.annotation}"
        parts.append(f"{p.kind.name}:{p.name}{note}{default}")
    return f"({', '.join(parts)}) -> {sig.return_annotation}"


def describe(obj: Any) -> str:
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return "exception(" + ",".join(b.__name__ for b in obj.__bases__) + ")"
        if issubclass(obj, BaseModel):
            return "model(" + ",".join(obj.model_fields) + ")"
        if dataclasses.is_dataclass(obj):
            return "dataclass(" + ",".join(f.name for f in dataclasses.fields(obj)) + ")"
        return "class"
    if callable(obj):
        return signature(obj)
    return type(obj).__name__


def resolve(name: str) -> Any:
    # the three functions are described from their definition: `datacopilot.audit` is also the
    # sub-package (made callable), whose signature is the generic one of any callable
    return getattr(api, name) if name in api.__all__ else getattr(dc, name)


def test_the_public_surface_matches_the_snapshot() -> None:
    assert {name: describe(resolve(name)) for name in dc.__all__} == SURFACE


def test_all_is_sorted_unique_and_resolvable() -> None:
    assert list(dc.__all__) == sorted(dc.__all__, key=str.casefold) or list(dc.__all__) == sorted(
        dc.__all__
    )
    assert len(set(dc.__all__)) == len(dc.__all__)
    for name in dc.__all__:
        assert getattr(dc, name) is not None


def test_nothing_public_is_missing_from_all() -> None:
    public = {
        name
        for name in dir(dc)
        if not name.startswith("_")
        and not isinstance(getattr(dc, name), (types.ModuleType, __future__._Feature))
    }
    assert public <= set(dc.__all__), sorted(public - set(dc.__all__))


def test_unknown_attributes_raise_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        dc.nope  # noqa: B018
    with pytest.raises(ImportError):
        from datacopilot import nope  # type: ignore[attr-defined]  # noqa: F401


def test_api_module_exports_exactly_the_three_functions() -> None:
    assert api.__all__ == ["audit", "export", "load"]


def test_audit_is_callable_whether_or_not_the_subpackage_was_imported() -> None:
    code = (
        "import datacopilot as dc, types\n"
        "assert callable(dc.audit)\n"
        "import datacopilot.audit.engine\n"
        "import datacopilot.audit.registry as r\n"
        "assert isinstance(dc.audit, types.ModuleType) and callable(dc.audit)\n"
        "from datacopilot import audit\n"
        "assert audit is dc.audit and hasattr(audit, 'engine') and r.check_names()\n"
        "from datacopilot import api\n"
        "assert dc.audit.__call__.__func__.__name__ == '__call__'\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)  # noqa: S603
    assert done.returncode == 0, done.stderr


def test_importing_the_package_is_cheap() -> None:
    code = (
        "import sys, datacopilot\n"
        "heavy = {'pyarrow', 'numpy', 'pydantic', 'jinja2', 'typer', 'datacopilot.api'}\n"
        "assert not heavy & set(sys.modules), sorted(heavy & set(sys.modules))\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)  # noqa: S603
    assert done.returncode == 0, done.stderr


def test_deprecation_helper_surface() -> None:
    assert {
        n: signature(getattr(deprecation, n)) for n in deprecation.__all__
    } == DEPRECATION_SURFACE
