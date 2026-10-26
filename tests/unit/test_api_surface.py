"""The public API is frozen: any change of a name, signature or field list fails here.

A deliberate change regenerates api_surface.json in the same commit, with a CHANGELOG entry (before
1.0 a minor version may break the API; after, only a major one may).
"""

import __future__

import dataclasses
import inspect
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import datacopilot as dc
from datacopilot import api, deprecation

SNAPSHOT = Path(__file__).with_name("api_surface.json")
"""The pinned surface, one entry per name of ``__all__``: for a function its full signature, for a
pydantic model every field with its type and default, for a dataclass the same, for an exception
its whole MRO. Regenerate deliberately with ``UPDATE_API_SURFACE=1 uv run pytest
tests/unit/test_api_surface.py`` and review the diff."""
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


def _split_top(text: str) -> list[str]:
    """``text`` cut at the commas that are not inside brackets."""
    parts, depth, start = [], 0, 0
    for n, char in enumerate(text):
        depth += {"[": 1, "]": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(text[start:n].strip())
            start = n + 1
    return [*parts, text[start:].strip()]


def _unions(text: str) -> str:
    """``typing.Optional[X]`` and ``typing.Union[A, B]`` written as ``X | None`` and ``A | B``.

    Python 3.14 prints every union in the ``|`` form, earlier versions print a union that holds a
    ``Literal`` in the ``typing`` form; the surface is the same, so the snapshot uses one spelling.
    """
    for name in ("typing.Optional[", "typing.Union["):
        while (start := text.find(name)) != -1:
            depth, end = 0, start + len(name) - 1
            for end in range(start + len(name) - 1, len(text)):
                depth += {"[": 1, "]": -1}.get(text[end], 0)
                if depth == 0:
                    break
            members = _split_top(text[start + len(name) : end])
            if name == "typing.Optional[":
                members = [*members, "None"]
            text = text[:start] + " | ".join(members) + text[end + 1 :]
    return text


def _clean(text: str) -> str:
    text = re.sub(r" at 0x[0-9a-f]+", "", text)  # a function's address is not part of the surface
    return _unions(text)


def test_union_spellings_are_normalised() -> None:
    optional = "typing.Optional[typing.Literal['a', 'b']]=x"
    assert _unions(optional) == "typing.Literal['a', 'b'] | None=x"
    assert _unions("typing.Union[int, typing.Optional[str]]") == "int | str | None"
    assert _unions("dict[str, int] | None") == "dict[str, int] | None"


def _field(field: Any) -> str:
    if field.default is not PydanticUndefined:
        default = repr(field.default)
    elif field.default_factory is not None:
        default = "factory:" + getattr(field.default_factory, "__name__", "?")
    else:
        default = "required"
    return _clean(f"{field.annotation!r}={default}")


def describe(obj: Any) -> Any:
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return [c.__name__ for c in obj.__mro__]
        if issubclass(obj, BaseModel):
            return {name: _field(f) for name, f in obj.model_fields.items()}
        if dataclasses.is_dataclass(obj):
            return {
                f.name: _clean(
                    f"{f.type}="
                    + ("required" if f.default is dataclasses.MISSING else repr(f.default))
                )
                for f in dataclasses.fields(obj)
            }
        return "class"
    if callable(obj):
        return signature(obj)
    return type(obj).__name__


def resolve(name: str) -> Any:
    # the three functions are described from their definition: `datacopilot.audit` is also the
    # sub-package (made callable), whose signature is the generic one of any callable
    return getattr(api, name) if name in api.__all__ else getattr(dc, name)


def surface() -> dict[str, Any]:
    return {name: describe(resolve(name)) for name in sorted(dc.__all__)}


def test_the_public_surface_matches_the_snapshot() -> None:
    if os.environ.get("UPDATE_API_SURFACE"):
        SNAPSHOT.write_text(json.dumps(surface(), indent=1, sort_keys=True) + "\n")
    assert surface() == json.loads(SNAPSHOT.read_text())


def test_the_snapshot_pins_types_defaults_and_the_exception_hierarchy() -> None:
    pinned = json.loads(SNAPSHOT.read_text())
    assert pinned["Annotation"]["provenance"].endswith("='human'")
    assert pinned["Annotation"]["kind"].startswith("typing.Literal[")
    assert pinned["Summary" if "Summary" in pinned else "AuditRun"]
    assert pinned["FormatError"] == [
        "FormatError",
        "DataCopilotError",
        "Exception",
        "BaseException",
        "object",
    ]
    assert pinned["SettingsError"][:2] == ["SettingsError", "ValueError"]
    assert pinned["GateFailedError"][:2] == ["GateFailedError", "DataCopilotError"]
    assert "root" in pinned["Dataset"] and pinned["Dataset"]["root"].endswith("=None")


def test_load_and_export_are_the_api_functions_themselves() -> None:
    assert dc.load is api.load and dc.export is api.export


def test_audit_forwards_to_the_api_function(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, Any]] = []

    def fake(*args: Any, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        return "run"

    monkeypatch.setattr(api, "audit", fake)
    assert dc.audit("ds", fail_on=()) == "run"
    assert seen == [(("ds",), {"fail_on": ()})]


def test_audit_introspects_and_pickles_like_the_function() -> None:
    import pickle

    assert inspect.signature(dc.audit) == inspect.signature(api.audit)
    assert inspect.unwrap(dc.audit) is api.audit
    assert pickle.loads(pickle.dumps(dc.audit)) is dc.audit  # noqa: S301
    import functools

    assert pickle.loads(pickle.dumps(functools.partial(dc.audit, fail_on=()))).keywords == {  # noqa: S301
        "fail_on": ()
    }


def test_audit_can_be_sent_to_another_process() -> None:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1) as pool:
        assert pool.submit(inspect.signature, api.audit).result() == inspect.signature(api.audit)
        assert pool.submit(callable, dc.audit).result() is True


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


def test_api_module_exports_exactly_the_five_functions() -> None:
    assert api.__all__ == ["apply", "audit", "export", "load", "propose"]


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
