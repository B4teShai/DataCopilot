"""DataCopilot: audit, convert and fix detection and classification datasets.

The public API is the names in ``__all__``; see :mod:`datacopilot.api` for the contract. Names
are imported on first use, so ``import datacopilot`` (and ``datacopilot --version``) stays cheap.
"""

from __future__ import annotations

import importlib as _importlib
import types as _types
import typing as _typing

import datacopilot.audit as _audit_package

__version__ = "0.1.0"

if _typing.TYPE_CHECKING:
    from datacopilot.api import audit, export, load
    from datacopilot.core.errors import (
        ConversionLossError,
        DataCopilotError,
        FormatError,
        LicenseError,
        ModelError,
        StoreError,
        ValidationError,
    )
    from datacopilot.core.schema import (
        Annotation,
        AuditRun,
        Category,
        Dataset,
        DatasetManifest,
        Finding,
        Image,
    )
    from datacopilot.io.report import ConversionReport
    from datacopilot.settings import SettingsError

__all__ = [
    "Annotation",
    "AuditRun",
    "Category",
    "ConversionLossError",
    "ConversionReport",
    "DataCopilotError",
    "Dataset",
    "DatasetManifest",
    "Finding",
    "FormatError",
    "Image",
    "LicenseError",
    "ModelError",
    "SettingsError",
    "StoreError",
    "ValidationError",
    "__version__",
    "audit",
    "export",
    "load",
]

_LAZY = {
    "export": "datacopilot.api",
    "load": "datacopilot.api",
    "ConversionReport": "datacopilot.io.report",
    "SettingsError": "datacopilot.settings",
    **{
        name: "datacopilot.core.errors"
        for name in (
            "ConversionLossError",
            "DataCopilotError",
            "FormatError",
            "LicenseError",
            "ModelError",
            "StoreError",
            "ValidationError",
        )
    },
    **{
        name: "datacopilot.core.schema"
        for name in (
            "Annotation",
            "AuditRun",
            "Category",
            "Dataset",
            "DatasetManifest",
            "Finding",
            "Image",
        )
    },
}


class _CallableModule(_types.ModuleType):
    """``datacopilot.audit`` is the audit sub-package and, called, the API function.

    Importing a sub-package binds its name on the parent, so a function called ``audit`` in this
    namespace would be replaced by the sub-package at the first ``import datacopilot.audit``.
    Making the sub-package callable keeps ``datacopilot.audit(dataset)`` working either way.
    """

    def __call__(self, *args: _typing.Any, **kwargs: _typing.Any) -> _typing.Any:
        return _importlib.import_module("datacopilot.api").audit(*args, **kwargs)


_audit_package.__class__ = _CallableModule


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'datacopilot' has no attribute {name!r}")
    value = getattr(_importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
