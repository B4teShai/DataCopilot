"""Exception hierarchy. Every expected failure is a DataCopilotError; the CLI maps it to exit 1."""

from __future__ import annotations


class DataCopilotError(Exception):
    """Base class for failures whose message is safe and useful to show to a user."""


class FormatError(DataCopilotError):
    """A dataset file or directory is unreadable, malformed, unsafe or in an unknown format."""


class ValidationError(DataCopilotError):
    """Rows violate a data-model invariant (see docs/architecture/03-data-model.md)."""


class StoreError(DataCopilotError):
    """A dataset.dcp directory cannot be read, written or is incompatible."""


class ConversionLossError(DataCopilotError):
    """A conversion dropped fields while strict mode was requested."""


class LicenseError(DataCopilotError):
    """A model or plugin license is outside ``DATACOPILOT_ALLOWED_LICENSES``."""


class ModelError(DataCopilotError):
    """A model cannot be resolved, downloaded, verified, loaded or run."""
