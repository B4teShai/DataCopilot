"""ConversionReport: every value a conversion could not carry is counted, never silently lost."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, Field, PositiveInt, field_validator

from datacopilot.core.errors import ConversionLossError

MAX_COLUMNS = 512
MAX_COLUMN_CHARS = 160
_COLUMN = re.compile(r"[A-Za-z0-9_.-]+")


class ConversionReport(BaseModel, frozen=True):
    """Outcome counts of one read or write.

    Attributes:
        dropped: ``column -> number of values`` that were present but that the target format
            cannot hold (for example ``annotations.is_crowd``, ``images.source.segmentation``).
        skipped: ``reason -> number of rows`` that were left out entirely (for example
            ``images.file_missing``).
    """

    source_format: str
    target_format: str
    rows_read: int = Field(ge=0)
    rows_written: int = Field(ge=0)
    dropped: dict[str, PositiveInt] = Field(default_factory=dict)
    skipped: dict[str, PositiveInt] = Field(default_factory=dict)

    @field_validator("dropped", "skipped", mode="before")
    @classmethod
    def _without_zeros(cls, value: object) -> object:
        """A count of zero is nothing happening: it is not a column (a negative one is an error)."""
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if not (isinstance(v, int) and v == 0)}
        return value

    @field_validator("dropped", "skipped")
    @classmethod
    def _plain_columns(cls, value: dict[str, int]) -> dict[str, int]:
        """Columns are names, never data: a hostile file must not reach logs or error lines."""
        if len(value) > MAX_COLUMNS:
            raise ValueError("too many distinct columns")
        for key in value:
            if len(key) > MAX_COLUMN_CHARS or not _COLUMN.fullmatch(key):
                raise ValueError("a column is not a plain name")
        return value

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    def is_lossless(self) -> bool:
        """True when nothing was dropped and no row was skipped."""
        return not self.dropped and not self.skipped

    def merged_with(
        self, dropped: Mapping[str, int] | None = None, skipped: Mapping[str, int] | None = None
    ) -> ConversionReport:
        """A copy with extra counts added (used to fold reader losses into the report)."""
        return self.model_copy(
            update={
                "dropped": _add(self.dropped, dropped or {}),
                "skipped": _add(self.skipped, skipped or {}),
            }
        )


def _add(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    out = dict(a)
    for key, count in b.items():
        out[key] = out.get(key, 0) + count
    return out


def enforce_strict(report: ConversionReport, *, strict: bool) -> None:
    """Raise if ``strict`` and the report shows any loss.

    Raises:
        ConversionLossError: naming every dropped column and skip reason with its count.
    """
    if not strict or report.is_lossless():
        return
    parts = [f"{k}={v}" for k, v in sorted(report.dropped.items())]
    parts += [f"{k}={v}" for k, v in sorted(report.skipped.items())]
    raise ConversionLossError(
        f"conversion {report.source_format} -> {report.target_format} is lossy: " + ", ".join(parts)
    )
