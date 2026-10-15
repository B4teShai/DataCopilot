"""ConversionReport: every value a conversion could not carry is counted, never silently lost."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field, PositiveInt

from datacopilot.core.errors import ConversionLossError


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
