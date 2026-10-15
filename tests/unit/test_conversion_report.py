import pytest

from datacopilot.core.errors import ConversionLossError
from datacopilot.io.report import ConversionReport, enforce_strict


def report(**kw: object) -> ConversionReport:
    base: dict[str, object] = {
        "source_format": "coco",
        "target_format": "yolo",
        "rows_read": 3,
        "rows_written": 3,
    }
    return ConversionReport(**{**base, **kw})  # type: ignore[arg-type]


def test_report_without_drops_is_lossless() -> None:
    r = report()
    assert r.is_lossless() and r.total_dropped == 0 and r.dropped == {} and r.skipped == {}


def test_any_dropped_or_skipped_value_makes_it_lossy() -> None:
    assert not report(dropped={"annotations.is_crowd": 2}).is_lossless()
    assert not report(skipped={"images.file_missing": 1}).is_lossless()


def test_total_dropped_sums_columns() -> None:
    assert report(dropped={"a": 2, "b": 3}).total_dropped == 5


def test_counts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        report(dropped={"a": 0})


def test_strict_passes_a_lossless_report_and_fails_a_lossy_one() -> None:
    enforce_strict(report(), strict=True)
    enforce_strict(report(dropped={"a": 1}), strict=False)
    with pytest.raises(ConversionLossError, match=r"annotations.is_crowd=2.*images.file_missing=1"):
        enforce_strict(
            report(dropped={"annotations.is_crowd": 2}, skipped={"images.file_missing": 1}),
            strict=True,
        )


def test_merge_adds_counts() -> None:
    a = report(dropped={"x": 1}, skipped={"s": 1})
    b = report(dropped={"x": 2, "y": 1})
    merged = a.merged_with(b.dropped, b.skipped)
    assert merged.dropped == {"x": 3, "y": 1} and merged.skipped == {"s": 1}
