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


def test_counts_must_be_positive_and_a_zero_is_not_a_column() -> None:
    with pytest.raises(ValueError):
        report(dropped={"a": -1})
    assert report(dropped={"a": 0, "b": 2}, skipped={"c": 0}).dropped == {"b": 2}
    assert report(skipped={"c": 0}).is_lossless()


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


# ------------------------------------------------- attacker-controlled keys never become counters
def test_safe_key_accepts_plain_names_and_replaces_everything_else() -> None:
    from datacopilot.io.common import MAX_KEY_CHARS, safe_key

    assert safe_key("date_captured") == "date_captured" and safe_key("a.b-c_1") == "a.b-c_1"
    for hostile in (
        "",
        "x" * (MAX_KEY_CHARS + 1),
        "a b",
        "\x1b[31mred",
        "::error::x",
        "é",
        "a\nb",
        5,
        None,
    ):
        assert safe_key(hostile) == "invalid_key", hostile
    assert safe_key("x" * MAX_KEY_CHARS) == "x" * MAX_KEY_CHARS


def test_distinct_counters_are_capped_and_aggregated_into_other() -> None:
    from collections import Counter

    from datacopilot.io.common import MAX_COUNTERS, count

    counter: Counter[str] = Counter()
    for n in range(MAX_COUNTERS * 3):
        count(counter, f"images.source.k{n}")
    assert len(counter) <= MAX_COUNTERS + 1
    assert counter["images.other"] == MAX_COUNTERS * 2
    assert sum(counter.values()) == MAX_COUNTERS * 3
    count(counter, "images.source.k0")  # an existing counter keeps counting
    assert counter["images.source.k0"] == 2


def test_report_refuses_keys_that_are_not_plain_names() -> None:
    import pytest

    from datacopilot.io.report import ConversionReport

    ok = {"annotations.source.area": 3}
    ConversionReport(source_format="a", target_format="b", rows_read=1, rows_written=1, dropped=ok)
    for bad in ("\x1b[31m", "::error::x", "a b", "x" * 200, ""):
        with pytest.raises(ValueError):
            ConversionReport(
                source_format="a", target_format="b", rows_read=1, rows_written=1, dropped={bad: 1}
            )
    with pytest.raises(ValueError):
        ConversionReport(
            source_format="a", target_format="b", rows_read=1, rows_written=1,
            skipped={f"k{n}": 1 for n in range(2000)},
        )  # fmt: skip


def test_hostile_source_keys_in_a_coco_export_become_safe_report_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json
    from pathlib import Path

    from fixtures import synth

    from datacopilot import convert
    from datacopilot.store import DatasetStore

    root = synth.build_coco(Path(tmp_path) / "coco")
    path = root / "annotations" / "instances_train.json"
    doc = json.loads(path.read_text())
    doc["images"][0]["\x1b[31m::error::pwn"] = 1
    for n in range(500):
        doc["annotations"][0][f"k{n}"] = n
    doc["‮top"] = 1
    doc["categories"][0]["evil key"] = 1
    path.write_text(json.dumps(doc))
    first = convert.ingest(root, "coco", Path(tmp_path) / "a.dcp")
    assert all(k.replace(".", "").replace("_", "").isalnum() for k in first.report.dropped)
    ds = DatasetStore.open(Path(tmp_path) / "a.dcp").read()
    report = convert.export(ds, "yolo", Path(tmp_path) / "out")
    keys = list(report.dropped) + list(first.report.dropped)
    assert all(len(k) < 160 and k.isprintable() and ":" not in k and " " not in k for k in keys)
    assert len(report.dropped) <= 210 and "annotations.other" in report.dropped
    assert "images.source.invalid_key" in report.dropped
