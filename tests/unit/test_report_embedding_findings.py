"""The reports render the embedding findings: descriptions, escaping, SARIF shape, no paths."""

from __future__ import annotations

import json
from pathlib import Path

from fixtures.embedfakes import near, random_unit
from fixtures.embedscene import Row, scene
from sarif_shape import validate_sarif

from datacopilot.audit import engine
from datacopilot.report import render
from datacopilot.report.rules import describe_rule

BASE = random_unit(b"report")


def test_every_new_rule_has_a_human_description() -> None:
    for check, rule in (
        ("near_duplicates", "near_duplicate"),
        ("leakage", "cross_split_near_duplicate"),
        ("leakage", "group_overlap"),
        ("outliers", "outlier"),
        ("outliers", "classes_skipped"),
        ("outliers", "images_without_class"),
        ("near_duplicates", "phash_not_applied"),
        ("leakage", "embedder_error"),
        ("leakage", "images_not_embedded"),
        ("outliers", "embedder_missing"),
        ("near_duplicates", "embedding_cache_rejected"),
        ("near_duplicates", "embedding_cache_mismatch"),
    ):
        assert not describe_rule(check, rule).startswith(f"{check}:"), (check, rule)


def test_a_run_with_leakage_renders_in_all_three_formats(tmp_path: Path) -> None:
    rows = [
        Row("<b>a</b>.png", "train", BASE, group="g<script>"),
        Row("b.png", "test", near(BASE, 0.1), group="g<script>"),
        *[Row(f"c{k}.png", "train") for k in range(8)],
    ]
    ds, _, ctx = (
        scene(tmp_path / "data", rows)
        if (tmp_path / "data").mkdir() is None
        else (None, None, None)
    )  # type: ignore[misc]
    run = engine.run_audit(ds, "all", ctx, ["leakage"])  # type: ignore[arg-type]
    out = tmp_path / "out"
    out.mkdir()
    render(run, html=out / "r.html", json=out / "r.json", sarif=out / "r.sarif")
    html = (out / "r.html").read_text()
    assert "Near-identical images in different splits" in html
    assert "<script>" not in html and "&lt;" in html  # escaped, not interpreted
    validate_sarif(json.loads((out / "r.sarif").read_text()))
    doc = json.loads((out / "r.json").read_text())
    rules = {f["payload"]["rule"] for f in doc["findings"]}
    assert {"cross_split_near_duplicate", "group_overlap"} <= rules
    for name in ("r.html", "r.json", "r.sarif"):
        assert str(tmp_path) not in (out / name).read_text()  # no absolute paths anywhere
