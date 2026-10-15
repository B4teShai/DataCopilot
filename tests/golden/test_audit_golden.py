"""A fixed dataset audited with every CPU check must produce byte-identical reports.

Regenerate the expected files deliberately with ``UPDATE_GOLDEN=1 uv run pytest tests/golden``
and review the diff: a changed finding or report layout is a decision, not noise.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fixtures.auditdata import Img, build
from fixtures.synth import png_bytes

from datacopilot import __version__
from datacopilot.audit.context import CheckContext
from datacopilot.audit.engine import run_audit
from datacopilot.core.schema import AuditRun, Dataset
from datacopilot.report import html_report, json_report, sarif_report

DATA = Path(__file__).parent / "data" / "audit"
RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
TWIN = png_bytes(16, 16, 7)


def fixed_dataset(root: Path) -> Dataset:
    specs = [
        Img(
            "a.png",
            64,
            48,
            "train",
            [(0, 8, 6, 40, 30), (0, 8.2, 6.1, 40, 30.1), (1, 1, 1, 60, 40)],
        ),
        Img("b.png", 80, 60, "train", [(0, 0, 0, 1, 40), (2, 0, 0, 80, 60)], seed=2),
        Img("c.png", 16, 16, "train", data=TWIN, image_id="00000000000000c1"),
        Img("d.png", 16, 16, "test", data=TWIN, image_id="00000000000000d1"),
        Img("e.png", 50, 50, "val", [(0, 12.5, 7.25, 31.5, 44.0)], declared=(60, 50), seed=5),
        Img("f.png", 32, 32, "train", seed=6, write=False),
        Img("g.png", 32, 32, "train", [(0, 2, 2, 30, 30)], seed=7, path="s3://bucket/g.png"),
        Img("h.png", 64, 48, "test", [(1, 4, 4, 60, 44)], seed=8),
    ]
    for n in range(10):  # a long tail in train: class 0 dominates
        specs.append(Img(f"t{n}.png", 40, 40, "train", [(0, 5, 5, 35, 35)], seed=20 + n))
    return build(root, specs)


def normalised(run: AuditRun) -> AuditRun:
    return run.model_copy(
        update={
            "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "timings_ms": dict.fromkeys(run.checks, 0),
        }
    )


@pytest.fixture
def run(tmp_path: Path) -> AuditRun:
    root = tmp_path / "ds"
    ds = fixed_dataset(root)
    ctx = CheckContext(run_id=RUN, root=root, cache_dir=tmp_path / "cache", workers=2)
    return normalised(run_audit(ds, "all", ctx, fail_on=["critical"]))


GOLDEN_VERSION = "0.0.1.dev0"
"""The package version the golden files were written with. Reports name the tool version; it is
replaced by this value before comparing, so a release does not rewrite every golden file."""


def check(name: str, text: str) -> None:
    text = text.replace(f'"{__version__}"', f'"{GOLDEN_VERSION}"').replace(
        f"datacopilot {__version__}", f"datacopilot {GOLDEN_VERSION}"
    )
    target = DATA / name
    if os.environ.get("UPDATE_GOLDEN"):
        target.write_text(text, encoding="utf-8")
    assert target.read_text(encoding="utf-8") == text, f"{name} changed; see the module docstring"


def test_json_report_matches_the_golden_file(run: AuditRun) -> None:
    check("report.json", json_report.render_json(run))


def test_sarif_report_matches_the_golden_file(run: AuditRun) -> None:
    check("report.sarif", sarif_report.render_sarif(run))


def test_html_report_matches_the_golden_file(run: AuditRun) -> None:
    check("report.html", html_report.render_html(run))


def test_every_check_contributes_findings_with_the_documented_severities(run: AuditRun) -> None:
    seen = {(f.check, str(f.payload["rule"]), f.severity) for f in run.findings}
    assert {
        ("integrity", "file_missing", "CRITICAL"),
        ("integrity", "size_mismatch", "CRITICAL"),
        ("integrity", "remote_not_checked", "WARN"),
        ("duplicates", "exact_duplicate", "CRITICAL"),
        ("duplicates", "content_changed", "CRITICAL"),
        ("imbalance", "long_tail", "WARN"),
        ("imbalance", "class_missing_in_split", "WARN"),
        ("boxes", "duplicate_box", "WARN"),
        ("boxes", "tiny_box", "WARN"),
    } <= seen
    assert run.checks == ("integrity", "duplicates", "imbalance", "boxes") and not run.passed


RUN_B = "9999ffff-8888-4777-8666-5555eeee4444"


def audit_texts(tmp_path: Path, name: str, run_id: str, workers: int) -> dict[str, str]:
    root = tmp_path / name
    ds = fixed_dataset(root)
    ctx = CheckContext(run_id=run_id, root=root, cache_dir=tmp_path / f"c{name}", workers=workers)
    run = normalised(run_audit(ds, "all", ctx, fail_on=["critical"]))
    return {
        "json": json_report.render_json(run),
        "sarif": sarif_report.render_sarif(run),
        "html": html_report.render_html(run),
    }


def test_two_runs_with_different_run_ids_are_identical_after_the_documented_normalisation(
    tmp_path: Path,
) -> None:
    """Contract (03-data-model.md): the only run-specific content is the run id (top-level
    ``run_id``, every finding's ``run_id``, the SARIF ``correlationGuid``, the HTML text),
    ``started_at`` and ``timings_ms``. Replace the run id and the output is byte-identical."""
    a = audit_texts(tmp_path, "a", RUN, workers=1)
    b = audit_texts(tmp_path, "b", RUN_B, workers=3)
    for fmt in a:
        assert a[fmt] != b[fmt], f"{fmt} must carry the run id"
        assert RUN not in b[fmt] and RUN_B not in a[fmt]
        assert a[fmt].replace(RUN, "<run-id>") == b[fmt].replace(RUN_B, "<run-id>"), fmt
    assert RUN in a["json"].split('"findings"')[1]  # findings carry the run id too
    assert json.loads(a["sarif"])["runs"][0]["automationDetails"]["id"] == "datacopilot/audit/"
