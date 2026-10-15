"""Static HTML report: score card, class histogram, split-by-class table, findings table.

Safety properties, each covered by a test:

* Rendered by Jinja2 with autoescaping always on, in a sandboxed environment; the template is a
  package file and no dataset text is ever used as template source.
* Every dataset-derived string passes through :func:`~datacopilot.report.text.clean_text` first
  (control, bidirectional and surrogate characters become visible escapes; length is bounded).
* No script, no event-handler or ``style`` attributes, no links, forms, images or remote
  resources. The histogram is inline SVG whose numbers are computed here.
* A ``Content-Security-Policy`` meta tag denies everything except the one inline stylesheet,
  which is allowed by its SHA-256 hash, so no ``unsafe-inline`` is needed.
* The findings table is sorted by severity across all checks before it is capped at
  ``MAX_HTML_FINDINGS`` rows; what is omitted is counted per severity. The JSON report is complete.
"""

from __future__ import annotations

import base64
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from datacopilot import __version__
from datacopilot.core.schema import SEVERITIES, TEST_ONLY_SUFFIX, AuditRun
from datacopilot.report.rules import describe_rule
from datacopilot.report.text import clean_text, describe_payload

MAX_HTML_FINDINGS = 2_000
"""Findings rows rendered in HTML."""
MAX_HISTOGRAM_BARS = 30
"""Classes drawn in the histogram (the most frequent; the rest are summed into one bar)."""
MAX_TABLE_CLASSES = 200
"""Class rows in the split-by-class table."""
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
_SPLIT_ORDER = ("train", "val", "test", "unassigned")
_BAR_AREA = 420
_ROW = 20

CSS = (
    ":root{color-scheme:light dark;--bg:#fff;--fg:#1b1f24;--muted:#5b6570;--line:#d6dbe1;"
    "--crit:#b42318;--warn:#b54708;--info:#175cd3;--ok:#067647;--bar:#4f6bed}"
    "@media (prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e6e9ed;--muted:#9aa5b1;"
    "--line:#2c333b;--crit:#f97066;--warn:#fdb022;--info:#84adff;--ok:#47cd89;--bar:#8098f9}}"
    "body{margin:0;padding:16px;background:var(--bg);color:var(--fg);"
    "font:14px/1.45 system-ui,sans-serif}"
    "main{max-width:1100px;margin:0 auto}h1{font-size:20px}h2{font-size:16px;margin-top:28px}"
    "table{border-collapse:collapse;width:100%;font-size:13px}"
    "th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;"
    "vertical-align:top;overflow-wrap:anywhere}th{color:var(--muted);font-weight:600}"
    "td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}"
    ".cards{display:flex;flex-wrap:wrap;gap:12px}"
    ".card{border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:96px}"
    ".card b{display:block;font-size:22px}.muted{color:var(--muted)}"
    ".critical{color:var(--crit)}.warn{color:var(--warn)}.info{color:var(--info)}"
    ".pass{color:var(--ok)}.fail{color:var(--crit)}"
    "svg{max-width:100%;height:auto}svg text{fill:var(--fg);font:12px system-ui,sans-serif}"
    "svg rect.bar{fill:var(--bar)}code{font-family:ui-monospace,monospace;font-size:12px}"
)

_env = SandboxedEnvironment(
    loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _csp() -> str:
    digest = base64.b64encode(hashlib.sha256(CSS.encode("utf-8")).digest()).decode("ascii")
    return f"default-src 'none'; style-src 'sha256-{digest}'; base-uri 'none'; form-action 'none'"


def _histogram(run: AuditRun) -> list[dict[str, Any]]:
    summary = run.summary
    if summary is None or summary.classes_omitted:
        return []
    totals: dict[str, int] = {}
    for per_class in summary.class_counts.values():
        for cid, n in per_class.items():
            totals[cid] = totals.get(cid, 0) + n
    names = {str(c.id): c.name for c in summary.classes}
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], int(kv[0])))
    shown, rest = ranked[: MAX_HISTOGRAM_BARS - 1], ranked[MAX_HISTOGRAM_BARS - 1 :]
    rows = [(clean_text(names.get(cid, cid), 40), n) for cid, n in shown]
    if rest:
        rows.append((f"{len(rest)} other classes", sum(n for _, n in rest)))
    peak = max((n for _, n in rows), default=1) or 1
    return [
        {
            "label": label,
            "count": n,
            "y": i * _ROW,
            "width": round(_BAR_AREA * n / peak, 2),
        }
        for i, (label, n) in enumerate(rows)
    ]


def _split_table(run: AuditRun) -> dict[str, Any] | None:
    summary = run.summary
    if summary is None or summary.classes_omitted:
        return None
    splits = [s for s in _SPLIT_ORDER if s in summary.class_counts or s in summary.split_images]
    rows = [
        {
            "name": clean_text(c.name, 60),
            "id": c.id,
            "counts": [summary.class_counts.get(s, {}).get(str(c.id), 0) for s in splits],
        }
        for c in summary.classes[:MAX_TABLE_CLASSES]
    ]
    return {
        "splits": splits,
        "images": [summary.split_images.get(s, 0) for s in splits],
        "rows": rows,
        "omitted": max(0, len(summary.classes) - MAX_TABLE_CLASSES),
    }


def render_html(run: AuditRun) -> str:
    """The run as a self-contained HTML page (see the module docstring for its guarantees)."""
    per_check = {name: 0 for name in run.checks}
    for f in run.findings:
        per_check[f.check] = per_check.get(f.check, 0) + 1
    ordered = sorted(run.findings, key=lambda f: _SEVERITY_RANK[f.severity])  # stable
    shown = ordered[:MAX_HTML_FINDINGS]
    omitted_count = Counter(f.severity for f in ordered[MAX_HTML_FINDINGS:])
    omitted = ", ".join(f"{omitted_count[s]} {s.lower()}" for s in SEVERITIES if omitted_count[s])
    rows = []
    for f in shown:
        rule = str(f.payload.get("rule", "finding"))
        rows.append(
            {
                "severity": f.severity,
                "severity_class": f.severity.lower(),
                "check": f.check,
                "rule": clean_text(rule, 64),
                "summary": clean_text(describe_rule(f.check, rule), 120),
                "image_id": f.image_id or "",
                "file_name": clean_text(f.payload.get("file_name", ""), 120),
                "detail": describe_payload(f.payload),
            }
        )
    context = {
        "csp": _csp(),
        "css": CSS,
        "version": __version__,
        "run_id": run.run_id,
        "dataset_version": run.dataset_version,
        "started_at": run.started_at.isoformat(),
        "embedder": clean_text(run.embedder, 80) if run.embedder else None,
        "embedder_test_only": bool(run.embedder and run.embedder.endswith(TEST_ONLY_SUFFIX)),
        "passed": run.passed,
        "gated": bool(run.fail_on),
        "fail_on": ", ".join(clean_text(t, 40) for t in run.fail_on) or "none",
        "counts": run.counts,
        "checks": [
            {"name": name, "findings": per_check[name], "ms": run.timings_ms.get(name)}
            for name in run.checks
        ],
        "summary": run.summary,
        "histogram": _histogram(run),
        "histogram_height": max(1, len(_histogram(run))) * _ROW,
        "bar_area": _BAR_AREA,
        "split_table": _split_table(run),
        "rows": rows,
        "total_findings": len(run.findings),
        "shown_findings": len(shown),
        "omitted": omitted,
    }
    return _env.get_template("report.html.j2").render(context)
