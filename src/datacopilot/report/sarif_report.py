"""SARIF 2.1.0 report for code-scanning uploads.

One result per finding. Severity maps to level (CRITICAL error, WARN warning, INFO note). Locations
carry the percent-encoded base name of the file relative to the symbolic base ``DATASET_IMAGES``
(no repository path is invented) and the image and annotation ids as logical locations. Results are
sorted by severity and capped at ``MAX_SARIF_RESULTS`` with a summary result for the rest.
``automationDetails.id`` is stable (``datacopilot/audit/``); the run id is the
``correlationGuid``. Rule ids are ``<check>/<rule>`` restricted to ``[a-z0-9_]``. All text goes
through ``clean_text`` and the
output is ASCII JSON, so nothing a dataset contains can break out of a string or reorder text.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any
from urllib.parse import quote

from datacopilot import __version__
from datacopilot.core.schema import SEVERITIES, AuditRun, Finding, Severity
from datacopilot.report.rules import describe_rule
from datacopilot.report.text import clean_text, describe_payload

SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"
_LEVEL = {"CRITICAL": "error", "WARN": "warning", "INFO": "note"}
_RANK = {s: i for i, s in enumerate(SEVERITIES)}
_SLUG_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
MAX_URI_NAME_CHARS = 255
MAX_SARIF_RESULTS = 5_000
"""Results written to SARIF (most severe first); one summary result counts the rest."""
BASE_ID = "DATASET_IMAGES"
TRUNCATED_RULE = "datacopilot/results_truncated"


def _slug(text: object) -> str:
    raw = str(text).lower()[:64]
    slug = "".join(c if c in _SLUG_CHARS else "_" for c in raw)
    return slug or "finding"


def _rule_of(finding: Finding) -> tuple[str, str]:
    rule = _slug(finding.payload.get("rule", "finding"))
    return f"{finding.check}/{rule}", rule


def _location(finding: Finding) -> dict[str, Any]:
    """Logical locations (image and annotation ids) plus, when the finding names a file, its base
    name relative to the symbolic base ``DATASET_IMAGES`` (no repository path is invented)."""
    location: dict[str, Any] = {}
    name = finding.payload.get("file_name")
    if isinstance(name, str) and name:
        location["physicalLocation"] = {
            "artifactLocation": {
                "uri": quote(name[:MAX_URI_NAME_CHARS], safe="", errors="replace"),
                "uriBaseId": BASE_ID,
            }
        }
    logical = []
    if finding.image_id:
        logical.append({"name": finding.image_id, "kind": "image"})
    if finding.ann_id:
        logical.append({"name": clean_text(finding.ann_id, 128), "kind": "annotation"})
    if not logical and "physicalLocation" not in location:
        logical.append({"name": "dataset.dcp", "kind": "module"})
    if logical:
        location["logicalLocations"] = logical
    return location


def _message(finding: Finding, rule: str) -> str:
    text = describe_rule(finding.check, str(finding.payload.get("rule", rule)))
    detail = describe_payload(finding.payload, 400)
    return clean_text(f"{text} ({detail})" if detail else text, 600)


def render_sarif(run: AuditRun) -> str:
    """The run as SARIF 2.1.0 JSON text (ASCII, sorted rules, results in finding order)."""
    ordered = sorted(run.findings, key=lambda f: _RANK[f.severity])
    kept = ordered if len(ordered) <= MAX_SARIF_RESULTS else ordered[: MAX_SARIF_RESULTS - 1]
    omitted = ordered[len(kept) :]
    rule_info: dict[str, dict[str, Any]] = {}
    top: dict[str, Severity] = {}
    for f in kept:
        rid, slug = _rule_of(f)
        rule_info.setdefault(
            rid,
            {
                "id": rid,
                "shortDescription": {"text": clean_text(describe_rule(f.check, slug), 200)},
            },
        )
        if rid not in top or _RANK[f.severity] < _RANK[top[rid]]:
            top[rid] = f.severity
    for rid, entry in rule_info.items():
        entry["defaultConfiguration"] = {"level": _LEVEL[top[rid]]}
    rules = [rule_info[k] for k in sorted(rule_info)]
    index = {r["id"]: n for n, r in enumerate(rules)}
    results = []
    for f in kept:
        rid, slug = _rule_of(f)
        results.append(
            {
                "ruleId": rid,
                "ruleIndex": index[rid],
                "level": _LEVEL[f.severity],
                "message": {"text": _message(f, slug)},
                "locations": [_location(f)],
                "partialFingerprints": {"findingId/v1": f.finding_id},
                "properties": {"severity": f.severity, "check": f.check},
            }
        )
    if omitted:
        worst = omitted[0].severity
        tally = Counter(f.severity for f in omitted)
        by_severity = ", ".join(f"{tally[sev]} {sev.lower()}" for sev in SEVERITIES if tally[sev])
        rules.append(
            {
                "id": TRUNCATED_RULE,
                "shortDescription": {"text": "Results were omitted from this report"},
                "defaultConfiguration": {"level": _LEVEL[worst]},
            }
        )
        results.append(
            {
                "ruleId": TRUNCATED_RULE,
                "ruleIndex": len(rules) - 1,
                "level": _LEVEL[worst],
                "message": {
                    "text": f"{len(omitted)} more findings omitted ({by_severity}); see JSON report"
                },
                "locations": [{"logicalLocations": [{"name": "dataset.dcp", "kind": "module"}]}],
                "partialFingerprints": {"findingId/v1": f"truncated-{len(omitted)}"},
                "properties": {"severity": worst, "check": "datacopilot"},
            }
        )
    doc = {
        "$schema": SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "datacopilot", "version": __version__, "rules": rules}},
                "automationDetails": {"id": "datacopilot/audit/", "correlationGuid": run.run_id},
                "originalUriBaseIds": {
                    BASE_ID: {
                        "description": {
                            "text": "Directory of the dataset images (location not disclosed)."
                        }
                    }
                },
                "results": results,
                "properties": {
                    "datasetVersion": run.dataset_version,
                    "passed": run.passed,
                    "critical": run.counts.critical,
                    "warn": run.counts.warn,
                    "info": run.counts.info,
                },
            }
        ],
    }
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
