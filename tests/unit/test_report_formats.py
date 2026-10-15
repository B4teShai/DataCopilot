"""JSON, HTML and SARIF reports: structure, escaping of hostile names, determinism."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
import structlog
from sarif_shape import validate_sarif

from datacopilot.audit.findings import make_finding
from datacopilot.core.errors import DataCopilotError
from datacopilot.core.schema import AuditRun, Counts, Finding, Summary, SummaryClass
from datacopilot.report import html_report, json_report, render, sarif_report

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
VERSION = "a" * 32
MISSING_A = {"rule": "file_missing", "file_name": "a.png"}
HOSTILE = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "&amp; &lt; &#x3c;",
    "javascript:alert(1)",
    "</td></tr></table><h1>pwned</h1>",
    "' onmouseover='alert(1)",
    "evil‮\u0000name\x1b[31m.png",
    "{{ 7*7 }} {% raw %} ${7*7}",
]


def findings(*specs: tuple[Any, ...]) -> tuple[Finding, ...]:
    return tuple(
        make_finding(RUN, check, sev, image_id=image_id, ann_id=ann_id, payload=payload)
        for check, sev, image_id, ann_id, payload in specs
    )


def make_run(
    fs: tuple[Finding, ...], *, names: tuple[str, ...] = ("cat", "dog"), **kw: Any
) -> AuditRun:
    tally = {s: sum(f.severity == s for f in fs) for s in ("CRITICAL", "WARN", "INFO")}
    base: dict[str, Any] = {
        "run_id": RUN,
        "dataset_version": VERSION,
        "started_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        "checks": ("integrity", "boxes"),
        "counts": Counts(critical=tally["CRITICAL"], warn=tally["WARN"], info=tally["INFO"]),
        "passed": tally["CRITICAL"] == 0,
        "fail_on": ("critical",),
        "timings_ms": {"integrity": 12, "boxes": 3},
        "summary": Summary(
            task="detection",
            image_count=10,
            annotation_count=30,
            split_images={"train": 8, "val": 2},
            classes=tuple(SummaryClass(id=i + 1, name=n) for i, n in enumerate(names)),
            class_counts={
                "train": {str(i + 1): 10 + i for i in range(len(names))},
                "val": {"1": 2},
            },
        ),
        "findings": fs,
    }
    return AuditRun(**{**base, **kw})


def hostile_run() -> AuditRun:
    img = "0" * 16
    fs = [
        ("integrity", "CRITICAL", img, None, {"rule": "file_missing", "file_name": name})
        for name in HOSTILE
    ]
    fs += [
        ("boxes", "WARN", f"{n + 1:016x}", f"ann-{name}", {"rule": name, "note": name, "k": [name]})
        for n, name in enumerate(HOSTILE)
    ]
    fs.append(("integrity", "INFO", None, None, {"rule": "remote_not_checked", "count": 2}))
    return make_run(findings(*fs), names=tuple(HOSTILE[:4]) + ("plain",))


# ------------------------------------------------------------------ JSON
def test_json_report_is_the_audit_run_and_round_trips() -> None:
    run = hostile_run()
    text = json_report.render_json(run)
    assert AuditRun.model_validate_json(text) == run
    assert text.isascii() and text.endswith("\n")
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_json_report_survives_hostile_strings_and_is_valid_json() -> None:
    parsed = json.loads(json_report.render_json(hostile_run()))
    names = {f["payload"].get("file_name") for f in parsed["findings"]}
    assert HOSTILE[0] in names and HOSTILE[6] in names


# ----------------------------------------------------------------- SARIF
def test_sarif_has_the_21_shape_and_one_result_per_finding() -> None:
    run = hostile_run()
    doc = json.loads(sarif_report.render_sarif(run))
    validate_sarif(doc)
    assert doc["version"] == "2.1.0" and len(doc["runs"][0]["results"]) == len(run.findings)


def test_sarif_levels_follow_severity() -> None:
    run = make_run(
        findings(
            ("integrity", "CRITICAL", None, None, {"rule": "a"}),
            ("integrity", "WARN", None, None, {"rule": "b"}),
            ("integrity", "INFO", None, None, {"rule": "c"}),
        )
    )
    levels = {
        r["ruleId"]: r["level"]
        for r in json.loads(sarif_report.render_sarif(run))["runs"][0]["results"]
    }
    assert levels == {"integrity/a": "error", "integrity/b": "warning", "integrity/c": "note"}


def test_sarif_rules_are_deduplicated_and_indexed() -> None:
    run = make_run(
        findings(
            (
                "integrity",
                "CRITICAL",
                "1" * 16,
                None,
                {"rule": "file_missing", "file_name": "a.png"},
            ),
            (
                "integrity",
                "CRITICAL",
                "2" * 16,
                None,
                {"rule": "file_missing", "file_name": "b.png"},
            ),
            ("boxes", "WARN", "3" * 16, "x", {"rule": "tiny_box"}),
        )
    )
    doc = json.loads(sarif_report.render_sarif(run))
    rules = doc["runs"][0]["tool"]["driver"]["rules"]
    assert [r["id"] for r in rules] == ["boxes/tiny_box", "integrity/file_missing"]
    validate_sarif(doc)


def test_sarif_rule_ids_from_hostile_rule_names_are_restricted_to_a_safe_alphabet() -> None:
    doc = json.loads(sarif_report.render_sarif(hostile_run()))
    for rule in doc["runs"][0]["tool"]["driver"]["rules"]:
        assert re.fullmatch(r"[a-z0-9_]+/[a-z0-9_]{1,64}", rule["id"]), rule["id"]


def test_sarif_uris_are_percent_encoded_relative_references() -> None:
    doc = json.loads(sarif_report.render_sarif(hostile_run()))
    uris = {
        loc["physicalLocation"]["artifactLocation"]["uri"]
        for r in doc["runs"][0]["results"]
        for loc in r["locations"]
        if "physicalLocation" in loc
    }
    assert all(re.fullmatch(r"[A-Za-z0-9._~%/-]+", u) for u in uris), uris


def test_sarif_messages_contain_no_control_characters_and_are_bounded() -> None:
    doc = json.loads(sarif_report.render_sarif(hostile_run()))
    for r in doc["runs"][0]["results"]:
        msg = r["message"]["text"]
        assert len(msg) < 1500 and all(
            ord(c) >= 32 and ord(c) not in range(0x7F, 0xA0) for c in msg
        )
        assert not any(c in msg for c in "‮⁦\ud800")


def test_sarif_output_is_ascii_json_without_nan() -> None:
    text = sarif_report.render_sarif(hostile_run())
    assert text.isascii()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard constant {c}"))


def test_sarif_fingerprint_is_the_finding_id() -> None:
    run = hostile_run()
    doc = json.loads(sarif_report.render_sarif(run))
    got = {r["partialFingerprints"]["findingId/v1"] for r in doc["runs"][0]["results"]}
    assert got == {f.finding_id for f in run.findings}


def test_sarif_for_an_empty_run_is_still_valid() -> None:
    validate_sarif(json.loads(sarif_report.render_sarif(make_run(()))))


# ------------------------------------------------------------------ HTML
class Collector(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "rect", "line", "path", "circle"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.style = ""
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)
        self._in_style = tag == "style"

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> over {self.stack[-3:]}")
        else:
            self.stack.pop()
        self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.style += data
        self.text.append(data)


def parse(text: str) -> Collector:
    c = Collector()
    c.feed(text)
    c.close()
    return c


def test_html_is_well_formed_and_contains_the_score_card() -> None:
    page = html_report.render_html(hostile_run())
    c = parse(page)
    assert c.errors == [] and c.stack == []
    body = " ".join(c.text)
    assert VERSION in body and RUN in body and "FAILED" in body
    assert "integrity" in body and "boxes" in body


def test_html_has_a_strict_csp_and_no_active_or_remote_content() -> None:
    c = parse(html_report.render_html(hostile_run()))
    tags = {t for t, _ in c.tags}
    assert not tags & {"script", "iframe", "object", "embed", "link", "img", "form", "base", "a"}
    metas = [
        a
        for t, a in c.tags
        if t == "meta" and (a.get("http-equiv") or "").lower() == "content-security-policy"
    ]
    assert len(metas) == 1
    policy = metas[0]["content"] or ""
    assert "default-src 'none'" in policy and "script-src" not in policy.replace("default-src", "")
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    for _, attrs in c.tags:
        for name, value in attrs.items():
            assert not name.startswith("on") and name not in {
                "style",
                "href",
                "src",
                "srcset",
                "action",
            }
            assert "://" not in (value or "")


def test_html_csp_style_hash_matches_the_inline_stylesheet() -> None:
    import base64

    c = parse(html_report.render_html(hostile_run()))
    (policy,) = [a["content"] for t, a in c.tags if t == "meta" and a.get("http-equiv") is not None]
    digest = base64.b64encode(hashlib.sha256(c.style.encode("utf-8")).digest()).decode()
    assert f"style-src 'sha256-{digest}'" in (policy or "") and c.style.startswith(":root")


def test_html_hostile_names_appear_only_as_text_never_as_markup() -> None:
    page = html_report.render_html(hostile_run())
    c = parse(page)
    text = "\n".join(c.text)
    assert HOSTILE[0] in text and HOSTILE[1] in text and HOSTILE[4] in text
    assert "<script" not in page and "<img" not in page and "<h1>pwned" not in page
    assert "onerror=alert" in page  # present only inside escaped text
    assert re.search(r"&lt;script&gt;alert\(1\)&lt;/script&gt;", page)
    assert "{{ 7*7 }}" in text


def test_html_control_and_surrogate_characters_are_escaped_and_the_page_encodes() -> None:
    page = html_report.render_html(hostile_run())
    page.encode("utf-8")
    assert "‮" not in page and "\x00" not in page and "\x1b" not in page
    assert "\\u202e" in page and "\\u0000" in page


def test_html_has_no_template_injection() -> None:
    page = html_report.render_html(hostile_run())
    assert "{{ 7*7 }}" in page and "${7*7}" in page  # shown verbatim, never evaluated


def test_html_histogram_is_svg_with_bounded_bars_and_labels_are_escaped() -> None:
    names = tuple(f"<b>c{i}</b>" for i in range(80))
    run = make_run((), names=names)
    c = parse(html_report.render_html(run))
    rects = [a for t, a in c.tags if t == "rect"]
    assert 0 < len(rects) <= html_report.MAX_HISTOGRAM_BARS
    assert all(float(a["width"] or 0) >= 0 for a in rects)
    assert "<b>c" not in html_report.render_html(run)


def test_html_findings_table_is_capped_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(html_report, "MAX_HTML_FINDINGS", 3)
    fs = findings(*[("boxes", "WARN", f"{n:016x}", None, {"rule": f"r{n}"}) for n in range(10)])
    c = parse(html_report.render_html(make_run(fs)))
    rows = [a for t, a in c.tags if t == "tr" and a.get("class") == "finding"]
    assert len(rows) == 3
    assert "3 of 10" in " ".join(c.text)


def test_html_split_by_class_table_lists_counts() -> None:
    c = parse(html_report.render_html(make_run(())))
    text = " ".join(" ".join(c.text).split())
    assert "train" in text and "val" in text and "cat" in text and "dog" in text
    assert "11" in text  # dog in train: 10 + 1


def test_html_for_a_run_without_summary_or_findings() -> None:
    page = html_report.render_html(make_run((), summary=None))
    assert parse(page).errors == [] and "PASSED" in page


def test_html_contains_no_file_system_paths() -> None:
    fs = findings(
        ("integrity", "CRITICAL", "1" * 16, None, {"rule": "file_missing", "file_name": "a.png"})
    )
    page = html_report.render_html(make_run(fs))
    assert "/Users/" not in page and "/home/" not in page


# ------------------------------------------------------------------ render
def test_render_writes_only_the_requested_files_and_returns_them(tmp_path: Path) -> None:
    run = hostile_run()
    out = render(run, html=tmp_path / "r.html", json=None, sarif=tmp_path / "r.sarif")
    assert out == [tmp_path / "r.html", tmp_path / "r.sarif"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.html", "r.sarif"]
    assert render(run, html=None, json=None, sarif=None) == []


def test_render_creates_parents_overwrites_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "r.json"
    render(make_run(()), html=None, json=target, sarif=None)
    first = target.read_text()
    render(make_run((), passed=False), html=None, json=target, sarif=None)
    assert target.read_text() != first
    assert [p.name for p in target.parent.iterdir()] == ["r.json"]


def test_render_replaces_a_symlink_instead_of_writing_through_it(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    link = tmp_path / "r.html"
    os.symlink(victim, link)
    render(make_run(()), html=link, json=None, sarif=None)
    assert victim.read_text() == "keep" and not link.is_symlink()


def test_render_logs_one_event_per_file_with_the_file_name_only(tmp_path: Path) -> None:
    with structlog.testing.capture_logs() as logs:
        render(make_run(()), html=tmp_path / "r.html", json=tmp_path / "r.json", sarif=None)
    assert [(e["event"], e["format"], e["path"]) for e in logs] == [
        ("report_written", "html", "r.html"),
        ("report_written", "json", "r.json"),
    ]
    assert str(tmp_path) not in str(logs)


def test_render_reports_unwritable_targets_as_a_domain_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(DataCopilotError, match="cannot write"):
        render(make_run(()), html=blocker / "r.html", json=None, sarif=None)
    assert not any(p.name.startswith("r.html") for p in tmp_path.iterdir())


def test_reports_are_deterministic_for_one_run() -> None:
    run = hostile_run()
    assert html_report.render_html(run) == html_report.render_html(run)
    assert sarif_report.render_sarif(run) == sarif_report.render_sarif(run)
    assert json_report.render_json(run) == json_report.render_json(run)


def test_no_report_format_contains_an_absolute_path_from_the_dataset(tmp_path: Path) -> None:
    private_dir = "/Users/someone/private-project"
    fs = findings(("integrity", "CRITICAL", "1" * 16, None, MISSING_A))
    run = make_run(fs)
    for text in (
        json_report.render_json(run),
        sarif_report.render_sarif(run),
        html_report.render_html(run),
    ):
        assert private_dir not in text and "/Users/" not in text


def test_html_headline_says_no_gate_applied_when_there_is_no_gate() -> None:
    fs = findings(("integrity", "CRITICAL", "1" * 16, None, MISSING_A))
    page = html_report.render_html(make_run(fs, fail_on=(), passed=True))
    assert "no gate applied" in page and "PASSED" not in page and "FAILED" not in page
    gated = html_report.render_html(make_run(fs, passed=False))
    assert "FAILED" in gated and "no gate applied" not in gated
