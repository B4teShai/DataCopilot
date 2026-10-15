"""A strict structural check of the SARIF 2.1.0 subset DataCopilot emits.

Mirrors the constraints of the official schema (https://json.schemastore.org/sarif-2.1.0.json) for
every object we produce: required members, member types, enumerations, and the rule index
cross-reference. Unknown members are rejected so an accidental field is caught.
"""

from __future__ import annotations

from typing import Any

LEVELS = {"none", "note", "warning", "error"}


def _obj(value: Any, where: str, required: set[str], optional: set[str]) -> dict[str, Any]:
    assert isinstance(value, dict), f"{where} must be an object"
    missing = required - value.keys()
    assert not missing, f"{where} lacks {sorted(missing)}"
    extra = value.keys() - required - optional
    assert not extra, f"{where} has unexpected members {sorted(extra)}"
    return value


def _text(value: Any, where: str, *, non_empty: bool = True) -> str:
    assert isinstance(value, str), f"{where} must be a string"
    assert value or not non_empty, f"{where} must not be empty"
    return value


def validate_sarif(doc: Any) -> None:
    root = _obj(doc, "log", {"version", "runs"}, {"$schema"})
    assert root["version"] == "2.1.0"
    if "$schema" in root:
        assert root["$schema"].startswith("https://") and root["$schema"].endswith(
            "sarif-2.1.0.json"
        )
    runs = root["runs"]
    assert isinstance(runs, list) and len(runs) == 1
    run = _obj(
        runs[0],
        "run",
        {"tool", "results"},
        {"automationDetails", "properties", "originalUriBaseIds"},
    )
    tool = _obj(run["tool"], "tool", {"driver"}, set())
    driver = _obj(tool["driver"], "driver", {"name", "version", "rules"}, {"informationUri"})
    _text(driver["name"], "driver.name")
    _text(driver["version"], "driver.version")
    rules = driver["rules"]
    assert isinstance(rules, list)
    ids = []
    for n, rule in enumerate(rules):
        r = _obj(rule, f"rules[{n}]", {"id", "shortDescription", "defaultConfiguration"}, {"name"})
        ids.append(_text(r["id"], "rule.id"))
        _text(_obj(r["shortDescription"], "shortDescription", {"text"}, set())["text"], "text")
        cfg = _obj(r["defaultConfiguration"], "defaultConfiguration", {"level"}, set())
        assert cfg["level"] in LEVELS
    assert len(ids) == len(set(ids)), "rule ids must be unique"
    if "automationDetails" in run:
        details = _obj(run["automationDetails"], "automationDetails", {"id"}, {"correlationGuid"})
        _text(details["id"], "id")
    assert isinstance(run["results"], list)
    for n, result in enumerate(run["results"]):
        where = f"results[{n}]"
        r = _obj(
            result,
            where,
            {"ruleId", "ruleIndex", "level", "message", "locations", "partialFingerprints"},
            {"properties"},
        )
        assert r["level"] in LEVELS
        assert isinstance(r["ruleIndex"], int) and 0 <= r["ruleIndex"] < len(ids)
        assert ids[r["ruleIndex"]] == r["ruleId"], f"{where} ruleIndex does not match ruleId"
        _text(_obj(r["message"], f"{where}.message", {"text"}, set())["text"], "message.text")
        locations = r["locations"]
        assert isinstance(locations, list) and locations
        for loc in locations:
            lo = _obj(loc, "location", set(), {"physicalLocation", "logicalLocations"})
            assert "physicalLocation" in lo or lo.get("logicalLocations"), "empty location"
            if "physicalLocation" in lo:
                phys = _obj(lo["physicalLocation"], "physicalLocation", {"artifactLocation"}, set())
                art = _obj(phys["artifactLocation"], "artifactLocation", {"uri"}, {"uriBaseId"})
                uri = _text(art["uri"], "uri")
                assert all(ord(c) > 32 and c not in '"<>\\^`{|}' for c in uri), "uri not encoded"
                assert "://" not in uri and not uri.startswith("/"), "uri must be relative"
                assert "uriBaseId" in art, "a relative uri needs a base id"
            for logical in lo.get("logicalLocations", []):
                lg = _obj(logical, "logicalLocation", {"name", "kind"}, set())
                _text(lg["name"], "name")
                _text(lg["kind"], "kind")
        fp = _obj(r["partialFingerprints"], "partialFingerprints", {"findingId/v1"}, set())
        _text(fp["findingId/v1"], "fingerprint")
        if "properties" in r:
            assert isinstance(r["properties"], dict)
