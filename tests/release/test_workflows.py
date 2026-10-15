"""Workflow lint: the release pipeline's safety rules, enforced on every workflow file.

A rule that matters here is one a reviewer could miss in a diff: an action that floats on a tag, a
job that can publish and also runs code from the build, an untrusted expression inside a shell
script, a token that outlives the job.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
FILES = sorted(WORKFLOWS.glob("*.yml"))
SHA = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?@[0-9a-f]{40}$")
# contexts an outsider can influence (branch names, titles, bodies, labels, refs) or that carry
# secrets; any `${{ }}` inside a script is an injection or a leak, so values go through `env:`
EXPRESSION = re.compile(r"\$\{\{")
PINNED_INSTALL = re.compile(r"(?:uvx|pip install|pipx run|npx)\b")


def load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["on"] = document.pop(True, document.get("on"))  # YAML 1.1 reads `on` as True
    assert isinstance(document, dict)
    return document


def steps(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(job, step) for job, body in document["jobs"].items() for step in body["steps"]]


def test_the_expected_workflows_exist() -> None:
    assert {"ci.yml", "release.yml"} <= {p.name for p in FILES}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_third_party_action_is_pinned_to_a_full_commit_sha(path: Path) -> None:
    for job, step in steps(load(path)):
        uses = step.get("uses")
        if uses and not uses.startswith("./"):
            assert SHA.fullmatch(uses), f"{path.name}:{job}: {uses} is not pinned to a full SHA"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_expression_inside_a_shell_script(path: Path) -> None:
    for job, step in steps(load(path)):
        assert not EXPRESSION.search(step.get("run", "")), f"{path.name}:{job}: {step.get('name')}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_pull_request_target_and_no_workflow_run_trigger(path: Path) -> None:
    triggers = load(path)["on"]
    names = (
        set(triggers)
        if isinstance(triggers, dict)
        else set([triggers] if isinstance(triggers, str) else triggers)
    )
    assert not {"pull_request_target", "workflow_run", "issue_comment"} & names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_permissions_are_declared_and_minimal(path: Path) -> None:
    document = load(path)
    assert "permissions" in document, "declare top-level permissions"
    top = document["permissions"]
    assert top == {} or top == {"contents": "read"}
    for job, body in document["jobs"].items():
        granted = body.get("permissions", top)
        assert granted != "write-all" and "write-all" not in str(granted)
        for scope, level in (granted or {}).items():
            if level == "write":
                assert scope == "id-token", f"{path.name}:{job} grants {scope}: write"
        if (granted or {}).get("id-token") == "write":
            assert path.name == "release.yml" and job == "publish"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_checkout_never_persists_credentials(path: Path) -> None:
    for job, step in steps(load(path)):
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False, f"{path.name}:{job}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_tool_installs_are_pinned(path: Path) -> None:
    for job, step in steps(load(path)):
        for line in step.get("run", "").splitlines():
            if PINNED_INSTALL.search(line) and "-r " not in line:
                assert "==" in line, f"{path.name}:{job}: unpinned install: {line}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_secret_is_referenced_and_nothing_pipes_the_network_into_a_shell(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "secrets." not in text.replace("secrets.GITHUB_TOKEN", "") or path.name == "ci.yml"
    assert not re.search(r"(curl|wget)[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh", text)


def test_release_is_triggered_by_version_tags_only() -> None:
    triggers = load(WORKFLOWS / "release.yml")["on"]
    assert set(triggers) == {"push"}
    assert set(triggers["push"]) == {"tags"}
    (pattern,) = triggers["push"]["tags"]
    assert pattern.startswith("v") and "*" not in pattern.replace("[0-9]+", "")


def test_release_builds_without_secrets_and_publishes_from_a_protected_environment() -> None:
    document = load(WORKFLOWS / "release.yml")
    assert document["permissions"] == {}
    build, publish = document["jobs"]["build"], document["jobs"]["publish"]
    assert build["permissions"] == {"contents": "read"}
    assert "environment" not in build and "id-token" not in build["permissions"]
    assert publish["needs"] == "build" or publish["needs"] == ["build"]
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"]["name"] == "pypi"
    assert "refs/tags/v" in publish["if"]
    uses = [s.get("uses", "") for s in publish["steps"]]
    assert any(u.startswith("actions/download-artifact@") for u in uses)
    assert any(u.startswith("pypa/gh-action-pypi-publish@") for u in uses)
    assert not any(u.startswith("actions/checkout@") for u in uses), (
        "publish runs no repository code"
    )
    assert not any("run" in s and "pip" in s["run"] for s in publish["steps"])
    build_uses = [s.get("uses", "") for s in build["steps"]]
    assert not any(u.startswith("pypa/gh-action-pypi-publish@") for u in build_uses)
    assert any(u.startswith("actions/upload-artifact@") for u in build_uses)


def test_release_checks_the_tag_against_the_version_and_logs_digests() -> None:
    document = load(WORKFLOWS / "release.yml")
    runs = "\n".join(s.get("run", "") for _, s in steps(document))
    assert "check_release_tag.py" in runs and "sha256sum" in runs
    build = document["jobs"]["build"]
    tag_steps = [s for s in build["steps"] if "check_release_tag.py" in s.get("run", "")]
    assert tag_steps and tag_steps[0]["env"]["TAG"] == "${{ github.ref_name }}"
    names = [s.get("name", "") for s in build["steps"]]
    assert names.index(tag_steps[0]["name"]) < min(
        i for i, n in enumerate(names) if "uild" in n and "tag" not in n.lower()
    )


def test_release_uploads_the_exact_artifact_the_publish_job_downloads() -> None:
    document = load(WORKFLOWS / "release.yml")
    up = next(
        s for s in document["jobs"]["build"]["steps"] if "upload-artifact" in s.get("uses", "")
    )
    down = next(
        s for s in document["jobs"]["publish"]["steps"] if "download-artifact" in s.get("uses", "")
    )
    assert up["with"]["name"] == down["with"]["name"]
    assert up["with"]["if-no-files-found"] == "error"


def test_ci_has_the_supply_chain_and_packaging_jobs() -> None:
    document = load(WORKFLOWS / "ci.yml")
    assert {"test", "docs", "package", "audit", "lock"} <= set(document["jobs"])
    runs = "\n".join(s.get("run", "") for _, s in steps(document))
    needles = (
        "uv lock --check",
        "mkdocs build --strict",
        "twine check --strict",
        "pip-audit==",
        "verify_install.py",
        "scripts/gen_cli_reference.py --check",
    )
    for needle in needles:
        assert needle in runs, needle


def test_the_lint_itself_catches_the_mistakes_it_exists_for(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "on: pull_request_target\npermissions: write-all\njobs:\n  j:\n    runs-on: x\n"
        "    steps:\n      - uses: actions/checkout@v4\n      - run: echo ${{ github.head_ref }}\n"
    )
    document = load(bad)
    assert "pull_request_target" in str(document["on"])
    assert not SHA.fullmatch(document["jobs"]["j"]["steps"][0]["uses"])
    assert EXPRESSION.search(document["jobs"]["j"]["steps"][1]["run"])
    assert document["permissions"] == "write-all"
