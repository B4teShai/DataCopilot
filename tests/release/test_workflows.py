"""Workflow lint: the release pipeline's safety rules, enforced on every workflow file.

Every rule is a function returning the problems it finds, so the lint can be run against a hostile
fixture (``test_the_lint_catches_every_mistake_it_exists_for``) as well as against the real files:
a rule that never fires is worse than no rule.
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
EXPRESSION = re.compile(r"\$\{\{")
SECRETS_CONTEXT = re.compile(r"\bsecrets\b\s*(?:\.|\[)", re.I)
PINNED_INSTALL = re.compile(r"(?:uvx|pip3? install|pipx run|npx)\b")
PIPE_TO_SHELL = [
    re.compile(r"(?:curl|wget)\b[^|;]*(?:\\\n[^|;]*)*\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b"),
    re.compile(r"\b(?:ba|z|da|k)?sh\b[^\n]*<\(\s*(?:curl|wget)\b"),
    re.compile(r"\b(?:ba|z|da|k)?sh\s+-c\s+[\"']?\$\(\s*(?:curl|wget)\b"),
    re.compile(r"\|\s*(?:sudo\s+)?(?:python3?|node|perl|ruby)\s+-\s*$", re.M),
]
SCRIPT_ACTIONS = ("actions/github-script@",)
FORBIDDEN_PUBLISH_KEYS = {"skip-existing", "skip_existing", "repository-url", "repository_url"}
FORBIDDEN_TRIGGERS = {"pull_request_target", "workflow_run", "issue_comment"}


def load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    document["on"] = document.pop(True, document.get("on"))  # YAML 1.1 reads `on` as True
    return document


def steps(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(job, step) for job, body in document["jobs"].items() for step in body["steps"]]


def strings(node: Any) -> list[str]:
    """Every string in a parsed document, keys included."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [t for k, v in node.items() for t in (strings(k) + strings(v))]
    if isinstance(node, list):
        return [t for item in node for t in strings(item)]
    return []


# ----------------------------------------------------------------------------------- the rules
def unpinned_actions(document: dict[str, Any]) -> list[str]:
    return [
        f"{job}: {step['uses']}"
        for job, step in steps(document)
        if step.get("uses")
        and not step["uses"].startswith("./")
        and not SHA.fullmatch(step["uses"])
    ]


def expressions_in_scripts(document: dict[str, Any]) -> list[str]:
    found = [f"{job}: {step.get('name')}" for job, step in steps(document)
             if EXPRESSION.search(step.get("run", ""))]  # fmt: skip
    for job, step in steps(document):  # script-type actions run their `with` values as code
        if str(step.get("uses", "")).startswith(SCRIPT_ACTIONS):
            found += [f"{job}: script input {k}" for k, v in step.get("with", {}).items()
                      if EXPRESSION.search(str(v))]  # fmt: skip
    return found


def dangerous_triggers(document: dict[str, Any]) -> list[str]:
    triggers = document["on"]
    names = (
        set(triggers)
        if isinstance(triggers, dict)
        else set([triggers] if isinstance(triggers, str) else triggers)
    )
    problems = sorted(names & FORBIDDEN_TRIGGERS)
    if isinstance(triggers, dict) and "inputs" in (triggers.get("workflow_dispatch") or {}):
        problems.append("workflow_dispatch with inputs")
    return problems


def permission_problems(document: dict[str, Any], name: str) -> list[str]:
    problems = []
    if "permissions" not in document:
        return ["no top-level permissions"]
    top = document["permissions"]
    if top not in ({}, {"contents": "read"}):
        problems.append(f"top-level permissions {top!r}")
    for job, body in document["jobs"].items():
        granted = body.get("permissions", top)
        if "write-all" in str(granted):
            problems.append(f"{job}: write-all")
        for scope, level in (granted or {}).items() if isinstance(granted, dict) else []:
            if level == "write" and scope != "id-token":
                problems.append(f"{job}: {scope}: write")
        if isinstance(granted, dict) and granted.get("id-token") == "write":
            if not (name == "release.yml" and job == "publish"):
                problems.append(f"{job}: id-token: write outside the release publish job")
    return problems


def credential_problems(document: dict[str, Any]) -> list[str]:
    problems = []
    for job, step in steps(document):
        uses = str(step.get("uses", ""))
        if uses.startswith("actions/checkout@"):
            with_ = step.get("with", {})
            if with_.get("persist-credentials") is not False:
                problems.append(f"{job}: checkout persists credentials")
            if "ref" in with_:
                problems.append(f"{job}: checkout with an explicit ref")
    return problems


def unpinned_installs(document: dict[str, Any]) -> list[str]:
    problems = []
    for job, step in steps(document):
        for line in step.get("run", "").splitlines():
            if PINNED_INSTALL.search(line) and "-r " not in line and "==" not in line:
                problems.append(f"{job}: {line.strip()}")
    return problems


def _inherits(node: Any) -> bool:
    if isinstance(node, dict):
        return node.get("secrets") == "inherit" or any(_inherits(v) for v in node.values())
    if isinstance(node, list):
        return any(_inherits(v) for v in node)
    return False


def secret_problems(document: dict[str, Any]) -> list[str]:
    problems = [s for s in strings(document) if SECRETS_CONTEXT.search(s)]
    return problems + (["secrets: inherit"] if _inherits(document) else [])


def pipe_to_shell(path_text: str) -> list[str]:
    return [p.pattern for p in PIPE_TO_SHELL if p.search(path_text)]


def environment_file_problems(document: dict[str, Any], name: str) -> list[str]:
    problems = []
    for job, step in steps(document):
        run = step.get("run", "")
        if "GITHUB_ENV" in run or "GITHUB_PATH" in run:
            problems.append(f"{job}: writes to the environment or the path of later steps")
        if "GITHUB_OUTPUT" in run and not (
            name == "release.yml" and job == "build" and step.get("id") == "digests"
        ):
            problems.append(f"{job}: writes a step output outside the digest step")
    return problems


# ------------------------------------------------------------------------------- real files
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_third_party_action_is_pinned_to_a_full_commit_sha(path: Path) -> None:
    assert unpinned_actions(load(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_expression_inside_a_shell_script_or_a_script_action(path: Path) -> None:
    assert expressions_in_scripts(load(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_dangerous_trigger(path: Path) -> None:
    assert dangerous_triggers(load(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_permissions_are_declared_and_minimal(path: Path) -> None:
    assert permission_problems(load(path), path.name) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_checkout_never_persists_credentials_or_takes_a_ref(path: Path) -> None:
    assert credential_problems(load(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_tool_installs_are_pinned(path: Path) -> None:
    assert unpinned_installs(load(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_workflow_uses_a_secret(path: Path) -> None:
    """Nothing here needs one: publishing is trusted publishing, tests are offline."""
    assert secret_problems(load(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_nothing_pipes_the_network_into_an_interpreter(path: Path) -> None:
    document = load(path)
    text = "\n".join(step.get("run", "") for _, step in steps(document))
    assert pipe_to_shell(text) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_step_writes_the_environment_of_later_steps(path: Path) -> None:
    assert environment_file_problems(load(path), path.name) == []


def test_the_expected_workflows_exist() -> None:
    assert {"ci.yml", "release.yml", "audit.yml"} <= {p.name for p in FILES}


def test_release_is_triggered_by_version_tags_only() -> None:
    triggers = load(WORKFLOWS / "release.yml")["on"]
    assert set(triggers) == {"push"}
    assert set(triggers["push"]) == {"tags"}
    (pattern,) = triggers["push"]["tags"]
    assert pattern.startswith("v") and "*" not in pattern.replace("[0-9]+", "")


def test_release_build_job_is_secretless_read_only_and_checks_ancestry_and_tag() -> None:
    document = load(WORKFLOWS / "release.yml")
    assert document["permissions"] == {}
    build = document["jobs"]["build"]
    assert build["permissions"] == {"contents": "read"}
    assert "environment" not in build
    checkout = next(s for s in build["steps"] if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["fetch-depth"] == 0, "the ancestry check needs the full history"
    ancestry = [s for s in build["steps"] if "merge-base --is-ancestor" in s.get("run", "")]
    assert len(ancestry) == 1
    assert "origin/main" in ancestry[0]["run"] and "GITHUB_SHA" in ancestry[0]["run"]
    tag = [s for s in build["steps"] if "check_release_tag.py" in s.get("run", "")]
    assert tag and tag[0]["env"]["TAG"] == "${{ github.ref_name }}"
    names = [s.get("name", "") for s in build["steps"]]
    first_build = next(i for i, n in enumerate(names) if n.startswith("Build"))
    assert names.index(ancestry[0]["name"]) < first_build
    assert names.index(tag[0]["name"]) < first_build
    build_uses = [s.get("uses", "") for s in build["steps"]]
    assert not any(u.startswith("pypa/gh-action-pypi-publish@") for u in build_uses)
    runs = "\n".join(s.get("run", "") for s in build["steps"])
    assert "pip-audit" in runs and "sha256sum" in runs and "twine check --strict" in runs


def test_release_build_exports_the_digest_as_a_job_output() -> None:
    build = load(WORKFLOWS / "release.yml")["jobs"]["build"]
    assert build["outputs"] == {"digest": "${{ steps.digests.outputs.digest }}"}
    step = next(s for s in build["steps"] if s.get("id") == "digests")
    assert "GITHUB_OUTPUT" in step["run"] and "sha256sum" in step["run"]


def test_release_publish_job_is_exactly_download_verify_publish() -> None:
    document = load(WORKFLOWS / "release.yml")
    publish = document["jobs"]["publish"]
    assert publish["needs"] in ("build", ["build"])
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"]["name"] == "pypi"
    assert "refs/tags/v" in publish["if"]
    download, verify, upload = publish["steps"]
    assert download["uses"].startswith("actions/download-artifact@")
    assert set(download["with"]) == {"name", "path"}
    assert "uses" not in verify and verify["name"] == "Verify the artifact"
    assert set(verify["env"]) == {"TAG", "EXPECTED_DIGEST"}
    assert verify["env"]["EXPECTED_DIGEST"] == "${{ needs.build.outputs.digest }}"
    assert verify["env"]["TAG"] == "${{ github.ref_name }}"
    assert upload["uses"].startswith("pypa/gh-action-pypi-publish@")
    assert set(upload["with"]) == {"packages-dir"}
    assert not FORBIDDEN_PUBLISH_KEYS & set(upload.get("with", {}))
    assert len(publish["steps"]) == 3, "the publish job runs nothing else"


def test_release_verify_step_checks_the_digest_and_the_exact_file_set() -> None:
    publish = load(WORKFLOWS / "release.yml")["jobs"]["publish"]
    run = publish["steps"][1]["run"]
    assert "EXPECTED_DIGEST" in run and "sha256sum" in run
    assert "datacopilot-${version}-py3-none-any.whl" in run
    assert "datacopilot-${version}.tar.gz" in run
    assert "exit 1" in run and "TAG" in run


def test_release_uploads_the_exact_artifact_the_publish_job_downloads() -> None:
    document = load(WORKFLOWS / "release.yml")
    up = next(
        s for s in document["jobs"]["build"]["steps"] if "upload-artifact" in s.get("uses", "")
    )
    down = document["jobs"]["publish"]["steps"][0]
    assert up["with"]["name"] == down["with"]["name"]
    assert up["with"]["if-no-files-found"] == "error"


def test_release_has_no_manual_trigger_and_no_publish_escape_hatches() -> None:
    text = (WORKFLOWS / "release.yml").read_text()
    for needle in ("workflow_dispatch", "skip-existing", "skip_existing", "repository-url",
                   "repository_url", "workflow_call"):  # fmt: skip
        assert needle not in text, needle


def test_ci_has_the_supply_chain_and_packaging_jobs() -> None:
    document = load(WORKFLOWS / "ci.yml")
    assert {"test", "docs", "package", "audit", "lock"} <= set(document["jobs"])
    runs = "\n".join(s.get("run", "") for _, s in steps(document))
    needles = (
        "uv lock --check",
        "mkdocs build --strict",
        "twine check --strict",
        "pip-audit --requirement",
        "verify_install.py",
        "scripts/gen_cli_reference.py --check",
    )
    for needle in needles:
        assert needle in runs, needle
    assert "uvx" not in runs, "the audit tool comes from the lock file"


def test_ci_audits_once_per_supported_python_with_the_locked_tool() -> None:
    audit = load(WORKFLOWS / "ci.yml")["jobs"]["audit"]
    assert audit["strategy"]["matrix"]["python"] == ["3.10", "3.14"]
    run = "\n".join(s.get("run", "") for s in audit["steps"])
    assert "uv run --locked --group release --python" in run and "pip-audit" in run


def test_ci_package_job_does_not_use_a_cache() -> None:
    package = load(WORKFLOWS / "ci.yml")["jobs"]["package"]
    setup = next(s for s in package["steps"] if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert setup["with"]["enable-cache"] is False


def test_the_nightly_audit_is_scheduled_minimal_and_audits_both_pythons() -> None:
    document = load(WORKFLOWS / "audit.yml")
    assert set(document["on"]) <= {"schedule", "workflow_dispatch"} and "schedule" in document["on"]
    assert document["permissions"] == {"contents": "read"}
    job = document["jobs"]["audit"]
    assert job["strategy"]["matrix"]["python"] == ["3.10", "3.14"]
    assert "pip-audit" in "\n".join(s.get("run", "") for s in job["steps"])


# ------------------------------------------------------------------- the lint's own test
BAD = """
on:
  pull_request_target:
  workflow_dispatch:
    inputs:
      who: {}
permissions: write-all
jobs:
  j:
    runs-on: x
    permissions:
      contents: write
      id-token: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
      - uses: actions/github-script@v7
        with:
          script: console.log('${{ github.event.pull_request.title }}')
      - run: echo ${{ github.head_ref }}
      - env:
          TOKEN: ${{ secrets.PYPI_TOKEN }}
        run: |
          echo "x=1" >> "$GITHUB_ENV"
          echo "y=1" >> "$GITHUB_OUTPUT"
          pip install twine
          curl -fsSL https://example.com/install.sh \\
            | sh
          bash <(curl -s https://example.com/x)
"""


def test_the_lint_catches_every_mistake_it_exists_for(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text(BAD)
    document = load(bad)
    text = "\n".join(step.get("run", "") for _, step in steps(document))
    assert unpinned_actions(document)
    assert len(expressions_in_scripts(document)) >= 2
    assert {"pull_request_target", "workflow_dispatch with inputs"} <= set(
        dangerous_triggers(document)
    )
    problems = permission_problems(document, "bad.yml")
    assert any("write-all" in p for p in problems) and any("id-token" in p for p in problems)
    assert any("contents: write" in p for p in problems)
    assert len(credential_problems(document)) == 2
    assert unpinned_installs(document)
    assert secret_problems(document)
    assert len(pipe_to_shell(text)) >= 2
    assert len(environment_file_problems(document, "release.yml")) == 2


def test_the_test_matrix_is_every_python_the_package_declares() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    project = tomllib.loads((WORKFLOWS.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    prefix = "Programming Language :: Python :: "
    declared = sorted(
        (c[len(prefix):] for c in project["project"]["classifiers"]
         if c.startswith(prefix + "3.") and c[len(prefix) + 2:].isdigit()),
        key=lambda v: int(v.split(".")[1]),
    )  # fmt: skip
    matrix = load(WORKFLOWS / "ci.yml")["jobs"]["test"]["strategy"]["matrix"]["python"]
    assert matrix == declared and declared[0] == "3.10" and "3.14" in declared
