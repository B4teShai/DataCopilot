# Release runbook

Publishing to PyPI is irreversible (a version number can never be reused, and a name can never be
given back cleanly). **Nothing in this repository publishes anything on its own, and publishing
needs the maintainer's explicit approval.** The pipeline is `.github/workflows/release.yml`; this
page is what a maintainer does around it.

## One-time setup and evidence (maintainer, mandatory)

The workflow cannot enforce repository settings: GitHub creates environments unprotected, so
without the settings below a pushed tag would publish without anyone approving it. The workflow
therefore refuses to build a commit that is not an ancestor of `origin/main`, and the maintainer
records the evidence below **before the first release and after any change of maintainers or
repository settings**. Paste the output of each command into the release notes of the first release.

1. **PyPI trusted publisher.** On PyPI create the project `datacopilot` (or a pending publisher
   before the first upload) and add a trusted publisher bound to: owner `B4teShai`, repository
   `DataCopilot`, workflow `release.yml`, environment `pypi`. No API token is created or stored.
   Evidence: a screenshot or the publisher's page on PyPI (there is no read-only API for it).
2. **The `pypi` environment has required reviewers.** Read-only check:

   ```bash
   gh api repos/B4teShai/DataCopilot/environments/pypi --jq '.protection_rules'
   ```

   The output must contain a rule of type `required_reviewers` with at least one reviewer. Also
   enable "Prevent self-review" (`prevent_self_review: true` in the same response), otherwise the
   person who pushes the tag can approve their own deployment. An empty `protection_rules` means
   the environment is unprotected: do not tag.
3. **The environment only deploys from version tags.**

   ```bash
   gh api repos/B4teShai/DataCopilot/environments/pypi/deployment-branch-policies --jq '.branch_policies'
   gh api repos/B4teShai/DataCopilot/environments/pypi --jq '.deployment_branch_policy'
   ```

   `deployment_branch_policy` must be `custom_branch_policies` with a policy of type `tag` and name
   `v*` (and no branch policy).
4. **A tag ruleset on `refs/tags/v*`.** Use repository rulesets (the legacy "tag protection rules" are
   deprecated). Read-only check:

   ```bash
   gh api repos/B4teShai/DataCopilot/rulesets --jq '.[] | select(.target=="tag") | {name, enforcement}'
   gh api repos/B4teShai/DataCopilot/rulesets/<id> --jq '{conditions: .conditions.ref_name, rules: [.rules[].type]}'
   ```

   The ruleset must be `active`, include `refs/tags/v*`, and its rules must include `creation`,
   `deletion` and `non_fast_forward` (restrict creation, deletion and force-push; bypass only for
   the maintainers).
5. **Branch protection of `main`** (the workflow's ancestry check is only as strong as `main`).

TestPyPI note: the name `datacopilot` on TestPyPI already belongs to an unrelated project by another
author, so a TestPyPI rehearsal under this name is impossible. Do not try to take it over or
publish under another name. Rehearse locally instead (below); the local rehearsal exercises the same
build, metadata check and clean-environment install.

## Rehearse locally

From a clean checkout of the commit to release:

<!-- skip: maintainer procedure; the same checks run in tests/release and the CI job `package` -->
```bash
uv sync --locked --group release
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov=datacopilot --cov-fail-under=85
uv lock --check
uv build --out-dir dist
uv run twine check --strict dist/*
sha256sum dist/*          # shasum -a 256 on macOS; record the digests in the release notes
uv run python scripts/verify_install.py dist/datacopilot-*.whl --python 3.10 --python 3.14
```

`verify_install.py` creates a throwaway virtual environment per interpreter, installs the wheel
into it, and runs `datacopilot --version` and `datacopilot doctor`. `tests/release/` builds the wheel
and the sdist and asserts their contents (license and NOTICE present, `py.typed` present, no tests,
benchmarks, plans, assistant or editor configuration directories, `.env` files, caches or golden data, no heavy dependency in the core
`Requires-Dist`, correct metadata).

## Cut the release

1. Update `CHANGELOG.md` (move the entries of `Unreleased` under the new version and date), set
   `__version__` in `src/datacopilot/__init__.py` and **remove the `> **Release status:**` line from
   `README.md` and `docs/guide/install.md`** (they say the package is not published; the README is the
   package's long description, so the tagged commit must not carry the line, and
   `scripts/check_release_tag.py` refuses a tag while the README still does). Commit as
   `chore(release): X.Y.Z`.
2. Wait for CI (`ci.yml`) to be green on that commit.
3. **Ask for approval to publish.** Then tag the commit: `git tag -s vX.Y.Z` and push the tag. The
   tag must equal `__version__` (the workflow refuses a mismatch).
4. `release.yml` builds the wheel and sdist in a job with no secrets and read-only permissions,
   prints their SHA-256 digests, and uploads them as one artifact. The `publish` job, which alone has
   `id-token: write`, runs in the `pypi` environment, waits for a reviewer, downloads that exact
   artifact and publishes it with trusted publishing.
5. Verify from a clean environment: `pip install datacopilot==X.Y.Z` then `datacopilot doctor`. Download
   what the index serves and compare it with the digests recorded in the build summary:
   `pip download datacopilot==X.Y.Z --no-deps --only-binary :all: -d check/` (the wheel) and
   `pip download datacopilot==X.Y.Z --no-deps --no-binary :all: -d check/` (the sdist), then
   `shasum -a 256 check/*`; both must equal the recorded wheel and sdist digests.

## If something is wrong

A published file cannot be replaced. Yank the release on PyPI (it stays downloadable by exact
version but is not selected by default), fix forward and publish `X.Y.(Z+1)`. If a credential or the
trusted-publisher configuration may be compromised, remove the publisher on PyPI first.

## Build backend

The build environment is created outside the lock file, so a floating `hatchling>=...` would let
whatever the index serves at build time into the release. `pyproject.toml` pins the backend and every
one of its dependencies to exact versions (`[build-system].requires`), and
`tests/release/test_build_backend.py` fails on any requirement that is not an exact pin. The pins are
reviewed and bumped like any dependency (`uv build` in the rehearsal exercises them). Pinning versions
does not pin hashes: an index serving different bytes under a pinned version is not detected here; that
residual risk is accepted and is why the build runs in a job without secrets or write permissions.

## What the pipeline guarantees

- No long-lived token exists. The build job cannot publish; the publish job cannot run on a branch, and
  does nothing but download, verify and publish.
- The build refuses a commit that is not an ancestor of `origin/main`, and a tag that differs from
  `__version__`.
- Every third-party action is pinned to a full commit SHA, and `tests/release/test_workflows.py` fails
  on an unpinned action, `pull_request_target`, a `${{ }}` expression inside a script (or a
  script-type action's input), a secret reference, a missing minimal `permissions` block, a write to
  `GITHUB_ENV`, a pipe from the network into a shell, `skip-existing` or `repository-url`.
- The locked runtime dependencies are audited with `pip-audit` (the locked version of the tool) in the
  release build, in CI for Python 3.10 and 3.14, and nightly (`audit.yml`).
- The publish job recomputes the digest of the downloaded files and compares it with the digest the
  build job exported as a job output, and requires exactly one wheel and one sdist named from the tag.
  This detects a corrupted or swapped artifact in transit through the artifact store; it does **not**
  detect a malicious build, because the same workflow produced both values. What stops a malicious
  build is review of what reaches `main`, the pins above and the protected environment.
