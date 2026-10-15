# Release runbook

Publishing to PyPI is irreversible (a version number can never be reused, and a name can never be
given back cleanly). **Nothing in this repository publishes anything on its own, and publishing
needs the maintainer's explicit approval.** The pipeline is `.github/workflows/release.yml`; this
page is what a maintainer does around it.

## One-time setup (maintainer)

1. **PyPI trusted publisher.** On PyPI create the project `datacopilot` (or a pending publisher
   before the first upload) and add a trusted publisher: owner and repository of this project,
   workflow `release.yml`, environment `pypi`. No API token is created or stored anywhere; the
   workflow has none.
2. **GitHub environment `pypi`.** In the repository settings create the environment with required
   reviewers (the maintainers) and restrict deployment to tags. The publish job waits for approval.
3. **Tag protection.** Protect tags matching `v*` so only maintainers can create them.

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
uv run python scripts/verify_install.py dist/datacopilot-*.whl --python 3.10 --python 3.13
```

`verify_install.py` creates a throwaway virtual environment per interpreter, installs the wheel
into it, and runs `datacopilot --version` and `datacopilot doctor`. `tests/release/` builds the wheel
and the sdist and asserts their contents (license and NOTICE present, `py.typed` present, no tests,
benchmarks, plans, assistant or editor configuration directories, `.env` files, caches or golden data, no heavy dependency in the core
`Requires-Dist`, correct metadata).

## Cut the release

1. Update `CHANGELOG.md` (move the entries of `Unreleased` under the new version and date) and set
   `__version__` in `src/datacopilot/__init__.py`. Commit as `chore(release): X.Y.Z`.
2. Wait for CI (`ci.yml`) to be green on that commit.
3. **Ask for approval to publish.** Then tag the commit: `git tag -s vX.Y.Z` and push the tag. The
   tag must equal `__version__` (the workflow refuses a mismatch).
4. `release.yml` builds the wheel and sdist in a job with no secrets and read-only permissions,
   prints their SHA-256 digests, and uploads them as one artifact. The `publish` job, which alone has
   `id-token: write`, runs in the `pypi` environment, waits for a reviewer, downloads that exact
   artifact and publishes it with trusted publishing.
5. Verify from a clean environment: `pip install datacopilot==X.Y.Z` then `datacopilot doctor`.
   Compare the installed files' digests with the recorded ones.

## If something is wrong

A published file cannot be replaced. Yank the release on PyPI (it stays downloadable by exact
version but is not selected by default), fix forward and publish `X.Y.(Z+1)`. If a credential or the
trusted-publisher configuration may be compromised, remove the publisher on PyPI first.

## What the pipeline guarantees

- No long-lived token exists. The build job cannot publish; the publish job cannot run on a branch.
- Every third-party action is pinned to a full commit SHA, and `tests/release/test_workflows.py`
  fails on an unpinned action, `pull_request_target`, a `${{ }}` expression inside a `run:` block or a
  missing minimal `permissions` block.
- The tag is compared with `__version__` before anything is built.
