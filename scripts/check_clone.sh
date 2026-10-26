#!/usr/bin/env bash
# Install DataCopilot from a fresh clone and run what the README promises.
#
#     scripts/check_clone.sh [PYTHON] [WORKDIR]
#
# Clones the committed HEAD of this repository (uncommitted changes are not part of a clone) into
# WORKDIR (default: a new temporary directory) and installs it with `uv sync --frozen --no-dev` for
# PYTHON (default: python3) into a fresh environment, so the verified dependencies are exactly the
# committed uv.lock (the network is used to download them only). Then it runs `datacopilot --version`,
# `datacopilot doctor`, the tutorial's dataset generator and every command of the README quick start,
# comparing each exit code with the one the README states. The README's pip route resolves the newest
# versions an index serves on the day and is not what this check verifies. Caches (uv, DataCopilot)
# and HOME live inside WORKDIR, so nothing is written to the user's caches. Needs git and uv.
# Exit 0 when every step passed, 1 when one failed, 2 on a usage error.
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: check_clone.sh [PYTHON] [WORKDIR]" >&2
  exit 2
fi
python_bin="${1:-python3}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ge 2 ]]; then
  work="$2"
  if [[ -e "$work" && -n "$(ls -A "$work")" ]]; then
    echo "error: $work is not empty" >&2
    exit 2
  fi
  mkdir -p "$work"
else
  work="$(mktemp -d "${TMPDIR:-/tmp}/dc-clone.XXXXXX")"
fi
work="$(cd "$work" && pwd)"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

export HOME="$work/home" XDG_CACHE_HOME="$work/home/xdg" PIP_CACHE_DIR="$work/pip-cache"
export UV_CACHE_DIR="$work/uv-cache" UV_PROJECT_ENVIRONMENT="$work/venv" UV_PYTHON_DOWNLOADS=never
export DATACOPILOT_CACHE_DIR="$work/home/cache" PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH VIRTUAL_ENV CI DATACOPILOT_CACHE_KEY UV_FROZEN UV_LOCKED
mkdir -p "$HOME"

command -v uv >/dev/null || fail "uv is needed to install from the lock file (https://docs.astral.sh/uv/)"
git clone --quiet --no-hardlinks "$repo" "$work/src"
echo "cloned $(git -C "$work/src" rev-parse --short HEAD) into $work/src"
python_path="$(command -v "$python_bin")" || fail "no interpreter $python_bin"
(cd "$work/src" && uv sync --frozen --no-dev --quiet --python "$python_path") ||
  fail "uv sync --frozen of the clone"
venv_python="$work/venv/bin/python"
echo "python $("$venv_python" -c 'import platform; print(platform.python_version())') (locked dependencies)"
program="$work/venv/bin/datacopilot"

"$program" --version || fail "datacopilot --version"
"$program" doctor >"$work/doctor.txt" || fail "datacopilot doctor (see $work/doctor.txt)"
echo "doctor: exit 0"

# The tutorial's first <!-- run --> python block generates the demo dataset `shop/`.
generator="$work/generate.py"
awk '
  /^<!-- run -->$/ { armed = 1; next }
  armed && /^```python$/ { inside = 1; next }
  inside && /^```$/ { exit }
  inside { print }
  /^[^[:space:]]/ && !inside { armed = 0 }
' "$work/src/docs/tutorial.md" >"$generator"
[[ -s "$generator" ]] || fail "no generator block found in docs/tutorial.md"
mkdir -p "$work/run"
(cd "$work/run" && "$venv_python" -I "$generator" >/dev/null) || fail "tutorial generator"

# Every `$ datacopilot ...` line of the README's console block, with the exit code it states.
commands=0
while IFS= read -r line; do
  command="${line#\$ }"
  expected=0
  if [[ "$command" == *"#"* ]]; then
    comment="${command#*#}"
    command="${command%%#*}"
    if [[ "$comment" =~ exit\ ([0-9]+) ]]; then
      expected="${BASH_REMATCH[1]}"
    fi
  fi
  read -r -a argv <<<"$command"
  [[ "${argv[0]}" == "datacopilot" ]] || fail "unexpected README command: $command"
  set +e
  (cd "$work/run" && "$program" "${argv[@]:1}" >/dev/null 2>"$work/stderr.txt")
  code=$?
  set -e
  echo "datacopilot ${argv[*]:1} -> exit $code"
  [[ "$code" == "$expected" ]] || fail "expected exit $expected (see $work/stderr.txt)"
  commands=$((commands + 1))
done < <(awk '/^```console$/ { inside = 1; next } inside && /^```$/ { exit } inside && /^\$ / { print }' \
  "$work/src/README.md")
[[ "$commands" -ge 4 ]] || fail "found $commands quick-start commands in the README"
echo "ok: clone, install, --version, doctor and $commands quick-start commands"
