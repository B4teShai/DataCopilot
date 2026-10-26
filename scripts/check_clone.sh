#!/usr/bin/env bash
# Install DataCopilot from a fresh clone and run what the README promises.
#
#     scripts/check_clone.sh [PYTHON] [WORKDIR]
#
# Clones the committed HEAD of this repository (uncommitted changes are not part of a clone) into
# WORKDIR (default: a new temporary directory), creates an empty virtual environment with PYTHON
# (default: python3), runs `pip install .` in the clone (the network is used for the dependencies
# only), then runs `datacopilot --version`, `datacopilot doctor`, the tutorial's dataset generator and
# every command of the README quick start, comparing each exit code with the one the README states.
# Caches (pip, DataCopilot) and HOME live inside WORKDIR, so nothing is written to the user's caches.
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
export DATACOPILOT_CACHE_DIR="$work/home/cache" PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH VIRTUAL_ENV CI DATACOPILOT_CACHE_KEY
mkdir -p "$HOME"

git clone --quiet --no-hardlinks "$repo" "$work/src"
echo "cloned $(git -C "$work/src" rev-parse --short HEAD) into $work/src"
"$python_bin" -m venv "$work/venv" || fail "cannot create a virtual environment with $python_bin"
venv_python="$work/venv/bin/python"
echo "python $("$venv_python" -c 'import platform; print(platform.python_version())')"
"$venv_python" -m pip install --quiet "$work/src" || fail "pip install of the clone"
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
