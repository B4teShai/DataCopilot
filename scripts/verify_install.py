r"""Install a built wheel into a throwaway virtual environment and run the installed program.

    python scripts/verify_install.py dist/datacopilot-X.Y.Z-py3-none-any.whl \
        --python 3.10 --python 3.13

For every interpreter: create an empty environment (``uv venv``), install the wheel and its
dependencies, run ``datacopilot --version`` and ``datacopilot doctor`` and require exit code 0 and
the right version. Nothing is installed into the current environment and nothing is uploaded.
Exit 0 when every interpreter passed, 1 when one failed, 2 on a usage error.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # noqa: S404  # runs uv and the installed program
import sys
import tempfile
from pathlib import Path

_WHEEL_VERSION = re.compile(r"^datacopilot-([^-]+)-")
_PYTHON = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
TIMEOUT_SECONDS = 600


def _run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
        **kwargs,  # type: ignore[call-overload]
    )


def verify(wheel: Path, python: str, uv: str) -> bool:
    expected = _WHEEL_VERSION.match(wheel.name)
    with tempfile.TemporaryDirectory(prefix="dc-verify-") as tmp:
        env = Path(tmp) / "venv"
        steps = [
            [uv, "venv", "--python", python, str(env)],
            [uv, "pip", "install", "--python", str(env / "bin" / "python"), str(wheel)],
        ]
        for argv in steps:
            done = _run(argv)
            if done.returncode != 0:
                print(f"python {python}: {' '.join(argv[:3])} failed\n{done.stderr[-600:]}")
                return False
        program = str(env / "bin" / "datacopilot")
        for args in (["--version"], ["doctor"]):
            done = _run([program, *args])
            print(f"python {python}: datacopilot {' '.join(args)} -> exit {done.returncode}")
            print(done.stdout.rstrip())
            if done.returncode != 0:
                print(done.stderr[-600:])
                return False
            if args == ["--version"] and expected and expected.group(1) not in done.stdout:
                print(
                    f"python {python}: version {done.stdout.strip()!r} is not {expected.group(1)}"
                )
                return False
        return True


def main(argv: list[str]) -> int:
    wheel: Path | None = None
    pythons: list[str] = []
    args = iter(argv)
    for arg in args:
        if arg == "--python":
            pythons.append(next(args, ""))
        elif wheel is None and not arg.startswith("-"):
            wheel = Path(arg)
        else:
            print("usage: verify_install.py WHEEL [--python X.Y]...", file=sys.stderr)
            return 2
    if wheel is None or not wheel.is_file() or not all(_PYTHON.fullmatch(p) for p in pythons):
        print("usage: verify_install.py WHEEL [--python X.Y]...", file=sys.stderr)
        return 2
    uv = shutil.which("uv")
    if uv is None:
        print("error: uv is required", file=sys.stderr)
        return 2
    results = [verify(wheel.resolve(), python, uv) for python in pythons or ["3.12"]]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
