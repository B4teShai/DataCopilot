"""Refuse a release whose tag is not exactly ``v`` + ``__version__``.

    python scripts/check_release_tag.py "$TAG"

Exit 0 when the tag equals ``v<__version__>``, 1 when it does not (or the version is a development
version or has a leading zero, which PyPI would normalise, or the README still carries the
"Release status" line that says the package is not published: that file is the package's long
description), 2 on a usage error. The tag arrives as an argument, never inside a script body.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "src" / "datacopilot" / "__init__.py"
_VERSION = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']\s*$""", re.M)
_RELEASE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
README = Path(__file__).resolve().parents[1] / "README.md"
STATUS_MARKER = "> **Release status:**"


def read_version(init: Path) -> str:
    match = _VERSION.search(init.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"no __version__ in {init.name}")
    return match.group(1)


def main(argv: list[str], init: Path = INIT, readme: Path = README) -> int:
    if len(argv) != 1:
        print("usage: check_release_tag.py TAG", file=sys.stderr)
        return 2
    try:
        version = read_version(init)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not _RELEASE.fullmatch(version):
        print(f"error: __version__ {version!r} is not a release version (X.Y.Z)", file=sys.stderr)
        return 1
    if argv[0] != f"v{version}":
        print(f"error: tag {argv[0]!r} does not match __version__ {version!r}", file=sys.stderr)
        return 1
    try:
        stale = STATUS_MARKER in readme.read_text(encoding="utf-8")
    except OSError:
        print(f"error: cannot read {readme.name}", file=sys.stderr)
        return 1
    if stale:
        print(
            f"error: {readme.name} still has the release status line; remove it in the release "
            "commit (docs/engineering/release.md)",
            file=sys.stderr,
        )
        return 1
    print(f"tag {argv[0]} matches __version__ {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
