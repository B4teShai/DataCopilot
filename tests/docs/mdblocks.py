"""Read the fenced code blocks of a Markdown page together with their test markers.

A fence may be preceded (on the previous non-blank line) by an HTML comment, invisible in the
rendered site:

* ``<!-- run -->``: the block is executed by ``test_tutorial.py``;
* ``<!-- skip: reason -->``: the block is not executed, and says why;
* ``<!-- output -->``: the block is the expected standard output of the preceding ``run`` block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MARKER = re.compile(r"^<!--\s*(run|output|skip:\s*\S.*?)\s*-->$")
FENCE = re.compile(r"^```(\w*)\s*$")
COMMAND_LANGUAGES = frozenset(
    {"bash", "console", "sh", "shell", "zsh", "python", "py", "yaml", "yml"}
)


@dataclass(frozen=True)
class Block:
    path: Path
    line: int
    lang: str
    text: str
    marker: str | None  # "run", "output", "skip" or None

    @property
    def where(self) -> str:
        return f"{self.path.name}:{self.line}"


def blocks(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8").splitlines()
    found: list[Block] = []
    marker: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        match = MARKER.match(line.strip())
        if match:
            marker = match.group(1).split(":")[0]
            index += 1
            continue
        opened = FENCE.match(line)
        if opened and opened.group(1):
            start = index
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].rstrip() != "```":
                body.append(lines[index])
                index += 1
            found.append(Block(path, start + 1, opened.group(1), "\n".join(body) + "\n", marker))
            marker = None
        elif line.strip():
            marker = None if not line.startswith("```") else marker
        index += 1
    return found
