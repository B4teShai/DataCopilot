"""Edit the CVAT files of a review export the way a reviewer (or an attacker) would."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # tests read the XML this package wrote
from collections.abc import Callable
from pathlib import Path


def xml_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.xml"))


def edit(directory: Path, change: Callable[[ET.Element], None]) -> None:
    for path in xml_files(directory):
        tree = ET.parse(path)  # noqa: S314  # our own output
        change(tree.getroot())
        tree.write(path, encoding="utf-8", xml_declaration=True)


def proposal_boxes(root: ET.Element) -> dict[str, ET.Element]:
    found: dict[str, ET.Element] = {}
    for box in root.iter("box"):
        for attr in box.findall("attribute"):
            if attr.get("name") == "proposal_id" and attr.text:
                found[attr.text] = box
    return found


def decide(directory: Path, decisions: dict[str, str]) -> None:
    def change(root: ET.Element) -> None:
        for pid, box in proposal_boxes(root).items():
            if pid in decisions:
                for attr in box.findall("attribute"):
                    if attr.get("name") == "decision":
                        attr.text = decisions[pid]

    edit(directory, change)


def move(directory: Path, pid: str, delta: float) -> None:
    def change(root: ET.Element) -> None:
        box = proposal_boxes(root).get(pid)
        if box is not None:
            for key in ("xtl", "ytl"):
                box.set(key, str(float(box.get(key, "0")) + delta))

    edit(directory, change)


def set_attribute(directory: Path, pid: str, name: str, value: str | None) -> None:
    def change(root: ET.Element) -> None:
        box = proposal_boxes(root).get(pid)
        if box is None:
            return
        for attr in box.findall("attribute"):
            if attr.get("name") == name:
                if value is None:
                    box.remove(attr)
                else:
                    attr.text = value

    edit(directory, change)
