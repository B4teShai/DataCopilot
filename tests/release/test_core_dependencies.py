"""The core install (what `pip install datacopilot` pulls) contains no heavy dependency."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
HEAVY = re.compile(
    "^("
    + "to"
    + "rch|"
    + "to"
    + "rchvision|triton|nvidia-.*|transformers|tokenizers|safetensors|huggingface-hub|"
    r"faiss.*|opencv.*|tensorflow.*|jax.*|scipy|pandas|" + "pola" + "rs|pillow|onnx.*|accelerate)$"
)


def core_closure() -> set[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"]}
    seen: set[str] = set()
    todo = [d["name"] for d in packages["datacopilot"].get("dependencies", [])]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo += [d["name"] for d in packages[name].get("dependencies", [])]
    return seen


def test_the_core_closure_in_the_lock_has_no_heavy_package() -> None:
    closure = core_closure()
    assert {"numpy", "pyarrow", "pydantic", "typer"} <= closure
    assert not [n for n in closure if HEAVY.match(n)], sorted(closure)


def test_the_embed_extra_is_where_the_heavy_packages_live() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    package = next(p for p in lock["package"] if p["name"] == "datacopilot")
    embed = {d["name"] for d in package["optional-dependencies"]["embed"]}
    assert {"transformers", "faiss-cpu", "safetensors"} <= embed
    core = {d["name"] for d in package.get("dependencies", [])}
    assert not core & embed
