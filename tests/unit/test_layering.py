"""ADR-0004 enforced by reading imports: downward layers, adapters behind registries."""

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "datacopilot"
CORE_DEPENDENCIES = {"click", "jinja2", "numpy", "pyarrow", "pydantic", "structlog", "typer"}
ALLOWED_TOP_LEVEL = set(sys.stdlib_module_names) | CORE_DEPENDENCIES | {"datacopilot", "__future__"}
UPPER = ("cli", "api", "convert", "doctor", "audit", "split", "propose", "patch", "report")
ADAPTER_FREE = {"datacopilot.io.registry", "datacopilot.io.report", "datacopilot.io.files"}


def imports(path: Path, *, top_level_only: bool = False) -> set[str]:
    tree = ast.parse(path.read_text())
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def allowed(name: str, prefixes: set[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def modules(sub: str) -> list[Path]:
    target = SRC / sub
    if target.is_dir():
        return sorted(target.rglob("*.py"))
    return [SRC / f"{sub}.py"] if (SRC / f"{sub}.py").exists() else []


@pytest.mark.parametrize("path", modules("core"), ids=lambda p: p.name)
def test_domain_imports_only_stdlib_pydantic_numpy_and_itself(path: Path) -> None:
    for name in imports(path):
        root = name.split(".")[0]
        allowed_roots = set(sys.stdlib_module_names) | {"pydantic", "numpy", "datacopilot"}
        assert root in allowed_roots | {"__future__"}, f"{path.name} imports {name}"
        if root == "datacopilot":
            assert name.startswith("datacopilot.core"), f"{path.name} imports {name}"


@pytest.mark.parametrize(
    "path",
    modules("io") + modules("store") + modules("embed") + modules("models"),
    ids=lambda p: p.name,
)
def test_infrastructure_never_imports_application_or_interface(path: Path) -> None:
    for name in imports(path):
        assert not any(
            name == f"datacopilot.{u}" or name.startswith(f"datacopilot.{u}.") for u in UPPER
        ), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", modules("io"), ids=lambda p: p.name)
def test_io_does_not_depend_on_the_store(path: Path) -> None:
    assert not any(n.startswith("datacopilot.store") for n in imports(path))


def test_application_reaches_adapters_only_through_the_registry() -> None:
    for name in imports(SRC / "convert.py"):
        if name.startswith("datacopilot.io."):
            assert allowed(name, ADAPTER_FREE), f"convert.py imports adapter {name}"
    for path in modules("cli"):
        for name in imports(path):
            if name.startswith("datacopilot.io."):
                assert allowed(name, ADAPTER_FREE), f"{path.name} imports adapter {name}"


@pytest.mark.parametrize("path", modules("audit") + modules("report"), ids=lambda p: p.name)
def test_audit_and_report_never_import_adapters_or_the_interface(path: Path) -> None:
    for name in imports(path):
        if name.startswith("datacopilot.io."):
            assert allowed(name, ADAPTER_FREE), f"{path.name} imports adapter {name}"
        assert not allowed(name, {"datacopilot.cli", "datacopilot.api"}), f"{path.name}: {name}"


@pytest.mark.parametrize("path", [SRC / "logging.py", SRC / "settings.py"], ids=lambda p: p.name)
def test_cross_cutting_modules_import_no_layer(path: Path) -> None:
    for name in imports(path):
        assert allowed(name, {"datacopilot.settings"}) or not name.startswith("datacopilot"), name


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_only_core_dependencies_are_imported_at_module_top_level(path: Path) -> None:
    roots = {n.split(".")[0] for n in imports(path, top_level_only=True)}
    assert roots <= ALLOWED_TOP_LEVEL, sorted(roots - ALLOWED_TOP_LEVEL)


def test_declared_core_dependencies_are_exactly_the_documented_set() -> None:
    text = (SRC.parents[1] / "pyproject.toml").read_text()
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    names = {
        line.strip().strip(",").strip('"').split(">")[0]
        for line in block.splitlines()
        if ">=" in line
    }
    assert names == CORE_DEPENDENCIES


@pytest.mark.parametrize("path", modules("embed") + modules("models"), ids=lambda p: p.name)
def test_model_runtime_never_imports_the_application_layers_or_other_adapters(path: Path) -> None:
    for name in imports(path):
        assert not allowed(name, {f"datacopilot.{u}" for u in UPPER}), f"{path.name}: {name}"
        if name.startswith("datacopilot.io."):
            assert allowed(name, ADAPTER_FREE), f"{path.name} imports adapter {name}"
        assert not name.startswith("datacopilot.store"), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", modules("models"), ids=lambda p: p.name)
def test_models_does_not_depend_on_embed(path: Path) -> None:
    assert not any(n.startswith("datacopilot.embed") for n in imports(path))


@pytest.mark.parametrize("path", modules("audit") + modules("cli"), ids=lambda p: p.name)
def test_application_reaches_model_backends_only_through_the_embedder_registry(path: Path) -> None:
    for name in imports(path):
        assert not allowed(name, {"datacopilot.embed.backends", "datacopilot.embed.hashembed"}), (
            f"{path.name} imports backend {name}"
        )
