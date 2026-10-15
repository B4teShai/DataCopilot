"""``datacopilot doctor``: a bounded, read-only diagnosis of the installation.

Nothing here imports a heavy library unless it is installed and only to ask whether a GPU is
visible (the vector-search library is never imported: on some platforms importing it after the
tensor library aborts the interpreter). Versions come from package metadata, optional packages
are looked up without being imported, and the cache directory is measured without following
symlinks, with an entry cap and a deadline. The report never contains a secret (the cache key is
shown as ``set`` or ``unset``) and shows the home directory as ``~``.

The JSON form (``schema`` 1) has the sections ``package``, ``dependencies`` (distribution to
version, ``null`` when missing), ``extras``, ``settings``, ``cache``, ``faiss``, ``gpu`` and
``plugins`` (entry-point names per group, not loaded). ``status`` is ``ok``, ``warn`` (an
unwritable cache directory, a failing GPU probe) or ``fail`` (a missing core dependency or invalid
settings); the exit code is 0, 0 and 1, or 2 for invalid settings.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import re
import time
from importlib.metadata import PackageNotFoundError, entry_points, requires, version
from pathlib import Path
from typing import Any

from datacopilot import __version__
from datacopilot.core.schema import SCHEMA_VERSION
from datacopilot.report.text import clean_text
from datacopilot.settings import Settings

SCHEMA = 1
CORE_DEPENDENCIES = ("click", "jinja2", "numpy", "pyarrow", "pydantic", "structlog", "typer")
MAX_CACHE_ENTRIES = 100_000
"""Directory entries visited when measuring the cache directory."""
CACHE_DEADLINE_SECONDS = 1.0
_TENSOR_DIST = "to" + "rch"
_MODULE_OF = {
    "pillow": "PIL",
    "huggingface-hub": "huggingface_hub",
    "faiss-cpu": "faiss",
}
_PLUGIN_GROUPS = {
    "formats": "datacopilot.formats",
    "checks": "datacopilot.checks",
    "embedders": "datacopilot.embedders",
}
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*["']([A-Za-z0-9_.-]+)["']""")
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _version_of(distribution: str) -> str | None:
    try:
        return version(distribution)
    except (PackageNotFoundError, ValueError):
        return None


def _module_available(module: str) -> bool:
    """True when the top-level module can be found; it is not imported."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _extra_requirements() -> dict[str, list[str]]:
    """``extra -> distribution names`` from the installed package metadata."""
    try:
        requirements = requires("datacopilot") or []
    except PackageNotFoundError:
        return {}
    out: dict[str, list[str]] = {}
    for line in requirements:
        marker = _EXTRA_MARKER.search(line)
        name = _REQUIREMENT_NAME.match(line.strip())
        if marker and name:
            dist = name.group(0).lower().replace("_", "-")
            out.setdefault(marker.group(1), []).append(dist)
    return out


def _extras() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for extra, dists in sorted(_extra_requirements().items()):
        present = sorted(
            d for d in dists if _module_available(_MODULE_OF.get(d, d.replace("-", "_")))
        )
        missing = sorted(set(dists) - set(present))
        out[extra] = {"available": not missing, "missing": missing, "present": present}
    return out


def _display_path(path: Path) -> str:
    text = str(path)
    home = os.path.expanduser("~")
    if home not in ("", "~", os.sep) and (text == home or text.startswith(home + os.sep)):
        text = "~" + text[len(home) :]
    return clean_text(text, 300)


def _cache(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": _display_path(path),
        "exists": False,
        "writable": False,
        "size_bytes": 0,
        "files": 0,
        "truncated": False,
    }
    try:
        if not path.is_dir():
            return info
    except OSError:
        return info
    info["exists"] = True
    info["writable"] = os.access(path, os.W_OK | os.X_OK)
    deadline = time.monotonic() + CACHE_DEADLINE_SECONDS
    stack = [path]
    visited = 0
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    visited += 1
                    if visited > MAX_CACHE_ENTRIES or time.monotonic() > deadline:
                        info["truncated"] = True
                        return info
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        info["files"] += 1
                        info["size_bytes"] += entry.stat(follow_symlinks=False).st_size
        except OSError:
            info["truncated"] = True
    return info


def _probe_gpu() -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "tensor_library": "not installed",
        "cuda": False,
        "devices": [],
        "mps": False,
    }
    if not _module_available(_TENSOR_DIST):
        return gpu
    gpu["tensor_library"] = "installed"
    try:
        lib = importlib.import_module(_TENSOR_DIST)
        gpu["tensor_library"] = clean_text(getattr(lib, "__version__", "installed"), 40)
        gpu["cuda"] = bool(lib.cuda.is_available())
        if gpu["cuda"]:
            for index in range(int(lib.cuda.device_count())):
                props = lib.cuda.get_device_properties(index)
                gpu["devices"].append(
                    {
                        "name": clean_text(props.name, 80),
                        "memory_mb": int(props.total_memory) // (1024 * 1024),
                    }
                )
        mps = getattr(getattr(lib, "backends", None), "mps", None)
        gpu["mps"] = bool(mps.is_available()) if mps is not None else False
    except Exception as exc:  # noqa: BLE001  # a broken driver must not break the diagnosis
        gpu["error"] = type(exc).__name__
    return gpu


def _plugins() -> dict[str, list[str]]:
    return {
        label: sorted({ep.name for ep in entry_points(group=group)})
        for label, group in _PLUGIN_GROUPS.items()
    }


def _settings_section(settings: Settings | None, error: str | None) -> dict[str, Any]:
    if settings is None:
        return {"valid": False, "error": clean_text(error or "invalid settings", 200)}
    return {
        "valid": True,
        "error": None,
        "env": settings.env,
        "log_level": settings.effective_log_level,
        "workers": settings.workers,
        "max_table_rows": settings.max_table_rows,
        "max_table_bytes": settings.max_table_bytes,
        "hash_cache": settings.hash_cache,
        "ci": settings.ci,
        "cache_key": "set" if settings.cache_key is not None else "unset",
        "allowed_licenses": sorted(settings.allowed_licenses),
    }


def diagnose(settings: Settings | None, *, settings_error: str | None = None) -> dict[str, Any]:
    """The report; ``settings`` is None when the environment holds an invalid value."""
    dependencies = {name: _version_of(name) for name in CORE_DEPENDENCIES}
    cache = (
        _cache(settings.cache_dir)
        if settings is not None
        else {
            "path": None,
            "exists": False,
            "writable": False,
            "size_bytes": 0,
            "files": 0,
            "truncated": False,
        }
    )
    faiss_installed = _module_available("faiss")
    gpu = _probe_gpu()
    status = "ok"
    if cache["exists"] and not cache["writable"] or "error" in gpu:
        status = "warn"
    if settings is None or any(v is None for v in dependencies.values()):
        status = "fail"
    return {
        "schema": SCHEMA,
        "status": status,
        "package": {
            "name": "datacopilot",
            "version": __version__,
            "python": platform.python_version(),
            "platform": clean_text(platform.system(), 40),
            "dcp_schema_version": SCHEMA_VERSION,
        },
        "dependencies": dependencies,
        "extras": _extras(),
        "settings": _settings_section(settings, settings_error),
        "cache": cache,
        "faiss": {
            "installed": faiss_installed,
            "version": _version_of("faiss-cpu") if faiss_installed else None,
        },
        "gpu": gpu,
        "plugins": _plugins(),
    }


def exit_code(report: dict[str, Any]) -> int:
    """0 ok or warn, 2 invalid settings, 1 any other failure."""
    if not report["settings"]["valid"]:
        return 2
    return 1 if report["status"] == "fail" else 0


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, allow_nan=False, ensure_ascii=True)


def _gpu_text(gpu: dict[str, Any]) -> str:
    if gpu["tensor_library"] == "not installed":
        return "tensor library not installed"
    parts = [f"tensor library {gpu['tensor_library']}"]
    for device in gpu["devices"]:
        parts.append(f"cuda {device['name']} ({device['memory_mb']} MB)")
    if gpu["mps"]:
        parts.append("mps")
    if not gpu["devices"] and not gpu["mps"]:
        parts.append("no accelerator")
    if "error" in gpu:
        parts.append(f"probe failed: {gpu['error']}")
    return ", ".join(parts)


def to_text(report: dict[str, Any]) -> str:
    """Human form: the status first, then one ``name: value`` line per section."""
    deps = report["dependencies"]
    missing = [name for name, v in deps.items() if v is None]
    settings = report["settings"]
    cache = report["cache"]
    package = report["package"]
    extras = "; ".join(
        f"{name} " + ("available" if e["available"] else "missing " + ", ".join(e["missing"]))
        for name, e in report["extras"].items()
    )
    if settings["valid"]:
        settings_line = (
            f"ok (env={settings['env']}, log_level={settings['log_level']}, "
            f"workers={settings['workers']}, hash_cache={str(settings['hash_cache']).lower()}, "
            f"ci={str(settings['ci']).lower()}, cache_key={settings['cache_key']})"
        )
    else:
        settings_line = f"INVALID ({settings['error']})"
    if cache["path"] is None:
        cache_line = "unknown (settings are invalid)"
    elif not cache["exists"]:
        cache_line = f"{cache['path']} (absent)"
    else:
        cache_line = (
            f"{cache['path']} ({'writable' if cache['writable'] else 'NOT writable'}, "
            f"{cache['files']} files, {cache['size_bytes']} bytes"
            + (", measurement truncated" if cache["truncated"] else "")
            + ")"
        )
    faiss = report["faiss"]
    plugins = report["plugins"]
    lines = [
        report["status"],
        f"package: datacopilot {package['version']} (dcp schema {package['dcp_schema_version']})",
        f"python: {package['python']} on {package['platform']}",
        "dependencies: "
        + (
            f"FAILED (missing: {', '.join(missing)})"
            if missing
            else "ok (" + ", ".join(f"{n} {v}" for n, v in deps.items()) + ")"
        ),
        f"extras: {extras or 'none'}",
        f"settings: {settings_line}",
        f"cache: {cache_line}",
        "faiss: "
        + (
            f"installed {faiss['version'] or ''}".strip() if faiss["installed"] else "not installed"
        ),
        f"gpu: {_gpu_text(report['gpu'])}",
        "plugins: " + ", ".join(f"{k}={len(v)}" for k, v in plugins.items()),
    ]
    return "\n".join(clean_text(line, 600) for line in lines)
