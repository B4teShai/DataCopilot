"""Gate fail-open fixes: unchecked images, unassigned leakage, default gate, plugin aborts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fixtures.auditdata import Img, build
from fixtures.synth import png_bytes

from datacopilot.audit import engine, registry
from datacopilot.audit.checks import duplicates, integrity
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.cli import app as cli
from datacopilot.core.errors import DataCopilotError, ValidationError
from datacopilot.core.schema import Annotation, Category, Dataset, Finding, Image, PluginSpec
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
SAME = png_bytes(16, 16, 7)


def ctx(root: Path, **kw: Any) -> CheckContext:
    return CheckContext(run_id=RUN, root=root, cache_dir=root / "cache", use_hash_cache=False, **kw)


def specs(remote: int, total: int) -> list[Img]:
    out = [Img(f"r{n}.png", write=False, seed=n, path=f"s3://b/{n}.png") for n in range(remote)]
    return out + [Img(f"l{n}.png", seed=100 + n) for n in range(total - remote)]


# ----------------------------------------------------------------- A1
def test_content_changed_is_critical(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png", seed=1)])
    (tmp_path / "images" / "a.png").write_bytes(png_bytes(64, 48, 99))
    (f,) = duplicates.plugin.run(ds, ctx(tmp_path))
    assert f.payload["rule"] == "content_changed" and f.severity == "CRITICAL"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_a_group_spanning_unassigned_and_a_real_split_is_critical(
    tmp_path: Path, split: str
) -> None:
    ds = build(
        tmp_path,
        [
            Img("a.png", 16, 16, split, data=SAME, image_id="1" * 16),
            Img("b.png", 16, 16, "unassigned", data=SAME, image_id="2" * 16),
        ],
    )
    (group,) = [
        f
        for f in duplicates.plugin.run(ds, ctx(tmp_path))
        if f.payload["rule"] == "exact_duplicate"
    ]
    assert group.severity == "CRITICAL" and group.payload["cross_split"] is True


# ----------------------------------------------------------------- A2
@pytest.mark.parametrize(
    ("remote", "total", "severity"),
    [(1, 4, "WARN"), (2, 4, "WARN"), (3, 4, "CRITICAL"), (4, 4, "CRITICAL")],
)
def test_unchecked_fraction_decides_warn_or_critical(
    tmp_path: Path, remote: int, total: int, severity: str
) -> None:
    ds = build(tmp_path, specs(remote, total))
    (a,) = integrity.plugin.run(ds, ctx(tmp_path))
    (b,) = [
        f
        for f in duplicates.plugin.run(ds, ctx(tmp_path))
        if f.payload["rule"] == "images_not_hashed"
    ]
    assert (a.severity, b.severity) == (severity, severity)
    assert a.payload["total"] == total and b.payload["total"] == total


def test_threshold_is_configurable_and_nothing_unchecked_is_silent(tmp_path: Path) -> None:
    ds = build(tmp_path, specs(1, 4))
    strict = ctx(tmp_path).model_copy(update={"config": CheckConfig(max_unchecked_fraction=0.2)})
    assert integrity.plugin.run(ds, strict)[0].severity == "CRITICAL"
    assert integrity.plugin.run(build(tmp_path / "x", specs(0, 3)), ctx(tmp_path)) == []


def test_require_local_images_makes_any_unchecked_image_critical(tmp_path: Path) -> None:
    ds = build(tmp_path, specs(1, 10))
    (f,) = integrity.plugin.run(ds, ctx(tmp_path, require_local_images=True))
    assert f.severity == "CRITICAL"
    g = [
        x
        for x in duplicates.plugin.run(ds, ctx(tmp_path, require_local_images=True))
        if x.payload["rule"] == "images_not_hashed"
    ]
    assert g[0].severity == "CRITICAL"


# ------------------------------------------------------------- CLI (A2, A3)
def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CI", raising=False)
    yield
    clear_request_context()


def store_with(tmp_path: Path, paths: list[str], name: str = "h.dcp") -> Path:
    images = [
        Image(image_id=f"{n:016x}", file_path=p, width=10, height=10) for n, p in enumerate(paths)
    ]
    anns = [
        Annotation(
            ann_id=f"a{n}",
            image_id=i.image_id,
            category_id=1,
            kind="bbox",
            bbox_xyxy_abs=(1, 1, 5, 5),
        )
        for n, i in enumerate(images)
    ]
    out = tmp_path / name
    DatasetStore.create(out, images, anns, [Category(id=1, name="c")], name="h", task="detection")
    return out


@pytest.mark.parametrize(
    "paths",
    [
        [f"http://evil.invalid/{n}.png" for n in range(4)],
        [f"/nonexistent/{n}.png" for n in range(4)],
    ],
)
def test_a_dataset_pointing_every_image_elsewhere_fails_the_default_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paths: list[str]
) -> None:
    dcp = store_with(tmp_path, paths)
    assert run(monkeypatch, "audit", dcp, "--report", tmp_path / "r.html") == 1


def test_require_local_images_flag_fails_on_a_single_unchecked_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "images"
    local.mkdir()
    paths = []
    for n in range(9):
        (local / f"{n}.png").write_bytes(png_bytes(10, 10, n))
        paths.append(str(local / f"{n}.png"))
    dcp = store_with(tmp_path, [*paths, "s3://bucket/x.png"])
    out = ["--report", str(tmp_path / "r.html")]
    # sizes differ from rows (10x10 matches), ids differ from content: use --checks integrity only
    assert run(monkeypatch, "audit", dcp, "--checks", "integrity", *out) == 0
    assert (
        run(monkeypatch, "audit", dcp, "--checks", "integrity", "--require-local-images", *out) == 1
    )


def test_default_gate_is_critical_and_none_opts_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dcp = store_with(tmp_path, ["/nonexistent/a.png"])
    out = ["--report", str(tmp_path / "r.html")]
    assert run(monkeypatch, "audit", dcp, *out) == 1
    assert json.loads((tmp_path / "r.json").read_text())["fail_on"] == ["critical"]
    assert run(monkeypatch, "audit", dcp, "--fail-on", "none", *out) == 0
    assert json.loads((tmp_path / "r.json").read_text())["fail_on"] == []
    assert run(monkeypatch, "audit", dcp, "--fail-on", "none,warn", *out) == 2


def test_empty_dataset_exits_1_with_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "e.dcp"
    DatasetStore.create(out, [], [], [Category(id=1, name="c")], name="e", task="detection")
    assert run(monkeypatch, "audit", out, "--fail-on", "none") == 1
    assert "no images" in capsys.readouterr().err


def test_run_audit_refuses_an_empty_dataset(tmp_path: Path) -> None:
    out = tmp_path / "e.dcp"
    store = DatasetStore.create(out, [], [], [Category(id=1, name="c")], name="e", task="detection")
    with pytest.raises(ValidationError, match="no images"):
        engine.run_audit(store.read(), "all", ctx(tmp_path))


def test_a_systemexit_from_the_command_body_can_not_become_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dcp = store_with(tmp_path, ["/nonexistent/a.png"])

    def boom(*_: Any, **__: Any) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(engine, "run_audit", boom)
    assert run(monkeypatch, "audit", dcp, "--report", tmp_path / "r.html") == 1


# ------------------------------------------------------- plugin aborts (A3)
class Aborting:
    def __init__(self, exc: BaseException) -> None:
        self.spec = PluginSpec(
            name="boxes",
            kind="check",
            license="Apache-2.0",
            extra=None,
            gpu_memory_gb=None,
            version="1",
        )
        self._exc = exc

    def run(self, ds: Dataset, c: CheckContext) -> list[Finding]:
        raise self._exc


class EP:
    def __init__(self, obj: object, error: BaseException | None = None) -> None:
        self.name, self.value, self.obj, self.error = "boxes", "m:p", obj, error

    def load(self) -> object:
        if self.error:
            raise self.error
        return self.obj


@pytest.mark.parametrize("exc", [SystemExit(0), SystemExit(None), GeneratorExit()])
def test_a_plugin_raising_a_base_exception_fails_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    ds = build(tmp_path, [Img("a.png")])
    monkeypatch.setattr(registry, "_entry_points", lambda: [EP(Aborting(exc))])
    with pytest.raises(DataCopilotError, match="aborted"):
        engine.run_audit(ds, ["boxes"], ctx(tmp_path))


def test_a_plugin_that_exits_while_loading_fails_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img("a.png")])
    monkeypatch.setattr(registry, "_entry_points", lambda: [EP(None, SystemExit(0))])
    with pytest.raises(DataCopilotError, match="aborted"):
        engine.run_audit(ds, ["boxes"], ctx(tmp_path))


def test_keyboard_interrupt_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img("a.png")])
    monkeypatch.setattr(registry, "_entry_points", lambda: [EP(Aborting(KeyboardInterrupt()))])
    with pytest.raises(KeyboardInterrupt):
        engine.run_audit(ds, ["boxes"], ctx(tmp_path))
