"""``audit --embedder`` and ``embed`` through the real CLI path (``cli.main``): gating, the license
guard, settings wiring and the fail-closed exit codes. The ``hash`` embedder needs no weights."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures import synth
from fixtures.embedfakes import FakeEmbedder
from fixtures.synth import png_bytes

from datacopilot.cli import app as cli
from datacopilot.core.schema import PluginSpec
from datacopilot.embed import backends
from datacopilot.embed import registry as embed_registry
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

ENV = (
    "DATACOPILOT_ENV",
    "DATACOPILOT_LOG_LEVEL",
    "DATACOPILOT_WORKERS",
    "DATACOPILOT_HASH_CACHE",
    "DATACOPILOT_ALLOWED_LICENSES",
    "DATACOPILOT_CACHE_KEY",
    "DATACOPILOT_CACHE_DIR",
    "XDG_CACHE_HOME",
    "CI",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))
    yield
    clear_request_context()


def run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> int:
    monkeypatch.setattr(sys, "argv", ["datacopilot", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def logs(err: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in err.splitlines() if line.startswith("{")]


def write_coco(root: Path, *, leak: bool, extra_train: int = 10) -> Path:
    """COCO export: ten train images, three test images and (when ``leak``) a near copy of a
    train image in the test split (the same file with a few bytes appended)."""
    images: dict[str, list[tuple[str, bytes]]] = {"train": [], "test": []}
    for n in range(extra_train):
        images["train"].append((f"train{n}.png", png_bytes(64, 48, n)))
    for n in range(3):
        images["test"].append((f"test{n}.png", png_bytes(64, 48, 100 + n)))
    if leak:
        images["test"].append(("leaked.png", png_bytes(64, 48, 3) + b"\x00\x01\x02\x03"))
    categories = [{"id": 1, "name": "cat"}]
    next_id = 1
    for split, members in images.items():
        entries, anns = [], []
        for name, data in members:
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "images" / name).write_bytes(data)
            entries.append({"id": next_id, "file_name": name, "width": 64, "height": 48})
            anns.append(
                {"id": next_id, "image_id": next_id, "category_id": 1, "bbox": [4, 4, 20, 20]}
            )
            next_id += 1
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "annotations" / f"instances_{split}.json").write_text(
            json.dumps({"images": entries, "annotations": anns, "categories": categories})
        )
    return root


def ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, *, leak: bool) -> Path:
    src = write_coco(tmp_path / f"{name}-src", leak=leak)
    out = tmp_path / f"{name}.dcp"
    assert run(monkeypatch, "ingest", src, "--format", "coco", "-o", out) == 0
    return out


@pytest.fixture
def leaky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return ingest(monkeypatch, tmp_path, "leaky", leak=True)


@pytest.fixture
def clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return ingest(monkeypatch, tmp_path, "clean", leak=False)


def reports(tmp_path: Path) -> list[str]:
    return [
        "--report",
        str(tmp_path / "r.html"),
        "--json",
        str(tmp_path / "r.json"),
        "--sarif",
        str(tmp_path / "r.sarif"),
    ]


def findings(tmp_path: Path) -> list[dict[str, Any]]:
    return json.loads((tmp_path / "r.json").read_text())["findings"]  # type: ignore[no-any-return]


# ------------------------------------------------------------ the gate on planted leakage
def test_planted_leakage_fails_the_leakage_gate_with_exit_1(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        monkeypatch,
        "audit",
        leaky,
        "--embedder",
        "hash",
        "--fail-on",
        "leakage",
        *reports(tmp_path),
    )
    out, err = capsys.readouterr()
    assert code == 1 and out.startswith("failed ")
    leaks = [f for f in findings(tmp_path) if f["check"] == "leakage"]
    assert [f["payload"]["rule"] for f in leaks] == ["cross_split_near_duplicate"]
    assert leaks[0]["severity"] == "CRITICAL" and leaks[0]["payload"]["splits"] == {
        "test": 1,
        "train": 1,
    }
    names = sorted(m["file_name"] for m in leaks[0]["payload"]["members"])
    assert names == ["leaked.png", "train3.png"]
    assert str(tmp_path) not in (tmp_path / "r.json").read_text()
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["embedder"].startswith("hash@") and "leakage" in report["checks"]
    assert not [r for r in logs(err) if r["message"] == "command_failed"]


def test_clean_data_passes_the_same_gate_with_exit_0(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        monkeypatch,
        "audit",
        clean,
        "--embedder",
        "hash",
        "--fail-on",
        "leakage",
        *reports(tmp_path),
    )
    assert code == 0 and capsys.readouterr().out.startswith("passed ")
    assert [f for f in findings(tmp_path) if f["check"] == "leakage"] == []


def test_the_default_gate_catches_the_leak_too(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "audit", leaky, "--embedder", "hash", *reports(tmp_path)) == 1


def test_without_a_gate_the_leak_is_reported_but_the_exit_is_0(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = run(
        monkeypatch, "audit", leaky, "--embedder", "hash", "--fail-on", "none", *reports(tmp_path)
    )
    assert code == 0
    assert any(f["check"] == "leakage" for f in findings(tmp_path))


def test_the_similarity_threshold_option_reaches_the_checks(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = run(
        monkeypatch,
        "audit",
        leaky,
        "--embedder",
        "hash",
        "--similarity-threshold",
        "0.999",
        "--fail-on",
        "leakage",
        *reports(tmp_path),
    )
    assert code == 0  # the planted pair is at about 0.996: stricter threshold, no leak


# ------------------------------------------------------------ selection and validation (exit 2)
def test_all_includes_the_embedding_checks_only_when_an_embedder_is_given(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "audit", clean, *reports(tmp_path)) == 0
    assert json.loads((tmp_path / "r.json").read_text())["checks"] == [
        "integrity",
        "duplicates",
        "imbalance",
        "boxes",
    ]
    assert json.loads((tmp_path / "r.json").read_text())["embedder"] is None
    assert run(monkeypatch, "audit", clean, "--embedder", "hash", *reports(tmp_path)) == 0
    assert set(json.loads((tmp_path / "r.json").read_text())["checks"]) >= {
        "near_duplicates",
        "leakage",
        "outliers",
    }


@pytest.mark.parametrize(
    "args",
    [
        ["--checks", "leakage"],  # an embedding check without an embedder
        ["--fail-on", "leakage"],  # a gate that could never fire
        ["--checks", "outliers", "--fail-on", "outliers"],
        ["--embedder", "hash", "--checks", "integrity,boxes"],  # an embedder nothing uses
        ["--embedder", "nope"],
        ["--embedder", "Hash"],
        ["--embedder", ""],
        ["--embedder", "hash", "--embed-batch", "0"],
        ["--embedder", "hash", "--embed-batch", "100000"],
        ["--embedder", "hash", "--embed-batch", "x"],
        ["--embed-batch", "8"],  # without an embedder
        ["--embedder", "hash", "--similarity-threshold", "0"],
        ["--embedder", "hash", "--similarity-threshold", "1.5"],
        ["--embedder", "hash", "--similarity-threshold", "-0.2"],
        ["--embedder", "hash", "--similarity-threshold", "nan"],
        ["--embedder", "hash", "--similarity-threshold", "inf"],
        ["--similarity-threshold", "0.9"],  # without an embedder
    ],
)
def test_bad_embedding_options_exit_2_before_anything_is_read_or_written(
    clean: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
) -> None:
    assert run(monkeypatch, "audit", clean, *args, *reports(tmp_path)) == 2
    out, err = capsys.readouterr()
    assert out == "" and "Traceback" not in err
    assert not (clean / "audits").exists() and not (clean / "embeddings").exists()
    assert not (tmp_path / "r.html").exists()


def test_fail_on_leakage_without_an_embedder_names_the_missing_option(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "audit", clean, "--fail-on", "leakage", *reports(tmp_path)) == 2
    assert "--embedder" in capsys.readouterr().err


# ------------------------------------------------------------ the license guard, end to end
def tripwires(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any download, load or even creation of a model fail the test."""
    touched: list[str] = []

    def trip(name: str) -> Any:
        def inner(*a: Any, **kw: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"{name} must not run")

        return inner

    monkeypatch.setattr(backends.HFEmbedderPlugin, "create", trip("create"))
    monkeypatch.setattr(backends.HFEmbedder, "_load", trip("_load"))
    for module in ("transformers", "huggingface_hub", backends.TENSOR_LIB):
        monkeypatch.setitem(sys.modules, module, None)
    return touched


def test_a_disallowed_model_is_refused_before_any_load_through_the_real_cli(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touched = tripwires(monkeypatch)
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "MIT,BSD-3-Clause")  # not Apache-2.0
    code = run(monkeypatch, "audit", clean, "--embedder", "dinov2_base", *reports(tmp_path))
    out, err = capsys.readouterr()
    assert code == 1 and out == ""
    line = next(ln for ln in err.splitlines() if ln.startswith("error:"))
    assert "dinov2_base" in line and "Apache-2.0" in line and "DATACOPILOT_ALLOWED_LICENSES" in line
    assert touched == []
    assert not (clean / "audits").exists() and not (tmp_path / "r.html").exists()
    assert [r for r in logs(err) if r["message"] == "command_failed"]


def test_the_default_allow_list_admits_apache_models(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[frozenset[str]] = []

    def create(self: backends.HFEmbedderPlugin, allowed: frozenset[str]) -> Any:
        seen.append(allowed)
        return FakeEmbedder()

    monkeypatch.setattr(backends.HFEmbedderPlugin, "create", create)
    assert run(monkeypatch, "audit", clean, "--embedder", "dinov2_base", *reports(tmp_path)) == 0
    assert seen == [frozenset({"apache-2.0", "mit", "bsd-3-clause"})]


def test_the_environment_value_is_what_reaches_the_guard(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[frozenset[str]] = []

    def create(self: backends.HFEmbedderPlugin, allowed: frozenset[str]) -> Any:
        seen.append(allowed)
        return FakeEmbedder()

    monkeypatch.setattr(backends.HFEmbedderPlugin, "create", create)
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "Apache-2.0, Custom-License")
    run(monkeypatch, "audit", clean, "--embedder", "siglip2_base", *reports(tmp_path))
    assert seen == [frozenset({"apache-2.0", "custom-license"})]


@pytest.mark.parametrize("value", ["", "MIT;GPL", "a,,b"])
def test_an_invalid_allow_list_exits_2_naming_the_variable(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", value)
    assert run(monkeypatch, "audit", clean, "--embedder", "hash") == 2
    assert "DATACOPILOT_ALLOWED_LICENSES" in capsys.readouterr().err


def test_the_embed_command_obeys_the_same_guard(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touched = tripwires(monkeypatch)
    monkeypatch.setenv("DATACOPILOT_ALLOWED_LICENSES", "MIT")
    assert run(monkeypatch, "embed", clean, "--embedder", "siglip2_base") == 1
    assert touched == [] and "DATACOPILOT_ALLOWED_LICENSES" in capsys.readouterr().err
    assert not (clean / "embeddings").exists()


def test_a_missing_extra_exits_1_naming_it(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for module in ("transformers", "huggingface_hub", backends.TENSOR_LIB):
        monkeypatch.setitem(sys.modules, module, None)
    code = run(monkeypatch, "audit", clean, "--embedder", "dinov2_small")
    err = capsys.readouterr().err
    assert code == 1 and "datacopilot[embed]" in err


# ------------------------------------------------------------ the cache
def test_the_second_audit_uses_the_cache_and_the_dataset_is_untouched(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    version = DatasetStore.open(leaky).manifest().version
    args = ["audit", leaky, "--embedder", "hash", "--fail-on", "leakage", *reports(tmp_path)]
    assert run(monkeypatch, *args) == 1
    first = [r for r in logs(capsys.readouterr().err) if r["message"] == "embeddings_computed"]
    assert first[0]["cache_hits"] == 0 and (leaky / "embeddings").is_dir()
    assert run(monkeypatch, *args) == 1
    second = [r for r in logs(capsys.readouterr().err) if r["message"] == "embeddings_computed"]
    assert second[0]["cache_hits"] == second[0]["image_count"] == 14
    assert DatasetStore.open(leaky).manifest().version == version


def test_in_ci_without_a_key_nothing_is_cached_and_the_verdict_is_unchanged(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    args = ["audit", leaky, "--embedder", "hash", "--fail-on", "leakage", *reports(tmp_path)]
    assert run(monkeypatch, *args) == 1
    assert not (leaky / "embeddings").exists()
    monkeypatch.setenv("DATACOPILOT_CACHE_KEY", "k" * 32)
    assert run(monkeypatch, *args) == 1
    assert (leaky / "embeddings").is_dir()  # the external key enabled the cache in CI


def test_no_embedding_cache_never_touches_the_dataset_directory(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = run(
        monkeypatch,
        "audit",
        leaky,
        "--embedder",
        "hash",
        "--no-embedding-cache",
        "--fail-on",
        "leakage",
        *reports(tmp_path),
    )
    assert code == 1 and not (leaky / "embeddings").exists()


def test_a_forged_cache_cannot_hide_the_leak(
    leaky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planted pair is made to look unrelated in the cache: the verdict must not change."""
    from datacopilot.embed import cache as ec

    monkeypatch.setenv("DATACOPILOT_CACHE_KEY", "k" * 32)
    args = ["audit", leaky, "--embedder", "hash", "--fail-on", "leakage", *reports(tmp_path)]
    assert run(monkeypatch, *args) == 1
    shard = next((leaky / "embeddings").rglob("shard-*.parquet"))
    rows = ec.EmbeddingCache.open(
        leaky / "embeddings",
        ec.CacheIdentity(
            "hash", "1", 128, "float16", DatasetStore.open(leaky).manifest().version, {}
        ),
        b"k" * 32,
    ).load()
    # an attacker without the key rewrites the shard with unrelated vectors
    forged = ec.encode_shard(
        ec.CacheIdentity(
            "hash", "1", 128, "float16", DatasetStore.open(leaky).manifest().version, {}
        ),
        list(rows.ids),
        ec.canonical_vectors(np.eye(len(rows.ids), 128, dtype=np.float32), "float16"),
        list(rows.digests),
    )
    shard.write_bytes(forged)
    assert run(monkeypatch, *args) == 1  # rejected (digest and MAC), recomputed, still caught
    assert any(f["check"] == "leakage" for f in findings(tmp_path))
    assert any(f["payload"]["rule"] == "embedding_cache_rejected" for f in findings(tmp_path))


# ------------------------------------------------------------ fail closed
@dataclass
class EP:
    name: str
    value: str
    obj: object
    loads: list[str] = field(default_factory=list)

    def load(self) -> object:
        return self.obj


class Plugin:
    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.spec = PluginSpec(
            name="broken",
            kind="embedder",
            license="Apache-2.0",
            extra=None,
            gpu_memory_gb=None,
            version="1",
        )

    def create(self, allowed_licenses: frozenset[str]) -> Any:
        return self.embedder


def install(monkeypatch: pytest.MonkeyPatch, embedder: Any) -> None:
    monkeypatch.setattr(
        embed_registry, "_entry_points", lambda: [EP("broken", "t:p", Plugin(embedder))]
    )


@pytest.mark.parametrize("error", [RuntimeError("boom"), MemoryError(), ValueError("x" * 5000)])
def test_an_embedder_error_fails_the_default_gate_and_the_leakage_gate(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    emb = FakeEmbedder()
    emb.fail_after, emb.error = 0, error
    install(monkeypatch, emb)
    for gate in ([], ["--fail-on", "leakage"], ["--fail-on", "warn"]):
        assert (
            run(monkeypatch, "audit", clean, "--embedder", "broken", *gate, *reports(tmp_path)) == 1
        )
    rules = {f["payload"]["rule"] for f in findings(tmp_path)}
    assert {"embedder_error", "images_not_embedded"} <= rules
    assert type(error).__name__ in (tmp_path / "r.json").read_text()
    assert "x" * 100 not in (tmp_path / "r.json").read_text()  # exception text never travels


def test_an_embedder_that_calls_sys_exit_zero_cannot_make_the_audit_pass(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    emb = FakeEmbedder()
    emb.fail_after, emb.error = 0, SystemExit(0)
    install(monkeypatch, emb)
    assert run(monkeypatch, "audit", clean, "--embedder", "broken", *reports(tmp_path)) == 1
    assert not (clean / "audits").exists() or not list((clean / "audits").iterdir())


def test_ctrl_c_during_embedding_exits_130(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = FakeEmbedder()
    emb.fail_after, emb.error = 0, KeyboardInterrupt()
    install(monkeypatch, emb)
    code = run(monkeypatch, "audit", clean, "--embedder", "broken", *reports(tmp_path))
    assert code in (1, 130) and code != 0


def test_images_that_cannot_be_embedded_fail_a_gate_on_the_check(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "clean-src" / "images" / "train0.png").unlink()
    code = run(
        monkeypatch,
        "audit",
        clean,
        "--embedder",
        "hash",
        "--checks",
        "leakage",
        "--fail-on",
        "leakage",
        *reports(tmp_path),
    )
    assert code == 1
    (miss,) = [f for f in findings(tmp_path) if f["payload"]["rule"] == "images_not_embedded"]
    assert miss["payload"]["reasons"] == {"file_missing": 1}


def test_an_embedder_returning_garbage_is_an_error_not_a_pass(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = FakeEmbedder()
    emb.mangle = lambda v: np.full_like(v, np.nan)
    install(monkeypatch, emb)
    assert run(monkeypatch, "audit", clean, "--embedder", "broken", *reports(tmp_path)) == 1
    assert any(f["payload"]["rule"] == "embedder_error" for f in findings(tmp_path))


def test_a_decode_failure_of_one_image_still_gates_on_the_check(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = FakeEmbedder()
    emb.decode_errors[png_bytes(64, 48, 100)] = "decode_failed"
    install(monkeypatch, emb)
    code = run(
        monkeypatch,
        "audit",
        clean,
        "--embedder",
        "broken",
        "--checks",
        "leakage",
        "--fail-on",
        "leakage",
        *reports(tmp_path),
    )
    assert code == 1
    (miss,) = [f for f in findings(tmp_path) if f["payload"]["rule"] == "images_not_embedded"]
    assert miss["payload"]["reasons"] == {"decode_failed": 1}


# ------------------------------------------------------------ `embed`
def test_embed_fills_the_cache_and_a_second_run_finds_everything_cached(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "embed", clean, "--embedder", "hash") == 0
    out, err = capsys.readouterr()
    assert "embedded=13 total=13" in out and "cached=0" in out
    assert [r for r in logs(err) if r["message"] == "embeddings_computed"]
    assert run(monkeypatch, "embed", clean, "--embedder", "hash") == 0
    assert "cached=13" in capsys.readouterr().out


def test_embed_exits_1_when_any_image_has_no_embedding(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean-src" / "images" / "test1.png").unlink()
    assert run(monkeypatch, "embed", clean, "--embedder", "hash") == 1
    out, err = capsys.readouterr()
    assert "embedded=12 total=13" in out and "file_missing=1" in err


def test_embed_exits_2_when_the_cache_cannot_be_used(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CI", "1")
    assert run(monkeypatch, "embed", clean, "--embedder", "hash") == 2  # nothing could be stored
    assert "ci_without_key" in capsys.readouterr().err
    assert run(monkeypatch, "embed", clean, "--embedder", "hash", "--no-embedding-cache") == 2


def test_embed_exits_1_when_the_dataset_is_read_only(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    os.chmod(clean, 0o555)  # noqa: S103
    try:
        code = run(monkeypatch, "embed", clean, "--embedder", "hash")
    finally:
        os.chmod(clean, 0o755)  # noqa: S103
    assert code == 1 and "could not be stored" in capsys.readouterr().err


def test_embed_exits_1_on_an_embedder_error(
    clean: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    emb = FakeEmbedder()
    emb.fail_after, emb.error = 0, RuntimeError("secret")
    install(monkeypatch, emb)
    assert run(monkeypatch, "embed", clean, "--embedder", "broken") == 1
    assert "secret" not in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--embedder", "nope"], ["--embed-batch", "0"], []])
def test_embed_usage_errors_exit_2(
    clean: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    assert run(monkeypatch, "embed", clean, *args) == 2


def test_embed_on_an_unreadable_dataset_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "embed", tmp_path / "missing.dcp", "--embedder", "hash") == 1


def test_synth_fixture_is_still_importable() -> None:
    assert synth.CATEGORIES
