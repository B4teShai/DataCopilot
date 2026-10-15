"""Resource exhaustion and special-file handling (JPEG fill, descriptors, pools, long strings)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fixtures.auditdata import Img, build
from fixtures.synth import jpeg_bytes, png_bytes
from pydantic import ValidationError as PydanticValidationError

from datacopilot.audit import pool
from datacopilot.audit.checks import duplicates, integrity
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.cli import app as cli
from datacopilot.core.errors import FormatError, StoreError
from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.io import files
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
FAST = 3.0


def ctx(root: Path, **kw: Any) -> CheckContext:
    cfg = kw.pop("config", CheckConfig())
    return CheckContext(
        run_id=RUN, root=root, cache_dir=root / "cache", use_hash_cache=False, config=cfg, **kw
    )


# ---------------------------------------------------------------- C1
def test_a_long_run_of_ff_fill_bytes_is_rejected_quickly(tmp_path: Path) -> None:
    p = tmp_path / "fill.jpg"
    with p.open("wb") as h:
        h.write(b"\xff\xd8")
        h.write(b"\xff" * (40 * 1024 * 1024))
    start = time.perf_counter()
    with pytest.raises(FormatError):
        files.image_header(p)
    assert time.perf_counter() - start < 1.0


def test_fill_bytes_within_the_bound_are_still_accepted(tmp_path: Path) -> None:
    p = tmp_path / "ok.jpg"
    good = jpeg_bytes(8, 6)
    p.write_bytes(good[:2] + b"\xff" * 30 + good[2:])
    assert files.image_header(p).width == 8


def test_header_walk_stops_after_a_total_byte_budget(tmp_path: Path) -> None:
    p = tmp_path / "big.jpg"
    segment = b"\xff\xe2\xff\xff" + b"\x00" * 65533  # 64 KiB APPn segments
    good = jpeg_bytes(8, 6)
    p.write_bytes(good[:2] + segment * 40 + good[2:])  # about 2.5 MiB before the frame header
    with pytest.raises(FormatError, match="header"):
        files.image_header(p)


# ---------------------------------------------------------------- C2
def special(tmp_path: Path, name: str, kind: str) -> Path:
    p = tmp_path / name
    if kind == "fifo":
        os.mkfifo(p)
    elif kind == "zero":
        p.symlink_to("/dev/zero")
    return p


@pytest.mark.parametrize("kind", ["fifo", "zero"])
def test_read_bounded_and_image_id_reject_special_files_without_hanging(
    tmp_path: Path, kind: str
) -> None:
    p = special(tmp_path, "x.json", kind)
    start = time.perf_counter()
    with pytest.raises(FormatError):
        files.read_bounded(p, 1000)
    with pytest.raises(FormatError):
        files.image_id(p)
    assert time.perf_counter() - start < FAST


@pytest.mark.parametrize("kind", ["fifo", "zero"])
def test_a_special_manifest_is_a_store_error(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "ds.dcp"
    root.mkdir()
    special(root, "manifest.json", kind)
    start = time.perf_counter()
    with pytest.raises(StoreError):
        DatasetStore.open(root)
    assert time.perf_counter() - start < FAST


def small_store(tmp_path: Path) -> DatasetStore:
    img = Image(image_id="0" * 16, file_path="/x/a.png", width=10, height=10)
    ann = Annotation(
        ann_id="a", image_id=img.image_id, category_id=1, kind="bbox", bbox_xyxy_abs=(1, 1, 5, 5)
    )
    return DatasetStore.create(
        tmp_path / "ds.dcp", [img], [ann], [Category(id=1, name="c")], name="d", task="detection"
    )


@pytest.mark.parametrize("kind", ["fifo", "zero"])
def test_a_special_table_file_is_a_store_error(tmp_path: Path, kind: str) -> None:
    store = small_store(tmp_path)
    table = tmp_path / "ds.dcp" / "annotations.parquet"
    table.unlink()
    special(tmp_path / "ds.dcp", "annotations.parquet", kind)
    start = time.perf_counter()
    with pytest.raises(StoreError):
        store.read()
    assert time.perf_counter() - start < FAST


@pytest.mark.parametrize("kind", ["fifo", "zero"])
def test_a_special_audit_file_is_a_store_error(tmp_path: Path, kind: str) -> None:
    store = small_store(tmp_path)
    run_id = "2222bbbb-2222-4333-8444-5555bbbb5555"
    (tmp_path / "ds.dcp" / "audits").mkdir()
    special(tmp_path / "ds.dcp" / "audits", f"{run_id}.json", kind)
    with pytest.raises(StoreError):
        store.read_audit(run_id)


def test_a_symlinked_manifest_is_refused(tmp_path: Path) -> None:
    store = small_store(tmp_path)
    real = tmp_path / "ds.dcp" / "manifest.json"
    moved = tmp_path / "moved.json"
    real.rename(moved)
    real.symlink_to(moved)
    with pytest.raises(StoreError):
        DatasetStore.open(store.path)


# ---------------------------------------------------------------- C3
def test_bounded_map_keeps_a_bounded_window_and_preserves_order() -> None:
    pulled = 0
    consumed = 0
    worst = 0

    def source() -> Any:
        nonlocal pulled
        for n in range(200):
            pulled += 1
            yield n

    for value in pool.bounded_map(lambda n: n * 2, source(), workers=2, window=6):
        consumed += 1
        worst = max(worst, pulled - consumed)
        assert value == (consumed - 1) * 2
    assert consumed == 200 and worst <= 7


def test_bounded_map_stops_submitting_at_the_deadline() -> None:
    results = list(
        pool.bounded_map(
            lambda n: (time.sleep(0.02), n)[1],
            range(1000),
            workers=1,
            window=2,
            deadline=time.monotonic() + 0.15,
        )
    )
    assert 1 <= len(results) < 100 and results == list(range(len(results)))


def test_bounded_map_propagates_errors_and_stops_workers() -> None:
    def fn(n: int) -> int:
        if n == 3:
            raise RuntimeError("boom")
        return n

    with pytest.raises(RuntimeError, match="boom"):
        list(pool.bounded_map(fn, range(50), workers=2, window=4))
    assert threading.active_count() < 20


def test_integrity_hits_its_time_budget_and_reports_audit_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(40)])
    real = files.image_header

    def slow(path: Path, **kw: Any) -> Any:
        time.sleep(0.05)
        return real(path, **kw)

    monkeypatch.setattr(integrity.files, "image_header", slow)
    findings = integrity.plugin.run(ds, ctx(tmp_path, config=CheckConfig(max_check_seconds=0.3)))
    (f,) = [x for x in findings if x.payload["rule"] == "audit_incomplete"]
    assert f.severity == "CRITICAL" and f.payload["reason"] == "time_budget"
    assert 0 < int(f.payload["checked"]) < 40 and f.payload["total"] == 40  # type: ignore[call-overload]


def test_incomplete_audit_is_only_a_warning_without_a_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(20)])
    real = files.image_header
    monkeypatch.setattr(
        integrity.files, "image_header", lambda p, **kw: (time.sleep(0.05), real(p, **kw))[1]
    )
    c = ctx(tmp_path, config=CheckConfig(max_check_seconds=0.2), incomplete_is_critical=False)
    (f,) = [x for x in integrity.plugin.run(ds, c) if x.payload["rule"] == "audit_incomplete"]
    assert f.severity == "WARN"


def test_duplicates_stops_at_the_byte_budget_and_says_so(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(10)])
    sizes = [(tmp_path / "images" / f"i{n}.png").stat().st_size for n in range(10)]
    findings = duplicates.plugin.run(
        ds, ctx(tmp_path, config=CheckConfig(max_hash_bytes=sum(sizes[:4]) + 1))
    )
    (f,) = [x for x in findings if x.payload["rule"] == "audit_incomplete"]
    assert f.payload["reason"] == "byte_budget" and f.payload["checked"] == 4
    (n,) = [x for x in findings if x.payload["rule"] == "images_not_hashed"]
    assert n.payload["reasons"] == {"byte_budget": 6}


def test_duplicates_budget_leaves_a_complete_run_alone(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(5)])
    assert duplicates.plugin.run(ds, ctx(tmp_path)) == []


# ---------------------------------------------------------------- C4
def test_image_path_and_names_have_hard_length_limits() -> None:
    ok = {"image_id": "0" * 16, "width": 1, "height": 1}
    Image(file_path="a" * 4096, **ok)
    with pytest.raises(PydanticValidationError):
        Image(file_path="a" * 4097, **ok)
    Category(id=1, name="n" * 65_536)
    with pytest.raises(PydanticValidationError):
        Category(id=1, name="n" * 65_537)
    base: dict[str, Any] = {"image_id": "0" * 16, "category_id": 1, "kind": "classification"}
    Annotation(ann_id="a" * 1024, **base)
    with pytest.raises(PydanticValidationError):
        Annotation(ann_id="a" * 1025, **base)


def test_make_finding_degrades_oversized_payload_values_instead_of_failing() -> None:
    from datacopilot.audit.findings import make_finding

    f = make_finding(
        RUN, "boxes", "WARN", payload={"rule": "r", "name": "x" * 100_000, "l": ["y" * 10**6]}
    )
    assert len(str(f.payload["name"])) <= 600 and len(f.model_dump_json()) < 5000


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("DATACOPILOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CI", raising=False)
    yield
    clear_request_context()


def test_an_8000_character_category_name_goes_through_the_cli_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_name = "L" * 8000
    root = tmp_path / "images"
    root.mkdir()
    images, anns = [], []
    for n in range(12):
        data = png_bytes(20, 20, n)
        (root / f"{n}.png").write_bytes(data)
        iid = __import__("hashlib").sha256(data).hexdigest()[:16]
        images.append(
            Image(
                image_id=iid, file_path=str(root / f"{n}.png"), width=20, height=20, split="train"
            )
        )
        cid = 1 if n < 10 else 2
        anns.append(
            Annotation(
                ann_id=f"a{n}",
                image_id=iid,
                category_id=cid,
                kind="bbox",
                bbox_xyxy_abs=(1, 1, 9, 9),
            )
        )
    cats = [Category(id=1, name=long_name), Category(id=2, name="x" * 5)]
    out = tmp_path / "long.dcp"
    DatasetStore.create(out, images, anns, cats, name="d", task="detection")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datacopilot",
            "audit",
            str(out),
            "--fail-on",
            "none",
            "--report",
            str(tmp_path / "r.html"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert (exc.value.code or 0) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    names = [c["name"] for c in report["summary"]["classes"]]
    assert len(names[0]) == 200
    tail = [f for f in report["findings"] if f["payload"].get("rule") == "long_tail"]
    assert tail and all(len(h["name"]) <= 100 for f in tail for h in f["payload"]["head"])
    assert long_name not in (tmp_path / "r.html").read_text()
