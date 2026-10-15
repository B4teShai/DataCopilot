"""Integrity check: files exist, are readable, have a parseable header that matches the row."""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fixtures.auditdata import Img, build
from fixtures.synth import jpeg_bytes, png_bytes

from datacopilot.audit import registry
from datacopilot.audit.checks import integrity
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.core.schema import Dataset, Finding

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"


def ctx(root: Path, **kw: Any) -> CheckContext:
    return CheckContext(run_id=RUN, root=root, cache_dir=root / "cache", **kw)


def run(ds: Dataset, root: Path, **kw: Any) -> list[Finding]:
    return integrity.plugin.run(ds, ctx(root, **kw))


def rules(findings: list[Finding]) -> Counter[str]:
    return Counter(str(f.payload["rule"]) for f in findings)


def test_plugin_is_registered_as_a_first_party_check() -> None:
    assert "integrity" in registry.check_names()
    assert registry.get_check("integrity").spec.license == "Apache-2.0"
    assert registry.get_check("integrity").spec.kind == "check"


def test_clean_dataset_has_no_findings(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png"), Img("b.png", 80, 60)])
    assert run(ds, tmp_path) == []


def test_missing_file_is_critical_and_names_the_image_not_the_directory(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png"), Img("gone.png", write=False)])
    (finding,) = run(ds, tmp_path)
    assert finding.severity == "CRITICAL" and finding.payload["rule"] == "file_missing"
    assert finding.image_id == ds.images[1].image_id
    assert finding.payload["file_name"] == "gone.png"
    assert str(tmp_path) not in finding.model_dump_json()


def test_declared_size_mismatch_is_critical_with_both_sizes(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png", 64, 48, declared=(100, 50))])
    (finding,) = run(ds, tmp_path)
    assert finding.severity == "CRITICAL" and finding.payload["rule"] == "size_mismatch"
    assert finding.payload["declared"] == [100, 50] and finding.payload["actual"] == [64, 48]


def test_exif_orientation_is_noted_and_swaps_the_compared_size(tmp_path: Path) -> None:
    rotated = Img("r.jpg", 20, 10, data=jpeg_bytes(10, 20, orientation=6))  # stored 10x20
    plain = Img("p.jpg", 20, 10, data=jpeg_bytes(20, 10, orientation=1))
    ds = build(tmp_path, [rotated, plain])
    (finding,) = run(ds, tmp_path)
    assert finding.severity == "INFO" and finding.payload["rule"] == "exif_orientation"
    assert finding.payload["orientation"] == 6 and finding.image_id == ds.images[0].image_id


def test_orientation_with_a_declared_size_that_ignores_it_is_a_mismatch(tmp_path: Path) -> None:
    img = Img("r.jpg", 10, 20, data=jpeg_bytes(10, 20, orientation=6))  # oriented 20x10
    ds = build(tmp_path, [img])
    assert rules(run(ds, tmp_path)) == Counter({"size_mismatch": 1, "exif_orientation": 1})


@pytest.mark.parametrize(
    ("data", "rule", "severity"),
    [
        (b"", "empty_file", "CRITICAL"),
        (b"\x89PNG\r\n\x1a\n\x00\x00", "invalid_header", "CRITICAL"),
        (b"\xff\xd8\xff", "invalid_header", "CRITICAL"),
        (b"not an image at all", "invalid_header", "CRITICAL"),  # ingest accepted it once
        (b"<html>404 Not Found</html>", "invalid_header", "CRITICAL"),
        (b"\x00" * 64, "invalid_header", "CRITICAL"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 20, "invalid_header", "CRITICAL"),
        (png_bytes(3, 3, 1).replace(b"IHDR", b"XXXX"), "invalid_header", "CRITICAL"),
    ],
)
def test_unreadable_or_unparseable_files(
    tmp_path: Path, data: bytes, rule: str, severity: str
) -> None:
    ds = build(tmp_path, [Img("x.png", 3, 3, data=data)])
    (finding,) = run(ds, tmp_path)
    assert (finding.payload["rule"], finding.severity) == (rule, severity)


def test_zero_dimension_header_is_invalid(tmp_path: Path) -> None:
    import struct

    data = png_bytes(1, 1, 0).replace(struct.pack(">II", 1, 1), struct.pack(">II", 0, 1))
    ds = build(tmp_path, [Img("z.png", 1, 1, data=data)])
    assert rules(run(ds, tmp_path)) == Counter({"invalid_header": 1})


def test_oversized_file_is_reported_and_not_read(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("big.png", 4, 4)])
    big = tmp_path / "images" / "big.png"
    with big.open("ab") as handle:
        handle.truncate(5000)
    cfg = CheckConfig(max_image_bytes=4096)
    findings = integrity.plugin.run(ds, ctx(tmp_path).model_copy(update={"config": cfg}))
    assert [(f.severity, f.payload["rule"]) for f in findings] == [("WARN", "file_too_large")]
    assert findings[0].payload["bytes"] == 5000


def test_directories_fifos_and_devices_are_not_regular_files_and_never_block(
    tmp_path: Path,
) -> None:
    ds = build(
        tmp_path,
        [
            Img("d.png", write=False, path=str(tmp_path)),
            Img("p.png", write=False, seed=2, path=str(tmp_path / "pipe")),
            Img("z.png", write=False, seed=3, path="/dev/zero"),
        ],
    )
    os.mkfifo(tmp_path / "pipe")
    start = time.perf_counter()
    findings = run(ds, tmp_path, allow_external_images=True)  # /dev/zero is outside the root
    assert time.perf_counter() - start < 2.0
    assert rules(findings) == Counter({"not_a_regular_file": 3})


def test_symlink_to_a_regular_image_is_followed(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png")])
    link = tmp_path / "images" / "link.png"
    link.symlink_to(tmp_path / "images" / "a.png")
    linked = ds.images[0].model_copy(update={"file_path": str(link)})
    assert run(Dataset(ds.manifest, (linked,), ds.annotations), tmp_path) == []


def test_relative_paths_resolve_against_the_dataset_root(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png", path="images/a.png")])
    assert run(ds, tmp_path) == []


@pytest.mark.parametrize(
    "bad", ["../outside.png", "images/../../x.png", "a\\b.png", "images/a\x00.png", "~/x.png"]
)
def test_hostile_relative_paths_are_reported_never_opened(tmp_path: Path, bad: str) -> None:
    ds = build(tmp_path, [Img("a.png", write=False, path=bad)])
    (finding,) = run(ds, tmp_path)
    assert finding.severity == "CRITICAL"
    assert finding.payload["rule"] in {"unsafe_path", "file_missing"}
    if bad != "~/x.png":
        assert finding.payload["rule"] == "unsafe_path"


def test_symlink_escaping_the_root_via_a_relative_path_is_unsafe(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.png").write_bytes(png_bytes(4, 4, 1))
    root = tmp_path / "root"
    (root).mkdir()
    (root / "link").symlink_to(outside)
    ds = build(root, [Img("a.png", 4, 4, write=False, path="link/x.png")])
    (finding,) = run(ds, root)
    assert finding.payload["rule"] == "unsafe_path"


def test_remote_uris_are_counted_not_opened(tmp_path: Path) -> None:
    ds = build(
        tmp_path,
        [
            Img("a.png", write=False, path="s3://bucket/a.png"),
            Img("b.png", write=False, seed=2, path="https://example.invalid/b.png"),
            Img("c.png", seed=3),
        ],
    )
    (finding,) = run(ds, tmp_path)  # 2 of 3 unchecked: above the 50 percent threshold
    assert finding.severity == "CRITICAL" and finding.payload == {
        "rule": "remote_not_checked",
        "count": 2,
        "total": 3,
    }
    assert finding.image_id is None


def test_markup_in_a_file_name_is_kept_verbatim_in_data(tmp_path: Path) -> None:
    name = '<img src=x onerror=alert(1)>"&.png'
    ds = build(tmp_path, [Img(name, write=False)])
    (finding,) = run(ds, tmp_path)
    assert finding.payload["file_name"] == name  # escaping is the renderers' job, not the data's


def test_overlong_file_names_are_truncated_in_payloads(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png", write=False, path=str(tmp_path / ("n" * 3000 + ".png")))])
    (finding,) = run(ds, tmp_path)
    assert len(str(finding.payload["file_name"])) <= 255


def test_results_do_not_depend_on_worker_count(tmp_path: Path) -> None:
    specs = [Img(f"i{n}.png", seed=n, declared=(9, 9) if n % 3 == 0 else None) for n in range(30)]
    ds = build(tmp_path, specs)
    assert run(ds, tmp_path, workers=1) == run(ds, tmp_path, workers=4)
    assert len(run(ds, tmp_path)) == 10


def test_no_pixel_data_is_decoded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = build(tmp_path, [Img("a.png")])
    reads: list[int] = []
    real = integrity.files.open_regular

    class Spy:
        def __init__(self, handle: Any) -> None:
            self._h = handle

        def __enter__(self) -> Spy:
            return self

        def __exit__(self, *exc: object) -> None:
            self._h.close()

        def read(self, n: int = -1) -> bytes:
            reads.append(n)
            return bytes(self._h.read(n))

        def seek(self, *args: int) -> int:
            return int(self._h.seek(*args))

    monkeypatch.setattr(integrity.files, "open_regular", lambda p, **kw: Spy(real(p, **kw)))
    assert run(ds, tmp_path) == []
    assert reads and all(0 < n <= 65_536 for n in reads)
