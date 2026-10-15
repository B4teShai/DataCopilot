"""Exact duplicates: streamed sha256, resumable cache, severity by split."""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import structlog
from fixtures.auditdata import Img, build, image_id_of
from fixtures.synth import png_bytes

from datacopilot.audit import registry
from datacopilot.audit.checks import duplicates
from datacopilot.audit.context import CheckConfig, CheckContext
from datacopilot.core.schema import Dataset, Finding

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
SAME = png_bytes(16, 16, 7)


def ctx(root: Path, **kw: Any) -> CheckContext:
    return CheckContext(run_id=RUN, root=root, cache_dir=root / "cache", **kw)


def run(ds: Dataset, root: Path, **kw: Any) -> list[Finding]:
    return duplicates.plugin.run(ds, ctx(root, **kw))


def groups(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.payload["rule"] == "exact_duplicate"]


def twin(name: str, split: str, n: int, **kw: Any) -> Img:
    """An image whose bytes equal SAME but whose row id differs, as a merge of two sources would."""
    return Img(name, 16, 16, split, data=SAME, image_id=f"{n:016x}", **kw)


def test_plugin_is_registered() -> None:
    assert "duplicates" in registry.check_names()
    assert registry.get_check("duplicates").spec.kind == "check"


def test_distinct_images_have_no_findings(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(5)])
    assert run(ds, tmp_path) == []


def test_duplicates_within_one_split_are_warn(tmp_path: Path) -> None:
    ds = build(tmp_path, [twin("a.png", "train", 1), twin("b.png", "train", 2)])
    (group,) = groups(run(ds, tmp_path))
    assert group.severity == "WARN" and group.payload["cross_split"] is False
    assert (
        group.payload["count"] == 2
        and group.payload["group_id"] == hashlib.sha256(SAME).hexdigest()[:12]
    )
    assert group.payload["splits"] == {"train": 2}


@pytest.mark.parametrize(
    ("splits", "severity"),
    [
        (("train", "test"), "CRITICAL"),
        (("train", "val"), "CRITICAL"),
        (("val", "test"), "CRITICAL"),
        (("train", "unassigned"), "CRITICAL"),  # unassigned next to a real split is leakage
        (("unassigned", "unassigned"), "WARN"),
        (("val", "val"), "WARN"),
    ],
)
def test_severity_depends_on_whether_the_group_spans_splits(
    tmp_path: Path, splits: tuple[str, str], severity: str
) -> None:
    ds = build(tmp_path, [twin("a.png", splits[0], 1), twin("b.png", splits[1], 2)])
    (group,) = groups(run(ds, tmp_path))
    assert group.severity == severity


def test_one_finding_per_group_with_bounded_members(tmp_path: Path) -> None:
    specs = [twin(f"t{n}.png", "train" if n < 25 else "test", n + 1) for n in range(30)]
    specs += [Img("other.png", seed=99)]
    ds = build(tmp_path, specs)
    (group,) = groups(run(ds, tmp_path))
    assert group.payload["count"] == 30 and group.payload["splits"] == {"train": 25, "test": 5}
    members = group.payload["members"]
    assert isinstance(members, list) and len(members) == duplicates.MAX_MEMBERS_LISTED
    assert group.image_id == min(i.image_id for i in ds.images[:30])


def test_each_group_gets_its_own_finding(tmp_path: Path) -> None:
    other = png_bytes(16, 16, 8)
    specs = [
        twin("a.png", "train", 1),
        twin("b.png", "train", 2),
        Img("c.png", 16, 16, data=other, image_id=f"{3:016x}"),
        Img("d.png", 16, 16, "test", data=other, image_id=f"{4:016x}"),
    ]
    found = groups(run(build(tmp_path, specs), tmp_path))
    assert sorted(f.severity for f in found) == ["CRITICAL", "WARN"]


def test_stale_image_id_is_reported_as_content_changed(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("a.png", seed=1), Img("b.png", seed=2)])
    (tmp_path / "images" / "b.png").write_bytes(png_bytes(64, 48, 200))  # edited after ingest
    (finding,) = run(ds, tmp_path)
    assert finding.severity == "CRITICAL" and finding.payload["rule"] == "content_changed"
    assert finding.image_id == ds.images[1].image_id
    assert finding.payload["actual"] == image_id_of(png_bytes(64, 48, 200))[:12]


def test_images_that_cannot_be_hashed_are_counted_by_reason_in_one_finding(
    tmp_path: Path,
) -> None:
    os.mkfifo(tmp_path / "pipe")
    ds = build(
        tmp_path,
        [
            Img("ok.png", seed=1),
            Img("gone.png", seed=2, write=False),
            Img("remote.png", seed=3, write=False, path="s3://b/x.png"),
            Img("evil.png", seed=4, write=False, path="../x.png"),
            Img("big.png", 16, 16, seed=5),
            Img("dev.png", seed=6, write=False, path=str(tmp_path / "pipe")),
        ],
    )
    (tmp_path / "images" / "big.png").write_bytes(b"\x00" * 5000)
    cfg = CheckConfig(max_image_bytes=4096)
    findings = duplicates.plugin.run(ds, ctx(tmp_path).model_copy(update={"config": cfg}))
    (finding,) = [f for f in findings if f.payload["rule"] == "images_not_hashed"]
    assert finding.severity == "CRITICAL" and finding.image_id is None  # 5 of 6 unchecked
    assert finding.payload["reasons"] == {
        "file_missing": 1,
        "remote": 1,
        "unsafe_path": 1,
        "file_too_large": 1,
        "not_a_regular_file": 1,
    }
    assert finding.payload["count"] == 5


def test_oversized_sparse_file_is_never_read(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img("big.png", 4, 4)])
    with (tmp_path / "images" / "big.png").open("ab") as handle:
        handle.truncate(3 * 1024**3)
    start = time.perf_counter()
    findings = run(ds, tmp_path)
    assert time.perf_counter() - start < 2.0
    assert [f.payload["rule"] for f in findings] == ["images_not_hashed"]


def test_files_are_read_in_bounded_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    big = png_bytes(16, 16, 1) + b"\x00" * (3 * duplicates.CHUNK_BYTES + 5)
    ds = build(tmp_path, [Img("a.png", 16, 16, data=big)])
    sizes: list[int] = []
    real = duplicates.files.open_regular

    class Spy:
        def __init__(self, handle: Any) -> None:
            self._h = handle

        def __enter__(self) -> Spy:
            return self

        def __exit__(self, *exc: object) -> None:
            self._h.close()

        def fileno(self) -> int:
            return int(self._h.fileno())

        def read(self, n: int = -1) -> bytes:
            sizes.append(n)
            return bytes(self._h.read(n))

    monkeypatch.setattr(duplicates.files, "open_regular", lambda p, **kw: Spy(real(p, **kw)))
    assert run(ds, tmp_path) == []
    assert sizes and max(sizes) == duplicates.CHUNK_BYTES and -1 not in sizes


def test_file_that_grows_beyond_the_limit_while_hashing_is_not_hashed(tmp_path: Path) -> None:
    path = tmp_path / "grow.bin"
    path.write_bytes(b"x" * 3000)
    digest, status = duplicates.hash_file(path, max_bytes=2000)
    assert digest is None and status == "file_too_large"
    assert duplicates.hash_file(path, max_bytes=3000) == (
        hashlib.sha256(b"x" * 3000).hexdigest(),
        "ok",
    )


# ------------------------------------------------------------------- cache
def spy_on_hashing(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    real = duplicates.hash_file

    def counting(path: Path, max_bytes: int, **kw: Any) -> tuple[str | None, str]:
        calls.append(path)
        return real(path, max_bytes, **kw)

    monkeypatch.setattr(duplicates, "hash_file", counting)
    return calls


def three(tmp_path: Path) -> Dataset:
    return build(
        tmp_path, [twin("a.png", "train", 1), twin("b.png", "test", 2), Img("c.png", seed=9)]
    )


def test_second_run_only_spot_checks_and_gives_the_same_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(30)])
    calls = spy_on_hashing(monkeypatch)
    first = run(ds, tmp_path)
    assert len(calls) == 30
    calls.clear()
    assert run(ds, tmp_path) == first and len(calls) == 10


def test_cache_is_keyed_by_path_size_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = three(tmp_path)
    run(ds, tmp_path)
    calls = spy_on_hashing(monkeypatch)
    target = tmp_path / "images" / "c.png"
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # same size, new mtime
    run(ds, tmp_path)
    assert calls.count(target) == 1  # a miss is hashed once (the other two are spot checks)
    calls.clear()
    target.write_bytes(target.read_bytes() + b"x")  # new size
    run(ds, tmp_path)
    assert calls.count(target) == 1


def test_cache_file_stores_no_paths_and_is_private(tmp_path: Path) -> None:
    ds = three(tmp_path)
    run(ds, tmp_path)
    cache_file = tmp_path / "cache" / duplicates.CACHE_FILE
    text = cache_file.read_text()
    assert str(tmp_path) not in text and "a.png" not in text
    assert all(len(line.split()) == 3 for line in text.splitlines())
    assert cache_file.stat().st_mode & 0o077 == 0


def test_corrupt_cache_lines_are_ignored_and_never_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = three(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / duplicates.CACHE_FILE).write_text(
        "garbage\n"
        + "g" * 32
        + " "
        + "0" * 64
        + "\n"
        + "0" * 32
        + " short\n"
        + "0" * 32
        + " "
        + "0" * 63
    )
    calls = spy_on_hashing(monkeypatch)
    assert len(groups(run(ds, tmp_path))) == 1 and len(calls) == 3


def test_oversized_cache_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = three(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with (cache_dir / duplicates.CACHE_FILE).open("wb") as handle:
        handle.truncate(duplicates.MAX_CACHE_BYTES + 1)
    calls = spy_on_hashing(monkeypatch)
    run(ds, tmp_path)
    assert len(calls) == 3


def test_interrupted_run_resumes_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(20)])
    real = duplicates.hash_file
    seen: list[Path] = []

    def dies_on_fourth(path: Path, max_bytes: int, **kw: Any) -> tuple[str | None, str]:
        seen.append(path)
        if len(seen) == 15:
            raise KeyboardInterrupt
        return real(path, max_bytes, **kw)

    monkeypatch.setattr(duplicates, "hash_file", dies_on_fourth)
    with pytest.raises(KeyboardInterrupt):
        run(ds, tmp_path)
    monkeypatch.setattr(duplicates, "hash_file", real)
    resumed = spy_on_hashing(monkeypatch)
    run(ds, tmp_path)
    assert len(resumed) == 16  # 6 new files + a 10-file spot check; not all 20 again


def test_unwritable_cache_directory_does_not_fail_the_check(tmp_path: Path) -> None:
    ds = three(tmp_path)
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the cache directory should be")
    context = CheckContext(run_id=RUN, root=tmp_path, cache_dir=blocker / "sub")
    with structlog.testing.capture_logs() as logs:
        findings = duplicates.plugin.run(ds, context)
    assert len(groups(findings)) == 1
    assert [e["event"] for e in logs] == ["audit_cache_unavailable"]
    assert logs[0]["log_level"] == "warning" and str(tmp_path) not in str(logs)


def test_results_do_not_depend_on_worker_count(tmp_path: Path) -> None:
    specs = [Img(f"i{n}.png", seed=n % 7) for n in range(40)]
    specs = [
        Img(s.name, 64, 48, data=png_bytes(64, 48, n % 7), image_id=f"{n:016x}")
        for n, s in enumerate(specs)
    ]
    ds = build(tmp_path, specs)
    one = run(ds, tmp_path, workers=1, use_hash_cache=False)
    four = run(ds, tmp_path, workers=4, use_hash_cache=False)
    assert one == four
    assert Counter(f.payload["rule"] for f in groups(one)) == {"exact_duplicate": 7}
