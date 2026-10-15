"""Authenticated, ownership-checked hash cache and its spot verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import structlog
from fixtures.auditdata import Img, build
from fixtures.synth import png_bytes

from datacopilot.audit.checks import duplicates
from datacopilot.audit.context import CheckContext
from datacopilot.audit.hashcache import HashCache, cache_key
from datacopilot.core.schema import Dataset, Finding

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
KEY = "a" * 32
DIGEST = "b" * 64


def open_cache(
    directory: Path, binding: str = "1|m1", key: bytes | None = None, in_ci: bool = False
) -> HashCache:
    return HashCache.open(directory, duplicates.CACHE_FILE, binding=binding, key=key, in_ci=in_ci)


def cache_file(directory: Path) -> Path:
    return directory / duplicates.CACHE_FILE


def test_entries_round_trip_and_the_directory_and_secret_are_private(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    assert open_cache(d).get(KEY) == DIGEST
    assert d.stat().st_mode & 0o077 == 0
    assert (d / "secret.key").stat().st_mode & 0o077 == 0
    assert cache_file(d).stat().st_mode & 0o077 == 0


def test_forged_digest_without_a_valid_tag_is_rejected(tmp_path: Path) -> None:
    d = tmp_path / "c"
    open_cache(d).flush()
    cache_file(d).write_text(f"{KEY} {DIGEST}\n{KEY} {DIGEST} {'0' * 64}\n")
    cache_file(d).chmod(0o600)
    assert open_cache(d).get(KEY) is None


def test_tampered_line_is_rejected_but_neighbours_survive(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put("1" * 32, "2" * 64)
    cache.put("3" * 32, "4" * 64)
    cache.flush()
    lines = cache_file(d).read_text().splitlines()
    lines[0] = lines[0].replace("2" * 64, "5" * 64)  # change the digest, keep the tag
    cache_file(d).write_text("\n".join(lines) + "\n")
    again = open_cache(d)
    assert again.get("1" * 32) is None and again.get("3" * 32) == "4" * 64


def test_entries_do_not_verify_under_another_tool_or_dataset_version(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d, "1|m1")
    cache.put(KEY, DIGEST)
    cache.flush()
    assert open_cache(d, "1|m1").get(KEY) == DIGEST
    assert open_cache(d, "1|m2").get(KEY) is None
    assert open_cache(d, "2|m1").get(KEY) is None


def test_another_secret_invalidates_every_entry(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    (d / "secret.key").write_bytes(os.urandom(32))
    assert open_cache(d).get(KEY) is None


def test_truncated_file_is_tolerated_and_appending_recovers(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put("1" * 32, "2" * 64)
    cache.flush()
    cache_file(d).write_bytes(cache_file(d).read_bytes()[:-20])  # crash mid-line
    again = open_cache(d)
    assert again.get("1" * 32) is None and again.untrusted_reason is None
    again.put("3" * 32, "4" * 64)
    again.flush()
    assert open_cache(d).get("3" * 32) == "4" * 64


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX ownership checks")
@pytest.mark.parametrize("mode", [0o770, 0o757, 0o777])
def test_group_or_world_writable_directory_is_ignored(tmp_path: Path, mode: int) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    d.chmod(mode)
    again = open_cache(d)
    assert again.get(KEY) is None and again.untrusted_reason == "directory_not_private"
    again.put(KEY, DIGEST)
    again.flush()  # a distrusted cache never writes
    d.chmod(0o700)
    assert open_cache(d).get(KEY) == DIGEST  # untouched on disk, still valid for its owner


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX ownership checks")
def test_group_writable_cache_file_and_secret_are_ignored(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    cache_file(d).chmod(0o664)
    assert open_cache(d).untrusted_reason == "file_not_private"
    cache_file(d).chmod(0o600)
    (d / "secret.key").chmod(0o640)
    assert open_cache(d).untrusted_reason == "secret_not_private"


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX ownership checks")
def test_directory_owned_by_someone_else_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "c"
    open_cache(d).flush()
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(d).st_uid + 1)
    assert open_cache(d).untrusted_reason == "directory_not_private"


def test_symlinked_directory_file_or_secret_is_ignored(tmp_path: Path) -> None:
    real = tmp_path / "real"
    open_cache(real).flush()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert open_cache(link).untrusted_reason == "directory_not_private"
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    target = tmp_path / "elsewhere"
    cache_file(d).rename(target)
    cache_file(d).symlink_to(target)
    assert open_cache(d).untrusted_reason == "file_not_private"
    cache_file(d).unlink()
    cache_file(d).write_text("")
    (d / "secret.key").unlink()
    (d / "secret.key").symlink_to(tmp_path / "x")
    assert open_cache(d).untrusted_reason == "secret_not_private"


def test_oversized_and_long_line_input_is_bounded(tmp_path: Path) -> None:
    d = tmp_path / "c"
    open_cache(d).flush()
    with cache_file(d).open("wb") as handle:
        handle.truncate(300 * 1024 * 1024)
    cache_file(d).chmod(0o600)
    assert open_cache(d).get(KEY) is None
    cache_file(d).write_text("x" * 10_000_000 + "\n")
    assert open_cache(d).get(KEY) is None


def test_discard_removes_the_file_and_later_writes_start_fresh(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    cache.discard()
    assert not cache_file(d).exists() and cache.get(KEY) is None


# ------------------------------------------------- check-level behaviour
def ctx(root: Path, **kw: Any) -> CheckContext:
    return CheckContext(run_id=RUN, root=root, cache_dir=root / "cache", **kw)


def run(ds: Dataset, root: Path, **kw: Any) -> list[Finding]:
    return duplicates.plugin.run(ds, ctx(root, **kw))


def hashed(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    real = duplicates.hash_file

    def counting(path: Path, max_bytes: int, **kw: Any) -> tuple[str | None, str]:
        calls.append(path)
        return real(path, max_bytes, **kw)

    monkeypatch.setattr(duplicates, "hash_file", counting)
    return calls


def poison(root: Path, ds: Dataset, image_index: int, fake: str) -> None:
    """Rewrite the genuine cache so one image's entry carries a forged digest with a valid tag.

    Models an attacker who obtained the secret; the spot check is the second line of defence.
    """
    from datacopilot.audit import hashcache

    cache = HashCache.open(
        root / "cache", duplicates.CACHE_FILE, binding=duplicates.binding_for(ds)
    )
    path = Path(os.path.realpath(ds.images[image_index].file_path))
    st = path.stat()
    cache.put(cache_key(path, st), fake)
    cache.flush()
    assert hashcache  # silence unused


def test_sampled_mismatch_discards_the_whole_cache_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = png_bytes(16, 16, 7)
    specs = [
        Img("a.png", 16, 16, data=same),
        Img("b.png", 16, 16, "test", data=same, image_id="1" * 16),
    ]
    specs += [Img(f"o{n}.png", seed=20 + n) for n in range(8)]  # 10 images: sample covers all
    ds = build(tmp_path, specs)
    run(ds, tmp_path)
    poison(tmp_path, ds, 1, hashlib.sha256(b"something else").hexdigest())
    calls = hashed(monkeypatch)
    with structlog.testing.capture_logs() as logs:
        findings = run(ds, tmp_path)
    rules = {str(f.payload["rule"]) for f in findings}
    assert "hash_cache_mismatch" in rules and "exact_duplicate" in rules  # duplicate not hidden
    (warn,) = [f for f in findings if f.payload["rule"] == "hash_cache_mismatch"]
    assert warn.severity == "WARN" and warn.payload["mismatched"] == 1
    assert len(calls) >= 10 + 0 and [e["event"] for e in logs] == ["audit_cache_discarded"]
    assert logs[0]["log_level"] == "warning" and str(tmp_path) not in str(logs)


def test_spot_check_samples_at_least_ten_hits_and_at_most_the_cap() -> None:
    assert duplicates.sample_size(3) == 3
    assert duplicates.sample_size(10) == 10
    assert duplicates.sample_size(500) == 10
    assert duplicates.sample_size(5000) == 50
    assert duplicates.sample_size(10_000_000) == duplicates.SAMPLE_MAX


def test_spot_check_rehashes_the_sample_on_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(30)])
    run(ds, tmp_path)
    calls = hashed(monkeypatch)
    assert run(ds, tmp_path) == []
    assert len(calls) == 10  # 1 percent of 30 is below the minimum of 10


def test_sample_is_deterministic_per_run_and_differs_between_runs(tmp_path: Path) -> None:
    keys = [f"{n:032x}" for n in range(1000)]
    a = duplicates.pick_sample(keys, "run-a")
    assert a == duplicates.pick_sample(keys, "run-a") and a != duplicates.pick_sample(keys, "run-b")
    assert len(a) == 10


def test_no_cache_option_neither_reads_nor_writes_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(4)])
    run(ds, tmp_path)
    before = sorted(p.name for p in (tmp_path / "cache").iterdir())
    size = cache_file(tmp_path / "cache").stat().st_size
    calls = hashed(monkeypatch)
    run(ds, tmp_path, use_hash_cache=False)
    assert len(calls) == 4  # nothing taken from the cache
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == before
    assert cache_file(tmp_path / "cache").stat().st_size == size
    fresh = tmp_path / "fresh"
    duplicates.plugin.run(
        ds, CheckContext(run_id=RUN, root=tmp_path, cache_dir=fresh, use_hash_cache=False)
    )
    assert not fresh.exists()


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX ownership checks")
def test_untrusted_cache_directory_logs_one_constant_warning_and_hashes_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(4)])
    run(ds, tmp_path)
    (tmp_path / "cache").chmod(0o777)
    calls = hashed(monkeypatch)
    with structlog.testing.capture_logs() as logs:
        run(ds, tmp_path)
    (tmp_path / "cache").chmod(0o700)
    assert len(calls) == 4
    assert [(e["event"], e["reason"]) for e in logs] == [
        ("audit_cache_untrusted", "directory_not_private")
    ]
    assert str(tmp_path) not in str(logs)


# ------------------------------------------------ B1: out-of-band key, CI
def test_external_key_authenticates_and_no_secret_file_is_created(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d, key=b"k" * 32)
    cache.put(KEY, DIGEST)
    cache.flush()
    assert open_cache(d, key=b"k" * 32).get(KEY) == DIGEST
    assert open_cache(d, key=b"x" * 32).get(KEY) is None  # another key
    assert open_cache(d).get(KEY) is None  # the in-directory secret is not the key
    assert [p.name for p in d.iterdir() if p.name == "secret.key"] == [
        "secret.key"
    ]  # from the last call


def test_external_key_never_touches_the_in_directory_secret(tmp_path: Path) -> None:
    d = tmp_path / "c"
    open_cache(d, key=b"k" * 32).flush()
    assert not (d / "secret.key").exists()


def test_in_ci_without_an_external_key_disables_the_cache(tmp_path: Path) -> None:
    d = tmp_path / "c"
    cache = open_cache(d, in_ci=True)
    assert cache.untrusted_reason == "ci_without_key" and not d.exists()
    cache.put(KEY, DIGEST)
    cache.flush()
    assert cache.get(KEY) == DIGEST and not d.exists()  # memory only, nothing persisted
    assert open_cache(d, key=b"k" * 32, in_ci=True).untrusted_reason is None


def test_check_logs_one_constant_warning_in_ci_without_a_key(tmp_path: Path) -> None:
    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(3)])
    context = CheckContext(run_id=RUN, root=tmp_path, cache_dir=tmp_path / "cache", in_ci=True)
    with structlog.testing.capture_logs() as logs:
        duplicates.plugin.run(ds, context)
    assert [(e["event"], e["reason"]) for e in logs] == [
        ("audit_cache_untrusted", "ci_without_key")
    ]
    assert not (tmp_path / "cache").exists()


def test_check_uses_the_external_key_in_ci(tmp_path: Path) -> None:
    from pydantic import SecretStr

    ds = build(tmp_path, [Img(f"i{n}.png", seed=n) for n in range(3)])
    context = CheckContext(
        run_id=RUN,
        root=tmp_path,
        cache_dir=tmp_path / "cache",
        in_ci=True,
        cache_key=SecretStr("k" * 32),
    )
    duplicates.plugin.run(ds, context)
    names = sorted(p.name for p in (tmp_path / "cache").iterdir())
    assert names == [duplicates.CACHE_FILE]
    assert "k" * 32 not in repr(context) and "k" * 32 not in context.model_dump_json()


# ------------------------------------------------ B2: identity in the key
def test_key_changes_with_ctime_inode_and_device(tmp_path: Path) -> None:
    import dataclasses

    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    st = f.stat()
    base = cache_key(f, st)

    def tweak(**kw: int) -> os.stat_result:
        fields = list(st)
        names = {"st_ino": 1, "st_dev": 2}
        for k, v in kw.items():
            fields[names[k]] = v
        return os.stat_result(fields)  # type: ignore[arg-type]

    assert cache_key(f, tweak(st_ino=st.st_ino + 1)) != base
    assert cache_key(f, tweak(st_dev=st.st_dev + 1)) != base
    before = st.st_ctime_ns
    os.chmod(f, 0o600)
    os.chmod(f, 0o644)
    assert f.stat().st_ctime_ns != before and cache_key(f, f.stat()) != base
    assert dataclasses  # silence unused


def test_a_file_swapped_in_with_the_same_size_and_mtime_changes_the_key(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"AAAA")
    old = f.stat()
    key = cache_key(f, old)
    other = tmp_path / "other"
    other.write_bytes(b"BBBB")
    os.utime(other, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.replace(other, f)
    assert cache_key(f, f.stat()) != key  # new inode and ctime, same size and mtime


def test_hash_file_refuses_a_file_that_is_not_the_one_that_was_keyed(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"AAAA")
    key = cache_key(f, f.stat())
    assert duplicates.hash_file(f, 100, nofollow=True, expected_key=key)[1] == "ok"
    other = tmp_path / "o"
    other.write_bytes(b"BBBB")
    os.replace(other, f)
    assert duplicates.hash_file(f, 100, nofollow=True, expected_key=key) == (
        None,
        "changed_during_audit",
    )


def test_changed_during_audit_is_counted_and_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = build(tmp_path, [Img("a.png", seed=1), Img("b.png", seed=2)])
    real = duplicates.hash_file

    def flaky(path: Path, max_bytes: int, **kw: Any) -> tuple[str | None, str]:
        if path.name == "a.png":
            return None, "changed_during_audit"
        return real(path, max_bytes, **kw)

    monkeypatch.setattr(duplicates, "hash_file", flaky)
    findings = run(ds, tmp_path)
    (f,) = [x for x in findings if x.payload["rule"] == "images_not_hashed"]
    assert f.payload["reasons"] == {"changed_during_audit": 1}
    monkeypatch.setattr(duplicates, "hash_file", real)
    calls = hashed(monkeypatch)
    run(ds, tmp_path)
    assert any(p.name == "a.png" for p in calls)  # a was not cached by the interrupted run


# ------------------------------------------------ B3: bounds
def test_oversized_cache_is_untrusted_and_never_grows(tmp_path: Path) -> None:
    d = tmp_path / "c"
    open_cache(d).flush()
    with cache_file(d).open("wb") as handle:
        handle.truncate(hashcache_limit() + 1)
    cache_file(d).chmod(0o600)
    cache = open_cache(d)
    assert cache.untrusted_reason == "file_too_large"
    cache.put(KEY, DIGEST)
    cache.flush()
    assert cache_file(d).stat().st_size == hashcache_limit() + 1


def hashcache_limit() -> int:
    from datacopilot.audit import hashcache

    assert hashcache.MAX_CACHE_BYTES <= 64 * 1024 * 1024
    return hashcache.MAX_CACHE_BYTES


def test_a_huge_line_is_skipped_without_loading_it_and_later_entries_survive(
    tmp_path: Path,
) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put("1" * 32, "2" * 64)
    cache.flush()
    good = cache_file(d).read_bytes()
    cache_file(d).write_bytes(b"x" * 5_000_000 + b"\n" + good)
    assert open_cache(d).get("1" * 32) == "2" * 64


@pytest.mark.parametrize("size", [0, 1, 31, 33])
def test_a_zero_byte_or_wrong_sized_secret_is_recreated(tmp_path: Path, size: int) -> None:
    d = tmp_path / "c"
    cache = open_cache(d)
    cache.put(KEY, DIGEST)
    cache.flush()
    (d / "secret.key").write_bytes(b"s" * size)
    (d / "secret.key").chmod(0o600)
    again = open_cache(d)
    assert again.untrusted_reason is None and again.get(KEY) is None
    assert (d / "secret.key").stat().st_size == 32
    again.put(KEY, DIGEST)
    again.flush()
    assert open_cache(d).get(KEY) == DIGEST
