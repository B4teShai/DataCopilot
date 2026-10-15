"""Authenticated, resumable cache of file hashes keyed by file identity.

The cache is an optimisation that sits on the integrity path of a CI gate, so it is treated as
untrusted input and defended in layers:

1. *Ownership.* The cache directory, the secret and the cache file must be regular entries owned
   by the current user, not symlinks (``O_NOFOLLOW``), and not group- or world-writable (the
   directory and the file) or readable (the secret). Otherwise the cache is ignored for the run
   and ``untrusted_reason`` says why; nothing is written to a distrusted cache.
2. *Authentication.* Each line is ``<key> <sha256> <tag>`` where ``tag`` is HMAC-SHA256 over key
   and digest. The MAC key is derived from a secret and a *binding* string (tool version and
   dataset version). The secret is either supplied out of band (``DATACOPILOT_CACHE_KEY``, never
   stored next to the cache: this is the only mode that defends a cache restored from an untrusted
   source) or, for local interactive use, a random 32-byte ``secret.key`` (mode 0600) beside the
   cache. The in-directory secret only stops blind pre-seeding by someone who cannot read the
   directory; anyone who can restore the cache directory can restore the secret with it. So in CI
   (the ``CI`` variable is set) a cache without an external key is disabled entirely
   (``ci_without_key``).
3. *Identity.* The key is a hash of the absolute path, size, mtime, ctime, inode and device taken
   from ``fstat`` of the descriptor that is hashed (``duplicates.hash_file`` re-checks it), so a
   file replaced after it was keyed is never cached under the old identity.
4. *Bounds.* A cache over ``MAX_CACHE_BYTES`` is untrusted and never appended to; lines are
   fixed-length hex fields read at most ``_MAX_LINE`` bytes at a time (a longer line is skipped
   without being loaded); a truncated last line is tolerated.

The key contains no path in clear text.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from pathlib import Path

MAX_CACHE_BYTES = 64 * 1024 * 1024
"""A larger cache is untrusted and never appended to (about 400,000 entries; R50k needs 8 MB)."""
_MAX_LINE = 256
SECRET_FILE = "secret.key"  # noqa: S105  # a file name, not a secret
_SECRET_BYTES = 32
_HEX = frozenset("0123456789abcdef")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def cache_key(path: Path, st: os.stat_result) -> str:
    """32-hex key of a file version: path, size, mtime, ctime, inode and device.

    ``st`` must come from ``fstat`` of the descriptor that is read, so the key describes the bytes
    that are hashed and a replaced file (new inode, ctime) never matches an old entry.
    """
    material = (
        f"{os.path.abspath(path)}\x00{st.st_size}\x00{st.st_mtime_ns}\x00{st.st_ctime_ns}"
        f"\x00{st.st_ino}\x00{st.st_dev}"
    ).encode("utf-8", "surrogatepass")
    return hashlib.sha256(material).hexdigest()[:32]


def _is_hex(text: str, length: int) -> bool:
    return len(text) == length and _HEX.issuperset(text)


def _owned_by_me(info: os.stat_result) -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is None or info.st_uid == geteuid()


def _private_dir(directory: Path) -> bool:
    try:
        info = os.lstat(directory)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and _owned_by_me(info) and info.st_mode & 0o022 == 0


def _open_private(path: Path, flags: int, forbidden_mode_bits: int, mode: int = 0o600) -> int:
    """Open ``path`` without following symlinks; the descriptor must be a regular file owned by
    this user with none of ``forbidden_mode_bits`` set. Raises OSError otherwise."""
    fd = os.open(path, flags | _NOFOLLOW | getattr(os, "O_BINARY", 0), mode)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not _owned_by_me(info)
            or info.st_mode & forbidden_mode_bits
        ):
            raise PermissionError("not private")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _load_secret(directory: Path) -> bytes | None:
    """Read or create ``secret.key``; a zero-byte or wrongly sized file is replaced.

    Returns None when the file is a symlink, not private or owned by someone else.
    """
    path = directory / SECRET_FILE
    for _ in range(3):
        try:
            fd = _open_private(path, os.O_RDONLY, 0o077)
        except FileNotFoundError:
            try:
                fd = _open_private(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o077)
            except FileExistsError:
                continue
            except OSError:
                return None
            try:
                os.write(fd, secrets.token_bytes(_SECRET_BYTES))
                os.fsync(fd)
            finally:
                os.close(fd)
            continue
        except OSError:
            return None
        try:
            data = os.read(fd, _SECRET_BYTES + 1)
        finally:
            os.close(fd)
        if len(data) == _SECRET_BYTES:
            return data
        try:
            path.unlink()  # damaged secret: every entry under it is invalid anyway
        except OSError:
            return None
    return None


def resolve_secret(
    directory: Path, key: bytes | None, in_ci: bool
) -> tuple[bytes | None, str | None]:
    """The MAC secret for a cache kept in ``directory`` and why none is available.

    ``key`` is the out-of-band secret (``DATACOPILOT_CACHE_KEY``). Without it, ``in_ci`` yields no
    secret (``ci_without_key``); otherwise a private ``secret.key`` beside the cache is used and
    created on first use. The directory must be private to this user, else no secret is returned
    (``directory_not_private``, ``secret_not_private``).

    Raises:
        OSError: the directory cannot be created.
    """
    if key is None and in_ci:
        return None, "ci_without_key"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not _private_dir(directory):
        return None, "directory_not_private"
    secret = key if key is not None else _load_secret(directory)
    if secret is None:
        return None, "secret_not_private"
    return secret, None


class HashCache:
    """In-memory view of a verified cache file with buffered, append-only persistence.

    Use :meth:`open` (verified, on disk) or :meth:`disabled` (memory only, never reads or writes).
    """

    def __init__(self) -> None:
        self._file: Path | None = None
        self._mac_key = b""
        self._entries: dict[str, str] = {}
        self._pending: list[str] = []
        self._needs_newline = False
        self.untrusted_reason: str | None = None

    @classmethod
    def disabled(cls) -> HashCache:
        return cls()

    @classmethod
    def open(
        cls,
        directory: Path,
        file_name: str,
        *,
        binding: str,
        key: bytes | None = None,
        in_ci: bool = False,
    ) -> HashCache:
        """Open (creating it privately if absent) the cache ``directory/file_name``.

        ``key`` is the out-of-band MAC secret. Without it, ``in_ci`` disables persistence
        (``untrusted_reason == "ci_without_key"``); otherwise the in-directory secret is used.

        Raises:
            OSError: the directory cannot be created.
        """
        cache = cls()
        secret, reason = resolve_secret(directory, key, in_ci)
        if secret is None:
            cache.untrusted_reason = reason
            return cache
        cache._mac_key = hashlib.sha256(secret + b"\x00" + binding.encode("utf-8")).digest()
        path = directory / file_name
        try:
            cache._load(path)
        except OSError:
            cache.untrusted_reason = "file_not_private"
            return cache
        if cache.untrusted_reason is None:
            cache._file = path
        return cache

    def _tag(self, key: str, digest: str) -> str:
        return hmac.new(
            self._mac_key, f"{key} {digest}".encode("ascii"), hashlib.sha256
        ).hexdigest()

    def _load(self, path: Path) -> None:
        try:
            fd = _open_private(path, os.O_RDONLY, 0o022)
        except FileNotFoundError:
            return
        with os.fdopen(fd, "rb") as handle:
            if os.fstat(handle.fileno()).st_size > MAX_CACHE_BYTES:
                self.untrusted_reason = "file_too_large"
                return
            consumed = 0
            while consumed <= MAX_CACHE_BYTES:
                raw = handle.readline(_MAX_LINE)
                if not raw:
                    break
                consumed += len(raw)
                if not raw.endswith(b"\n"):
                    if len(raw) >= _MAX_LINE:  # over-long line: skip the rest without loading it
                        while raw and not raw.endswith(b"\n") and consumed <= MAX_CACHE_BYTES:
                            raw = handle.readline(_MAX_LINE)
                            consumed += len(raw)
                        continue
                    self._needs_newline = True  # truncated last line
                self._accept(raw.decode("ascii", errors="replace").rstrip("\n"))

    def _accept(self, line: str) -> None:
        parts = line.split(" ")
        if len(parts) != 3:
            return
        key, digest, tag = parts
        if (
            _is_hex(key, 32)
            and _is_hex(digest, 64)
            and _is_hex(tag, 64)
            and hmac.compare_digest(tag, self._tag(key, digest))
        ):
            self._entries[key] = digest

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, digest: str) -> None:
        self._entries[key] = digest
        if self._file is None:
            return
        self._pending.append(f"{key} {digest} {self._tag(key, digest)}\n")
        if len(self._pending) >= 1000:
            self.flush()

    def flush(self) -> None:
        """Append buffered entries; an I/O error disables further writes instead of raising."""
        if not self._pending or self._file is None:
            self._pending.clear()
            return
        prefix = "\n" if self._needs_newline else ""
        data = (prefix + "".join(self._pending)).encode("ascii")
        self._pending.clear()
        self._needs_newline = False
        try:
            fd = _open_private(self._file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o022)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
            finally:
                os.close(fd)
        except OSError:
            self._file = None

    def discard(self) -> None:
        """Forget every entry and delete the file; later puts start a fresh, authenticated file."""
        self._entries.clear()
        self._pending.clear()
        self._needs_newline = False
        if self._file is not None:
            try:
                self._file.unlink(missing_ok=True)
            except OSError:
                self._file = None
