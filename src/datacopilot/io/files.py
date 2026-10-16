"""Safe file helpers shared by the format adapters: bounded reads, path containment, hashing and
image-header parsing without a imaging library.

Every function treats its inputs as untrusted: paths are resolved and kept inside a root,
sizes are bounded before bytes are read into memory, and image headers are parsed with
length checks so a crafted file cannot cause large allocations.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from datacopilot.core.errors import FormatError

MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_CVAT_BYTES = 64 * 1024 * 1024
"""One CVAT export document (a dataset's boxes in one file); the parser also caps its elements."""
MAX_LABEL_STUDIO_BYTES = 64 * 1024 * 1024
"""One Label Studio JSON export; deeper structure limits are applied after parsing."""
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif"})
UNSUPPORTED_EXTENSIONS = frozenset({".webp", ".tif", ".tiff"})
"""Image files DataCopilot recognises as pictures but cannot identify; every reader refuses them."""
_CHUNK = 1024 * 1024
_BMP_INFO_HEADERS = frozenset({40, 52, 56, 64, 108, 124})
MAX_FILL_BYTES = 64
"""Consecutive 0xFF fill bytes tolerated before a JPEG marker."""
MAX_HEADER_BYTES = 1024 * 1024
"""Bytes a JPEG header walk may consume (reads and skipped segment bodies) before giving up."""
MAX_JPEG_SEGMENTS = 4096
"""JPEG segments walked before the frame header must have appeared (bounds header parsing)."""


def read_bounded(path: Path, limit: int) -> bytes:
    """Read a whole regular file, refusing larger files and FIFOs, devices and directories.

    The type and size checks run on the opened, non-blocking descriptor and the read is bounded
    by ``limit``, so ``/dev/zero`` or a FIFO behind a symlink can neither exhaust memory nor hang.
    """
    try:
        with open_regular(path) as handle:
            if os.fstat(handle.fileno()).st_size > limit:
                raise FormatError(f"{path.name} is larger than {limit} bytes")
            data = handle.read(limit + 1)
    except OSError:
        raise FormatError(f"cannot read {path.name}") from None
    if len(data) > limit:
        raise FormatError(f"{path.name} is larger than {limit} bytes")
    return data


def safe_join(root: Path, rel: str) -> Path:
    """Join an untrusted relative path onto ``root``; the result must stay inside ``root``.

    Raises:
        FormatError: for absolute paths, ``..`` components, backslashes, NUL bytes, or a
            resolved location (following symlinks) outside ``root``.
    """
    parts = PurePosixPath(rel).parts
    if (
        not rel
        or "\x00" in rel
        or "\\" in rel
        or rel.startswith("/")
        or any(p == ".." for p in parts)
    ):
        raise FormatError(f"unsafe path {rel!r}: must be relative and stay inside the dataset")
    base = root.resolve()
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        raise FormatError(f"unsafe path {rel!r}: resolves outside the dataset")
    return target


def image_id(path: Path) -> str:
    """``sha256(file bytes)[:16]``, streamed so large images are never fully in memory."""
    digest = hashlib.sha256()
    try:
        with open_regular(path) as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
    except OSError:
        raise FormatError(f"cannot read image {path.name}") from None
    return digest.hexdigest()[:16]


def _read_exact(handle: BinaryIO, n: int, path: Path) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise FormatError(f"{path.name}: truncated or invalid image header")
    return data


def _exif_orientation(exif: bytes) -> int:
    """Orientation tag (1-8) from an APP1 Exif payload, 1 when absent or malformed."""
    if exif[:6] != b"Exif\x00\x00" or len(exif) < 14:
        return 1
    tiff = exif[6:]
    order = {b"II": "<", b"MM": ">"}.get(tiff[:2])
    if order is None:
        return 1
    (offset,) = struct.unpack(order + "I", tiff[4:8])
    if offset + 2 > len(tiff):
        return 1
    (count,) = struct.unpack(order + "H", tiff[offset : offset + 2])
    for i in range(min(count, 256)):
        start = offset + 2 + 12 * i
        entry = tiff[start : start + 12]
        if len(entry) < 12:
            return 1
        tag, kind = struct.unpack(order + "HH", entry[:4])
        if tag == 0x0112 and kind == 3:
            (value,) = struct.unpack(order + "H", entry[8:10])
            return value if 1 <= value <= 8 else 1
    return 1


class UnsupportedImageError(FormatError):
    """The file does not start with the magic bytes of a format whose header can be parsed."""


@dataclass(frozen=True)
class ImageHeader:
    """What the file header declares, without decoding any pixels.

    ``width`` and ``height`` are the stored frame size; ``orientation`` is the EXIF value (1-8,
    1 when absent). :attr:`oriented_size` swaps the sides for orientations 5-8.
    """

    format: str
    width: int
    height: int
    orientation: int = 1

    @property
    def oriented_size(self) -> tuple[int, int]:
        return (self.height, self.width) if self.orientation >= 5 else (self.width, self.height)


def open_regular(path: Path, *, nofollow: bool = False) -> BinaryIO:
    """Open ``path`` for reading only if it is a regular file.

    Symlinks are followed unless ``nofollow`` is set (``O_NOFOLLOW`` where the platform has it;
    callers that resolved the path themselves pass it so a link swapped in afterwards fails).

    The type check runs on the opened descriptor and the open itself is non-blocking, so a FIFO
    or device named by a hostile dataset can neither hang the process nor stream endless data.

    Raises:
        FormatError: ``cannot read`` for missing or unreadable paths, ``not a regular file``.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if nofollow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise FormatError(f"cannot read {path.name}") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise FormatError(f"{path.name}: not a regular file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _jpeg_header(handle: BinaryIO, path: Path) -> ImageHeader:
    orientation = 1
    for _ in range(MAX_JPEG_SEGMENTS):
        byte = _read_exact(handle, 1, path)
        if byte != b"\xff":
            raise FormatError(f"{path.name}: invalid JPEG structure")
        marker = _read_exact(handle, 1, path)[0]
        fill = 0
        while marker == 0xFF:
            fill += 1
            if fill > MAX_FILL_BYTES:
                raise FormatError(f"{path.name}: invalid JPEG structure (fill bytes)")
            marker = _read_exact(handle, 1, path)[0]
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0xD9, 0xDA):
            break
        (length,) = struct.unpack(">H", _read_exact(handle, 2, path))
        if length < 2:
            raise FormatError(f"{path.name}: invalid JPEG segment")
        body_len = length - 2
        if marker == 0xE1:
            orientation = _exif_orientation(_read_exact(handle, body_len, path))
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            body = _read_exact(handle, body_len, path)
            if len(body) < 5:
                raise FormatError(f"{path.name}: invalid JPEG frame header")
            height, width = struct.unpack(">HH", body[1:5])
            return ImageHeader("jpeg", width, height, orientation)
        else:
            handle.seek(body_len, 1)
        if handle.tell() > MAX_HEADER_BYTES:
            raise FormatError(f"{path.name}: JPEG header is too long")
    raise FormatError(f"{path.name}: JPEG has no frame header")


def _parse_header(handle: BinaryIO, path: Path) -> ImageHeader:
    head = handle.read(8)
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        rest = _read_exact(handle, 16, path)
        if rest[4:8] != b"IHDR":
            raise FormatError(f"{path.name}: invalid PNG header")
        width, height = struct.unpack(">II", rest[8:16])
        header = ImageHeader("png", width, height)
    elif head.startswith(b"\xff\xd8"):
        handle.seek(2)
        header = _jpeg_header(handle, path)
    elif head.startswith((b"GIF87a", b"GIF89a")):
        handle.seek(6)
        width, height = struct.unpack("<HH", _read_exact(handle, 4, path))
        header = ImageHeader("gif", width, height)
    elif head.startswith(b"BM"):
        handle.seek(14)
        (dib_size,) = struct.unpack("<I", _read_exact(handle, 4, path))
        if dib_size == 12:  # BITMAPCOREHEADER: unsigned 16-bit sides
            width, height = struct.unpack("<HH", _read_exact(handle, 4, path))
            header = ImageHeader("bmp", width, height)
        elif dib_size in _BMP_INFO_HEADERS:  # signed 32-bit width and height
            width, height = struct.unpack("<ii", _read_exact(handle, 8, path))
            header = ImageHeader("bmp", width, abs(height))
        else:
            raise FormatError(f"{path.name}: unsupported BMP header size")
    else:
        raise UnsupportedImageError(f"{path.name}: unsupported or invalid image format")
    if header.width <= 0 or header.height <= 0:
        raise FormatError(f"{path.name}: image has a zero dimension")
    return header


def image_header(path: Path, *, nofollow: bool = False) -> ImageHeader:
    """Format, stored size and EXIF orientation from the file header (PNG, JPEG, GIF, BMP).

    Reads a few dozen bytes for PNG, GIF and BMP, and walks at most ``MAX_JPEG_SEGMENTS`` JPEG
    segments (skipping bodies by seeking), so the cost does not depend on the file size.

    Raises:
        UnsupportedImageError: the magic bytes are none of the supported formats.
        FormatError: unreadable, not a regular file, truncated, malformed or zero-sized.
    """
    try:
        with open_regular(path, nofollow=nofollow) as handle:
            return _parse_header(handle, path)
    except OSError:
        raise FormatError(f"cannot read image {path.name}") from None


def identify(path: Path, *, nofollow: bool = False) -> tuple[ImageHeader, str]:
    """Header and :func:`image_id` of one file read through **one** descriptor.

    Opening the path twice (once for the size, once for the digest) would let a file swapped in
    between make the two disagree; here both come from the same opened, regular file. With
    ``nofollow`` the final path component must not be a symlink (callers pass paths they already
    resolved, so a link that appears afterwards fails instead of being followed).

    Raises:
        FormatError: as :func:`image_header`, or the file cannot be read.
    """
    try:
        with open_regular(path, nofollow=nofollow) as handle:
            header = _parse_header(handle, path)
            handle.seek(0)
            digest = hashlib.sha256()
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
    except OSError:
        raise FormatError(f"cannot read image {path.name}") from None
    return header, digest.hexdigest()[:16]


def image_size(path: Path) -> tuple[int, int]:
    """Pixel ``(width, height)`` from the file header (PNG, JPEG, GIF, BMP).

    JPEG EXIF orientations 5-8 swap the sides, matching the data model ("after EXIF
    orientation"). Only the header is read.

    Raises:
        FormatError: for unreadable, truncated, unsupported or zero-sized images.
    """
    return image_header(path).oriented_size


def unique_name(name: str, used: set[str], salt: str) -> str:
    """Return ``name`` or, if taken, ``name`` with ``_<salt>`` before the suffix; records it."""
    candidate = name
    if candidate in used:
        p = PurePosixPath(name)
        candidate = str(p.with_name(f"{p.stem}_{salt}{p.suffix}"))
    used.add(candidate)
    return candidate


def copy_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` (creating parents); a missing source is a FormatError."""
    if not src.is_file():
        raise FormatError(f"source file is missing: {src.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src, dst)
    except OSError:
        raise FormatError(f"cannot copy {src.name}") from None


def refuse_unsupported(name: str) -> None:
    """Raise if ``name`` is a picture in a format the header reader cannot identify.

    An image is never skipped silently: a reader either reads it or stops with this error, so a
    dataset cannot lose photos that the audit would then never see.
    """
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in UNSUPPORTED_EXTENSIONS:
        raise FormatError(
            f"{PurePosixPath(name).name}: unsupported image format ({suffix}); "
            "convert it to JPEG, PNG, BMP or GIF"
        )


TAIL_BYTES = 4096
"""Bytes read from the end of a file by :func:`is_cut_off` (one block, whatever the file size)."""


def is_cut_off(path: Path, header: ImageHeader, *, nofollow: bool = False) -> bool:
    """True if a JPEG, PNG or GIF lacks the marker that ends a complete file.

    JPEG ends with ``FFD9``, PNG with an ``IEND`` chunk and GIF with ``3B``; trailing padding
    after them is allowed (the last :data:`TAIL_BYTES` are searched). A file cut off by an
    interrupted copy keeps its header, so the header check alone cannot see it. BMP has no end
    marker and is never reported. This reads one block from the end of the file, no pixels.
    """
    marker = {"jpeg": b"\xff\xd9", "png": b"IEND", "gif": b";"}.get(header.format)
    if marker is None:
        return False
    with open_regular(path, nofollow=nofollow) as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - TAIL_BYTES))
        tail = handle.read(TAIL_BYTES)
    if header.format == "gif":
        return tail.rstrip(b"\x00")[-1:] != marker
    return marker not in tail
