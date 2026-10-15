"""Bounded image-header reader and safe open used by the integrity check."""

import os
import struct
import time
from pathlib import Path

import pytest
from fixtures.synth import jpeg_bytes, png_bytes

from datacopilot.core.errors import FormatError
from datacopilot.io import files


def test_header_reports_format_dimensions_and_orientation(tmp_path: Path) -> None:
    jp = tmp_path / "a.jpg"
    jp.write_bytes(jpeg_bytes(10, 20, orientation=6))
    header = files.image_header(jp)
    assert (header.format, header.width, header.height, header.orientation) == ("jpeg", 10, 20, 6)
    assert header.oriented_size == (20, 10)
    pn = tmp_path / "a.png"
    pn.write_bytes(png_bytes(7, 5, 1))
    header = files.image_header(pn)
    assert (header.format, header.oriented_size, header.orientation) == ("png", (7, 5), 1)


def test_image_size_still_applies_orientation(tmp_path: Path) -> None:
    jp = tmp_path / "a.jpg"
    jp.write_bytes(jpeg_bytes(10, 20, orientation=8))
    assert files.image_size(jp) == (20, 10)


def test_unsupported_magic_is_a_distinct_error(tmp_path: Path) -> None:
    p = tmp_path / "a.webp"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 20)
    with pytest.raises(files.UnsupportedImageError):
        files.image_header(p)
    broken = tmp_path / "b.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    with pytest.raises(FormatError) as info:
        files.image_header(broken)
    assert not isinstance(info.value, files.UnsupportedImageError)


def test_jpeg_with_thousands_of_segments_is_bounded_and_fast(tmp_path: Path) -> None:
    p = tmp_path / "many.jpg"
    p.write_bytes(jpeg_bytes(8, 8, segments=files.MAX_JPEG_SEGMENTS + 10))
    start = time.perf_counter()
    with pytest.raises(FormatError, match="no frame header"):
        files.image_header(p)
    assert time.perf_counter() - start < 1.0
    ok = tmp_path / "ok.jpg"
    ok.write_bytes(jpeg_bytes(8, 8, segments=files.MAX_JPEG_SEGMENTS - 2))
    assert files.image_header(ok).width == 8


def test_header_read_never_reads_much_of_a_huge_file(tmp_path: Path) -> None:
    p = tmp_path / "huge.png"
    with p.open("wb") as handle:
        handle.write(png_bytes(4, 4, 0)[:33])
        handle.truncate(2 * 1024**3)  # sparse 2 GiB
    start = time.perf_counter()
    assert files.image_header(p).oriented_size == (4, 4)
    assert time.perf_counter() - start < 1.0


def test_open_regular_rejects_fifos_directories_and_devices_without_blocking(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    start = time.perf_counter()
    with pytest.raises(FormatError, match="not a regular file"):
        files.open_regular(fifo)
    assert time.perf_counter() - start < 1.0
    with pytest.raises(FormatError, match="not a regular file"):
        files.open_regular(tmp_path)
    with pytest.raises(FormatError, match="not a regular file"):
        files.open_regular(Path("/dev/zero"))
    with pytest.raises(FormatError, match="cannot read"):
        files.open_regular(tmp_path / "missing.png")


def test_exif_orientation_out_of_range_is_ignored(tmp_path: Path) -> None:
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg_bytes(10, 20, orientation=9))
    assert files.image_header(p).orientation == 1


def test_truncated_png_header_is_a_format_error(tmp_path: Path) -> None:
    p = tmp_path / "t.png"
    p.write_bytes(struct.pack(">8s", b"\x89PNG\r\n\x1a\n") + b"\x00\x00")
    with pytest.raises(FormatError):
        files.image_header(p)
