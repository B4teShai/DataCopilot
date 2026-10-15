import struct
from pathlib import Path

import pytest
from fixtures.synth import png_bytes

from datacopilot.core.errors import FormatError
from datacopilot.io import files


def jpeg(width: int, height: int, orientation: int | None = None) -> bytes:
    out = b"\xff\xd8"
    if orientation is not None:
        tiff = b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", 1)
        tiff += struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + b"\x00\x00\x00\x00"
        exif = b"Exif\x00\x00" + tiff
        out += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    sof = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    return out + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9"


def test_png_size(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(png_bytes(33, 17, 1))
    assert files.image_size(p) == (33, 17)


def test_gif_and_bmp_size(tmp_path: Path) -> None:
    g = tmp_path / "a.gif"
    g.write_bytes(b"GIF89a" + struct.pack("<HH", 12, 7) + b"\x00" * 8)
    assert files.image_size(g) == (12, 7)
    b = tmp_path / "a.bmp"
    b.write_bytes(
        b"BM" + b"\x00" * 12 + struct.pack("<I", 40) + struct.pack("<ii", 20, -9) + b"\x00" * 8
    )
    assert files.image_size(b) == (20, 9)


def test_jpeg_size_and_exif_orientation(tmp_path: Path) -> None:
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg(10, 20))
    assert files.image_size(p) == (10, 20)
    p.write_bytes(jpeg(10, 20, orientation=6))  # rotated 90 degrees: sides swap
    assert files.image_size(p) == (20, 10)
    p.write_bytes(jpeg(10, 20, orientation=1))
    assert files.image_size(p) == (10, 20)


@pytest.mark.parametrize(
    "data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n\x00", b"\xff\xd8"]
)
def test_unreadable_images_raise_format_error(tmp_path: Path, data: bytes) -> None:
    p = tmp_path / "x.png"
    p.write_bytes(data)
    with pytest.raises(FormatError, match="x.png"):
        files.image_size(p)


def test_zero_sized_png_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "z.png"
    p.write_bytes(png_bytes(1, 1, 0).replace(struct.pack(">II", 1, 1), struct.pack(">II", 0, 1)))
    with pytest.raises(FormatError):
        files.image_size(p)


def test_image_id_is_sha256_prefix_of_bytes_and_ignores_location(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_bytes(png_bytes(8, 8, 3))
    (tmp_path / "sub" / "moved.png").write_bytes(png_bytes(8, 8, 3))
    import hashlib

    expected = hashlib.sha256(png_bytes(8, 8, 3)).hexdigest()[:16]
    assert (
        files.image_id(tmp_path / "a.png")
        == expected
        == files.image_id(tmp_path / "sub" / "moved.png")
    )


@pytest.mark.parametrize(
    "rel", ["../x", "a/../../x", "/etc/passwd", "", "a\x00b", "a\\b", "..", "a/./b/../../.."]
)
def test_safe_join_rejects_traversal_and_absolute_paths(tmp_path: Path, rel: str) -> None:
    with pytest.raises(FormatError, match="unsafe"):
        files.safe_join(tmp_path, rel)


def test_safe_join_accepts_nested_relative_paths(tmp_path: Path) -> None:
    assert files.safe_join(tmp_path, "a/b.png") == tmp_path.resolve() / "a" / "b.png"


def test_safe_join_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(FormatError, match="unsafe"):
        files.safe_join(root, "link/secret.png")


def test_read_bounded_enforces_the_limit(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"x" * 10)
    assert files.read_bounded(p, 10) == b"x" * 10
    with pytest.raises(FormatError, match="larger than"):
        files.read_bounded(p, 9)
    with pytest.raises(FormatError, match="cannot read"):
        files.read_bounded(tmp_path / "missing", 10)


def test_unique_name_adds_a_suffix_on_collision() -> None:
    used: set[str] = set()
    assert files.unique_name("a.png", used, "1234abcd") == "a.png"
    assert files.unique_name("a.png", used, "1234abcd") == "a_1234abcd.png"
    assert used == {"a.png", "a_1234abcd.png"}


def test_copy_image_copies_bytes(tmp_path: Path) -> None:
    src = tmp_path / "s.png"
    src.write_bytes(png_bytes(4, 4, 1))
    files.copy_file(src, tmp_path / "out" / "deep" / "d.png")
    assert (tmp_path / "out" / "deep" / "d.png").read_bytes() == src.read_bytes()
    with pytest.raises(FormatError, match="missing"):
        files.copy_file(tmp_path / "nope.png", tmp_path / "o.png")


# ------------------------------------------------ one descriptor for the identity of an image
def test_identify_returns_header_and_id_from_one_open(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from fixtures import synth

    path = tmp_path / "a.png"
    synth.write_png(path, 40, 30, 7)
    opened: list[Path] = []
    real = files.open_regular

    def spy(p: Path, *, nofollow: bool = False):  # type: ignore[no-untyped-def]
        opened.append(p)
        assert nofollow is True
        return real(p, nofollow=nofollow)

    monkeypatch.setattr(files, "open_regular", spy)
    header, digest = files.identify(path, nofollow=True)
    assert (header.width, header.height) == (40, 30) and len(opened) == 1
    monkeypatch.undo()
    assert digest == files.image_id(path) and header == files.image_header(path)


def test_identify_refuses_a_symlink_and_a_special_file(tmp_path: Path) -> None:
    import os

    from fixtures import synth

    real = tmp_path / "a.png"
    synth.write_png(real, 8, 8, 1)
    link = tmp_path / "link.png"
    link.symlink_to(real)
    with pytest.raises(FormatError):
        files.identify(link, nofollow=True)
    assert files.identify(link, nofollow=False)[1] == files.image_id(real)
    fifo = tmp_path / "f.png"
    os.mkfifo(fifo)
    with pytest.raises(FormatError):
        files.identify(fifo, nofollow=True)


def test_make_image_opens_the_resolved_file_without_following_a_late_symlink(
    tmp_path: Path,
) -> None:
    from fixtures import synth

    from datacopilot.io.common import make_image

    real = tmp_path / "a.png"
    synth.write_png(real, 8, 8, 1)
    link = tmp_path / "swapped.png"
    link.symlink_to(real)
    with pytest.raises(FormatError):
        make_image(
            link, split="unassigned", source={}, ref="x"
        )  # a link swapped in after safe_join
    image = make_image(real, split="unassigned", source={}, ref="x")
    assert (image.width, image.height) == (8, 8)
