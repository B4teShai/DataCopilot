"""Bounded image decoding and deterministic preprocessing (needs Pillow; a dev dependency)."""

from __future__ import annotations

import io
import struct
import time
import zlib

import numpy as np
import pytest
from fixtures.synth import png_bytes

from datacopilot.embed import imaging
from datacopilot.embed.protocol import ImageDecodeError

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def encode(mode: str, size: tuple[int, int], fmt: str, color: object = 0) -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, format=fmt)  # type: ignore[arg-type]
    return out.getvalue()


def png_claiming(width: int, height: int) -> bytes:
    """A PNG whose header declares ``width`` x ``height`` but whose data is a few bytes."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + chunk(b"IEND", b"")
    )


def test_decodes_png_jpeg_gif_and_bmp_to_rgb() -> None:
    for fmt in ("PNG", "JPEG", "GIF", "BMP"):
        arr = imaging.decode_rgb(encode("RGB", (20, 10), fmt, (10, 20, 30)), 0)
        assert arr.shape == (10, 20, 3) and arr.dtype == np.uint8, fmt


def test_grayscale_palette_and_alpha_images_become_rgb() -> None:
    assert imaging.decode_rgb(png_bytes(8, 6, 1), 0).shape == (6, 8, 3)
    assert imaging.decode_rgb(encode("P", (4, 4), "PNG"), 0).shape == (4, 4, 3)
    assert imaging.decode_rgb(encode("RGBA", (4, 4), "PNG", (1, 2, 3, 4)), 0).shape == (4, 4, 3)


def test_exif_orientation_is_applied() -> None:
    img = Image.new("RGB", (20, 10), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees: the stored 20x10 displays as 10x20
    out = io.BytesIO()
    img.save(out, format="JPEG", exif=exif)
    assert imaging.decode_rgb(out.getvalue(), 0).shape == (20, 10, 3)


def test_a_declared_size_over_the_pixel_limit_is_refused_before_decoding() -> None:
    data = png_claiming(40_000, 40_000)  # 1.6 billion pixels declared, 100 bytes of data
    start = time.perf_counter()
    with pytest.raises(ImageDecodeError) as err:
        imaging.decode_rgb(data, 3)
    assert err.value.reason == "too_many_pixels" and err.value.index == 3
    assert time.perf_counter() - start < 1.0


def test_the_limit_is_exact() -> None:
    data = encode("RGB", (30, 20), "PNG")
    assert imaging.decode_rgb(data, 0, max_pixels=600).shape == (20, 30, 3)
    with pytest.raises(ImageDecodeError) as err:
        imaging.decode_rgb(data, 0, max_pixels=599)
    assert err.value.reason == "too_many_pixels"


def test_the_global_pillow_limit_is_restored_and_the_decoder_never_warns(
    recwarn: pytest.WarningsRecorder,
) -> None:
    before = Image.MAX_IMAGE_PIXELS
    imaging.decode_rgb(encode("RGB", (8, 8), "PNG"), 0)
    with pytest.raises(ImageDecodeError):
        imaging.decode_rgb(png_claiming(5_000, 5_000), 0, max_pixels=1_000_000)
    assert Image.MAX_IMAGE_PIXELS == before
    assert not [w for w in recwarn if "DecompressionBomb" in str(w.category)]


def test_a_size_between_one_and_two_times_the_limit_is_an_error_not_a_warning() -> None:
    data = encode("RGB", (30, 20), "PNG")  # 600 px
    with pytest.raises(ImageDecodeError) as err:
        imaging.decode_rgb(data, 0, max_pixels=400)  # 1.5x
    assert err.value.reason == "too_many_pixels"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00" * 64,
        b"not an image at all",
        encode("RGB", (50, 50), "PNG")[:60],  # truncated inside the data
        encode("RGB", (50, 50), "JPEG")[:80],
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
    ],
)
def test_corrupt_input_is_a_decode_error_with_a_constant_reason(data: bytes) -> None:
    with pytest.raises(ImageDecodeError) as err:
        imaging.decode_rgb(data, 7)
    assert err.value.reason in {"decode_failed", "too_many_pixels"} and err.value.index == 7
    assert "Traceback" not in str(err.value)


@pytest.mark.parametrize(
    "data",
    [
        b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n",  # Pillow would call Ghostscript
        b"<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/>",
        b"II*\x00\x08\x00\x00\x00",  # TIFF: not an accepted dataset format
    ],
)
def test_formats_outside_the_dataset_set_are_never_handed_to_their_decoders(data: bytes) -> None:
    with pytest.raises(ImageDecodeError) as err:
        imaging.decode_rgb(data, 0)
    assert err.value.reason == "decode_failed"


def test_a_gif_with_many_frames_decodes_only_the_first() -> None:
    frames = [Image.new("RGB", (8, 8), (i, i, i)) for i in range(50)]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:])
    assert imaging.decode_rgb(out.getvalue(), 0).shape == (8, 8, 3)


# ------------------------------------------------------------ preprocessing
def test_dinov2_preprocessing_resizes_the_short_side_then_centre_crops() -> None:
    pre = imaging.Preprocess(
        size=224,
        resize_short=256,
        resample="bicubic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    arr = np.full((300, 600, 3), 128, dtype=np.uint8)
    out = imaging.preprocess(arr, pre)
    assert out.shape == (3, 224, 224) and out.dtype == np.float32
    expected = (128 / 255 - np.array(pre.mean, dtype=np.float32)) / np.array(
        pre.std, dtype=np.float32
    )
    np.testing.assert_allclose(out[:, 100, 100], expected, atol=1e-4)


def test_siglip_preprocessing_resizes_to_the_square_without_cropping() -> None:
    pre = imaging.Preprocess(
        size=224, resize_short=None, resample="bilinear", mean=(0.5,) * 3, std=(0.5,) * 3
    )
    arr = np.zeros((100, 300, 3), dtype=np.uint8)
    arr[:, 150:] = 255  # right half white: a crop would lose the edge, a squash keeps both halves
    out = imaging.preprocess(arr, pre)
    assert out.shape == (3, 224, 224)
    assert out[0, 112, 5] < -0.95 and out[0, 112, 218] > 0.95


def test_preprocessing_is_deterministic_and_does_not_mutate_its_input() -> None:
    pre = imaging.Preprocess(
        size=32, resize_short=40, resample="bicubic", mean=(0.0,) * 3, std=(1.0,) * 3
    )
    arr = np.random.default_rng(0).integers(0, 256, (50, 70, 3), dtype=np.uint8)
    copy = arr.copy()
    a, b = imaging.preprocess(arr, pre), imaging.preprocess(arr, pre)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(arr, copy)


def test_tiny_and_extreme_aspect_images_still_produce_the_target_size() -> None:
    pre = imaging.Preprocess(
        size=32, resize_short=40, resample="bicubic", mean=(0.0,) * 3, std=(1.0,) * 3
    )
    for shape in ((1, 1, 3), (1, 5000, 3), (5000, 1, 3)):
        assert imaging.preprocess(np.zeros(shape, dtype=np.uint8), pre).shape == (3, 32, 32)
