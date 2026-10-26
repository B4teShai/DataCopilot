"""Bounded image decoding and deterministic preprocessing for the model backends.

Pillow is imported inside the functions (it is an ``[embed]`` dependency). Decoding is defended
against hostile images:

- only the formats a dataset can contain (JPEG, PNG, GIF, BMP) are handed to a decoder, so
  Pillow's other plugins (EPS, which can start Ghostscript, and the rest) never see the bytes;
- the declared pixel count is compared with ``max_pixels`` before any pixel is decoded, and
  Pillow's own decompression-bomb limit is set to the same value for the duration of the call
  (restored afterwards; the module global is process-wide, so decoding is not thread-safe);
- truncated files are errors, only the first frame of an animation is decoded, and the failure
  reason is a constant code, never text from the file.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from datacopilot.core.errors import ModelError
from datacopilot.embed.protocol import ImageDecodeError

MAX_DECODE_PIXELS = 64_000_000
"""Largest image (width x height) that is decoded; 8,000 x 8,000. RGB decoding of such an image
needs about 200 MB, which is the per-image memory bound."""
MAX_RESIZED_LONG_SIDE = 4096
"""Longest side after the aspect-preserving resize; a 1 x 60,000,000 strip must not become a
multi-gigabyte array (only matters beyond 16:1 aspect ratios)."""
_FORMATS = ("JPEG", "PNG", "GIF", "BMP")
_EXIF_ORIENTATION = 0x0112


_INSTALL = 'install the extra "datacopilot[embed]" (in a clone: uv sync --extra embed)'
_BACKGROUND = (255, 255, 255)
"""Colour under transparent pixels: alpha is composited on white, so what an image hides under
alpha 0 can neither change nor carry an embedding."""
_MODES = frozenset(
    {
        "1",
        "L",
        "P",
        "RGB",
        "RGBA",
        "LA",
        "PA",
        "RGBa",
        "La",
        "CMYK",
        "YCbCr",
        "I",
        "I;16",
        "I;16L",
        "I;16B",
        "F",
    }
)
_SIXTEEN_BIT = frozenset({"I", "I;16", "I;16L", "I;16B"})
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})


def _pil() -> Any:
    """The ``PIL`` package, or a ``ModelError`` that names the extra (never a bare ImportError)."""
    try:
        import PIL.Image
        import PIL.ImageOps
    except ImportError:
        raise ModelError(f"image decoding needs Pillow: {_INSTALL}") from None
    return PIL


def _to_rgb(img: Any, index: int) -> NDArray[np.uint8]:
    """Pixels of a loaded image as ``(h, w, 3)`` uint8 without silent clipping or lost alpha.

    16-bit integer images (Pillow's ``I;16`` and ``I``) are divided by 257 so 65535 maps to 255
    instead of saturating; float images are taken as 0 to 255 and clipped; alpha (including palette
    and colour-key transparency) is composited on a fixed white background.
    """
    from PIL import Image

    mode = img.mode
    if mode not in _MODES:
        raise ImageDecodeError(index, "decode_failed")
    if mode in _SIXTEEN_BIT or mode == "F":
        raw = np.asarray(img, dtype=np.float64)
        gray = np.clip(raw, 0, 65535) / 257.0 if mode in _SIXTEEN_BIT else np.clip(raw, 0, 255)
        plane = np.rint(gray).astype(np.uint8)
        return np.stack([plane, plane, plane], axis=-1)
    if mode in _ALPHA_MODES or "transparency" in img.info:
        rgba = img.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (*_BACKGROUND, 255))
        return np.array(Image.alpha_composite(base, rgba).convert("RGB"), dtype=np.uint8)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def decode_rgb(
    data: bytes, index: int, *, max_pixels: int = MAX_DECODE_PIXELS
) -> NDArray[np.uint8]:
    """Decode ``data`` to an ``(h, w, 3)`` uint8 array, EXIF orientation applied.

    Only known pixel modes are accepted (see ``_to_rgb`` for the scaling and compositing rules) and
    only single-frame images: an animation has no single picture to embed, so it is refused
    (``multi_frame``) and reported as not embedded.

    Raises:
        ImageDecodeError: ``too_many_pixels`` (declared size over ``max_pixels``), ``multi_frame``
            or ``decode_failed`` (anything else); ``index`` is echoed back for the caller.
        ModelError: Pillow is not installed (the message names the extra).
    """
    _pil()
    from PIL import Image, ImageOps

    if not data:
        raise ImageDecodeError(index, "decode_failed")
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=_FORMATS) as img:
                width, height = img.size
                if width * height > max_pixels:
                    raise ImageDecodeError(index, "too_many_pixels")
                if getattr(img, "n_frames", 1) > 1:
                    raise ImageDecodeError(index, "multi_frame")
                img.load()
                return _to_rgb(ImageOps.exif_transpose(img), index)
    except ImageDecodeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageDecodeError(index, "too_many_pixels") from None
    except Exception:  # noqa: BLE001  # untrusted parser: any failure is a bad image
        raise ImageDecodeError(index, "decode_failed") from None
    finally:
        Image.MAX_IMAGE_PIXELS = previous


@dataclass(frozen=True)
class Preprocess:
    """Deterministic preprocessing of one model family.

    Attributes:
        size: side of the square model input.
        resize_short: resize the short side to this many pixels (aspect preserved), then centre
            crop to ``size``; None resizes straight to ``size`` x ``size``.
        resample: ``bicubic`` or ``bilinear`` (Pillow's, so results do not depend on a model
            library version).
        mean: per-channel mean of the normalisation.
        std: per-channel standard deviation of the normalisation.
    """

    size: int
    resize_short: int | None
    resample: Literal["bicubic", "bilinear"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def preprocess(rgb: NDArray[np.uint8], pre: Preprocess) -> NDArray[np.float32]:
    """``(h, w, 3)`` uint8 to a ``(3, size, size)`` float32 array; the input is not modified."""
    _pil()
    from PIL import Image

    method = Image.Resampling.BICUBIC if pre.resample == "bicubic" else Image.Resampling.BILINEAR
    image = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    width, height = image.size
    if pre.resize_short is None:
        image = image.resize((pre.size, pre.size), method)
    else:
        short = min(width, height)
        scale = pre.resize_short / short
        new_w = min(max(int(width * scale), 1), MAX_RESIZED_LONG_SIDE)
        new_h = min(max(int(height * scale), 1), MAX_RESIZED_LONG_SIDE)
        new_w, new_h = max(new_w, pre.size), max(new_h, pre.size)
        image = image.resize((new_w, new_h), method)
        left, top = (new_w - pre.size) // 2, (new_h - pre.size) // 2
        image = image.crop((left, top, left + pre.size, top + pre.size))
    array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(pre.mean, dtype=np.float32)
    std = np.asarray(pre.std, dtype=np.float32)
    normalised: NDArray[np.float32] = ((array - mean) / std).transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(normalised)


def dhash(data: bytes, index: int, *, max_pixels: int = MAX_DECODE_PIXELS) -> int:
    """64-bit difference hash of an image: luma resized to 9 x 8, one bit per horizontal step.

    Decoding is as bounded as :func:`decode_rgb`, whose errors it raises. Two pictures that differ
    by recompression, small brightness changes or mild resizing differ in few bits.
    """
    _pil()
    from PIL import Image

    rgb = decode_rgb(data, index, max_pixels=max_pixels)
    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = Image.fromarray(np.clip(luma + 0.5, 0, 255).astype(np.uint8), "L")
    small = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits of two hashes."""
    return (a ^ b).bit_count()
