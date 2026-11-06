"""YOLO-OBB reader and writer: the Ultralytics oriented-box label format (ADR-0011).

Layout as ``yolo`` (``data.yaml``, ``images/[split/]``, ``labels/[split/]<name>.txt``); each label
line is ``class x1 y1 x2 y2 x3 y3 x4 y4 [conf]``, the four corners normalised to 0..1.

Read: each line becomes an ``obb`` annotation whose corners keep their order. A corner up to
``CLIP_TOLERANCE_PX`` (0.001 pixel) outside the image is snapped onto the edge; further out, or
corners that do not form a convex quadrilateral (crossed, repeated or collinear), is a
``FormatError`` naming the file and line. Corners are rounded to the 7-decimal grid the writer
prints, so ingest, export, ingest is a fixed point. A five-field (axis-aligned) line is refused
with a pointer to the ``yolo`` format.

Write: an ``obb`` row is written as its corners; a ``bbox`` row as the four corners of its box,
clockwise on screen from the top-left (no information is lost; read back it is an ``obb`` row).
A corner that moves by more than ``PRECISION_TOLERANCE_PX`` is counted in
``annotations.obb_xyxyxyxy_abs``.
"""

from __future__ import annotations

from collections import Counter

from datacopilot.core import geometry
from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Annotation, Image, Kind, PluginSpec
from datacopilot.io.yolo import (
    _CANONICAL_ROUNDS,
    CLIP_TOLERANCE_PX,
    PRECISION_TOLERANCE_PX,
    YoloFormat,
    _fmt,
)

NAME = "yolo_obb"


def _snap(value: float, size: int, where: str) -> float:
    if value < -CLIP_TOLERANCE_PX or value > size + CLIP_TOLERANCE_PX:
        raise FormatError(f"{where}: a corner lies outside the image")
    return min(max(value, 0.0), float(size))


def _canonical(corners: list[float], image: Image, where: str) -> geometry.Quad:
    """Corners on the 7-decimal grid the writer prints, snapped into the image.

    Each coordinate is rounded independently, and 0 and the image size are on the grid, so one
    round normally settles it; the loop guards the rare value whose print and read differ again.
    """
    sizes = [image.width, image.height] * 4
    out = [_snap(v * s, s, where) for v, s in zip(corners, sizes, strict=True)]
    for _ in range(_CANONICAL_ROUNDS):
        back = [_snap(float(_fmt(v / s)) * s, s, where) for v, s in zip(out, sizes, strict=True)]
        if back == out:
            break
        out = back
    return tuple(out)  # type: ignore[return-value]


class YoloObbFormat(YoloFormat):
    spec = PluginSpec(
        name=NAME, kind="format", license="Apache-2.0", extra=None, gpu_memory_gb=None, version="1"
    )

    @staticmethod
    def _check_fields(tokens: list[str], where: str) -> None:
        if len(tokens) in (5, 6):
            raise FormatError(
                f"{where}: an axis-aligned line (class cx cy w h); read it with the yolo format"
            )
        if len(tokens) not in (9, 10):
            raise FormatError(f"{where}: expected 'class x1 y1 x2 y2 x3 y3 x4 y4 [conf]'")

    @staticmethod
    def _shape(
        values: list[float], image: Image, where: str
    ) -> tuple[Kind, dict[str, object], list[float]]:
        corners = _canonical(values[:8], image, where)
        try:
            geometry.check_quad(corners)
        except ValueError:
            raise FormatError(
                f"{where}: the corners do not form a convex quadrilateral (crossed, repeated or "
                "collinear corners)"
            ) from None
        return "obb", {"obb_xyxyxyxy_abs": corners}, values[8:]

    @staticmethod
    def _geometry(ann: Annotation, image: Image, dropped: Counter[str]) -> list[str]:
        """The four corners, normalised; a ``bbox`` row's corners are those of its box."""
        assert ann.bbox_xyxy_abs is not None  # noqa: S101  # checked by the caller
        corners = ann.obb_xyxyxyxy_abs or geometry.xyxy_to_quad(ann.bbox_xyxy_abs)
        sizes = [image.width, image.height] * 4
        text = [_fmt(v / s) for v, s in zip(corners, sizes, strict=True)]
        moved = max(abs(float(t) * s - v) for t, v, s in zip(text, corners, sizes, strict=True))
        if moved > PRECISION_TOLERANCE_PX:
            dropped["annotations.obb_xyxyxyxy_abs"] += 1
        return text


plugin = YoloObbFormat()
