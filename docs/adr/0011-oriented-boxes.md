# ADR-0011: Oriented bounding boxes

**Status:** accepted
**Date:** 2026-11-06

## Context

The first consumer that trains on rotated objects uses Ultralytics `yolo obb`: one label line per object,
`class x1 y1 x2 y2 x3 y3 x4 y4`, four corners normalised to 0..1, with the usual `data.yaml`. Its
products stand at any angle on a shelf, so an axis-aligned box around them is mostly background and two
neighbours' boxes overlap even when the objects do not. The data model had only `bbox` (axis-aligned) and
`classification`, and the YOLO reader refused any line longer than six fields.

Constraints that apply:

- Conversions are lossless or count what they lose ([ADR-0006](0006-dataset-format.md)); the YOLO-OBB text
  must round-trip, corner order included.
- Existing dataset versions must not change: a dataset without oriented boxes keeps its content hash, and
  every `.dcp` written so far still reads. CV-Pilot records the version.
- The core dependency allow-list ([ADR-0003](0003-repo-and-tooling.md)) has no geometry library; the domain
  layer may use NumPy only ([ADR-0004](0004-layering-and-dependency-rule.md)).
- Ultralytics turns every quadrilateral into its minimum-area rectangle (`xywhr`) when it trains, so a label
  that is not a rectangle is silently changed by training.

## Options considered

### Representation

- **A: centre, size and angle (`xywhr`).** Compact and what Ultralytics trains on; but a quadrilateral that is
  not a rectangle, and the original corner order, cannot be written back, so YOLO-OBB would not round-trip.
- **B: the four corners in absolute pixels, in their original order, plus the enclosing axis-aligned box.**
  Lossless for YOLO-OBB; every piece of code that only needs a rectangle in the image (crops, the image-size
  invariant, other formats) keeps using `bbox_xyxy_abs`.
- **C: a general polygon kind.** Covers segmentation later, but every check would need general polygons now.

### Task or kind

- **A: a new task `obb`**, as Ultralytics has. Every place that switches on `task == "detection"` (checks,
  writers, propose, split) would need a second branch, and other formats would refuse the dataset outright
  instead of keeping the enclosing box and counting the loss.
- **B: a new annotation kind `obb` inside the `detection` task.** Writers of other formats keep working on the
  enclosing box; the format, not the task, says "this is YOLO-OBB".

### Store and version

- **A: add a nullable column and keep `schema_version` 1.** Every table written from now on, even without an
  oriented box, has a column older packages reject with a vague "columns do not match".
- **B: bump `schema_version` to 2 for every dataset.** The version hash includes `dcp:<schema_version>`, so
  every existing version would change. Forbidden.
- **C: one generation per dataset: the oldest that can hold its rows.**

### Rotated IoU

- **A: shapely.** Correct and fast, but a new core dependency (GEOS underneath) outside the allow-list.
- **B: rasterise the two shapes.** Approximate, and memory grows with box size.
- **C: convex clipping (Sutherland-Hodgman) in NumPy, vectorised over box pairs.**

## Decision

Representation B, kind B, store C, IoU C.

1. **Row.** `Annotation.kind` gains `obb`. An `obb` row has `obb_xyxyxyxy_abs`, eight floats: the corners in
   pixels, in the order the source gave them. They must form a strictly convex quadrilateral (all four corners
   turn the same way; this refuses crossed "bow-tie" orders, repeated or collinear corners and zero area),
   either drawing direction. `bbox_xyxy_abs` must equal the box that encloses the corners (it is derived when
   omitted), so invariant 4 (inside the image) covers every corner. `bbox` and `classification` rows have no
   corners.
2. **One kind of box per dataset.** A detection dataset holds `bbox` rows or `obb` rows, never both
   (`validate_tables`). An axis-aligned box written as YOLO-OBB is four corners; read back it is an `obb` row.
3. **Generations.** `schema_version` 1 is the original annotations table; 2 adds the nullable column
   `obb_xyxyxyxy_abs` (list of float64) after `bbox_xyxy_abs`. A dataset is written in the oldest generation
   that holds its rows (`schema_version_for`), so a dataset without oriented boxes is written exactly as
   before. The reader picks the table schema from the manifest; a package that knows only generation 1 refuses
   a generation-2 dataset with "Upgrade datacopilot". The hash procedure leaves `obb_xyxyxyxy_abs` out of an
   annotation row while it is null, so versions of datasets without oriented boxes are unchanged
   (`tests/golden/test_version_golden.py`, `tests/golden/test_legacy_store.py`).
4. **Formats.** A new format `yolo_obb` (reader and writer, entry point like `yolo`) reads and writes
   `class x1 y1 x2 y2 x3 y3 x4 y4 [conf]` with the YOLO layout and `data.yaml`. Corners are normalised and
   written with 7 decimals; a corner up to 0.001 pixel outside the image is snapped onto the edge, further out
   is an error, and a degenerate quadrilateral is an error naming the file and line. Every other writer
   writes the enclosing box and counts `annotations.obb_xyxyxyxy_abs` as dropped, so `--strict` fails. The
   `yolo` reader names `yolo_obb` when it meets a nine-field line.
5. **Checks.** `boxes` handles both kinds. For `obb` rows: `invalid_box`, `out_of_bounds` (any corner) and
   `zero_area` (a degenerate quadrilateral) as before; `tiny_box` and `extreme_aspect` use the sides of the
   minimum-area rectangle, not the enclosing box; a new WARN rule `obb_not_rectangular` reports a box whose area
   is below `min_obb_rectangularity` (default 0.9) of that rectangle, because training replaces it by the
   rectangle; `duplicate_box` and `conflicting_class` use the rotated IoU. `integrity`, `duplicates`,
   `imbalance` and the embedding checks (`near_duplicates`, `leakage`, `outliers`) work on images and classes
   and are unchanged. `label_noise` crops the enclosing box of an oriented box (more background than the
   object, documented, not rotated).
6. **Geometry.** `core/obb.py`: polygon areas, minimum-area rectangles (one side on a polygon edge, four
   candidates) and rotated IoU, NumPy only. Pairs are clipped only where the enclosing boxes overlap and an
   upper bound on the IoU exceeds the caller's threshold; clipping runs in chunks of 65,536 pairs. Tests check
   it with property tests against an exact rational oracle that uses a different algorithm.
7. **Propose** refuses a dataset with oriented boxes: the proposers return axis-aligned boxes, and accepting one
   would mix the kinds.

## Consequences

- A YOLO-OBB dataset ingests, audits and exports back to identical label text (up to the 7-decimal grid), and
  its version is stable within generation 2.
- Every version written before stays valid and every existing `.dcp` reads; generation-1 files are written
  byte for byte as before. A generation-2 dataset needs this package or newer.
- Converting oriented boxes to COCO, YOLO, VOC, CVAT XML or Label Studio keeps only the enclosing box, counted.
- Forbidden: mixing `bbox` and `obb` rows in one dataset; non-convex or self-intersecting corner orders.
- Follow-up: read and write rotated boxes in CVAT XML (`rotation`) and Label Studio (`rotation`), which are
  still skipped and counted; rotated box crops for `label_noise`; proposers that return oriented boxes;
  general polygons.

## Links

- [ADR-0006](0006-dataset-format.md), [ADR-0010](0010-split-and-box-crops.md)
- [../architecture/03-data-model.md](../architecture/03-data-model.md), [../guide/formats.md](../guide/formats.md)
