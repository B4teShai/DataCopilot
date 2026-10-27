# Formats

`datacopilot formats` lists the six formats. Each can read and write. A detection format holds
detection datasets only, ImageFolder holds classification only, and Label Studio holds either.

| Format | Layout read | Layout written | Notes |
|---|---|---|---|
| `coco` | `annotations.json`, `annotations/*.json` (split from `train`, `val` or `test` in the file name) or one file | `annotations.json` or `annotations/instances_<split>.json`, plus `images/` | Boxes are `xywh`. Keeps `iscrowd`, `score`, `group_id` and every other key in `source`. |
| `yolo` | `data.yaml`, `images/<split>/`, `labels/<split>/` | the same | Boxes are normalised, written with 7 decimals. |
| `voc` | `Annotations/*.xml`, `JPEGImages/`, `ImageSets/Main/<split>.txt`, optional `labelmap.txt` | the same | Keeps `difficult`, `truncated`, `occluded` and other elements. |
| `cvat_xml` | `annotations.xml`, `annotations/*.xml` (split from `<subset>` or the file name) or one file | `annotations.xml` or `annotations/<split>.xml`, plus `images/` | CVAT for images 1.1. |
| `label_studio` | `annotations.json`, `annotations/*.json` or one file (the "JSON" export) | `annotations.json` or `annotations/<split>.json`, plus `images/` | Detection and classification. |
| `imagefolder` | `<class>/<image>` or `<split>/<class>/<image>` | the same | Classification; class ids follow the sorted class names. |

## CVAT for images 1.1

Read: images, boxes, labels, attributes, `occluded`, the box source (`manual` is human, `auto` is
model) and the task subset as the split.

Counted, never dropped silently: polygons, polylines, points, ellipses, masks, skeletons, cuboids,
tags, rotated boxes and every `<track>`. Export metadata such as ids, dates and owner is neither
kept nor counted.

## Label Studio JSON

Read: `rectanglelabels` results (percent of the image, converted to pixels) and `choices` results
(classification). `annotations` become human rows; the first `predictions` entry becomes model rows
with their `score`. The labeling configuration is never read.

- One annotation per task is read (the first with `ground_truth`, else the lowest `id`). Cancelled
  annotations and the rest are counted, not merged.
- Polygons, brushes, keypoints, text areas, relations, multi-label and rotated shapes are counted
  as unsupported. Images that are not local files (`http`, `s3`, `data:`) are counted and never
  fetched.
- Only an allow-list of fields is kept. Every other key (`completed_by`, usernames, comments,
  `meta`) is dropped and counted by name. Task `data` fields other than `image` are dropped unless
  you pass `--keep-task-data`. The allow-list cannot know that a label name or file name holds
  personal data.
- A same-format round trip reproduces the original layout and keeps every kept field.

## What each conversion loses

The [loss matrix](../architecture/03-data-model.md#format-mapping-and-loss-matrix) lists, for every
pair of formats, which values cannot be carried. Converting to the same format loses nothing:
ingest, export, ingest gives the first version again. In practice COCO ids, areas and segmentations
do not survive YOLO or VOC, and Label Studio result ids do not survive the others. Use `--strict`
to make any loss a failure.

YOLO text keeps 7 decimals, so the YOLO reader rounds each box to that grid (at most 0.00004
pixel). This is a documented rounding, not a loss, and it keeps the version stable.

## Input checks

- A picture the header reader cannot identify (`.webp`, `.tif`, `.tiff`) is refused with one error
  naming the file (exit 1), never skipped. Convert such files to JPEG or PNG first.
- A file cut off after its header becomes a `truncated_file` warning at audit time.
- Paths cannot leave the dataset folder, XML with a DOCTYPE is rejected, and boxes must lie inside
  their image.
- `ingest --group-by "re:^images/(?P<g>[^/_]+)"` assigns a `group_id` from the file path. The
  pattern must pass the safety rules in [Split](split.md#groups) or `ingest` refuses it.

Exit codes are in [Settings and exit codes](../reference/settings.md#exit-codes).
