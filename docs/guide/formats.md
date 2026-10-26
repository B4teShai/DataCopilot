# Formats

`datacopilot formats` lists the registered formats. Each can read and write; a detection format only
holds detection datasets and ImageFolder only classification (Label Studio holds either).

| Format | Layout read | Layout written | Notes |
|---|---|---|---|
| `coco` | `annotations.json`, `annotations/*.json` (split from a `train`, `val` or `test` word in the file name) or one file | `annotations.json` or `annotations/instances_<split>.json`, `images/` | boxes `xywh`; keeps `iscrowd`, `score`, `group_id` and every other key in `source` |
| `yolo` | `data.yaml`, `images/<split>/`, `labels/<split>/` | the same | normalised boxes written with 7 decimals; a box that moves by more than 0.001 pixel is counted |
| `voc` | `Annotations/*.xml`, `JPEGImages/`, `ImageSets/Main/<split>.txt`, optional `labelmap.txt` | the same | `difficult`, `truncated`, `occluded` and other elements are kept |
| `cvat_xml` | `annotations.xml`, `annotations/*.xml` (split from `<subset>` or the file name) or one file | `annotations.xml` or `annotations/<split>.xml`, `images/` | CVAT for images 1.1 |
| `label_studio` | `annotations.json`, `annotations/*.json` or one file (the "JSON" export) | `annotations.json` or `annotations/<split>.json`, `images/` | detection and classification |
| `imagefolder` | `<class>/<image>` or `<split>/<class>/<image>` | the same | classification; class ids follow the sorted class names |

## CVAT for images 1.1 XML

Read: images, boxes (`xtl ytl xbr ybr`), labels with colour and type, attributes, `occluded`,
`source` (`manual` is human, `auto` is model) and the task subset as the split. Image sizes come from
the file header; a different declared size is counted.

Counted, never dropped silently: polygons, polylines, points, ellipses, masks, skeletons, cuboids,
tags, rotated boxes and every `<track>` (`skipped`, `annotations.unsupported.<shape>`), label
attribute definitions that carry more than a name (`categories.attributes`). Export metadata such as
ids, dates and owner describes the export, not the data, and is neither kept nor counted. Box
coordinates must be finite and inside the image, sizes positive whole numbers, and image names stay
inside the dataset directory.

## Label Studio JSON

Read: `rectanglelabels` results (percent of the original size, converted to pixels of the real
image and rounded to 1e-6 pixel) and `choices` results (classification). `annotations` become human
rows and the first `predictions` entry becomes model rows with their `score`. The dataset is a
detection dataset if any box was read, else classification. Classes are the label names that occur;
the labelling configuration is never read.

Multiple annotators are never merged: one annotation per task is read (the first with
`ground_truth`, else the lowest `id`); cancelled annotations and every other annotation or
prediction are counted (`annotations.cancelled`, `annotations.of_other_annotators`,
`annotations.of_other_predictions`). Polygons, brushes, keypoints, text areas, relations,
multi-label and rotated rectangles and rotated images are counted as unsupported. Images that are
not local files (`http`, `s3`, `data:`) are counted as `images.remote_uri` and never fetched. Predictions
of a task that has no kept annotation are not read as rows (`annotations.prediction_of_unlabeled_task`),
and the task type follows the human rows (boxes, else choices) before the model rows.

Personal and free-form data: only an allow-list of fields is read, stored, reported and re-emitted
(task `created_at`, `updated_at`, `inner_id`; annotation `id`, `ground_truth`, `was_cancelled`,
`lead_time`, `created_at`, `updated_at`; prediction `id`, `model_version`, `score`, `created_at`,
`updated_at`; result `id`, `from_name`, `to_name`, `origin`). Every other key (`completed_by`,
`created_username`, `updated_by`, `comment_authors`, `meta`, `file_upload` and so on) is dropped and
counted by name, and task `data` fields other than `image` are dropped and counted too unless you pass
`--keep-task-data` (they are then stored and re-emitted as they are). The allow-list cannot know that a
label name, an image file name or an allowed text value contains personal data.

A same-format round trip reproduces the original file layout (one document per original file under
`annotations/`, the form of the image reference, empty predictions) and is field-identical for the kept
fields. A category that no row uses cannot be carried (Label Studio has no list of classes): the writer
counts it as `categories.unused`. The reference `images/<name>` is a file path, not a URL Label Studio can
serve; use the `/data/local-files/?d=...` form in the source if you need that.

## What each conversion loses

The [loss matrix](../architecture/03-data-model.md#format-mapping-and-loss-matrix) lists, for every
pair of formats, the columns that cannot be carried over and how many values were lost on the
synthetic fixtures; `tests/golden/test_roundtrip_matrix.py` asserts every cell. Converting to the
same format loses nothing and reads back field-identical with an equal `manifest.version`: ingest,
export, ingest gives the version of the first ingest (a fixed point after one pass, tested for every
format on a fixture with boxes on all four edges, tiny boxes and more than ten boxes on one image, and
on COCO128). YOLO text keeps 7 decimals of the normalised box, so the YOLO reader snaps every box to
what that text can hold (a box that overshoots the image edge by less than 0.001 pixel is first snapped
onto the edge, then rounded to the 7-decimal grid, a change of at most 0.00004 pixel); this is a documented
rounding, not a loss, and it is applied at ingest so that the version is stable. In
practice: COCO ids, areas and segmentations do not survive YOLO or VOC, CVAT `z_order` does not
survive COCO, YOLO, VOC or Label Studio, Label Studio result ids and annotation metadata do not
survive the others, and classes are re-numbered where a format has no category ids. Use `--strict`
to make any of these a failure.

## Input checks

A picture in a format the header reader cannot identify (`.webp`, `.tif`, `.tiff`) is **refused** by
every reader with one error naming the file (exit 1); it is never skipped silently, so a dataset cannot
lose photos that the audit would then never see. Convert such files to JPEG or PNG first. A file that
looks like a supported picture but has a bad header is refused the same way, and one that is cut off
after its header (a JPEG without `FFD9`, a PNG without `IEND`, a GIF without its trailer) is a
`truncated_file` warning of the `integrity` check at audit time (one 4 KiB read from the end of each file).

Input is validated: paths cannot leave the dataset directory, XML with a DOCTYPE is rejected, boxes
must lie inside their image and identical image content is reported by file name. `ingest --group-by
"re:^images/(?P<g>[^/_]+)"` assigns a group (for example a camera or a session) from the file path,
so that leakage findings can tell which images belong together; the pattern must pass the safety rules
described in [Split](split.md#groups) (bounded, no overlapping quantifiers), or `ingest` refuses it. ImageFolder is classification only.
The exit codes are in [Settings](../reference/settings.md#exit-codes).
