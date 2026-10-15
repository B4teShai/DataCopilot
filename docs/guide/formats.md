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
not local files (`http`, `s3`, `data:`) are counted as `images.remote_uri` and never fetched. Personal
fields of `completed_by` are never stored.

## What each conversion loses

The [loss matrix](../architecture/03-data-model.md#format-mapping-and-loss-matrix) lists, for every
pair of formats, the columns that cannot be carried over and how many values were lost on the
synthetic fixtures; `tests/golden/test_roundtrip_matrix.py` asserts every cell. Converting to the
same format loses nothing and reads back field-identical with an equal `manifest.version`. In
practice: COCO ids, areas and segmentations do not survive YOLO or VOC, CVAT `z_order` does not
survive COCO, YOLO, VOC or Label Studio, Label Studio result ids and annotation metadata do not
survive the others, and classes are re-numbered where a format has no category ids. Use `--strict`
to make any of these a failure.
