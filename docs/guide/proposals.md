# Propose, review, apply

Datasets are rarely complete: a box was forgotten, or a class was added late. `propose` asks a
detection model to look for your dataset's classes in every image and writes down the boxes it
thinks are missing. A person reviews them. Only accepted ones are written, as a **new** dataset
version. Nothing is applied on its own and the old version is never edited.

Only missing boxes (`add_box`) are proposed, by Grounding DINO tiny (Apache-2.0). You need the `vlm`
extra (see [Install](install.md)) and a one-time download of about 690 MB of model weights. See
[Status](../status.md) for what else exists.

## Four commands

<!-- skip: needs the [vlm] extra and a model download -->
```bash
datacopilot propose shop.dcp                                  # prints set=ID
datacopilot review export shop.dcp --proposals ID --to cvat -o review
# open review/ in CVAT; type accepted or rejected in the `decision` of each proposal
datacopilot review import shop.dcp review --proposals ID --from cvat    # prints a new set=ID2
datacopilot apply shop.dcp --proposals ID2 -o shop_v2.dcp     # prints the new version
```

`shop.dcp` is unchanged afterwards. `shop_v2.dcp` lists it as its parent in the lineage, every new box
has provenance `proposal_accepted`, and the reviewed set is copied into `shop_v2.dcp/proposals/`.

## What `propose` does

For every image the model is asked about every class name and alias of the dataset, or about
`--classes a,b`. A candidate becomes a proposal only if:

- its numbers are finite, it lies inside the image and it is at least 2 pixels wide and high;
- its label is one of the phrases that were asked about;
- its score reaches `--min-score` (default 0.35), or `--min-score-for CLASS=SCORE`;
- the image has no box of the same class there already (IoU 0.5), and no better candidate of the
  same class overlaps it that much.

Everything dropped is counted by reason. Class names become lower-case words before they reach the
model. A name with a full stop, other punctuation or non-ASCII letters is skipped and counted; give
such a class an ASCII alias. When the model names two classes at once, the candidate is counted and
dropped, not guessed, so a few real detections are lost.

The result is a **proposal set** in `shop.dcp/proposals/`, named by a fingerprint of its content and
bound to the dataset version and the model that made it. An edited file stops matching its name. The
name proves the file was not changed, not who wrote it: keep the folder as private as your labels.
The model's license is checked before the dataset is read.

A run that did not reach every image (time budget, unreadable image) exits 1. Its partial set is kept
and marked incomplete, and `apply` refuses it unless you pass `--allow-incomplete`.

## Review in CVAT

`review export` writes your own boxes and, as a separate layer, the proposals (class name plus
`__proposed`), with the images, as CVAT for images 1.1. Each proposal has the attributes
`proposal_id`, `score` and `decision`. The reviewer types `accepted` or `rejected` in `decision`.
Moving a box first makes it an edited proposal. Deleting a box decides nothing.

`review import` reads the decisions and stores a **new** set that names the reviewed one as its
parent. Your own boxes in the file are ignored and counted: a review never edits a label. The whole
file is refused if it:

- names an unknown, repeated or foreign proposal;
- changes a class or uses another decision word;
- moves an accepted box outside its image, or holds a rotated box;
- lists an image missing from the folder;
- decides nothing.

## What `apply` refuses

`apply` writes the accepted proposals into a new dataset and nothing else. It refuses, writing
nothing:

- a set made for another version of the dataset, or edited after it was written;
- a set that did not come through `review import`, or is not a review of the set it names as parent;
- a set from the test double `test_boxes` (unless `--allow-test-proposer`);
- a model that is not the pinned revision of the registered model, or whose license is no longer
  allowed;
- an incomplete run (without `--allow-incomplete`) or a set with nothing accepted;
- an output that exists or lies inside the source dataset.

## From Python

<!-- skip: needs the [vlm] extra and a model download -->
```python
import datacopilot as dc

ps = dc.propose("shop.dcp", "grounding_dino", min_score=0.4)  # stored in shop.dcp/proposals/
print(len(ps.proposals), ps.incomplete)
manifest = dc.apply("shop.dcp", reviewed_set_id, "shop_v2.dcp")  # after the review above
```

## How good is it?

Weak on precision. On COCO128 (929 boxes) with 178 boxes deleted at random as an answer key and the
default threshold, the model proposed 215 boxes. 84 matched a deleted box (precision 0.39, recall
0.47). More than half of the proposals were not deleted boxes: some are real objects COCO never
labelled, some have the wrong class, some only partly fit.

The test is friendlier than real life. The reviewer knew the answer key, and randomly deleted boxes
are easier than the small and crowded objects real datasets miss. Treat the recall as optimistic.
Raising the threshold trades recall for precision. The sweep, the figures and the method are in the
[changelog](https://github.com/B4teShai/DataCopilot/blob/main/CHANGELOG.md), and what was left out
on purpose is in [ADR-0008](../adr/0008-propose-review-apply.md).
