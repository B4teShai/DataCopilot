# Propose, review, apply

Datasets are almost never complete: a box was forgotten, or a class was added late. `propose` asks a
detection model to look for the classes of your dataset in every image and writes down the boxes it
thinks are missing. A person reviews them, and only the accepted ones are written, as a **new** version
of the dataset. Nothing is ever applied on its own, and the old version is never edited.

This page describes what is built: boxes that are missing (`add_box`), found by Grounding DINO (tiny,
Apache-2.0), reviewed through CVAT files. It needs the `vlm` extra and downloads about 690 MB of model
weights once (see [Install](install.md)).

## Six commands

<!-- skip: needs the [vlm] extra and a model download -->
```bash
datacopilot propose shop.dcp                                  # prints set=ID
datacopilot review export shop.dcp --proposals ID --to cvat -o review
# open review/ in CVAT; type accepted or rejected in the `decision` of each proposal
datacopilot review import shop.dcp review --proposals ID --from cvat    # prints a new set=ID2
datacopilot apply shop.dcp --proposals ID2 -o shop_v2.dcp     # prints the new version
```

`shop.dcp` is unchanged afterwards; `shop_v2.dcp` records it as its parent in `lineage`, and every new
box has provenance `proposal_accepted`.

## What `propose` does

For every image the model is asked about every class name (and alias) of the dataset, or about
`--classes a,b`. A candidate becomes a proposal only if:

- its numbers are finite, it lies inside the image (a pixel of overshoot is clipped) and it is at least
  2 pixels wide and high;
- its label is one of the phrases that were asked about;
- its score reaches `--min-score` (default 0.35), or `--min-score-for CLASS=SCORE` for that class;
- the image has no box of the same class at the same place already (IoU 0.5), and no better candidate of the
  same class overlaps it that much.

Everything dropped is counted by reason in the set. Class names are turned into lower-case words before
they reach the model; a name with punctuation or non-ASCII letters is skipped and counted (a model prompt
must not be rewritable by a class name).

The result is a **proposal set** in `shop.dcp/proposals/`, named by a fingerprint of its content, bound to
the dataset version and to the model, revision and file checksums that produced it. Edit the file and it
stops matching its own name. `propose` takes the license check first: a model whose license is not on
`DATACOPILOT_ALLOWED_LICENSES` is refused before the dataset is read.

A run that did not reach every image (the `--time-budget` ended it, an image could not be read) exits with
code 1, but its partial set is kept and marked incomplete; `apply` refuses it unless you pass
`--allow-incomplete`.

## Review in CVAT

`review export` writes the dataset's own boxes and, as a separate layer, the proposals (the class name plus
`__proposed`), with the images, as CVAT for images 1.1. Each proposal has the attributes `proposal_id`,
`score` and `decision`. The reviewer types `accepted` or `rejected` in `decision`; moving a box first makes an
edited proposal. Deleting a box decides nothing, so a half-saved file cannot reject anything.

`review import` reads the decisions and stores a **new** set that names the reviewed one as its parent. The
dataset's own boxes in the file are ignored and counted: a review never edits a label. The whole file is
refused if it names an unknown, repeated or foreign proposal, changes a class, uses another decision word,
moves an accepted box outside its image, or decides nothing.

## What `apply` refuses

`apply` writes the accepted proposals into a new dataset and nothing else. It refuses, writing nothing:

- a set made for another version of the dataset, or edited after it was written;
- a model that is not the pinned revision and files of the registered model, or whose license is no longer
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

Measured for real in `notebooks/03_propose_review_apply.ipynb`: on COCO128 (929 boxes) a seeded 19 percent
(178 boxes) were deleted as an answer key, and the real Grounding DINO tiny model, run on a CPU (about 7
minutes for 128 photos), proposed boxes with the default threshold 0.35.

| Result at the default threshold | |
|---|---|
| Proposals | 215 |
| Matching a deleted box (IoU 0.5, same class) | 84 |
| Precision (right / proposals) | 0.39 |
| Recall (found / deleted) | 0.47 |

**The precision is poor.** More than half of the proposals do not match a deleted box. Some of them are
correct objects that COCO never labelled (so the figure is a lower bound), some are the wrong class and some
are boxes that only partly fit; that is why a person reviews every proposal and nothing is applied on its own.
Raising the threshold trades recall for precision (at 0.7 precision was 0.96 and recall 0.15), but that sweep
was tuned on the same data it is graded on, so treat it as optimistic. Accepting only the matches brought 84 of
the 178 deleted boxes back and restored part of the classes that the deletion had removed from the validation
split (30 classes in the complete dataset, 27 with holes, 29 after applying); the audit's findings did not
change in kind. The notebook also draws six of the model's most confident mistakes.

## Not built

Proposals that change a class, delete or refine a box; SAM 3 and Qwen3-VL proposers; consensus between
models; calibration of scores; review in FiftyOne; evidence crops. [ADR-0008](../adr/0008-propose-review-apply.md)
says why.
