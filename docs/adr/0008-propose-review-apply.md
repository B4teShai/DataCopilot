# ADR-0008: Propose, review, apply

**Status:** accepted
**Date:** 2026-10-17

## Context

The product promise is not only to find mistakes in a dataset but to help fix them. A model can find
boxes that a dataset is probably missing, but model output is untrusted: it hallucinates, it can be
steered by a class name, and a file that carries it can be forged. Two earlier rounds of review of
checkpoint trust showed that every added mechanism produces new findings, so the first version of the
fix workflow has to be the smallest one that delivers the promise and can be proven on real data. The
rules that shape it:

- model output never becomes a label without a person's decision, and a label that exists is never edited;
- everything a model, a reviewer or a file can influence is validated before it is stored;
- applying writes a new dataset and records where it came from; no existing version is touched;
- the model is license-checked before it is loaded, loaded from pinned files only, and the same pin is
  checked again when proposals are applied.

## Options considered

### Option A: write model output straight into the dataset with provenance `model`

Simplest. But the audit then trusts it (or excludes it) with no human in the loop, and a wrong box is
indistinguishable from a right one once it is in the table.

### Option B: proposals as a stored, reviewable set; a person decides in an annotation tool; apply writes a new dataset

One more file format and two more commands, but each step can be checked, refused and replayed.

### Option C: Option B plus several models voting (consensus), SAM masks, a language model checking classes, FiftyOne review

The roadmap's full picture. More recall and a nicer review UI, but every model adds a download, a license,
a pin and a failure mode before the basic loop has been measured on real data.

## Decision

Option B, with the smallest useful content.

1. **One proposal kind, `add_box`.** A proposal is a box with a class, a score, the phrase that produced it
   and a status (`pending`, `accepted`, `rejected`) and an `edited` flag.
2. **A proposal set is one JSON file, content-addressed.** `proposals/<set_id>.json`, where `set_id` is
   the digest of the rest of the document, recomputed on every build and read. A review never edits a set:
   it writes a new one that names its parent. The model identity (id, revision, weight hashes) and the
   dataset version are recorded once per set. A set is bounded (200,000 proposals, 128 MiB read).
3. **One detector, Grounding DINO tiny, behind the `[vlm]` extra.** Registered under
   `datacopilot.proposers` like formats and embedders, pinned to a full commit with the sha256 of the
   weights, the configuration and the tokenizer and processor files; `.safetensors` only, no remote code,
   license check before any import or download. A dependency-free `test_boxes` proposer serves the tests.
4. **Validation of every candidate** (finite, score in (0, 1], a label that was asked for, inside the
   image, at least 2 px), a cap per image and per set, and a time budget. Dropped candidates are counted by
   reason in the set. A run that did not reach every image is marked incomplete and cannot be applied
   without `--allow-incomplete`.
5. **Prompts come from class names, but only after normalisation** to lower-case ASCII words; a name that
   could split or rewrite a prompt is skipped and counted.
6. **Review reuses the CVAT writer and reader.** Proposals are a layer (`<class>__proposed`) with the
   attributes `proposal_id`, `score`, `decision`. A deleted box decides nothing, so truncation cannot reject
   anything. Unknown, repeated or foreign ids, changed classes, invalid decisions and boxes edited outside
   the image refuse the whole file.
7. **`apply` appends only.** It calls `DatasetStore.create` (never `write_version`; the output directory is
   created exclusively), with the parent's lineage plus the parent's version; the output must be a new path
   outside the source (the source is a required argument). It refuses a set for another version, a set that did
   not come through `review import` or is not exactly a review of its parent, a test double (unless
   `--allow-test-proposer`), a set that claims the identity of a registered model it was not made by, an
   unpinned or unlicensed model, an incomplete run and a set with nothing accepted. Each new row records the
   model, revision, set and whether the box was edited; an edited box loses the model's score; the reviewed set
   and its parent are copied into the new dataset so the provenance survives deleting the source.
   A set id is a digest, not a signature: it detects change, not authorship. Whoever can write into the dataset
   directory can write a self-consistent set; the review marker and the parent check make that deliberate, not
   accidental.
8. **Exports verify images.** Every exporter (and `review export`) resolves each image path inside the dataset
   directory, refuses links to elsewhere, caps the size and requires the content to equal `image_id`, or
   exports nothing; this closed a hole in all exporters, found in review.
9. **Proposals are not a version.** They do not change `manifest.version`; only `apply` does.

### What was deliberately NOT built

- **`change_class`, `delete_box`, `refine_box` and `flag_ambiguous` proposals.** Each needs an answer to
  "what does accepting do to a human label", and the first loop should prove that adding boxes works
  before anything edits a label. Only `add_box` exists.
- **SAM 3 and Qwen3-VL proposers.** Each adds a gated or large download and a license to review; the
  detector alone has to be measured first.
- **FiftyOne review.** CVAT files cover the round trip with no extra dependency; a second tool doubles the
  untrusted-file surface.
- **Consensus gating between models.** With one model there is nothing to agree with.
- **Score calibration.** A raw detector score is not a probability; the notebook shows how it relates to
  precision on one dataset, and a threshold is the user's choice.
- **Evidence crops, `reviewer_id`, a `latest` alias for a set id, an `--accepted-only` flag** (apply only
  ever applies accepted proposals) **and the base-size Grounding DINO checkpoint.**
- **GPU-specific code paths** beyond the CPU/CUDA choice the embedders already make.
- **Guessing a class from a merged label.** The decoder sometimes names two classes in one box; those are
  counted (`ambiguous_label`) and dropped, not resolved.

### Accepted risks

- A file can change between the checksum check and the model library reading it (time of check to time of use);
  the files are in the user's own cache directory. Copying the pinned files to a private directory first is a
  small follow-up.
- Memory use of parsing a 128 MiB proposal file is estimated, not measured; the proposal count is capped at
  200,000 and the file at 128 MiB.
- `apply` reads the dataset under the table limits but writes the new dataset without re-applying them (it adds
  at most 200,000 rows).
- The time budget is checked between images, not between the prompt passes of one image (at most about 8
  passes for 512 phrases).
- The reviewer in the notebook is an oracle and the deletions are random, so its recall is optimistic.

## Consequences

- Easier: every step can be tested without a model (fakes and the `test_boxes` proposer), every refusal has
  a test, and the real behaviour is measured in `notebooks/03_propose_review_apply.ipynb` on COCO128 with a
  held-out answer key.
- Harder: a new proposer is a card in the registry plus a module; a new proposal kind changes the set
  format and `apply`.
- Honest limit: held-out precision is a lower bound (a correct box that the dataset never had counts as
  wrong), and the proposals of a small detector are noisy; a person must review every one.
- Forbidden: applying anything that is not accepted, editing a set in place, reading a set without checking
  its digest, loading a model whose license is not allowed.
- Follow-up: the proposal kinds above, more proposers, FiftyOne, consensus and calibration once there are
  numbers to justify each.

## Links

- [ADR-0005](0005-plugin-system.md) (proposers as plugins)
- [ADR-0006](0006-dataset-format.md) (dataset layout, append-only versions)
- [../architecture/03-data-model.md](../architecture/03-data-model.md) (ProposalSet)
- [../architecture/04-flows.md](../architecture/04-flows.md) (Flow B)
- [../guide/proposals.md](../guide/proposals.md)
