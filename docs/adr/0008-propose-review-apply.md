# ADR-0008: Propose, review, apply

**Status:** accepted
**Date:** 2026-10-17

## Context

The product promise is not only to find mistakes but to help fix them. A model can find boxes a dataset
is probably missing, but model output is untrusted: it hallucinates, a class name can steer it, and a
file that carries it can be forged. Earlier reviews of checkpoint trust showed that every added
mechanism produces new findings, so the first fix workflow is the smallest one that delivers the promise
and can be proven on real data. Its rules:

- model output never becomes a label without a person's decision, and an existing label is never edited;
- everything a model, a reviewer or a file can influence is validated before it is stored;
- applying writes a new dataset and records its origin; no existing version is touched;
- the model is license-checked before loading, loaded from pinned files only, and the pin is checked
  again at apply time.

## Options considered

- **A: write model output straight into the dataset with provenance `model`.** Simplest, but the audit
  then trusts or excludes it with no human in the loop, and a wrong box looks like a right one.
- **B: a stored, reviewable proposal set; a person decides in an annotation tool; apply writes a new
  dataset.** One more file format and two more commands, but each step can be checked, refused and replayed.
- **C: B plus several models voting, SAM masks, a language model checking classes, FiftyOne review.** More
  recall and a nicer UI, but every model adds a download, a license, a pin and a failure mode before the
  basic loop is measured.

## Decision

Option B, with the smallest useful content.

1. **One proposal kind, `add_box`:** a box with a class, a score, the phrase that produced it, a status
   (`pending`, `accepted`, `rejected`) and an `edited` flag.
2. **A set is one content-addressed JSON file,** `proposals/<set_id>.json`, where `set_id` is the digest of
   the rest, recomputed on every build and read. A review writes a new set naming its parent. Model
   identity (id, revision, weight hashes) and dataset version are recorded once per set. At most 200,000
   proposals and 128 MiB read.
3. **One detector, Grounding DINO tiny, behind `vlm`,** registered under `datacopilot.proposers`, pinned
   to a full commit with the SHA-256 of weights, configuration, tokenizer and processor files. Safetensors
   only, no remote code, license check before any import or download. A dependency-free `test_boxes`
   proposer serves tests.
4. **Every candidate is validated** (finite, score in (0, 1], a label that was asked for, inside the
   image, at least 2 px), with a cap per image and per set and a time budget. Drops are counted by reason.
   An incomplete run is marked and cannot be applied without `--allow-incomplete`.
5. **Prompts come from class names after normalisation** to lower-case ASCII words. A name that could
   split or rewrite a prompt is skipped and counted.
6. **Review reuses the CVAT writer and reader.** Proposals are a layer (`<class>__proposed`) with
   `proposal_id`, `score` and `decision`. A deleted box decides nothing. Unknown, repeated or foreign ids,
   changed classes, invalid decisions and boxes edited outside the image refuse the whole file.
7. **`apply` appends only.** It calls `DatasetStore.create` (never `write_version`) into an exclusively
   created new path outside the source, with the parent's lineage plus the parent's version. It refuses a
   set for another version, a set that did not come through `review import` or is not a review of its
   parent, a test double (unless `--allow-test-proposer`), a set claiming a registered model's identity it
   was not made by, an unpinned or unlicensed model, an incomplete run and a set with nothing accepted.
   Each new row records model, revision, set and whether the box was edited; an edited box loses the
   score. The reviewed set and its parent are copied into the new dataset. A set id is a digest, not a
   signature: it detects change, not authorship, so whoever can write the dataset folder can write a
   self-consistent set.
8. **Exports verify images.** Every exporter and `review export` resolves each image path inside the
   dataset folder, refuses links elsewhere, caps size and requires the content to equal `image_id`, or
   exports nothing. This closed a hole in all exporters found in review.
9. **Proposals are not a version.** Only `apply` changes `manifest.version`.

### Deliberately not built

- **`change_class`, `delete_box`, `refine_box`, `flag_ambiguous`.** Each needs an answer to "what does
  accepting do to a human label", and adding boxes should be proven first.
- **SAM 3 and Qwen3-VL proposers:** each adds a large or gated download and a license to review.
- **FiftyOne review:** CVAT files cover the round trip, and a second tool doubles the untrusted-file surface.
- **Consensus between models** (with one model there is nothing to agree with) and **score calibration**
  (a raw score is not a probability; the threshold is the user's choice).
- **Evidence crops, `reviewer_id`, a `latest` alias, `--accepted-only`, the base-size checkpoint and
  GPU-specific code paths.**
- **Guessing a class from a merged label.** Two classes in one box are counted (`ambiguous_label`) and dropped.

### Accepted risks

- A file can change between the checksum and the model library reading it; the files are in the user's
  cache folder. Copying them to a private folder first is a small follow-up.
- Parsing a 128 MiB proposal file is estimated, not measured. `apply` writes without re-applying table
  limits (it adds at most 200,000 rows).
- The time budget is checked between images, not between the prompt passes of one image.
- The notebook's reviewer is an oracle and its deletions are random, so its recall is optimistic.

## Consequences

- Easier: every step is testable without a model (fakes, `test_boxes`), every refusal has a test, and the
  real behaviour is measured on COCO128 with a held-out answer key.
- Harder: a new proposer is a registry card plus a module; a new proposal kind changes the set format and
  `apply`.
- Honest limit: held-out precision is a lower bound (a correct box the dataset never had counts as
  wrong), the proposals of a small detector are noisy, and a person must review every one.
- Forbidden: applying anything not accepted, editing a set in place, reading a set without checking its
  digest, loading a model whose license is not allowed.
- Follow-up: the kinds above, more proposers, FiftyOne, consensus and calibration once numbers justify them.

## Links

- [ADR-0005](0005-plugin-system.md) (proposers as plugins)
- [ADR-0006](0006-dataset-format.md) (dataset layout, append-only versions)
- [../architecture/03-data-model.md](../architecture/03-data-model.md) (ProposalSet)
- [../architecture/04-flows.md](../architecture/04-flows.md) (Flow B)
- [../guide/proposals.md](../guide/proposals.md)
