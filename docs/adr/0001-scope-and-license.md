# ADR-0001: Scope and license

**Status:** accepted
**Date:** 2026-10-14

## Context

Teams that train detection and classification models hold labels in several export
formats (COCO, YOLO, VOC, CVAT XML, Label Studio JSON, ImageFolder) and move them between
tools. Three problems recur and no open-source tool covers all of them:

1. Conversion between formats loses fields silently (attributes, occlusion flags, track
   ids, `iscrowd`, CVAT groups). No tool publishes what is lost per format pair.
2. Dataset quality is checked interactively, not as a gate. Nothing returns a non-zero exit
   code when a split leaks or a class is missing from the validation set.
3. Model-generated label fixes (missing boxes, class corrections) are either applied
   blindly or not produced at all, and model licenses are not checked before use.

The landscape as of October 2026: FiftyOne (Apache-2.0) is the best dataset browser and
needs MongoDB and an App session; LightlyStudio (Apache-2.0, 1.1.0) is a UI-centric
curate-label-QA tool with a planned plugin system; Cleanlab and CleanVision (AGPL-3.0)
score label and image issues; Datumaro (MIT) is the broadest converter with no audit layer;
supervision (MIT) covers a few formats; Encord Active was archived in August 2025; fastdup
is CC BY-NC-ND and unusable commercially; X-AnyLabeling (GPL-3.0) is a labeler.

The first user is a retail smart-fridge pipeline: frames from the same door-open session
must not be split across train and validation, near-identical SKU packaging is confused,
occluded products need box checks. The sibling project CV-Pilot records a dataset version
for every training run and needs a stable content hash to record.

Constraints: the core must install on CPU-only machines without PyTorch; the project must
be usable commercially without legal review of each dependency; review of proposals stays
in the tools people already use.

## Options considered

### Option A: a FiftyOne-style browser and curation application

Build a web UI with a sample grid, embeddings view and in-app label editing.

Trade-offs: the most visible product; but it competes head-on with FiftyOne and
LightlyStudio, needs a server process and a database, cannot run as a CI gate, and the
UI absorbs most engineering effort before any audit check exists.

### Option B: a Cleanlab / Datumaro-style library only

Ship Python functions for conversion and issue scoring, no CLI, no on-disk format, no
report.

Trade-offs: smallest surface; but scores without a stable artifact cannot gate CI, there
is no provenance for fixes, and every team re-implements the glue (manifest, versioning,
review export).

### Option C: headless auditor, lossless converter and reviewable proposals, CLI-first

A CLI plus library over one on-disk dataset format: readers and writers with a round-trip
contract, audit checks that produce findings and exit codes, proposals stored as data with
provenance and pushed to CVAT, FiftyOne or X-AnyLabeling for review.

Trade-offs: no UI of its own, so adoption depends on integrations; in return it runs on a
laptop and in CI, has no server, and every output is a file that can be diffed and stored.

### License options

- **Apache-2.0:** matches FiftyOne, LightlyStudio, DINOv2, Grounding DINO, Qwen3-VL and
  SigLIP2; explicit patent grant; acceptable to commercial users without review.
- **AGPL-3.0:** would allow shipping Cleanlab, CleanVision and Ultralytics as core
  dependencies; blocks most commercial adopters and contaminates any package that imports
  this one.
- **MIT:** maximally permissive; no patent grant, which matters for a project that wraps
  model inference.

## Decision

Option C with Apache-2.0.

Rules that reviewers check:

1. DataCopilot is a CLI and a library. It ships no GUI, annotation tool, dataset browser,
   trainer or hosted service. A read-only report viewer is the only UI ever considered.
2. Review of proposals happens in CVAT, FiftyOne or X-AnyLabeling. DataCopilot exports to
   them and imports decisions; it does not replace them.
3. Core dependencies are Apache-2.0, MIT, BSD, ISC, PSF or MPL-2.0 only. Copyleft
   libraries (Cleanlab, CleanVision, Ultralytics; AGPL-3.0/GPL-3.0) are never core and
   never vendored; they may be declared as an optional extra (`[cleanlab]`) and imported
   lazily inside the one adapter that needs them, and that adapter's `--help` text and
   the runtime docs state that using the extra makes the user's combined work subject to
   that license. Code that includes or derives from copyleft code (the YOLOE proposer)
   ships as a separately named package (`datacopilot-yoloe`) that depends on DataCopilot,
   never the other way round.
4. Model weights under custom or gated licenses (SAM License, DINOv3 License) are never
   vendored or redistributed. The model registry shows the license before first use and
   refuses models whose license is not in `DATACOPILOT_ALLOWED_LICENSES`.
5. fastdup (CC BY-NC-ND) is not used in any form.
6. `manifest.version`, the content hash of a dataset version, is the contract with
   CV-Pilot, which records it as `Repro.dataset_hash`. Its definition is in
   [ADR-0006](0006-dataset-format.md) and changes only with a schema version bump.

## Consequences

- Easier: CI gating (one process, exit codes, SARIF), reproducibility (content-hashed
  versions), commercial adoption (one permissive license, no server).
- Harder or forbidden: no interactive curation; any feature that needs a long-running
  process or a database is out of scope; copyleft libraries never appear at module top
  level or in the core lock, so the label-noise check must degrade cleanly (skipped with a
  `WARN`) when `[cleanlab]` is absent; derived copyleft code (YOLOE) stays outside this
  repository.
- Follow-up: the round-trip test matrix per format pair; the model registry with license
  metadata; `THIRD_PARTY_MODELS.md` entries added with each model; the `[cleanlab]` extra
  populated in `pyproject.toml` together with the lazy-import adapter and its license
  notice.

## Links

- [ADR-0005](0005-plugin-system.md) plugin system; [ADR-0006](0006-dataset-format.md)
  dataset format.
- [../architecture/01-overview.md](../architecture/01-overview.md)
- [../engineering/dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md)
