# Trust model

DataCopilot is meant to run on datasets you did not make, in CI where logs are public. It treats
these as untrusted: dataset files and `dataset.dcp` folders (including their caches), third-party
plugins, model weights and model output, and the network. The full threat table is in
[cross-cutting concerns](../architecture/06-cross-cutting.md#security-and-threat-model).

## Files you read

- **Paths** in annotation files must stay inside the dataset folder, symlinks included. Absolute
  paths and `..` are errors. Remote URIs (`http`, `s3`, `data:`) are never fetched, only counted.
- **XML** (VOC, CVAT) is parsed without entities: any DOCTYPE is rejected. Nesting, element count and
  file size are capped (8 MiB for VOC, 64 MiB for CVAT).
- **JSON** (COCO, Label Studio) is read through a size cap (256 MiB and 64 MiB). NaN and infinity are
  refused. Label Studio also refuses repeated keys.
- **Numbers** must be finite and in range, and boxes must lie inside their image.
- **Names** (classes, attributes) are data. Every output format escapes them, and control characters
  are stripped before they reach a terminal or a log.
- **Special files** (pipes, devices) are rejected.
- **Images named by a dataset** are copied only after their path is resolved inside the dataset
  folder and their content matches the recorded `image_id`. A hostile `.dcp` cannot use an export to
  copy `~/.ssh/id_rsa` into the output. `--allow-external-images` lifts the folder rule, never the
  content check.

## What never happens

No network call from a gate, no telemetry, no `trust_remote_code`, no pickled weights, no secret in a
log or report (the cache key shows as set or unset), and no image path with a user name in a log.
Label Studio fields outside a fixed allow-list (`completed_by`, usernames, comments, presigned URLs)
never reach the stored dataset or a report. The allow-list cannot see free text in a label name, a
file name or task `data` you asked to keep.

## Model output and review files

A model's boxes are suggestions, never labels. Each candidate is validated before it is stored and
counted if dropped. Class names become lower-case words before they reach a prompt. A proposal set
is named by a fingerprint of its content, so an edited set is refused, and it is bound to the dataset
version and the pinned model. Decisions come back through the hardened CVAT reader, and a file with
anything unexpected is refused whole. `apply` re-checks the license and the pinned model and writes
only a new dataset. See [Propose, review, apply](proposals.md).

## Caches

Caches are untrusted input too. Each entry is authenticated with a key from `DATACOPILOT_CACHE_KEY`
(or a private local secret outside CI). A cache that cannot be verified is ignored and reported, and
a sample of its entries is re-checked on every run. In CI, use `--no-hash-cache`, and give a restored
cache its key out of band.

## What it does not protect against

- A holder of the cache key can forge cache contents outside the re-checked sample.
- `image_id` is a 64-bit label of the file bytes, so two different files can be built with the same
  id. The embedding checks compare full digests; the exact-duplicate check and the version use the
  short id.
- A hostile image the decoders do not handle is bounded in size and time but is still decoded by
  Pillow when you use an embedder. Run untrusted datasets in a sandbox.

## Reporting

Report a vulnerability privately (a security advisory on the repository, or a message to a
maintainer). Do not open a public issue.
