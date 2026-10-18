# Trust model

DataCopilot is meant to run on datasets and exports you did not produce, in CI where logs are public.
It treats these as untrusted: dataset files and `dataset.dcp/` directories (including their caches),
third-party plugins, model weights and model output, and the network. The full threat table is in
[cross-cutting concerns](../architecture/06-cross-cutting.md#security-and-threat-model); this page
is the user-facing summary.

## Files you read

- **Paths** in annotation files are resolved and must stay inside the dataset directory, symlinks
  included. Absolute paths and `..` are errors, never followed. Remote URIs (`http`, `s3`, `data:`) are
  never fetched; they are counted.
- **XML** (Pascal VOC, CVAT) is parsed from `expat` events: any DOCTYPE or entity declaration is
  rejected, so there is no entity expansion and no external entity. Nesting is limited to 32 levels,
  elements to 1,000,000 per document, attributes to 256 per element, and the file to 8 MiB (VOC) or
  64 MiB (CVAT).
- **JSON** (COCO, Label Studio) is read through a bounded descriptor (256 MiB for COCO, 64 MiB for
  Label Studio), NaN and infinities are refused, and Label Studio also refuses repeated keys and bounds
  tasks, annotations and results per task.
- **Numbers** are validated: coordinates and percentages must be finite and in range, sizes positive
  whole numbers, boxes inside their image.
- **Names** (classes, attributes) are data: they are escaped by every serialiser (XML, JSON, the HTML
  report) and stripped of control characters before they reach a terminal or a log.
- **Special files** (FIFOs, devices) are rejected on the opened descriptor.
- **Images named by a dataset** are never copied blindly: `export` and `review export` resolve every image
  path inside the dataset's directory, refuse links to elsewhere, remote and unsafe paths, cap the size and
  require the content to match the recorded `image_id`, or export nothing (`--allow-external-images` lifts
  only the directory rule, never the content check). A hostile `.dcp` can therefore not use an export to copy
  `~/.ssh/id_rsa` into the output.

## What never happens

No network call from a gate; no telemetry; no `trust_remote_code`; no pickled weights; no secret in
a log or report (the cache key is shown as set or unset); no image path with a user name in a log
field; no Label Studio field outside a fixed allow-list (so no `completed_by`, usernames, e-mail addresses,
comments, upload names or presigned URLs) in the stored dataset, a report or a re-exported file. What the
allow-list cannot see is free text you put into a label name, an image file name or task `data` that you
asked to keep (`--keep-task-data`).

## Model output and review files

A model's boxes are suggestions, never labels. Each candidate is validated before it is stored (finite numbers,
inside the image, a label that was asked for) and counted if dropped; class names are turned into lower-case
words before they reach a model prompt; a proposal set is named by the digest of its content, so an edited set
is refused, and it is bound to the dataset version and to the pinned model it came from. Decisions come back
through the hardened CVAT reader and a file with an unknown, repeated or foreign id, a changed class or a box
outside its image is refused as a whole. `apply` writes a new dataset only, checks the license and the pinned
model again, and never writes into the source. See [Propose, review, apply](proposals.md).

## Caches

Caches are untrusted input too. Each entry is authenticated with a key from `DATACOPILOT_CACHE_KEY`
(or a private local secret outside CI); a cache that cannot be verified is ignored and reported, and a
sample of its entries is re-checked against the files on every run. A gate should use
`--no-hash-cache`; a restored cache must carry its key out of band.

## What it does not protect against

A holder of the cache key can forge cache contents outside the re-checked sample. `image_id` is a
64-bit label of the file bytes, so two different files can be constructed with the same id; the
embedding checks compare full digests, the exact-duplicate check and the version hash use the short
id. A dataset that is hostile in ways the image decoders do not handle is bounded in size and time but
is still decoded by Pillow when you use an embedder: run untrusted datasets in a sandbox.

## Reporting

Report a vulnerability privately (a security advisory on the repository, or a message to a
maintainer); do not open a public issue.
