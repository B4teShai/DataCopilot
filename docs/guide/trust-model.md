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

## What never happens

No network call from a gate; no telemetry; no `trust_remote_code`; no pickled weights; no secret in
a log or report (the cache key is shown as set or unset); no image path with a user name in a log
field; no personal fields copied from Label Studio's `completed_by`.

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
