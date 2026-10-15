# ADR-0006: Dataset format

**Status:** accepted
**Date:** 2026-10-14

## Context

Every command reads and writes one internal representation of a dataset: images,
annotations, categories, plus derived data (embeddings, audit runs, proposals). The
representation must: hold 50k images and 500k boxes on a laptop; be queryable without a
server; keep unmapped fields from any source format so conversion is lossless; carry a
stable content hash that CV-Pilot can record; keep history so that applying proposals
never destroys human labels; and survive an interrupted write. Images are large and often
live in object storage; copying them would double storage and break provenance.

## Options considered

### Option A: a directory of Parquet tables, queried with DuckDB, with a JSON manifest

Trade-offs: columnar, compressed, zero-copy into Arrow, DuckDB queries across tables
without loading everything; one directory is easy to copy, diff and version in object
storage. Costs: two dependencies (`pyarrow`, `duckdb`), no in-place row updates (which is
desired here), nested fields (`source`, `attributes`) need an explicit encoding.

### Option B: a single SQLite file

Trade-offs: one file, transactions, ubiquitous; but row-oriented storage is slow for
column scans over hundreds of thousands of boxes, vectors need blobs, and a single file
cannot be partially synced or diffed.

### Option C: JSON lines per table

Trade-offs: human-readable and trivially diffable; but 10 to 20 times larger than Parquet,
no typed columns, every query is a full parse, and embeddings are impractical.

### Option D: a document database service (MongoDB-style, as FiftyOne uses)

Trade-offs: flexible schema and rich queries; but requires a running server, which rules
out CI gates and laptops without setup, and is the main operational complaint about
FiftyOne.

## Decision

Option A. A dataset is a directory with the suffix `.dcp`:

```
dataset.dcp/
  manifest.json                     # DatasetManifest: version, schema_version, lineage
  images.parquet                    # Image rows
  annotations.parquet               # Annotation rows
  categories.parquet                # Category rows
  versions/<version>/               # immutable snapshot of the three tables + manifest
  embeddings/<model_id>@<revision>.parquet
  audits/<run_id>.json              # AuditRun with its Finding rows
  proposals/<set_id>.parquet        # Proposal rows (one set per propose run)
```

Rules that reviewers check:

1. Images are referenced by `file_path` (a local path or an `fsspec` URI). They are never
   copied into the directory.
2. `image_id` is `sha256(image bytes)[:16]` hex. It does not change when a file moves.
3. `manifest.version` is the content hash of the three tables as defined in
   [../architecture/03-data-model.md](../architecture/03-data-model.md#versioning-and-compatibility):
   canonical JSON per row, rows hashed, row hashes sorted, tables hashed in a fixed order,
   `file_path` excluded. Row order on disk does not affect the hash.
4. Versions are append-only. `apply`, `split` and `ingest` write a new snapshot under
   `versions/<version>/`, then replace the top-level tables and `manifest.json` by writing
   to a temporary name and renaming. Existing snapshots are never modified or deleted by a
   command. `manifest.lineage` lists parent versions, newest last.
5. Readers keep every field they cannot map in `source` (verbatim). Writers re-emit
   `source` fields the target format can hold. Every conversion returns a
   `ConversionReport` that counts dropped fields per column; `--strict` makes any drop a
   failure (exit code 1).
6. Embeddings, audits and proposals are derived data keyed by `image_id` and are not part
   of the version hash. They are kept across versions as long as their `image_id` still
   resolves.
7. The three tables use explicit Arrow schemas with `schema_version` in the manifest.
   Nested fields are stored as Arrow `struct`/`map` columns, never as pickled blobs.

## Consequences

- Easier: a CI gate needs only the directory and the core wheel; a dataset version can be
  pinned by one string; conversions are testable as golden round trips; chaos tests can
  truncate a temporary file and assert recovery.
- Harder or forbidden: no in-place edits of annotation rows (by design); each new version
  duplicates the three tables (a 500k-box annotations table is about 30 to 60 MB in
  Parquet, acceptable); very large nested `source` payloads inflate rows, so writers may
  compress them.
- Follow-up: `store/` module, the Arrow schemas, the hashing routine with a golden test,
  the chaos tier for interrupted writes, and a `datacopilot gc` command for pruning old
  snapshots (explicitly opt-in).

## Links

- [ADR-0001](0001-scope-and-license.md) (the version hash is the contract with CV-Pilot)
- [../architecture/03-data-model.md](../architecture/03-data-model.md)
