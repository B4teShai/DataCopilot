# Status

What works today. This page is the only place that says so; other pages describe how things work.
"Built" is implemented and tested. "Experimental" runs but its results are weak. "Planned" does not
run yet.

| Area | Status | What it covers |
|---|---|---|
| Convert formats | Built | COCO, YOLO, YOLO-OBB, VOC, CVAT, Label Studio and ImageFolder; losses counted. See [Formats](guide/formats.md). |
| Oriented boxes | Built | YOLO-OBB in and out, corner order kept; box checks by corners and rotated IoU. Not read from CVAT or Label Studio yet; label-noise crops are not rotated. See [ADR-0011](adr/0011-oriented-boxes.md). |
| Dataset folder and version | Built | `ingest` writes `dataset.dcp`; same content gives the same version. |
| CPU audit | Built | Integrity, exact duplicates, class imbalance, box sanity; reports in JSON, HTML and SARIF. |
| Image-model audit | Built | Near-duplicates, leakage and outliers with `--embedder`. See [Embeddings](guide/embeddings.md). |
| Label noise | Experimental | Box-crop leads are better than before but weak. See [Label noise](guide/label-noise.md). |
| Split | Built | Group-aware re-split into a new version. See [Split](guide/split.md). |
| Propose missing boxes | Experimental | Grounding DINO proposes `add_box` only; precision is low. See [Propose, review, apply](guide/proposals.md). |
| Review and apply | Built | CVAT round trip, then a new version with lineage. |
| Python API and `doctor` | Built | `load`, `audit`, `export`, `split`, `propose`, `apply`; `doctor` checks the install. |
| MCP server | Built | `datacopilot mcp` (extra `mcp`): dataset info, stored and fresh audit summaries; dataset-creating tools only when enabled. See [MCP server](guide/mcp.md). |
| More proposal kinds | Planned | Change a class, delete a box, refine a box; consensus between models. |
| More models and tools | Planned | SAM 3 and Qwen3-VL proposers, FiftyOne review, a Cleanlab label-noise backend. |
