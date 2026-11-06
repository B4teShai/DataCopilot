# MCP server

`datacopilot mcp` serves DataCopilot's dataset tools over the
[Model Context Protocol](https://modelcontextprotocol.io) on standard input and output, so a
coding agent or any other MCP client can inspect and audit datasets without shell access.

<!-- skip: installs an extra and starts a server that waits for an MCP client -->
```bash
uv sync --extra mcp              # in a clone; the extra holds the MIT-licensed `mcp` SDK
datacopilot mcp                  # started by the MCP client, not by hand
```

A client configuration (the file name and layout depend on the client):

```json
{
  "mcpServers": {
    "datacopilot": {
      "command": "datacopilot",
      "args": ["mcp"],
      "env": { "DATACOPILOT_MCP_ROOTS": "/data/datasets" }
    }
  }
}
```

## Tools

| Tool | Writes | What it returns |
|---|---|---|
| `list_formats` | no | Readers and writers with their license and extra. |
| `dataset_info` | no | Manifest (version, counts, classes, lineage), stored audits, proposal sets and their review status. |
| `audit_summary` | no | A stored audit: verdict, counts per check and severity, the worst findings, class balance. |
| `run_audit` | no | The same summary for an audit run now; nothing is stored. With an embedder it may download the model and fill the embedding cache. |
| `ingest` | new `.dcp` | Manifest and what the conversion dropped or skipped. |
| `propose` | proposal set | Set id, progress and status counts (needs the extra `vlm`). |
| `apply_proposals` | new `.dcp` | The new version's manifest. |
| `split` | new `.dcp` | The split report without the per-class table. |

## Safety rules

- **Read-only by default.** `ingest`, `propose`, `apply_proposals` and `split` exist only when the
  server starts with `DATACOPILOT_MCP_ALLOW_WRITE=1`. None of them changes or deletes an existing
  version; outputs that already exist are refused.
- **Path roots.** Every path must resolve, symlinks followed, inside `DATACOPILOT_MCP_ROOTS`
  (separated by `:`, or `;` on Windows; default: the directory the server started in). Relative
  paths start at the first root.
- **Summaries, never rows.** At most 50 findings per call, payloads cut to 300 characters, no image
  or annotation table.
- **Errors** come back as tool errors with a one-line reason. Logs are the usual JSON records on
  standard error, one `request_id` per tool call; an unexpected failure is logged once, with its
  traceback, and the client only sees its type.
- Label Studio task data is always dropped by `ingest`, and the test-only proposer and embedder are
  not offered.
