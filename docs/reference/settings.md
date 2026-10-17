# Settings and exit codes

## Settings

Settings come from environment variables. An invalid value is a usage error (exit 2) before any
work starts. Secrets are only read from the environment and never printed or logged.

| Variable | Meaning | Default |
|---|---|---|
| `DATACOPILOT_ENV` | `dev`, `staging` or `prod`; `DEBUG` logging is honoured only in `dev` | `prod` |
| `DATACOPILOT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `DATACOPILOT_MAX_TABLE_ROWS`, `DATACOPILOT_MAX_TABLE_BYTES` | limits applied to a `dataset.dcp` before it is decoded | 5,000,000 rows, 4 GiB |
| `DATACOPILOT_WORKERS` | threads for file checks (1 to 256) | CPU count |
| `DATACOPILOT_CACHE_DIR` | where caches live | `~/.cache/datacopilot` |
| `DATACOPILOT_HASH_CACHE` | `false` disables the file-hash cache (same as `--no-hash-cache`) | `true` |
| `DATACOPILOT_CACHE_KEY` | out-of-band key that authenticates caches; needed for the embedding cache in CI | unset |
| `DATACOPILOT_ALLOWED_LICENSES` | licenses a model may carry; checked before `propose` reads the dataset and again by `apply` | `Apache-2.0,MIT,BSD-3-Clause` |

Logs are JSON records on standard error with `@timestamp`, `level`, `message`, `service`, `env`,
`request_id` and `logger`. `datacopilot doctor` shows the validated settings.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success, or the audit gate passed (or no gate was given) |
| `1` | the command ran and failed: unreadable or invalid input, unknown format, output exists, `--strict` with dropped fields, or a `--fail-on` match |
| `2` | usage or configuration error |

The [CLI reference](cli.md) lists every command and option. Settings that matter in CI are explained
in [Audit in CI](../guide/audit-ci.md).
