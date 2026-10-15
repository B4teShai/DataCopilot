// Flow A: ingest -> audit -> report -> exit code (CI gate).
import { sequence } from "../../../tools/diagrams/lib/sequence.mjs";
import { note } from "../../../tools/diagrams/lib/scene.mjs";
import { c4Legend } from "../../../tools/diagrams/lib/c4.mjs";

const steps = [
  { from: "user", to: "cli", text: "ingest ./export --format coco" },
  { from: "cli", to: "io", text: "read(format, path, group_by)" },
  { from: "io", to: "io", text: "validate; map to schema; keep unmapped in source" },
  { from: "io", to: "store", text: "write images, annotations, manifest" },
  { from: "store", to: "cli", text: "manifest.version", kind: "return" },
  { divider: "audit" },
  { note: "audit ds.dcp --checks all --fail-on critical", at: "user" },
  { from: "cli", to: "audit", text: "run(checks, embedder, fail_on)" },
  { from: "audit", to: "store", text: "bounded read; verify hash" },
  { from: "audit", to: "audit", text: "run checks -> Finding rows" },
  { from: "audit", to: "audit", text: "--embedder: cache, sample check, kNN" },
  { from: "audit", to: "store", text: "write audits/<run_id>.json" },
  { from: "audit", to: "report", text: "render(AuditRun)" },
  { from: "report", to: "cli", text: "report.html / .json / .sarif", kind: "return" },
  { note: "exit 1 if a finding matches --fail-on; 2 on bad usage", at: "cli" },
  { from: "cli", to: "ci", text: "exit code 0 / 1 / 2 + SARIF" },
];
const totalH = 80 + 64 + 30 + steps.length * 64 + 40;

export default {
  elements: [
    sequence({
      title: "Flow A: ingest -> audit -> report -> exit code",
      actors: [
        { id: "user", label: "CV engineer", kind: "person" },
        { id: "cli", label: "cli/ (built)" },
        { id: "io", label: "io/ adapters" },
        { id: "store", label: "dataset.dcp/", kind: "store" },
        { id: "audit", label: "audit/" },
        { id: "report", label: "report/" },
        { id: "ci", label: "CI runner", kind: "external" },
      ],
      steps,
    }),
    c4Legend(40, totalH + 20),
    note("n1", "Ingest, the CPU audit checks, the gate and the reports are built; the embedding checks (--embedder) are built and fail closed. Logged events: dataset_ingested, conversion_loss, audit_check_completed, audit_completed, report_written, command_failed.", 40, totalH + 60, 14),
  ],
};
