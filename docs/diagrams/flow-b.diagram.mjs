// Flow B: propose -> review export -> apply (new dataset version with lineage).
import { sequence } from "../../../tools/diagrams/lib/sequence.mjs";
import { note } from "../../../tools/diagrams/lib/scene.mjs";
import { c4Legend } from "../../../tools/diagrams/lib/c4.mjs";

const steps = [
  { from: "user", to: "cli", text: "propose --proposer missing_boxes" },
  { from: "cli", to: "propose", text: "run(proposer, models, consensus)" },
  { from: "propose", to: "store", text: "load images, annotations, embeddings" },
  { from: "propose", to: "runtime", text: "infer(model_id, prompts)" },
  { from: "runtime", to: "runtime", text: "license guard; record weight + prompt hash" },
  { from: "runtime", to: "propose", text: "candidates + evidence crops", kind: "return" },
  { from: "propose", to: "propose", text: "consensus gate: IoU >= 0.7, agreement >= 2" },
  { from: "propose", to: "store", text: "write proposals/<set_id>.parquet" },
  { divider: "review" },
  { from: "cli", to: "tool", text: "review --to cvat: push proposals as a layer" },
  { from: "tool", to: "cli", text: "decisions: status, reviewer_id", kind: "return" },
  { divider: "apply" },
  { from: "cli", to: "propose", text: "apply --accepted-only" },
  { from: "propose", to: "store", text: "write new version; append lineage" },
  { note: "existing rows are never mutated; manifest.version is recomputed", at: "store" },
  { from: "store", to: "cli", text: "new manifest.version", kind: "return" },
];
const totalH = 80 + 64 + 30 + steps.length * 64 + 40;

export default {
  elements: [
    sequence({
      title: "Flow B: propose -> review -> apply",
      actors: [
        { id: "user", label: "CV engineer", kind: "person" },
        { id: "cli", label: "cli/ (ingest, export built)" },
        { id: "propose", label: "propose/ + patch/" },
        { id: "runtime", label: "model runtime (extras)" },
        { id: "store", label: "dataset.dcp/", kind: "store" },
        { id: "tool", label: "CVAT / FiftyOne", kind: "external" },
      ],
      steps,
    }),
    c4Legend(40, totalH + 20),
    note("n1", "Planned flow. Logged events: proposal_set_written, review_exported, review_imported, proposals_applied, command_failed.", 40, totalH + 60, 14),
  ],
};
