// Flow A: ingest, audit, report, exit code.
import { arrow, box, heading, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const F = { fontSize: 18 };
const col = (c) => 40 + c * 380;
const dataset = box("ds", col(0), 90, 260, 90, "Your dataset\nany of six formats", COLORS.external, F);
const dcp = box("dcp", col(1), 90, 260, 90, "dataset.dcp/\ntables and version", COLORS.store, F);
const other = box("other", col(2), 90, 260, 90, "Dataset in\nanother format", COLORS.built, F);
const audit = box("audit", col(1), 270, 260, 90, "audit\nchecks and the gate", COLORS.core, F);
const reports = box("reports", col(0), 450, 260, 90, "Reports\nHTML, JSON, SARIF", COLORS.built, F);
const exit = box("exit", col(2), 450, 260, 90, "Exit code\n0 pass, 1 fail, 2 usage", COLORS.built, F);

export default {
  elements: [
    heading("t", "Flow A: ingest, audit, report, exit code", 40, 20),
    dataset, dcp, other, audit, reports, exit,
    arrow(dataset, dcp, "ingest", F),
    arrow(dcp, other, "export", F),
    arrow(dcp, audit, "audit reads", F),
    arrow(audit, reports, "writes", F),
    arrow(audit, exit, "gate decides", F),
    note("cap", "Image models, when you ask for them, add the look-alike checks inside audit.", 40, 580, 18),
  ],
};
