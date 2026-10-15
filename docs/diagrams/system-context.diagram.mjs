import { grid, actor, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { system, external, c4Legend } from "../../../tools/diagrams/lib/c4.mjs";

// C4 level 1 for the pair: who uses the two systems, what they exchange, and the one
// contract that ties them together (DataCopilot manifest.version -> CV-Pilot Repro.dataset_hash).
// Layout: externals on the top row, the two systems in the middle, shared infrastructure below.
const g = grid({ x: 40, y: 110, colW: 230, rowH: 90, gapX: 40, gapY: 80 });
const at = (c, r, span = 1) => [g(c, r, span).x, g(c, r, span).y, g(c, r, span).w, g(c, r, span).h];

const raw = external("raw", ...at(0, 0), "Dataset exports", "COCO · YOLO · VOC · CVAT · LS");
const review = external("review", ...at(1, 0), "Review tools", "CVAT · FiftyOne");
const hub = external("hub", ...at(2, 0), "Model hubs", "HF Hub · gated weights");
const eng = actor("eng", g(3, 0).x + 35, g(3, 0).y, "CV engineer");
const frameworks = external("fw", ...at(4, 0), "Training frameworks", "Ultralytics · Lightning · HF");
const trackers = external("trk", ...at(5, 0), "Experiment trackers", "MLflow · W&B · Trackio");
const sched = external("sched", ...at(6, 0), "Schedulers / spot", "SkyPilot · SLURM");

const dc = system("dc", ...at(0, 2, 3), "DataCopilot", "audit · convert · propose fixes");
const cvp = system("cvp", ...at(4, 2, 3), "CV-Pilot", "regression gates · checkpoints");

const ci = external("ci", ...at(3, 3), "CI pipeline", "exit codes · SARIF");
const store = external("obj", ...at(0, 4, 7), "Object storage (S3 · GCS · Azure)", "images referenced by URI · checkpoints and artifacts");

export default {
  elements: [
    heading("t", "DataCopilot + CV-Pilot: system context", 40, 20),
    raw, review, hub, eng, frameworks, trackers, sched, dc, cvp, ci, store,
    arrow(raw, dc, "reads"),
    arrow(dc, review, "proposals ⇄ decisions"),
    arrow(hub, dc, "weights"),
    arrow(eng[0], dc, "audit · convert"),
    arrow(eng[0], cvp, "compare · gate"),
    arrow(frameworks, cvp, "hooks"),
    arrow(cvp, trackers, "forwards"),
    arrow(sched, cvp, "preemption notice"),
    arrow(dc, cvp, "manifest.version → Repro.dataset_hash"),
    arrow(ci, dc, "runs audit"),
    arrow(ci, cvp, "runs gate"),
    arrow(dc, store, "image URIs"),
    arrow(cvp, store, "checkpoints"),
    c4Legend(40, g(0, 5).y),
    note("n", "DataCopilot built: ingest, export, audit, embed, doctor and the load, audit, export API for six formats. Proposals, review and CV-Pilot features: see each project's architecture documents.", 760, g(0, 5).y + 2),
  ],
};
