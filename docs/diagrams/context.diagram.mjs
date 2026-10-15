// C4 level 1: DataCopilot in its environment.
import { grid, actor, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { system, external, c4Legend } from "../../../tools/diagrams/lib/c4.mjs";

const g = grid({ x: 40, y: 110, colW: 250, rowH: 100, gapX: 200, gapY: 50 });
const at = (c, r) => g(c, r);

// Row 0: people and the model hub.
const engineer = actor("engineer", at(0, 0).x, at(0, 0).y, "CV engineer", 250);
const hub = external("hub", at(1, 0).x, at(1, 0).y, 250, 100, "Hugging Face Hub", "gated SAM 3 / DINOv3 repos, HTTPS");
const reviewer = actor("reviewer", at(2, 0).x, at(2, 0).y, "Label reviewer", 250);

// Rows 1-2: the system in the middle, two rows tall.
const s = at(1, 1);
const sys = system("sys", s.x, s.y, s.w, s.h * 2 + 50, "DataCopilot", "CLI + Python library: ingest, audit, propose, convert");

// Left column.
const exports_ = external("exports", at(0, 1).x, at(0, 1).y, 250, 100, "Raw dataset exports", "COCO, YOLO, VOC, CVAT XML, Label Studio, ImageFolder");
const ci = external("ci", at(0, 2).x, at(0, 2).y, 250, 100, "CI runner", "GitHub Actions or similar");
const cvpilot = external("cvpilot", at(0, 3).x, at(0, 3).y, 250, 100, "CV-Pilot", "sibling project: training run companion");

// Right column.
const tools = external("tools", at(2, 1).x, at(2, 1).y, 250, 100, "Annotation tools", "CVAT, FiftyOne, X-AnyLabeling");
const vllm = external("vllm", at(2, 2).x, at(2, 2).y, 250, 100, "vLLM endpoint", "Qwen3-VL, OpenAI-style HTTP");

// Bottom.
const objstore = external("objstore", at(1, 3).x, at(1, 3).y, 250, 100, "Object storage", "S3 / GCS / Azure via fsspec");

export default {
  elements: [
    heading("t", "DataCopilot: system context (C4 L1)", 40, 20),
    engineer, hub, reviewer, sys, exports_, ci, cvpilot, tools, vllm, objstore,
    arrow(engineer[0], sys, "runs CLI / imports API"),
    arrow(reviewer[0], tools, "accepts / rejects"),
    arrow(sys, hub, "downloads weights"),
    arrow(sys, exports_, "reads, writes files"),
    arrow(ci, sys, "gate: exit code, SARIF"),
    arrow(cvpilot, sys, "reads manifest.version"),
    arrow(sys, tools, "proposals / decisions"),
    arrow(sys, vllm, "HTTP class check"),
    arrow(sys, objstore, "image bytes by URI"),
    c4Legend(40, 720),
    note("n1", "Arrows point from caller to callee and name the artifact or protocol.", 40, 760),
  ],
};
