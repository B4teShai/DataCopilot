// DataCopilot and the people and tools around it.
import { arrow, box, heading, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const F = { fontSize: 18 };
const side = { fontSize: 18 };
const engineer = box("eng", 40, 90, 240, 90, "CV engineer\nruns commands", COLORS.agent, side);
const files = box("files", 40, 250, 240, 110, "Dataset files\nCOCO, YOLO, VOC, CVAT,\nLabel Studio", COLORS.external, side);
const ci = box("ci", 40, 430, 240, 90, "CI runner\npass or fail", COLORS.external, side);
const dc = box("dc", 410, 160, 180, 280, "DataCopilot\ncommand line\nand Python\nlibrary", COLORS.core, { fontSize: 20, strokeWidth: 2 });
const cvat = box("cvat", 720, 90, 240, 90, "CVAT\nreviewer decides", COLORS.external, side);
const hub = box("hub", 720, 250, 240, 110, "Hugging Face Hub\nmodel weights", COLORS.external, side);
const pilot = box("pilot", 720, 430, 240, 90, "CV-Pilot\ntracks training runs", COLORS.external, side);

export default {
  elements: [
    heading("t", "DataCopilot and its surroundings", 40, 20),
    engineer, files, ci, dc, cvat, hub, pilot,
    arrow(engineer, dc, "runs", F),
    arrow(files, dc, "files", F),
    arrow(dc, ci, "exit code", F),
    arrow(dc, cvat, "review", F),
    arrow(dc, hub, "weights", F),
    arrow(pilot, dc, "version", F),
    note("cap", "Orange boxes are outside tools. Arrows point from caller to callee.", 40, 570, 18),
  ],
};
