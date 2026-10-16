import { arrow, box, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

// The one picture a newcomer sees first (README and docs home): a dataset goes in, DataCopilot
// converts, audits and reports, and two plain results come out. Three colours only:
// input orange, tool blue, result green. No legend is needed. Detailed views: the other diagrams.
const big = { fontSize: 20 };
const input = box("in", 40, 130, 300, 130, "Your dataset\nCOCO · YOLO · VOC ·\nCVAT · Label Studio", COLORS.external, big);
const tool = box("dc", 470, 130, 360, 130, "DataCopilot\n1 Convert  2 Audit  3 Report", COLORS.core, { ...big, strokeWidth: 2 });
const clean = box("out1", 990, 40, 320, 110, "Clean dataset\nin any format", COLORS.built, big);
const gate = box("out2", 990, 240, 320, 110, "CI passes or fails", COLORS.built, big);
const models = box("ml", 450, 330, 400, 110, "Image models (optional)\nfor duplicates and leaks", COLORS.core, { ...big, strokeStyle: "dashed" });

export default {
  elements: [
    input, tool, clean, gate, models,
    arrow(input, tool, "reads", big),
    arrow(tool, clean, "writes", big),
    arrow(tool, gate, "exit code", big),
    arrow(models, tool, "helps", big),
    note("cap", "Convert it, audit it, and fail the build if the data is bad.", 40, 480, 20),
  ],
};
