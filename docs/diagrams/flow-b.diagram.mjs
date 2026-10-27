// Flow B: propose, review, apply. A person decides; the old version is never changed.
import { arrow, box, heading, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const F = { fontSize: 18 };
const x = (c) => 40 + c * 380;
const y = (r) => 90 + r * 190;
const B = (id, c, r, text, bg) => box(id, x(c), y(r), 260, 90, text, bg, F);
const dcp = B("dcp", 0, 0, "dataset.dcp/\nyour dataset", COLORS.store);
const propose = B("propose", 1, 0, "propose\nmodel finds boxes", COLORS.core);
const set = B("set", 2, 0, "Proposal set\nstored in the .dcp", COLORS.store);
const exportR = B("export", 2, 1, "review export\nCVAT files", COLORS.core);
const person = B("person", 1, 1, "Reviewer\naccepts or rejects", COLORS.agent);
const importR = B("import", 0, 1, "review import\nnew proposal set", COLORS.core);
const apply = B("apply", 0, 2, "apply\naccepted boxes only", COLORS.core);
const out = B("out", 1, 2, "New dataset\nold one unchanged", COLORS.built);

export default {
  elements: [
    heading("t", "Flow B: propose, review, apply", 40, 20),
    dcp, propose, set, exportR, person, importR, apply, out,
    arrow(dcp, propose, "reads", F),
    arrow(propose, set, "writes", F),
    arrow(set, exportR, "exports", F),
    arrow(exportR, person, "opens", F),
    arrow(person, importR, "decisions", F),
    arrow(importR, apply, "reviewed set", F),
    arrow(apply, out, "writes", F),
    note("cap", "Nothing changes until a person accepts a box. Only missing boxes are proposed.", 40, 610, 18),
  ],
};
