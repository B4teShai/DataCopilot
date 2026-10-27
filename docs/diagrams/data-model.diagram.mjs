// What dataset.dcp/ stores and how the pieces relate. Names and key fields only; the page has the rest.
import { arrow, box, heading, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const F = { fontSize: 18 };
const col = (c) => 40 + c * 380;
const row = (r) => 90 + r * 190;
const E = (id, c, r, text, bg = COLORS.store) => box(id, col(c), row(r), 260, 110, text, bg, F);

const category = E("category", 0, 0, "Category\nid, name, aliases");
const manifest = E("manifest", 1, 0, "DatasetManifest\nversion, task, lineage", COLORS.core);
const auditRun = E("run", 2, 0, "AuditRun\nrun_id, dataset_version");
const annotation = E("annotation", 0, 1, "Annotation\nann_id, box, class");
const image = E("image", 1, 1, "Image\nimage_id, file_path");
const finding = E("finding", 2, 1, "Finding\ncheck, severity");
const embedding = E("embedding", 1, 2, "Embedding\nimage_id, vector");
const proposals = E("proposals", 2, 2, "ProposalSet\nset_id, proposals");

export default {
  elements: [
    heading("t", "What dataset.dcp/ stores", 40, 20),
    category, manifest, auditRun, annotation, image, finding, embedding, proposals,
    arrow(manifest, category, "has", F),
    arrow(manifest, image, "has", F),
    arrow(auditRun, manifest, "audits", F),
    arrow(auditRun, finding, "has", F),
    arrow(category, annotation, "names", F),
    arrow(image, annotation, "has", F),
    arrow(image, embedding, "has", F),
    arrow(finding, image, "points at", F),
    arrow(proposals, image, "boxes for", F),
    note("cap", "Each Image has 0 or more Annotations. Fields and keys are on the page.", 40, 620, 18),
  ],
};
