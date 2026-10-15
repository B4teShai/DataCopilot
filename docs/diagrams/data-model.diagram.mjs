// Entities stored in dataset.dcp/ and their relations. PK/FK in the field text; cardinality on arrows.
import { grid, entity, arrow, heading, note, legend, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const W = 300;
const g = grid({ x: 40, y: 110, colW: W, rowH: 310, gapX: 100, gapY: 50 });
const BUILT = new Set(["category", "manifest", "annotation", "image", "audit_run", "finding"]);
const E = (id, c, r, title, fields) =>
  entity(id, g(c, r).x, g(c, r).y, W, title, fields, BUILT.has(id) ? COLORS.built : COLORS.planned);

const category = E("category", 0, 0, "Category", [
  "id: int  PK",
  "name: str  UNIQUE",
  "aliases: list[str]",
  "attributes: dict[str, str]",
]);
const manifest = E("manifest", 1, 0, "DatasetManifest", [
  "version: str  PK (content hash)",
  "name: str",
  "schema_version: int",
  "task: detection|classification",
  "categories: list[Category]",
  "lineage: list[str]  FK->version",
  "image_count, annotation_count: int",
  "created_at: datetime",
]);
const auditRun = E("audit_run", 2, 0, "AuditRun", [
  "run_id: str  PK",
  "dataset_version: str  FK->Manifest",
  "started_at: datetime",
  "checks: list[str]",
  "embedder: str?",
  "counts: {critical, warn, info}",
  "passed: bool",
  "fail_on, timings_ms, summary",
]);
const annotation = E("annotation", 0, 1, "Annotation", [
  "ann_id: str  PK",
  "image_id: str  FK->Image",
  "category_id: int  FK->Category",
  "kind: bbox|classification",
  "bbox_xyxy_abs: float[4]?",
  "score: float?",
  "is_crowd: bool",
  "occluded: bool?",
  "truncated: bool?",
  "attributes: dict[str, scalar]",
  "provenance: human|model|",
  "            proposal_accepted",
  "source: dict  (verbatim)",
]);
const image = E("image", 1, 1, "Image", [
  "image_id: str  PK  sha256[:16]",
  "file_path: str  (path or URI)",
  "width: int",
  "height: int",
  "split: train|val|test|unassigned",
  "group_id: str?",
  "source: dict  (verbatim)",
]);
const finding = E("finding", 2, 1, "Finding", [
  "finding_id: str  PK",
  "run_id: str  FK->AuditRun",
  "check: str",
  "severity: CRITICAL|WARN|INFO",
  "image_id: str?  FK->Image",
  "ann_id: str?  FK->Annotation",
  "payload: dict",
]);
const proposal = E("proposal", 0, 2, "Proposal", [
  "proposal_id: str  PK",
  "set_id: str",
  "kind: add_box|change_class|",
  "      refine_box|delete_box|",
  "      flag_ambiguous",
  "target_ann_id: str?  FK->Annotation",
  "image_id: str  FK->Image",
  "payload: dict",
  "model_id: str, model_revision: str",
  "weight_hash: str, prompt_hash: str",
  "score: float",
  "evidence_crop_uri: str?",
  "status: pending|accepted|rejected",
  "reviewer_id: str?",
]);
const embedding = E("embedding", 1, 2, "Embedding", [
  "image_id: str  PK, FK->Image",
  "model_id: str  PK",
  "model_revision: str  PK",
  "vector: float32[dims]",
  "dims: int",
]);

export default {
  elements: [
    heading("t", "DataCopilot: data model (dataset.dcp/)", 40, 20),
    category, manifest, auditRun, annotation, image, finding, proposal, embedding,
    arrow(manifest.box, category.box, "1..*"),
    arrow(manifest.box, image.box, "1..*"),
    arrow(auditRun.box, manifest.box, "1 version"),
    arrow(auditRun.box, finding.box, "0..*"),
    arrow(category.box, annotation.box, "0..*"),
    arrow(image.box, annotation.box, "0..*"),
    arrow(image.box, embedding.box, "0..* per model"),
    arrow(finding.box, image.box, "0..1"),
    arrow(proposal.box, annotation.box, "0..1 target"),
    legend(40, 1250, [[COLORS.built, "entity (built)"], [COLORS.planned, "entity (planned)"]]),
    note("n1", "PK primary key. FK foreign key. ? nullable. Not drawn: Finding.ann_id -> Annotation (0..1), Proposal.image_id -> Image (1).", 40, 1290, 14),
  ],
};
