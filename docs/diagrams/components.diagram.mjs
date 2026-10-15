// C4 level 3: packages by layer (ADR-0004). Arrows point downward only:
// interface -> application -> domain / infrastructure (through registries).
import { grid, lane, arrow, elbow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { component, statusLegend } from "../../../tools/diagrams/lib/c4.mjs";

const W = 190, H = 80, GX = 50, GY = 50;
const X0 = 180; // leaves room for the lane titles left of column 0
const g = grid({ x: X0, y: 110, colW: W, rowH: H, gapX: GX, gapY: GY });
const p = (c, r, span = 1) => { const q = g(c, r, span); return [q.x, q.y, q.w, q.h]; };
const FULL = 6 * W + 5 * GX; // six columns
const laneX = 20, laneW = FULL + X0 + 20;
const laneY = (r) => g(0, r).y - 24;

// Vertical arrow from a's bottom to b's top at the x of the narrower box (computed, not hand-placed).
const vert = (a, b, text) => {
  const x = (a.width <= b.width ? a.x + a.width / 2 : b.x + b.width / 2);
  const y0 = a.y + a.height, y1 = b.y;
  return {
    type: "arrow", x, y: y0, width: 0, height: y1 - y0, points: [[0, 0], [0, y1 - y0]],
    start: { id: a.id }, end: { id: b.id }, endArrowhead: "arrow", strokeWidth: 1,
    ...(text ? { label: { text, fontSize: 14 } } : {}), _kind: "arrow",
  };
};

// Lanes, top to bottom.
const lInterface = lane("l_if", laneX, laneY(0), laneW, 250, "Interface");
const lApp = lane("l_app", laneX, laneY(2), laneW, 120, "Application");
const lDomain = lane("l_dom", laneX, laneY(3), laneW, 120, "Domain");
const lInfra = lane("l_inf", laneX, laneY(4), laneW, 250, "Infrastructure");
const lSide = lane("l_side", laneX + laneW + 30, laneY(0), 270, 770, "Cross-cutting, plugins");

// Interface.
const cli = component("cli", ...p(0, 0, 6), "cli/", "Typer commands. Built: doctor, ingest, export, formats, audit. Planned: split, propose, review, apply", true);
const api = component("api", ...p(0, 1, 6), "api.py", "public Python API: load, audit, split, propose, apply, export");

// Application (integer columns).
const convert = component("convert", ...p(0, 2), "convert.py", "ingest / export use case", true);
const audit = component("audit", ...p(1, 2), "audit/", "runner, gate, 4 checks", true);
const propose = component("propose", ...p(2, 2), "propose/", "proposers, consensus");
const patch = component("patch", ...p(3, 2), "patch/", "review export, apply");
const split = component("split", ...p(4, 2), "split/", "group-aware re-split");
const report = component("report", ...p(5, 2), "report/", "HTML, JSON, SARIF", true);

// Domain (half columns so vertical arrows pass between them).
const schema = component("schema", ...p(0.5, 3), "core/schema.py", "entities, invariants", true);
const geometry = component("geometry", ...p(1.5, 3), "core/geometry.py", "box conversions, clip, iou_xyxy", true);
const version = component("version", ...p(3.5, 3), "core/version.py", "content hash, lineage", true);

// Infrastructure row 1 (half columns) and the store spanning all columns.
const io = component("io", ...p(0.5, 4), "io/", "4 formats + registry", true);
const embed = component("embed", ...p(1.5, 4), "embed/", "backends, cache, FAISS");
const models = component("models", ...p(2.5, 4), "models/", "registry, licenses");
const integrations = component("integrations", ...p(3.5, 4), "integrations/", "fiftyone, cvat, cleanlab, lightly");
const store = component("store", ...p(0, 5, 6), "store/", "Parquet tables, manifest, atomic versions, audit runs built; DuckDB, embeddings, proposals planned", true);

// Side lane: cross-cutting modules (built) and plugin entry-point groups.
const sx = laneX + laneW + 30 + 25, sw = 220, sh = 56;
const logging = component("logging", sx, 120, sw, sh, "logging.py", "structlog JSON", true);
const settings = component("settings", sx, 190, sw, sh, "settings.py", "DATACOPILOT_* env", true);
const pFormats = component("p_formats", sx, 330, sw, sh, "datacopilot.formats", "-> io/ registry", true);
const pChecks = component("p_checks", sx, 400, sw, sh, "datacopilot.checks", "-> audit/ registry", true);
const pProposers = component("p_proposers", sx, 470, sw, sh, "datacopilot.proposers", "-> propose/ registry");
const pEmbedders = component("p_embedders", sx, 540, sw, sh, "datacopilot.embedders", "-> embed/ registry");

const useCases = [convert, audit, propose, patch, split, report];

export default {
  elements: [
    heading("t", "DataCopilot: components by layer (C4 L3)", 40, 20),
    lInterface, lApp, lDomain, lInfra, lSide,
    cli, api, ...useCases,
    schema, geometry, version, io, embed, models, integrations, store,
    logging, settings, pFormats, pChecks, pProposers, pEmbedders,
    note("n_side", "entry-point groups\n(PluginSpec: name, license,\nextra, gpu_memory_gb)", sx, 262, 13),
    // interface -> application
    arrow(cli, api, "calls"),
    ...useCases.map((u) => vert(api, u)),
    // application -> domain (stubs into the adjacent domain box)
    elbow(convert, schema), elbow(audit, schema), elbow(audit, geometry), elbow(propose, geometry),
    elbow(patch, version), elbow(split, version),
    // application -> infrastructure, through registries only
    elbow(convert, io), elbow(audit, embed), elbow(propose, embed), elbow(propose, models), elbow(patch, integrations),
    ...useCases.map((u) => vert(u, store)),
    statusLegend(40, 890),
    note("n1", "Vertical lines are imports: api.py -> every use case; use cases -> store/ and the adjacent domain / infrastructure box.\nEvery application and infrastructure package imports core/schema.py (infrastructure -> domain is not drawn: arrows point downward only).\nApplication reaches infrastructure through registries keyed by name. Cross-cutting modules import no layer.", 40, 930, 14),
  ],
};
