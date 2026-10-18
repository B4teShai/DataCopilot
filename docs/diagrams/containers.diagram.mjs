// C4 level 2: installable units inside the DataCopilot distribution.
import { grid, actor, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { container, store, external, boundary, c4Legend, statusLegend } from "../../../tools/diagrams/lib/c4.mjs";

const W = 240, H = 100;
const g = grid({ x: 360, y: 110, colW: W, rowH: H, gapX: 120, gapY: 80 });
const p = (c, r) => [g(c, r).x, g(c, r).y, W, H];

// Person above the boundary.
const engineer = actor("engineer", g(0, 0).x, g(0, 0).y, "CV engineer", W);

// Inside the boundary (cols 0-2, rows 1-3).
const api = container("api", ...p(0, 1), "Python API", "load, audit, export, propose, apply built", true);
const cli = container("cli", ...p(1, 1), "datacopilot CLI", "Typer; ingest, export, audit, embed, propose, review, apply built", true);
const plugins = container("plugins", ...p(2, 1), "Plugin host", "formats, checks, embedders, proposers built", true);
const adapters = container("adapters", ...p(0, 2), "Format adapters", "io/: 6 formats built; integrations/ planned", true);
const audit = container("audit", ...p(1, 2), "Audit engine", "checks, gate built", true);
const runtime = container("runtime", ...p(2, 2), "Model runtime", "embedders, Grounding DINO built", true);
const core = container("core", ...p(0, 3), "Core library", "schema, content hash, lineage, geometry", true);
const dcp = store("dcp", ...p(1, 3), "Dataset store", "dataset.dcp/: Parquet, manifest, audits", true);
const report = container("report", ...p(2, 3), "Report generator", "HTML, JSON, SARIF built", true);

const b = g(0, 1);
const box_ = boundary("sysb", b.x - 20, b.y - 36, 3 * W + 2 * 120 + 40, 3 * H + 2 * 80 + 56, "DataCopilot distribution");

// Externals around the boundary.
const exports_ = external("exports", ...p(-1, 1), "Raw dataset exports", "COCO, YOLO, VOC, CVAT XML, Label Studio, ImageFolder");
const tools = external("tools", ...p(-1, 2), "Annotation tools", "CVAT, FiftyOne, X-AnyLabeling");
const cvpilot = external("cvpilot", ...p(-1, 3), "CV-Pilot", "records manifest.version as dataset_hash");
const pkgs = external("pkgs", ...p(3, 1), "Plugin packages", "e.g. datacopilot-yoloe (AGPL, separate)");
const hub = external("hub", ...p(3, 2), "Model hubs", "Hugging Face Hub, vLLM endpoint");
const objstore = external("objstore", ...p(1, 4), "Object storage", "S3 / GCS / Azure via fsspec");
const ci = external("ci", ...p(2, 4), "CI runner", "consumes exit code + SARIF");

export default {
  elements: [
    heading("t", "DataCopilot: containers (C4 L2)", 40, 20),
    box_, engineer, api, cli, plugins, adapters, audit, runtime, core, dcp, report,
    exports_, tools, cvpilot, pkgs, hub, objstore, ci,
    arrow(engineer[0], cli, "runs"),
    arrow(engineer[0], api, "imports"),
    arrow(cli, api, "calls"),
    arrow(api, adapters, "load, export"),
    arrow(api, audit, "audit, propose, apply"),
    arrow(adapters, core, "maps to schema"),
    arrow(adapters, dcp, "writes tables"),
    arrow(adapters, exports_, "reads, writes files"),
    arrow(adapters, tools, "proposals / decisions"),
    arrow(audit, dcp, "reads tables"),
    arrow(audit, runtime, "embed, infer"),
    arrow(audit, report, "findings"),
    arrow(plugins, runtime, "registers"),
    arrow(plugins, pkgs, "entry points"),
    arrow(runtime, hub, "weights, HTTP"),
    arrow(core, cvpilot, "manifest.version"),
    arrow(dcp, objstore, "image URIs"),
    arrow(report, ci, "exit code, SARIF"),
    c4Legend(40, 880),
    statusLegend(40, 920),
    note("n1", "Built: ingest, export, audit and doctor (CLI, core library, six format adapters, registries, dataset store, four CPU checks and four embedding checks, DINOv2 and SigLIP2 embedders, HTML/JSON/SARIF reports), the load, audit and export API, settings, logging. Propose, review and apply (Grounding DINO, add_box only) are built; split is planned.", 40, 960),
  ],
};
