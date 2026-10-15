// Deployment view: where DataCopilot runs and what crosses machine boundaries.
import { grid, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { container, store, external, node, c4Legend, statusLegend } from "../../../tools/diagrams/lib/c4.mjs";

const W = 220, H = 80, GX = 180, GY = 40;
const g = grid({ x: 60, y: 110, colW: W, rowH: H, gapX: GX, gapY: GY });
const p = (c, r) => [g(c, r).x, g(c, r).y, W, H];
const nodeBox = (id, c, r, cols, rows, title) =>
  node(id, g(c, r).x - 20, g(c, r).y - 36, cols * W + (cols - 1) * GX + 40, rows * H + (rows - 1) * GY + 56, title);

// Developer laptop (CPU only).
const laptop = nodeBox("n_laptop", 0, 0, 2, 2, "Developer laptop (CPU, Python >= 3.10)");
const cli = container("cli", ...p(0, 0), "datacopilot CLI", "ingest, export, audit built", true);
const core = container("core", ...p(1, 0), "core + audit", "CPU checks, no torch", true);
const reportFiles = container("report", ...p(0, 1), "report files", "report.html / .json / .sarif", true);
const dcpLocal = store("dcp_local", ...p(1, 1), "dataset.dcp/", "Parquet + manifest", true);

// GPU machine (optional extras).
const gpu = nodeBox("n_gpu", 3, 0, 2, 2, "GPU machine (extras; 8-24 GB VRAM)");
const extras = container("extras", ...p(3, 0), "datacopilot[extras]", "[embed] [vlm] [sam3]");
const hfCache = store("hf_cache", ...p(4, 0), "model cache", "weights, embeddings");
const dcpGpu = store("dcp_gpu", ...p(3, 1), "dataset.dcp/", "same layout; adds embeddings/");
const runtime = container("runtime", ...p(4, 1), "model runtime", "torch, transformers");

// CI runner.
const ci = nodeBox("n_ci", 0, 3, 2, 1, "CI runner (CPU; audit as a gate)");
const ciAudit = container("ci_audit", ...p(0, 3), "datacopilot audit", "--fail-on critical --no-hash-cache", true);
const ciOut = container("ci_out", ...p(1, 3), "exit code + SARIF", "uploaded to code scanning", true);

// Services and the sibling project.
const cvpilot = external("cvpilot", ...p(2, 0), "CV-Pilot", "reads manifest.version");
const objstore = external("objstore", ...p(2, 1), "Object storage", "S3 / GCS / Azure (fsspec)");
const hub = external("hub", ...p(5, 0), "Hugging Face Hub", "gated repos; token via env");
const vllm = external("vllm", ...p(4, 3), "vLLM endpoint", "Qwen3-VL over HTTP (optional)");

export default {
  elements: [
    heading("t", "DataCopilot: runtime and deployment", 40, 20),
    laptop, gpu, ci,
    cli, core, reportFiles, dcpLocal, extras, hfCache, dcpGpu, runtime, ciAudit, ciOut,
    cvpilot, objstore, hub, vllm,
    arrow(cli, core, "in-process"),
    arrow(core, dcpLocal, "reads / writes"),
    arrow(cli, reportFiles, "writes"),
    arrow(dcpLocal, objstore, "images by URI; sync"),
    arrow(dcpGpu, objstore, "images by URI; sync"),
    arrow(dcpLocal, cvpilot, "manifest.version"),
    arrow(extras, runtime, "in-process"),
    arrow(runtime, dcpGpu, "embeddings, proposals"),
    arrow(hfCache, hub, "weights (HTTPS)"),
    arrow(runtime, vllm, "HTTP"),
    arrow(ciAudit, objstore, "dcp + images, read-only"),
    arrow(ciAudit, ciOut, "in-process"),
    c4Legend(40, 650),
    statusLegend(40, 690),
    note("n1", "Everything runs in one Python process per command; no daemon, no database server.\nEnv: DATACOPILOT_ENV, LOG_LEVEL, MAX_TABLE_*, WORKERS, CACHE_DIR, HASH_CACHE, ALLOWED_LICENSES, CACHE_KEY (built); HF token (planned, env only, never logged).", 40, 730, 14),
  ],
};
