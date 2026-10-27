// The four layers and the dependency rule: imports point down only.
import { arrow, box, lane, heading, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const F = { fontSize: 18 };
const L = (id, y, title) => lane(id, 20, y, 960, 130, title);

const interfaceLane = L("l_if", 70, "Interface");
const appLane = L("l_app", 250, "Application");
const infraLane = L("l_inf", 430, "Infrastructure");
const domainLane = L("l_dom", 610, "Domain");

const cli = box("cli", 300, 105, 400, 80, "cli/ and api.py\ncommands and the Python API", COLORS.core, F);
const convert = box("convert", 45, 285, 200, 80, "convert.py\ningest, export", COLORS.core, F);
const audit = box("audit", 275, 285, 200, 80, "audit/\nchecks and the gate", COLORS.core, F);
const flows = box("flows", 505, 285, 240, 80, "splits/, proposals/\nsplit, propose, apply", COLORS.core, F);
const report = box("report", 775, 285, 180, 80, "report/\nHTML, JSON, SARIF", COLORS.core, F);
const io = box("io", 45, 465, 200, 80, "io/\nsix formats", COLORS.store, F);
const store = box("store", 275, 465, 200, 80, "store/\ndataset.dcp tables", COLORS.store, F);
const embed = box("embed", 505, 465, 200, 80, "embed/\nembedders, kNN", COLORS.store, F);
const models = box("models", 735, 465, 220, 80, "models/, proposers/\nregistry, detector", COLORS.store, F);
const core = box("core", 250, 645, 500, 80, "core/\nschema, version, geometry, errors", COLORS.store, F);

export default {
  elements: [
    heading("t", "DataCopilot: four layers, imports point down", 20, 20),
    interfaceLane, appLane, infraLane, domainLane,
    cli, convert, audit, flows, report, io, store, embed, models, core,
    arrow(interfaceLane[0], appLane[0], "calls", { fontSize: 18 }),
    arrow(appLane[0], infraLane[0], "by name, through registries", { fontSize: 18 }),
    arrow(infraLane[0], domainLane[0], "use", { fontSize: 18 }),
    note("cap", "logging.py and settings.py sit beside the layers and import none of them.", 20, 780, 18),
  ],
};
