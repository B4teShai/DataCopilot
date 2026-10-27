# How DataCopilot is built

DataCopilot reads a dataset, converts and audits it, and gives you a clean dataset and a pass or
fail. This section is for people who will change the code. Read [Concepts](../guide/concepts.md)
first if a term is new, and [Status](../status.md) to see what runs today.

![DataCopilot reads your dataset, converts and audits it, and gives you a clean dataset and a pass or fail for CI](../diagrams/png/overview.png)

- You give it a dataset in COCO, YOLO, Pascal VOC, CVAT or Label Studio format.
- It reads the dataset into one internal format, a `dataset.dcp/` folder.
- It writes the dataset back out in any format and counts what a format cannot hold.
- It audits the dataset and writes a report. The exit code says pass or fail.
- Image models are optional. They add the near-duplicate and leakage checks.

| Page | Question it answers | Diagram |
|---|---|---|
| [01 Overview](01-overview.md) | Who uses it, what must it guarantee, what is out of scope? | `context` |
| [02 Components](02-components.md) | Which packages exist and what may import what? | `components` |
| [03 Data model](03-data-model.md) | What is stored, with which keys and rules? | `data-model` |
| [04 Flows](04-flows.md) | How do the main commands run, and how do they fail? | `flow-a`, `flow-b` |
| [05 Runtime](05-runtime.md) | Where does it run, and where do files and settings live? | none |
| [06 Cross-cutting](06-cross-cutting.md) | Logging, errors, security, licenses. | none |
| [Decisions](../adr/README.md) | Why is it this way? | none |

## How to read the diagrams

- They are generated from `docs/diagrams/*.diagram.mjs`. Each PNG embeds its scene, so you can drop
  it into [excalidraw.com](https://excalidraw.com) and edit it.
- Colors: blue is DataCopilot, orange an outside tool, purple a person, yellow stored data.
- Arrows point from the caller to the callee.

## Keeping this current

A change that adds a package, a public interface, a stored entity or an outside dependency updates
the matching page and diagram in the same pull request. The review checklist in
[workflow](../engineering/workflow.md) enforces this.
