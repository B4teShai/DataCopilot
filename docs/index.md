# DataCopilot

DataCopilot checks computer-vision datasets (object detection and classification) for mistakes
before you train on them. It converts between COCO, YOLO, Pascal VOC, CVAT and Label Studio, audits
a dataset, can fail your CI build, and can propose boxes your labels missed for a person to review.

![DataCopilot reads your dataset, converts and audits it, and gives you a clean dataset and a pass or fail for CI](diagrams/png/overview.png)

*A dataset goes in; a clean dataset and a pass or fail come out.*

## Start here

- [Install](guide/install.md), then the [tutorial](tutorial.md): ten minutes on a generated demo dataset.
- Unknown word? See [Concepts](guide/concepts.md).
- What works today: [Status](status.md).
- How it is built: [Architecture](architecture/README.md).
- Contribute: [Contributing](https://github.com/B4teShai/DataCopilot/blob/main/CONTRIBUTING.md) and the [engineering guide](engineering/README.md).
