# DataCopilot

DataCopilot finds and fixes mistakes in computer-vision datasets (object detection and
classification) before you train on them. It converts between COCO, YOLO, Pascal VOC, CVAT and
Label Studio without silently losing information, and it can fail your CI build when a dataset is bad.

It audits a dataset for broken boxes, duplicate and leaked images between train and test, class
imbalance and missing files. Everything runs on your machine from the command line or Python; no
server, and no GPU for the core.

![DataCopilot reads your dataset, converts and audits it, and gives you a clean dataset and a pass or fail for CI](diagrams/png/overview.png)

*A dataset goes in; a clean dataset and a pass or fail come out.*

## I want to try it

- [Install](guide/install.md)
- [Quick start](tutorial.md#1-generate-the-dataset): the first commands, on a generated demo dataset
- [Tutorial](tutorial.md): audit, fail a build and convert, in ten minutes

## I want to understand it

- [Concepts](guide/concepts.md)
- [How it works](architecture/README.md)
- [Formats](guide/formats.md)
- [Audit and CI](guide/audit-ci.md)
- [What is built and what is planned](status.md)

## I want to contribute

- [Contributing](https://github.com/B4teShai/DataCopilot/blob/main/CONTRIBUTING.md)
- [Engineering guide](engineering/README.md)
- [Architecture decisions](adr/README.md)
