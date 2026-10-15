# DataCopilot

Headless dataset doctor for detection and classification datasets: lossless format conversion, an
audit that can gate CI, and (planned) model-generated fix proposals that a person reviews before
they touch a label.

**Built and tested in 0.1:** `ingest` and `export` between COCO, YOLO, Pascal VOC, CVAT for images
1.1 XML, Label Studio JSON and ImageFolder, with a count of every field a conversion cannot carry;
`audit` with eight checks (integrity, exact duplicates, class imbalance, box sanity, and with an
embedder near-duplicates, split leakage and outliers) and reports in HTML, JSON and SARIF; `embed`;
`doctor`; and a three-function Python API. Label noise, split, propose, review and apply are not
implemented yet: see the [architecture](architecture/README.md) for what is planned.

## Where to start

- [Tutorial](tutorial.md): audit, gate and convert a synthetic dataset in ten minutes. Every
  command on the page is executed by the test suite.
- [Install](guide/install.md), [Concepts](guide/concepts.md), [Formats](guide/formats.md).
- [Audit in CI](guide/audit-ci.md), [Embeddings](guide/embeddings.md).
- [Trust model](guide/trust-model.md): what happens with files you do not control.
- [Python API](guide/python-api.md) and [CLI reference](reference/cli.md).
- Design documents: [architecture](architecture/README.md), [engineering guide](engineering/README.md),
  [decision records](adr/README.md).

DataCopilot is Apache-2.0 licensed; see `LICENSE` and `NOTICE` in the repository.
