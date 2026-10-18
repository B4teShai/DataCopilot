# DataCopilot

DataCopilot finds and fixes mistakes in computer-vision datasets (object detection and classification) before you train on them.

## What is it?

Wrong boxes, the same image in both train and test, and format conversions that quietly lose
data all ruin models, and nobody notices until the scores look strange. DataCopilot converts between
COCO, YOLO, Pascal VOC, CVAT and Label Studio and tells you what, if anything, a conversion could
not carry. It audits a dataset for broken boxes, duplicate and leaked images, class imbalance and
missing files, and it can fail your CI build when a dataset is bad. It runs on your machine from
the command line or Python: no server, and no GPU for the core.

![DataCopilot reads your dataset, converts and audits it, and gives you a clean dataset and a pass or fail for CI](https://raw.githubusercontent.com/B4teShai/DataCopilot/main/docs/diagrams/png/overview.png)

*A dataset goes in; a clean dataset and a pass or fail come out.*

## Quick start

Needs Python 3.10 or newer.

```bash
pip install datacopilot
```

> **Release status:** not yet published to PyPI; until then see the [release runbook](https://github.com/B4teShai/DataCopilot/blob/main/docs/engineering/release.md).

The [tutorial](https://github.com/B4teShai/DataCopilot/blob/main/docs/tutorial.md) generates a small demo dataset, `shop/`, with planted mistakes. Then:

```console
$ datacopilot ingest shop --format coco -o shop.dcp          # read a COCO dataset
9e96d9fb67b1d64fafe14c25ccdb43f0
$ datacopilot export shop.dcp --format yolo -o shop_yolo     # convert it to YOLO
shop_yolo
$ datacopilot audit shop.dcp                                 # check it: passes, with warnings
passed critical=0 warn=3 info=0 run_id=a57fdb53-ec2f-4652-8d69-a860e1970362
$ datacopilot audit shop.dcp --fail-on warn                  # stricter: exit 1, so CI fails
failed critical=0 warn=3 info=0 run_id=571c0043-41af-4ca4-b256-34bc66f25a61
```

## What you can do

- Convert formats: `datacopilot ingest` then `export` (COCO, YOLO, VOC, CVAT, Label Studio, folders of classes).
- Check a dataset: `datacopilot audit` finds broken boxes, missing files and class imbalance.
- Catch duplicates and leaks: identical images, and train images copied into test.
- Use it in CI: `--fail-on warn` turns findings into a non-zero exit code.
- Find look-alike images with image models: `datacopilot audit --embedder` (optional, `pip install "datacopilot[embed]"`).
- Spot boxes with a doubtful class: `label_noise` leads over the same embeddings (weak; see the guide).
- Find boxes your labels missed: `datacopilot propose`, then `review` in CVAT and `apply` (optional, `pip install "datacopilot[vlm]"`).
- Use it from Python: `dc.load`, `dc.audit`, `dc.export`, `dc.propose` and `dc.apply`.

## Learn more

- [Tutorial](https://github.com/B4teShai/DataCopilot/blob/main/docs/tutorial.md): ten minutes, every command runs
- [User guide](https://github.com/B4teShai/DataCopilot/tree/main/docs/guide)
- [How it works](https://github.com/B4teShai/DataCopilot/blob/main/docs/architecture/README.md)
- [Settings](https://github.com/B4teShai/DataCopilot/blob/main/docs/reference/settings.md)
- [What is built and what is planned](https://github.com/B4teShai/DataCopilot/blob/main/docs/status.md)
- [Contributing](https://github.com/B4teShai/DataCopilot/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/B4teShai/DataCopilot/blob/main/CHANGELOG.md)

## License

Apache-2.0. See [LICENSE](https://github.com/B4teShai/DataCopilot/blob/main/LICENSE).
