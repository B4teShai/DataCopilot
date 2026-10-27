# DataCopilot

DataCopilot finds and fixes mistakes in computer-vision datasets (object detection and classification) before you train on them.

## What is it?

A command-line tool and Python library for people who train detection or classification models.
Point it at a dataset folder and it reports what is wrong: broken or tiny boxes, files that are
missing, classes with almost no examples, and the same picture in both train and test (which makes
your test score look better than it is).

It also converts between COCO, YOLO, Pascal VOC, CVAT and Label Studio, and tells you what a
conversion could not carry. It can make your CI build fail when a dataset is bad.

It can propose a fix for one kind of mistake: boxes your labels missed. A person reviews every
proposal before anything changes. All other mistakes it reports, and you fix them.
It runs on your machine, with no server and no GPU for the core.

![DataCopilot reads your dataset, converts and audits it, and gives you a clean dataset and a pass or fail for CI](https://raw.githubusercontent.com/B4teShai/DataCopilot/main/docs/diagrams/png/overview.png)

*A dataset goes in; a clean dataset and a pass or fail come out.*

## Quick start

Needs Python 3.10 or newer. Install from a clone with [uv](https://docs.astral.sh/uv/) or pip:

```bash
git clone https://github.com/B4teShai/DataCopilot.git && cd DataCopilot
uv sync && source .venv/bin/activate    # or: python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

> **Release status:** not yet published to PyPI, so `pip install datacopilot` does not work; install from a clone as above.

To use it in your own project, pin a commit you have reviewed (a bare URL follows the default branch):
`uv add "git+https://github.com/B4teShai/DataCopilot@<full-commit-sha>"` or
`pip install "datacopilot @ git+https://github.com/B4teShai/DataCopilot@<full-commit-sha>"`.
Optional image models: `uv sync --extra embed` or `pip install -e ".[embed]"`.

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

`ingest` printed the dataset's version, a fingerprint of its content. `audit` printed pass or fail
and the findings per severity.

## What you can do

- Convert formats: `datacopilot ingest`, then `export`.
- Check a dataset: `datacopilot audit` finds broken boxes, missing files and class imbalance.
- Fail a CI build: `--fail-on warn` turns findings into a non-zero exit code.
- Find look-alike images with an image model: `datacopilot audit --embedder dinov2_small`
  (needs the `embed` extra). This catches copies in both train and test.
- Re-split without copies across sides: `datacopilot split`.
- Find boxes your labels missed: `datacopilot propose`, review them in CVAT, then `datacopilot apply`
  (needs the `vlm` extra).
- Use it from Python: `dc.load`, `dc.audit`, `dc.export`, `dc.split`, `dc.propose` and `dc.apply`.

## Learn more

- [Tutorial](https://github.com/B4teShai/DataCopilot/blob/main/docs/tutorial.md): ten minutes, every command runs
- [Terms explained](https://github.com/B4teShai/DataCopilot/blob/main/docs/guide/concepts.md)
- [How it works](https://github.com/B4teShai/DataCopilot/blob/main/docs/architecture/README.md)
- [What works today](https://github.com/B4teShai/DataCopilot/blob/main/docs/status.md)
- [Contributing](https://github.com/B4teShai/DataCopilot/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/B4teShai/DataCopilot/blob/main/CHANGELOG.md)

## License

Apache-2.0. See [LICENSE](https://github.com/B4teShai/DataCopilot/blob/main/LICENSE).
