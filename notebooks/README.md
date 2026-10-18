# Notebooks: learn DataCopilot on real data
- `01_tutorial.ipynb`: a story in 11 steps (install check, ingest, audit, conversions, CI gates, a real DINOv2-small look-alike search, a planted leak, the Python API), every result explained in plain English.
- `02_edge_cases.ipynb`: 26 broken copies of the same photos (corrupt files, unsafe paths, bad boxes, an XML bomb, private data, an unwritable folder, a blocked model licence) with a table of expected against actual results.
- `03_propose_review_apply.ipynb`: the real Grounding DINO model (extra `[vlm]`, about 690 MB once) finds boxes deleted from COCO128 (a seeded answer key), a simulated CVAT review, `apply`, honest precision and recall, a threshold sweep (tuned on the same data), six drawn mistakes and seven refusals; about 20 minutes on a CPU.
- `04_label_noise.ipynb`: the `label_noise` check on COCO128 with seeded planted class mistakes and the real DINOv2-small embeddings: precision, recall and false-positive rate against the planted answer key, an independent re-derivation of the rule, six drawn failures and a computed verdict (it is weak here, and the notebook says so); about a minute on a CPU.
- Data: COCO128 (128 COCO photos) is downloaded at run time into `$TMPDIR/datacopilot-notebooks` (change it with `DATACOPILOT_NOTEBOOK_DATA`); nothing is committed. COCO annotations are CC BY 4.0; each photo keeps its Flickr licence. Models: DINOv2-small (`[embed]`, about 90 MB) and Grounding DINO tiny (`[vlm]`, about 690 MB), both Apache-2.0, pinned by the model registry, run on the CPU. The network is needed once.
- Run (a minute each for 01 and 02; `--with` uses a throwaway environment, so `pyproject.toml` and `uv.lock` stay untouched; use `--extra vlm` for 03):
```bash
uv run --extra embed --with jupyterlab --with ipykernel --with matplotlib jupyter lab notebooks/
```
- Re-execute one in place: use `jupyter nbconvert --to notebook --execute --inplace notebooks/01_tutorial.ipynb` in place of `jupyter lab notebooks/`. The saved outputs are real; `tests/notebooks/test_notebooks.py` checks them without a network or a model.
