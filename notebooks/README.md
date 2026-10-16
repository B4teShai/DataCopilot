# Notebooks: learn DataCopilot on real data
- `01_tutorial.ipynb`: a story in 11 steps (install check, ingest, audit, conversions, CI gates, a real DINOv2-small look-alike search, a planted leak, the Python API), every result explained in plain English.
- `02_edge_cases.ipynb`: 25 broken copies of the same photos (corrupt files, unsafe paths, bad boxes, an XML bomb, private data, an unwritable folder, a blocked model licence) with a table of expected against actual results.
- Data: COCO128 (128 COCO photos) is downloaded at run time into `$TMPDIR/datacopilot-notebooks` (change it with `DATACOPILOT_NOTEBOOK_DATA`); nothing is committed. COCO annotations are CC BY 4.0; each photo keeps its Flickr licence.
- Model: DINOv2-small (Apache-2.0) through the `[embed]` extra, about 90 MB, pinned by the model registry, run on the CPU. The network is needed once.
- Run (about a minute each; `--with` uses a throwaway environment, so `pyproject.toml` and `uv.lock` stay untouched):
```bash
uv run --extra embed --with jupyterlab --with ipykernel --with matplotlib jupyter lab notebooks/
```
- Re-execute one in place: use `jupyter nbconvert --to notebook --execute --inplace notebooks/01_tutorial.ipynb` in place of `jupyter lab notebooks/`. The saved outputs are real; `tests/notebooks/test_notebooks.py` checks them without a network or a model.
