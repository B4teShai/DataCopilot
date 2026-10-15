"""Datasets whose embeddings are planted, for the embedding checks.

``scene`` writes one tiny PNG per row (distinct bytes), registers the requested vector for those
bytes in a :class:`FakeEmbedder` and returns the dataset, the embedder and a context whose provider
embeds through it. A row without a vector gets a random unit vector (unrelated to every other).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import SecretStr

from datacopilot.audit import embedding
from datacopilot.audit.context import CheckContext
from datacopilot.core.schema import Dataset
from fixtures.auditdata import Img, build
from fixtures.embedfakes import FakeEmbedder
from fixtures.synth import png_bytes

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"


@dataclass
class Row:
    name: str
    split: str = "train"
    vec: NDArray[np.float32] | None = None
    group: str | None = None
    cat: int | None = 0  # category index for one box, None for no annotation
    cats: tuple[int, ...] = ()  # several boxes (overrides ``cat``)


def scene(
    root: Path, rows: list[Row], *, batch_size: int = 8, **ctx_kw: Any
) -> tuple[Dataset, FakeEmbedder, CheckContext]:
    emb = FakeEmbedder()
    specs: list[Img] = []
    for n, row in enumerate(rows):
        data = png_bytes(64, 48, 1000 + n)
        if row.vec is not None:
            emb.plant(data, row.vec)
        cats = row.cats or (() if row.cat is None else (row.cat,))
        boxes = [(c, 1.0 + k, 1.0, 30.0 + k, 30.0) for k, c in enumerate(cats)]
        specs.append(Img(row.name, 64, 48, row.split, boxes=boxes, seed=1000 + n, group=row.group))
    ds = build(root, specs)
    (root / "ds.dcp").mkdir(exist_ok=True)
    base: dict[str, Any] = dict(
        run_id=RUN,
        root=root,
        cache_dir=root / "cache",
        dataset_dir=root / "ds.dcp",
        workers=2,
        embedder="fake",
        embeddings=embedding.PipelineProvider(emb, batch_size=batch_size),
        cache_key=SecretStr("k" * 32),
    )
    base.update(ctx_kw)
    return ds, emb, CheckContext(**base)
