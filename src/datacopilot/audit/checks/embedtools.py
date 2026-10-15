"""Shared pieces of the embedding checks: the fail-closed prologue, clusters and member lists.

Every embedding check starts with :func:`embedding_set`. It returns the shared
:class:`~datacopilot.audit.embedding.EmbeddingSet` plus the findings that describe how complete it
is: a missing embedder, an embedder error, images without an embedding, an incomplete run and a
distrusted cache are all explicit findings of the check, so that ``--fail-on <check>`` and the
severity gates see them whatever else the check goes on to find. Nothing here can make a check
return "no findings" because embedding failed.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from pydantic import JsonValue

from datacopilot.audit.context import CheckContext
from datacopilot.audit.embedding import EmbeddingSet
from datacopilot.audit.findings import make_finding
from datacopilot.audit.paths import display_name, unchecked_severity
from datacopilot.core.schema import Dataset, Finding, Image

MAX_MEMBERS_LISTED = 20
"""Images named in one cluster finding; ``count`` always carries the true size."""
GROUP_ID_HEX = 12
MAX_LABEL_CHARS = 64
REAL_SPLITS = ("train", "val", "test")


def short_digest(parts: Iterable[str]) -> str:
    """12-hex digest of ``parts`` (a stable short id for a cluster or a group)."""
    return hashlib.sha256("\n".join(parts).encode("utf-8", "surrogatepass")).hexdigest()[
        :GROUP_ID_HEX
    ]


def safe_label(value: str, limit: int = MAX_LABEL_CHARS) -> str:
    """``value`` cut to ``limit`` characters with control, bidirectional and format characters
    replaced, so user-chosen text (a group id) is safe to show anywhere."""
    out = []
    for char in value[: limit + 1]:
        bad = unicodedata.category(char) in ("Cc", "Cf", "Cs", "Zl", "Zp") or not char.isprintable()
        out.append("?" if bad else char)
    text = "".join(out)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def spans_splits(splits: Iterable[str]) -> bool:
    """True when images with these splits cross a split boundary.

    Same rule as the exact-duplicate check: two real splits, or a real split next to
    ``unassigned``, which is not a split but would be assigned to one later.
    """
    seen = set(splits)
    real = seen.intersection(REAL_SPLITS)
    return len(real) > 1 or (bool(real) and "unassigned" in seen)


def members_payload(members: Sequence[Image]) -> list[JsonValue]:
    """Up to ``MAX_MEMBERS_LISTED`` members as short ids, splits and base names only."""
    out: list[JsonValue] = []
    for i in members[:MAX_MEMBERS_LISTED]:
        out.append(
            {"image_id": i.image_id, "split": i.split, "file_name": display_name(i.file_path)}
        )
    return out


def split_counts(members: Iterable[Image]) -> dict[str, JsonValue]:
    counts = Counter(i.split for i in members)
    return {split: counts[split] for split in sorted(counts)}


def components(pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components (size >= 2) of the graph whose edges are ``pairs``.

    Iterative union-find with path halving; components and their members are sorted, so the
    result depends only on the edge set.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    grouped: dict[int, list[int]] = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(node)
    return sorted((sorted(g) for g in grouped.values()), key=lambda g: g[0])


def incomplete_finding(
    ctx: CheckContext, name: str, reason: str, checked: int, total: int
) -> Finding:
    return make_finding(
        ctx.run_id,
        name,
        "CRITICAL" if ctx.incomplete_is_critical else "WARN",
        payload={"rule": "audit_incomplete", "reason": reason, "checked": checked, "total": total},
    )


def embedding_set(
    ds: Dataset, ctx: CheckContext, name: str
) -> tuple[EmbeddingSet | None, list[Finding]]:
    """The embeddings of ``ds`` and the findings that qualify them (see the module docstring)."""
    provider = ctx.embeddings
    if provider is None:
        return None, [
            make_finding(ctx.run_id, name, "CRITICAL", payload={"rule": "embedder_missing"})
        ]
    es = provider.get(ds, ctx)
    findings: list[Finding] = []
    if es.error is not None:
        findings.append(
            make_finding(
                ctx.run_id,
                name,
                "CRITICAL",
                payload={
                    "rule": "embedder_error",
                    "error_type": es.error,
                    "embedded": len(es.ids),
                    "total": es.total,
                },
            )
        )
    if es.missing:
        findings.append(
            make_finding(
                ctx.run_id,
                name,
                unchecked_severity(
                    es.missing_count,
                    es.total,
                    max_fraction=ctx.config.max_unchecked_fraction,
                    require_local=ctx.require_local_images,
                ),
                payload={
                    "rule": "images_not_embedded",
                    "count": es.missing_count,
                    "total": es.total,
                    "reasons": {r: len(v) for r, v in sorted(es.missing.items())},
                },
            )
        )
    if es.incomplete is not None:
        findings.append(
            incomplete_finding(ctx, name, es.incomplete[0], es.incomplete[1], es.incomplete[2])
        )
    if es.cache_rejected:
        findings.append(
            make_finding(
                ctx.run_id,
                name,
                "WARN",
                payload={
                    "rule": "embedding_cache_rejected",
                    "state": es.cache_state,
                    "reasons": dict(sorted(es.cache_rejected.items())),
                },
            )
        )
    if es.spot_mismatched:
        findings.append(
            make_finding(
                ctx.run_id,
                name,
                "WARN",
                payload={
                    "rule": "embedding_cache_mismatch",
                    "sampled": es.spot_checked,
                    "mismatched": es.spot_mismatched,
                },
            )
        )
    return es, findings


def images_by_id(ds: Dataset) -> Mapping[str, Image]:
    return {i.image_id: i for i in ds.images}
