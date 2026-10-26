"""Pydantic models for the dataset tables and the invariants of docs/architecture/03-data-model.md.

Row-level invariants live in model validators; invariants that span rows (unique ids, foreign
keys, boxes inside their image) are checked by :func:`validate_tables`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from datacopilot.core.errors import GateFailedError, ValidationError

Split = Literal["train", "val", "test", "unassigned"]
Kind = Literal["bbox", "classification"]
Provenance = Literal["human", "model", "proposal_accepted"]
Task = Literal["detection", "classification"]
Severity = Literal["CRITICAL", "WARN", "INFO"]
SEVERITIES: tuple[Severity, ...] = ("CRITICAL", "WARN", "INFO")

SCHEMA_VERSION = 1
MAX_PATH_CHARS = 4096
"""Longest ``Image.file_path`` (the usual operating-system path limit)."""
MAX_NAME_CHARS = 65_536
"""Longest ``Category.name``; reports and findings show a truncated form."""
MAX_ID_CHARS = 1024
"""Longest ``Annotation.ann_id`` (and the ``Finding.ann_id`` that refers to it)."""
_MAX_REPORTED = 5
MAX_PAYLOAD_BYTES = 65_536
"""Largest ``Finding.payload`` in canonical JSON bytes; a class histogram of 1,000 classes fits."""
MAX_PAYLOAD_DEPTH = 6
"""Deepest nesting of a ``Finding.payload``; findings are flat records, not documents."""
MAX_PAYLOAD_NODES = 20_000
"""Most JSON values (containers and scalars) in one ``Finding.payload``."""


def _finite(value: JsonValue) -> JsonValue:
    """Reject NaN and infinities anywhere in a JSON-like value (they break canonical hashing)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    return value


def _bounded_payload(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Bound a finding payload by depth, node count and serialised size before it is stored.

    The walk is iterative, so a hostile plugin cannot trigger deep recursion here.
    """
    nodes = 0
    stack: list[tuple[JsonValue, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > MAX_PAYLOAD_DEPTH or nodes > MAX_PAYLOAD_NODES:
            raise ValueError("payload is nested too deeply or has too many values")
        if isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
    size = len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload is {size} bytes; the limit is {MAX_PAYLOAD_BYTES}")
    return value


Source = dict[str, Annotated[JsonValue, AfterValidator(_finite)]]
Payload = Annotated[
    dict[str, Annotated[JsonValue, AfterValidator(_finite)]], AfterValidator(_bounded_payload)
]
Scalar = str | int | float | bool


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    def to_row(self) -> dict[str, JsonValue]:
        """Plain JSON-compatible mapping of this row (tuples become lists)."""
        return self.model_dump(mode="json")


class Category(_Row):
    id: int
    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    aliases: tuple[str, ...] = ()
    attributes: dict[str, str] = Field(default_factory=dict)


class Image(_Row):
    image_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    file_path: str = Field(min_length=1, max_length=MAX_PATH_CHARS)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    split: Split = "unassigned"
    group_id: str | None = None
    source: Source = Field(default_factory=dict)


def group_key(label: str) -> str:
    """How group labels are compared: case-insensitively, without surrounding space and in NFKC
    form, so ``Cam1``, ``" cam1 "`` and a fullwidth ``Ｃam1`` are one group (the leakage check and
    ``split`` use this one definition)."""
    return unicodedata.normalize("NFKC", label).casefold().strip()


class Annotation(_Row):
    ann_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    image_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    category_id: int
    kind: Kind
    bbox_xyxy_abs: tuple[float, float, float, float] | None = None
    score: float | None = None
    is_crowd: bool = False
    occluded: bool | None = None
    truncated: bool | None = None
    attributes: dict[str, Scalar] = Field(default_factory=dict)
    provenance: Provenance = "human"
    source: Source = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Annotation:
        box = self.bbox_xyxy_abs
        if self.kind == "classification":
            if box is not None:
                raise ValueError("classification annotation must not have a box")
        else:
            if box is None:
                raise ValueError("bbox annotation requires bbox_xyxy_abs")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError("box must satisfy 0 <= x_min < x_max and 0 <= y_min < y_max")
        if self.score is not None and self.provenance == "human":
            raise ValueError("score must be null for human provenance")
        for value in self.attributes.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite attribute value")
        return self


_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class PluginSpec(BaseModel, frozen=True):
    """Metadata every registered plugin exposes before its implementation is used."""

    name: str = Field(pattern=_NAME.pattern)
    kind: Literal["format", "check", "proposer", "embedder"]
    license: str = Field(min_length=1)
    extra: str | None
    gpu_memory_gb: float | None
    version: str = Field(min_length=1)


class DatasetManifest(_Row):
    version: str = Field(pattern=r"^[0-9a-f]{32}$")
    schema_version: int = Field(ge=1)
    name: str = Field(min_length=1)
    task: Task
    categories: tuple[Category, ...]
    lineage: tuple[str, ...] = ()
    image_count: int = Field(ge=0)
    annotation_count: int = Field(ge=0)
    created_at: AwareDatetime


TEST_ONLY_SUFFIX = "-test-only"
"""Appended to ``AuditRun.embedder`` when the embedder is a test double, so every report format
says so (the JSON and SARIF carry the label, the HTML header spells it out)."""
_CHECK_NAME = r"^[a-z][a-z0-9_]{0,63}$"
_UUID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class Finding(_Row):
    """One problem a check found; embedded in :class:`AuditRun`."""

    finding_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    run_id: str = Field(pattern=_UUID)
    check: str = Field(pattern=_CHECK_NAME)
    severity: Severity
    image_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    ann_id: str | None = Field(default=None, min_length=1, max_length=MAX_ID_CHARS)
    payload: Payload = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_id(self) -> Finding:
        from datacopilot.core.version import finding_id

        expected = finding_id(self.check, self.image_id, self.ann_id, self.payload)
        if self.finding_id != expected:
            raise ValueError("finding_id does not match the finding's check, target and payload")
        return self


class Counts(_Row):
    """Findings per severity."""

    critical: int = Field(ge=0)
    warn: int = Field(ge=0)
    info: int = Field(ge=0)


MAX_SUMMARY_NAME_CHARS = 200
"""Category names are cut to this length in :class:`Summary`."""


class SummaryClass(_Row):
    id: int
    name: str = Field(max_length=MAX_SUMMARY_NAME_CHARS)


class Summary(_Row):
    """Dataset facts the reports need (histogram, split by class table), computed once per run.

    ``annotation_count`` and ``class_counts`` describe the rows the checks saw: model-generated rows
    (``provenance == "model"``) are excluded from an audit unless asked for, and
    ``excluded_model_annotations`` says how many were left out.
    ``class_counts`` maps ``split -> str(category_id) -> annotations``. When the dataset has more
    classes than the summary cap, ``classes`` and ``class_counts`` are empty and
    ``classes_omitted`` is true; the totals are always present.
    """

    task: Task
    image_count: int = Field(ge=0)
    annotation_count: int = Field(ge=0)
    split_images: dict[str, int]
    classes: tuple[SummaryClass, ...] = ()
    classes_omitted: bool = False
    class_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    excluded_model_annotations: int = Field(default=0, ge=0)


class AuditRun(_Row):
    """Result of one audit: the findings plus the facts needed to reproduce and gate on them.

    Attributes:
        run_id: the CLI ``request_id`` of the invocation (lowercase UUID).
        fail_on: the gate that was evaluated; ``passed`` is False when it matched.
        timings_ms: wall time per check, in run order. Not part of the determinism contract.
        summary: dataset totals and per-split class counts for the reports.
    """

    run_id: str = Field(pattern=_UUID)
    dataset_version: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at: AwareDatetime
    checks: tuple[str, ...]
    embedder: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
    counts: Counts
    passed: bool
    fail_on: tuple[str, ...] = ()
    timings_ms: dict[str, int] = Field(default_factory=dict)
    summary: Summary | None = None
    findings: tuple[Finding, ...] = ()

    def raise_for_gate(self) -> None:
        """Raise :class:`GateFailedError` if the gate that was evaluated matched.

        The command line turns a failed gate into exit code 1; the Python API records it in
        ``passed`` and leaves the decision to the caller, who calls this to get the same effect.
        """
        if not self.passed:
            c = self.counts
            raise GateFailedError(
                f"the audit gate failed: critical={c.critical} warn={c.warn} info={c.info} "
                f"(run_id {self.run_id})"
            )

    @model_validator(mode="after")
    def _check(self) -> AuditRun:
        seen: set[str] = set()
        for f in self.findings:
            if f.run_id != self.run_id:
                raise ValueError(f"finding {f.finding_id} belongs to run_id {f.run_id}")
            if f.finding_id in seen:
                raise ValueError(f"duplicate finding_id {f.finding_id}")
            seen.add(f.finding_id)
        tally = Counter(f.severity for f in self.findings)
        expected = Counts(critical=tally["CRITICAL"], warn=tally["WARN"], info=tally["INFO"])
        if self.counts != expected:
            raise ValueError(f"counts {self.counts.model_dump()} do not equal the findings")
        return self


class SplitShare(_Row):
    """How many images one split received against its target (``deviation`` is
    ``images / total - ratio``, rounded to 4 places)."""

    images: int = Field(ge=0)
    target_images: float = Field(ge=0)
    deviation: float


class ClassShare(_Row):
    """Annotations of one class in one split: ``share`` of the class's total and ``deviation``
    from the split's ratio (both rounded to 4 places)."""

    annotations: int = Field(ge=0)
    share: float = Field(ge=0, le=1)
    deviation: float


_HEX32 = r"^[0-9a-f]{32}$"


class SplitReport(_Row):
    """What ``split`` did: the new version, how images were grouped, and how close it came.

    Attributes:
        parent_version, dataset_version: the version that was split and the new one (its lineage
            ends with the parent).
        ratios: train, val and test fractions asked for; ``seed`` breaks ties between equal groups.
        stratify: ``category`` when classes were balanced too.
        group_key: the ``--by-group`` key; the new version's ``group_id`` holds its label.
        near_duplicate_threshold, embedder: set when near-duplicate clusters were grouped.
        group_count: groups assigned (an image without a key or near copy is a group of its own);
            ``multi_image_groups`` and ``largest_group`` describe the groups of two or more.
        key_groups: groups of two or more images that share a key; ``images_without_key`` had none.
        near_duplicate_pairs: image pairs at or above the threshold, all of them (not a top-k).
        groups_across_splits: groups whose images ended in more than one split; checked after the
            assignment and always 0 in a written report.
        moved_images: images whose split changed.
        splits: ``split -> SplitShare``; classes: ``split -> str(category_id) -> ClassShare`` for
            every class with annotations; ``max_class_deviation`` is the largest absolute class
            deviation (None without annotations).
    """

    parent_version: str = Field(pattern=_HEX32)
    dataset_version: str = Field(pattern=_HEX32)
    seed: int = Field(ge=0)
    ratios: tuple[float, float, float]
    stratify: Literal["category"] | None = None
    group_key: str
    near_duplicate_threshold: float | None = None
    embedder: str | None = None
    image_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    multi_image_groups: int = Field(ge=0)
    largest_group: int = Field(ge=0)
    key_groups: int = Field(ge=0)
    images_without_key: int = Field(ge=0)
    near_duplicate_pairs: int = Field(ge=0)
    groups_across_splits: int = Field(ge=0)
    moved_images: int = Field(ge=0)
    splits: dict[str, SplitShare]
    classes: dict[str, dict[str, ClassShare]]
    max_class_deviation: float | None = None


@dataclass(frozen=True)
class Dataset:
    """A validated dataset version: manifest plus the image and annotation tables.

    ``root`` is the directory the image paths are expected to lie in (the dataset's own directory
    for a source dataset, the parent of the ``.dcp`` directory for a stored one); the audit refuses
    to read images outside it. It is not part of the dataset's identity: it is excluded from
    equality and from the content version.
    """

    manifest: DatasetManifest
    images: tuple[Image, ...]
    annotations: tuple[Annotation, ...]
    root: Path | None = field(default=None, compare=False, repr=False)

    @property
    def categories(self) -> tuple[Category, ...]:
        return self.manifest.categories


def _duplicates(values: Sequence[object]) -> list[object]:
    return [v for v, n in Counter(values).items() if n > 1]


def validate_tables(
    images: Sequence[Image],
    annotations: Sequence[Annotation],
    categories: Sequence[Category],
    *,
    task: Task,
) -> None:
    """Check the cross-row invariants (3, 4, 6 and id uniqueness) of the three tables.

    Raises:
        ValidationError: listing up to five problems (naming ann_id or image_id), then a count.
    """
    problems: list[str] = []
    for label, values in (
        ("image_id", [i.image_id for i in images]),
        ("ann_id", [a.ann_id for a in annotations]),
        ("category id", [c.id for c in categories]),
        ("category name", [c.name for c in categories]),
    ):
        problems += [f"duplicate {label} {v!r}" for v in _duplicates(values)]
    by_id = {i.image_id: i for i in images}
    category_ids = {c.id for c in categories}
    expected: Kind = "bbox" if task == "detection" else "classification"
    for ann in annotations:
        ref = f"ann_id {ann.ann_id!r}"
        if ann.kind != expected:
            problems.append(f"{ref} has kind {ann.kind!r} but dataset task is {task!r}")
        image = by_id.get(ann.image_id)
        if image is None:
            problems.append(f"{ref} references unknown image_id {ann.image_id!r}")
        elif ann.bbox_xyxy_abs is not None:
            _, _, x2, y2 = ann.bbox_xyxy_abs
            if x2 > image.width or y2 > image.height:
                problems.append(
                    f"{ref} box {ann.bbox_xyxy_abs} outside image {image.image_id!r} "
                    f"({image.width}x{image.height})"
                )
        if ann.category_id not in category_ids:
            problems.append(f"{ref} references unknown category id {ann.category_id}")
    if problems:
        shown = "; ".join(problems[:_MAX_REPORTED])
        more = len(problems) - _MAX_REPORTED
        suffix = f"; and {more} more" if more > 0 else ""
        raise ValidationError(f"invalid dataset: {shown}{suffix}")
