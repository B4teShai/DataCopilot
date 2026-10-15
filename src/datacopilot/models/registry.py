"""Model registry: which weights exist, under which license, pinned to which revision.

Every card pins a full 40-hex commit of the Hugging Face repository and the sha256 of each
``.safetensors`` weight file at that commit, so a download is verified against values that were
reviewed into this repository, not against whatever the hub serves today. Only safetensors files
are ever listed; nothing here can name a pickle checkpoint.

:func:`get_model` is the license guard: it is called before any download or load and raises
:class:`~datacopilot.core.errors.LicenseError` when the card's license is not in the allow-list
(``DATACOPILOT_ALLOWED_LICENSES``). Weights are never vendored (THIRD_PARTY_MODELS.md).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from datacopilot.core.errors import FormatError, LicenseError, ModelError
from datacopilot.io import files

HASH_CHUNK_BYTES = 1024 * 1024
"""Read size while hashing weights; memory does not depend on the weight size."""
MAX_DIMS = 4096
"""Largest embedding width a card may declare (the cache and kNN limits assume it)."""
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.safetensors$")

Family = Literal["dinov2", "siglip2"]


class ModelCard(BaseModel, frozen=True, extra="forbid"):
    """Immutable description of one model.

    Attributes:
        model_id: registry name, also the embedder name (``snake_case``).
        repo: Hugging Face ``owner/name``.
        revision: full commit hash; a branch or tag is rejected (it can move).
        license: SPDX id or custom name, as shown in THIRD_PARTY_MODELS.md.
        gated: the hub requires the user to accept terms and use their own token.
        gpu_memory_gb: rough GPU need, or None for CPU-friendly.
        dims: width of the embedding this model produces.
        family: selects the preprocessing and the feature head in ``embed/backends.py``.
        weights_sha256: ``file name -> sha256`` of every weight file; verified after download.
    """

    model_id: str = Field(pattern=_ID.pattern)
    repo: str = Field(pattern=_REPO.pattern)
    revision: str = Field(pattern=_SHA1.pattern)
    license: str = Field(min_length=1, max_length=64)
    gated: bool
    gpu_memory_gb: float | None = Field(default=None, gt=0, le=1024)
    dims: int = Field(ge=1, le=MAX_DIMS)
    family: Family
    weights_sha256: dict[str, str] = Field(min_length=1, max_length=16)

    @field_validator("repo")
    @classmethod
    def _no_dot_segments(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("invalid repo")
        return value

    @field_validator("weights_sha256")
    @classmethod
    def _safetensors_only(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if not _FILE.fullmatch(name):
                raise ValueError("weights must be .safetensors files with plain names")
            if not _SHA256.fullmatch(digest):
                raise ValueError("weights_sha256 values must be 64 lower-case hex digits")
        return value


_CARDS: tuple[ModelCard, ...] = (
    ModelCard(
        model_id="dinov2_base",
        repo="facebook/dinov2-base",
        revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        license="Apache-2.0",
        gated=False,
        gpu_memory_gb=2.0,
        dims=768,
        family="dinov2",
        weights_sha256={
            "model.safetensors": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
        },
    ),
    ModelCard(
        model_id="dinov2_small",
        repo="facebook/dinov2-small",
        revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        license="Apache-2.0",
        gated=False,
        gpu_memory_gb=1.0,
        dims=384,
        family="dinov2",
        weights_sha256={
            "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
        },
    ),
    ModelCard(
        model_id="siglip2_base",
        repo="google/siglip2-base-patch16-224",
        revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        license="Apache-2.0",
        gated=False,
        gpu_memory_gb=4.0,
        dims=768,
        family="siglip2",
        weights_sha256={
            "model.safetensors": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
        },
    ),
)
_BY_ID = {card.model_id: card for card in _CARDS}


def list_models() -> list[ModelCard]:
    """All registered cards, sorted by ``model_id``."""
    return sorted(_CARDS, key=lambda c: c.model_id)


def has_model(model_id: str) -> bool:
    """True when ``model_id`` names a registered card (no license decision is made)."""
    return model_id in _BY_ID


def license_allowed(license_name: str, allowed_licenses: Collection[str]) -> bool:
    """Case-insensitive membership; an empty allow-list allows nothing."""
    wanted = license_name.casefold()
    return any(wanted == entry.casefold() for entry in allowed_licenses)


def require_license(subject: str, license_name: str, allowed_licenses: Collection[str]) -> None:
    """Raise unless ``license_name`` is allowed; ``subject`` names the model or plugin.

    Raises:
        LicenseError: the license is not in the allow-list. The message names the subject, the
            license and the variable that widens the list.
    """
    if not license_allowed(license_name, allowed_licenses):
        raise LicenseError(
            f"{subject} is licensed {license_name!r}, which is not in "
            "DATACOPILOT_ALLOWED_LICENSES; add it there if you have accepted its terms"
        )


def get_model(model_id: str, allowed_licenses: Collection[str]) -> ModelCard:
    """The card for ``model_id``, after the license guard has passed.

    Call this before any download or load of weights.

    Raises:
        ModelError: malformed or unknown id.
        LicenseError: the model's license is not allowed.
    """
    if not _ID.fullmatch(model_id):
        raise ModelError("invalid model id; use lower-case letters, digits and underscores")
    card = _BY_ID.get(model_id)
    if card is None:
        raise ModelError(f"unknown model {model_id!r}; known models: " + ", ".join(sorted(_BY_ID)))
    require_license(f"model {card.model_id!r}", card.license, allowed_licenses)
    return card


def weight_hash(local_path: Path) -> str:
    """sha256 hex of a weight file, streamed in 1 MiB chunks.

    The file is opened without following symlinks and must be a regular file.

    Raises:
        ModelError: the path is a symlink, a directory or cannot be read.
    """
    path = local_path
    digest = hashlib.sha256()
    try:
        with files.open_regular(path, nofollow=True) as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except (FormatError, OSError):
        raise ModelError(f"cannot read weight file {path.name!r}") from None
    return digest.hexdigest()
