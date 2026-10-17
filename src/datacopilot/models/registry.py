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

from pydantic import BaseModel, Field, field_validator, model_validator

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
_AUX_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(json|txt)$")

Family = Literal["dinov2", "siglip2", "grounding_dino"]
EMBEDDING_FAMILIES = frozenset({"dinov2", "siglip2"})


class ModelCard(BaseModel, frozen=True, extra="forbid"):
    """Immutable description of one model.

    Attributes:
        model_id: registry name, also the embedder name (``snake_case``).
        repo: Hugging Face ``owner/name``.
        revision: full commit hash; a branch or tag is rejected (it can move).
        license: SPDX id or custom name, as shown in THIRD_PARTY_MODELS.md.
        gated: the hub requires the user to accept terms and use their own token.
        gpu_memory_gb: rough GPU need, or None for CPU-friendly.
        dims: width of the embedding an embedder produces; None for a detector, which has none.
        family: selects the preprocessing and the feature head in ``embed/backends.py`` (or the
            detector in ``proposers/grounding_dino.py``).
        weights_sha256: ``file name -> sha256`` of every weight file; verified after download.
        weights_bytes: ``file name -> size`` of the same files; checked before they are hashed, so
            a hostile endpoint cannot make the audit hash an endless stream.
        config_sha256: sha256 of ``config.json`` at the pinned revision; a changed configuration
            (``auto_map`` entries, architecture) is refused, and the digest is part of the cache
            identity.
        aux_sha256: ``file name -> sha256`` of the other small files the model needs (tokenizer
            and preprocessor settings); only ``.json`` and ``.txt`` names are accepted, so a card
            can never name code or a pickle. Verified like the weights.
    """

    model_id: str = Field(pattern=_ID.pattern)
    repo: str = Field(pattern=_REPO.pattern)
    revision: str = Field(pattern=_SHA1.pattern)
    license: str = Field(min_length=1, max_length=64)
    gated: bool
    gpu_memory_gb: float | None = Field(default=None, gt=0, le=1024)
    dims: int | None = Field(default=None, ge=1, le=MAX_DIMS)
    family: Family
    weights_sha256: dict[str, str] = Field(min_length=1, max_length=16)
    weights_bytes: dict[str, int] = Field(min_length=1, max_length=16)
    config_sha256: str = Field(pattern=_SHA256.pattern)
    aux_sha256: dict[str, str] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def _sizes_match_files(self) -> ModelCard:
        if set(self.weights_bytes) != set(self.weights_sha256) or not all(
            0 < n <= 64 * 1024**3 for n in self.weights_bytes.values()
        ):
            raise ValueError("weights_bytes must give a positive size for every weight file")
        if (self.dims is not None) != (self.family in EMBEDDING_FAMILIES):
            raise ValueError("an embedder card has dims and a detector card has none")
        if "config.json" in self.aux_sha256:
            raise ValueError("config.json is pinned by config_sha256")
        return self

    @field_validator("aux_sha256")
    @classmethod
    def _data_files_only(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if not _AUX_FILE.fullmatch(name):
                raise ValueError("auxiliary files must be .json or .txt files with plain names")
            if not _SHA256.fullmatch(digest):
                raise ValueError("aux_sha256 values must be 64 lower-case hex digits")
        return value

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
        weights_bytes={"model.safetensors": 346345912},
        config_sha256="f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
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
        weights_bytes={"model.safetensors": 88249960},
        config_sha256="1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
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
        weights_bytes={"model.safetensors": 1500800904},
        config_sha256="fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
    ),
    ModelCard(
        model_id="grounding_dino",
        repo="IDEA-Research/grounding-dino-tiny",
        revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
        license="Apache-2.0",
        gated=False,
        gpu_memory_gb=2.0,
        family="grounding_dino",
        weights_sha256={
            "model.safetensors": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
        },
        weights_bytes={"model.safetensors": 689359096},
        config_sha256="eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a",
        aux_sha256={
            "preprocessor_config.json": "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60",  # noqa: E501
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "tokenizer_config.json": "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56",  # noqa: E501
            "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",  # noqa: E501
            "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            "added_tokens.json": "909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2",
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
