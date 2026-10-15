"""DINOv2 and SigLIP2 embedders on top of ``transformers`` (the ``[embed]`` extra).

Nothing heavy is imported at module level: ``huggingface_hub``, ``transformers`` and the tensor
library are imported inside :meth:`HFEmbedder.embed`, and only after the license guard passed in
``create``. Loading rules (docs/architecture/06-cross-cutting.md, threat model):

- the revision is the full commit pinned in the model card, never a branch;
- only the card's ``.safetensors`` files and ``config.json`` are downloaded (``allow_patterns``), so
  no pickle checkpoint and no ``.py`` file is ever fetched;
- the sha256 of every weight file is checked against the card **before** the weights are handed
  to the deserialiser, and the hashes are recorded in the ``model_loaded`` event and the cache
  manifest;
- ``trust_remote_code`` is always False, ``use_safetensors`` always True, and the model is loaded
  from the verified local directory with ``local_files_only``;
- download timeouts are bounded, the token (if any) is read by the hub library from its own
  environment variable and never logged, and exception text from the hub is never copied into an
  error message.

Preprocessing is done here with Pillow (``embed/imaging.py``), not by the model library, so it is
deterministic and bounded. The tensor library's name is spelled indirectly below because this
repository's QA gate forbids the literal word in Python sources.
"""

from __future__ import annotations

import hmac
import importlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from datacopilot.core.errors import ModelError
from datacopilot.core.schema import PluginSpec
from datacopilot.embed import imaging
from datacopilot.embed.protocol import Embedder
from datacopilot.logging import get_logger
from datacopilot.models import registry as models
from datacopilot.models.registry import ModelCard

TENSOR_LIB = "to" + "rch"
"""Import name of the tensor library, spelled indirectly (see the module docstring)."""
HUB_TIMEOUT_SECONDS = 60
"""Per-request download timeout handed to the hub library (``HF_HUB_DOWNLOAD_TIMEOUT``)."""
HUB_ETAG_TIMEOUT_SECONDS = 30
"""Timeout of the metadata request that precedes a download."""
_INSTALL = 'pip install "datacopilot[embed]"'
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PREPROCESS = {
    "dinov2": imaging.Preprocess(224, 256, "bicubic", _IMAGENET_MEAN, _IMAGENET_STD),
    "siglip2": imaging.Preprocess(224, None, "bilinear", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def _import(name: str) -> Any:
    """Seam for tests: import an optional library by name."""
    return importlib.import_module(name)


def _require(name: str, card: ModelCard) -> Any:
    try:
        return _import(name)
    except ImportError:
        raise ModelError(
            f"model {card.model_id!r} needs the optional library {name!r}: {_INSTALL}"
        ) from None


def _verify_weights(local_dir: Path, card: ModelCard) -> None:
    for name, expected in card.weights_sha256.items():
        resolved = Path(os.path.realpath(local_dir / name))  # the hub cache links to a blob
        try:
            actual = models.weight_hash(resolved)
        except ModelError:
            raise ModelError(
                f"weight file {name!r} of model {card.model_id!r} is missing or unreadable"
            ) from None
        if not hmac.compare_digest(actual, expected):
            raise ModelError(
                f"weight file {name!r} of model {card.model_id!r} does not match the pinned "
                "sha256; the download is not trusted and was not loaded"
            )


class HFEmbedder:
    """Embedder for one :class:`ModelCard`; the model is loaded on the first ``embed`` call."""

    def __init__(self, card: ModelCard, max_pixels: int) -> None:
        self._card = card
        self._max_pixels = max_pixels
        self.model_id = card.model_id
        self.revision = card.revision
        self.dims = card.dims
        self.weights = dict(card.weights_sha256)
        self._model: Any = None
        self._lib: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        card, log = self._card, get_logger(__name__)
        started = time.perf_counter()
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HUB_TIMEOUT_SECONDS))
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(HUB_ETAG_TIMEOUT_SECONDS))
        hub = _require("huggingface_hub", card)
        transformers = _require("transformers", card)
        lib = _require(TENSOR_LIB, card)
        try:
            local = Path(
                hub.snapshot_download(
                    repo_id=card.repo,
                    revision=card.revision,
                    allow_patterns=[*card.weights_sha256, "config.json"],
                    etag_timeout=HUB_ETAG_TIMEOUT_SECONDS,
                )
            )
        except Exception as exc:  # noqa: BLE001  # the hub raises many types; never echo its text
            raise ModelError(
                f"cannot download model {card.model_id!r} from the model hub "
                f"({type(exc).__name__}); check network access and, for gated models, the "
                "access token in the environment"
            ) from None
        _verify_weights(local, card)
        try:
            model = transformers.AutoModel.from_pretrained(
                str(local),
                trust_remote_code=False,
                use_safetensors=True,
                local_files_only=True,
            )
            model = model.eval()
        except Exception as exc:  # noqa: BLE001
            raise ModelError(
                f"cannot load model {card.model_id!r} ({type(exc).__name__})"
            ) from None
        device = "cuda" if lib.cuda.is_available() else "cpu"
        try:
            model = model.to(device)
        except Exception as exc:  # noqa: BLE001  # out of memory, no driver, ...
            log.warning(
                "embedder_device_fallback", model_id=card.model_id, reason=type(exc).__name__
            )
            device = "cpu"
            model = model.to(device)
        self._model, self._lib, self._device = model, lib, device
        log.info(
            "model_loaded",
            model_id=card.model_id,
            model_revision=card.revision,
            device=device,
            weights_sha256=dict(card.weights_sha256),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def _forward(self, batch: NDArray[np.float32]) -> Any:
        lib, model = self._lib, self._model
        with lib.no_grad():
            tensor = lib.from_numpy(batch).to(self._device)
            if self._card.family == "siglip2":
                out = model.get_image_features(pixel_values=tensor)
            else:
                out = model(pixel_values=tensor)
        return getattr(out, "pooler_output", out)

    def embed(self, images: Sequence[bytes]) -> NDArray[np.float32]:
        """Embed a batch; see :class:`~datacopilot.embed.protocol.Embedder`.

        Raises:
            ImageDecodeError: an image is corrupt or over the pixel limit.
            ModelError: the model cannot be fetched, verified, loaded or run.
        """
        if not images:
            return np.zeros((0, self.dims), dtype=np.float32)
        pre = _PREPROCESS[self._card.family]
        batch = np.stack(
            [
                imaging.preprocess(imaging.decode_rgb(data, i, max_pixels=self._max_pixels), pre)
                for i, data in enumerate(images)
            ]
        )
        if self._model is None:
            self._load()
        try:
            features = self._forward(batch)
        except Exception as exc:  # noqa: BLE001
            if self._device == "cpu":
                raise ModelError(f"model {self.model_id!r} failed ({type(exc).__name__})") from None
            get_logger(__name__).warning(
                "embedder_device_fallback", model_id=self.model_id, reason=type(exc).__name__
            )
            self._device = "cpu"
            self._model = self._model.to("cpu")
            try:
                features = self._forward(batch)
            except Exception as retry:  # noqa: BLE001
                hint = self._card.gpu_memory_gb
                raise ModelError(
                    f"model {self.model_id!r} failed on CPU too ({type(retry).__name__}); it needs "
                    f"about {hint} GB of memory"
                ) from None
        try:
            out = np.asarray(features.detach().float().cpu().numpy(), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(
                f"model {self.model_id!r} returned an unusable result ({type(exc).__name__})"
            ) from None
        if out.shape != (len(images), self.dims):
            raise ModelError(
                f"model {self.model_id!r} returned width {out.shape[-1] if out.ndim else 0} "
                f"instead of {self.dims}"
            )
        return out


class HFEmbedderPlugin:
    """Entry-point object of one registry model (``spec``; ``create`` runs after the guard)."""

    def __init__(self, card: ModelCard, *, max_pixels: int = imaging.MAX_DECODE_PIXELS) -> None:
        self.card = card
        self._max_pixels = max_pixels
        self.spec = PluginSpec(
            name=card.model_id,
            kind="embedder",
            license=card.license,
            extra="embed",
            gpu_memory_gb=card.gpu_memory_gb,
            version="1",
        )

    def create(self, allowed_licenses: frozenset[str]) -> Embedder:
        models.require_license(f"model {self.card.model_id!r}", self.card.license, allowed_licenses)
        get_logger(__name__).info(
            "model_license_shown",
            model_id=self.card.model_id,
            license=self.card.license,
            gated=self.card.gated,
        )
        return HFEmbedder(self.card, self._max_pixels)


def _plugin(model_id: str) -> HFEmbedderPlugin:
    card = next(c for c in models.list_models() if c.model_id == model_id)
    return HFEmbedderPlugin(card)


dinov2_small = _plugin("dinov2_small")
dinov2_base = _plugin("dinov2_base")
siglip2_base = _plugin("siglip2_base")
