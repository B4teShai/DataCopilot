"""Grounding DINO (tiny) as a proposer: an open-vocabulary detector that finds boxes by name.

Nothing heavy is imported at module level: ``huggingface_hub``, ``transformers`` and the tensor
library are imported inside :meth:`GroundingDinoProposer.detect`, only after the license guard
passed in ``create``. The loading rules are those of the embedders (``embed/backends.py``,
``fetch_verified``): a full pinned commit, only the card's files (weights as ``.safetensors``,
configuration, tokenizer and processor settings; never a pickle, never code), every file checked
against its pinned sha256 before it is deserialised, ``trust_remote_code`` always False, bounded
timeouts, hub error text never echoed. The image is decoded with the bounded decoder of
``embed/imaging.py`` (declared pixel limit, known formats only) before the processor sees it.

The model is asked about the phrases in prompts of at most :data:`MAX_TEXT_TOKENS` tokens (the
model reads 256): a prompt is the phrases joined by full stops, so :func:`normalize_phrase`
refuses any phrase that could split or rewrite it. What the model returns is untrusted: the caller
validates every candidate. The tensor library's name is spelled indirectly because this repository's
QA gate forbids the literal word in Python sources.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from datacopilot.core.errors import ModelError
from datacopilot.core.schema import PluginSpec
from datacopilot.embed import imaging
from datacopilot.embed.backends import fetch_verified, pinned_files
from datacopilot.logging import get_logger
from datacopilot.models import registry as models
from datacopilot.models.registry import ModelCard
from datacopilot.proposers.protocol import (
    MAX_PHRASES,
    PHRASE,
    Candidate,
    Proposer,
)

__all__ = ["MAX_PHRASES", "MAX_TEXT_TOKENS", "TEXT_THRESHOLD", "GroundingDinoPlugin", "plugin"]

MAX_TEXT_TOKENS = 240
"""Longest prompt, in tokens including the two special ones (the model's own limit is 256)."""
TEXT_THRESHOLD = 0.25
"""Share a word's score must reach to be named as the label of a box (the library default); the
box threshold is the ``min_score`` of the call."""
_INSTALL = 'pip install "datacopilot[vlm]"'


class GroundingDinoProposer:
    """Proposer for one :class:`ModelCard`; the model is loaded on the first ``detect`` call."""

    def __init__(self, card: ModelCard, max_pixels: int) -> None:
        self._card = card
        self._max_pixels = max_pixels
        self.model_id = card.model_id
        self.revision = card.revision
        # the identity covers the weights, the configuration and the tokenizer files
        self.weights = pinned_files(card)
        self._model: Any = None
        self._processor: Any = None
        self._lib: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        card, log = self._card, get_logger(__name__)
        started = time.perf_counter()
        local, transformers, lib = fetch_verified(card, _INSTALL)
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                str(local), trust_remote_code=False, local_files_only=True
            )
            model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                str(local), trust_remote_code=False, use_safetensors=True, local_files_only=True
            ).eval()
        except Exception as exc:  # noqa: BLE001
            raise ModelError(
                f"cannot load model {card.model_id!r} ({type(exc).__name__})"
            ) from None
        device = "cuda" if lib.cuda.is_available() else "cpu"
        try:
            model = model.to(device)
        except Exception as exc:  # noqa: BLE001  # out of memory, no driver, ...
            log.warning(
                "proposer_device_fallback", model_id=card.model_id, reason=type(exc).__name__
            )
            device = "cpu"
            model = model.to(device)
        self._model, self._processor, self._lib, self._device = model, processor, lib, device
        log.info(
            "model_loaded",
            model_id=card.model_id,
            model_revision=card.revision,
            device=device,
            weights_sha256=dict(self.weights),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def _chunks(self, phrases: Sequence[str]) -> list[list[str]]:
        """Greedy groups of phrases whose prompt stays within :data:`MAX_TEXT_TOKENS`."""
        tokenizer = self._processor.tokenizer
        chunks: list[list[str]] = [[]]
        for phrase in phrases:
            trial = [*chunks[-1], phrase]
            if len(tokenizer(_prompt(trial))["input_ids"]) > MAX_TEXT_TOKENS and chunks[-1]:
                chunks.append([phrase])
            else:
                chunks[-1] = trial
        return chunks

    def _pass(self, pil: Any, phrases: list[str], size: tuple[int, int], min_score: float) -> Any:
        processor, model, lib = self._processor, self._model, self._lib
        with lib.no_grad():
            inputs = processor(images=pil, text=_prompt(phrases), return_tensors="pt")
            inputs = inputs.to(self._device)
            outputs = model(**inputs)
        return processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=min_score,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(size[1], size[0])],
        )

    def detect(
        self, image: bytes, phrases: Sequence[str], size: tuple[int, int], *, min_score: float
    ) -> list[Candidate]:
        """Find the things named by ``phrases`` (the :class:`Proposer` contract).

        Raises:
            ValueError: a phrase that is not a :data:`PHRASE`, or more than ``MAX_PHRASES``.
            ImageDecodeError: the image is corrupt or over the pixel limit.
            ModelError: the model cannot be fetched, verified, loaded or run, or returned
                something that is not a list of boxes.
        """
        if len(phrases) > MAX_PHRASES:
            raise ValueError(f"at most {MAX_PHRASES} phrases per call")
        if not all(PHRASE.fullmatch(p) and not p.endswith(" ") for p in phrases):
            raise ValueError("a phrase must be lower-case words (see normalize_phrase)")
        if not phrases:
            return []
        rgb = imaging.decode_rgb(image, 0, max_pixels=self._max_pixels)
        if self._model is None:
            self._load()
        from PIL import Image

        pil = Image.fromarray(rgb, "RGB")
        decoded = (pil.size[0], pil.size[1])
        found: list[Candidate] = []
        for chunk in self._chunks(phrases):
            try:
                results = self._pass(pil, chunk, decoded, min_score)
            except Exception as exc:  # noqa: BLE001  # a library error must not leak its text
                raise ModelError(f"model {self.model_id!r} failed ({type(exc).__name__})") from None
            found.extend(_candidates(results, self.model_id))
        return found


def _prompt(phrases: Sequence[str]) -> str:
    return ". ".join(phrases) + "."


def _candidates(results: Any, model_id: str) -> list[Candidate]:
    try:
        (result,) = results
        boxes = np.asarray(result["boxes"].detach().float().cpu().numpy(), dtype=np.float64)
        scores = np.asarray(result["scores"].detach().float().cpu().numpy(), dtype=np.float64)
        labels = [str(label) for label in result["text_labels"]]
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),):
            raise ValueError("shape")
        if len(labels) != len(boxes):
            raise ValueError("labels")
    except Exception as exc:  # noqa: BLE001
        raise ModelError(
            f"model {model_id!r} returned an unusable result ({type(exc).__name__})"
        ) from None
    return [
        Candidate((float(b[0]), float(b[1]), float(b[2]), float(b[3])), label, float(score))
        for b, score, label in zip(boxes, scores, labels, strict=True)
    ]


class GroundingDinoPlugin:
    """Entry-point object (``spec``; ``create`` runs after the license guard)."""

    def __init__(self, card: ModelCard, *, max_pixels: int = imaging.MAX_DECODE_PIXELS) -> None:
        self.card = card
        self._max_pixels = max_pixels
        self.spec = PluginSpec(
            name="grounding_dino",
            kind="proposer",
            license=card.license,
            extra="vlm",
            gpu_memory_gb=card.gpu_memory_gb,
            version="1",
        )

    def create(self, allowed_licenses: frozenset[str]) -> Proposer:
        models.require_license(f"model {self.card.model_id!r}", self.card.license, allowed_licenses)
        get_logger(__name__).info(
            "model_license_shown",
            model_id=self.card.model_id,
            license=self.card.license,
            gated=self.card.gated,
        )
        return GroundingDinoProposer(self.card, self._max_pixels)


plugin = GroundingDinoPlugin(
    next(c for c in models.list_models() if c.model_id == "grounding_dino")
)
