"""Lazy ONNX Parakeet transcription with duration and confidence rejection."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from inspect_robots_voice._transcriber import (
    _SAMPLE_RATE,
    _is_audio_too_short,
    _is_hallucination,
)

_MIN_MEAN_LOGPROB = -2.5


class _TimestampedResult(Protocol):
    text: str
    logprobs: list[float] | None


class _Model(Protocol):
    def recognize(
        self, samples: npt.NDArray[np.float32], *, sample_rate: int
    ) -> _TimestampedResult: ...


ModelFactory = Callable[[str], _Model]


def _load_model(model: str) -> _Model:
    import onnx_asr

    loaded = onnx_asr.load_model(
        model,
        quantization="int8",
        providers=["CPUExecutionProvider"],
    )
    return cast(_Model, loaded.with_timestamps())


class ParakeetTranscriber:
    """Transcribe 16 kHz mono utterances and reject coarse low-confidence output."""

    def __init__(
        self,
        model: str,
        *,
        _model_factory: ModelFactory | None = None,
    ) -> None:
        factory = _load_model if _model_factory is None else _model_factory
        self._model = factory(model)

    def transcribe(self, audio: npt.ArrayLike) -> str | None:
        """Return accepted text, or ``None`` for silence and unreliable candidates."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if _is_audio_too_short(samples):
            return None
        result = self._model.recognize(samples, sample_rate=_SAMPLE_RATE)
        text = result.text.strip()
        if not text or _is_hallucination(text):
            return None
        logprobs = result.logprobs
        if logprobs and fmean(logprobs) < _MIN_MEAN_LOGPROB:
            return None
        return text
