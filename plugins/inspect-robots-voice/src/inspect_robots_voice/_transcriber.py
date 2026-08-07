"""Lazy faster-whisper transcription with silence and confidence rejection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

_SAMPLE_RATE = 16_000
_MIN_AUDIO_SECONDS = 0.4
_MAX_NO_SPEECH_PROB = 0.6
_MIN_AVG_LOGPROB = -1.0
_HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thank you",
        "you",
        "thanks for watching!",
        "thanks for watching",
        "bye.",
        "bye",
        "the end.",
        "the end",
        "[music]",
        "music",
    }
)


class _Segment(Protocol):
    text: str
    no_speech_prob: float
    avg_logprob: float


class _Model(Protocol):
    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        language: str,
        vad_filter: bool,
    ) -> tuple[Iterable[_Segment], object]: ...


ModelFactory = Callable[[str, str, str], _Model]


def _load_model(model: str, compute: str, asr_device: str) -> _Model:
    from faster_whisper import WhisperModel

    return cast(_Model, WhisperModel(model, device=asr_device, compute_type=compute))


class WhisperTranscriber:
    """Transcribe 16 kHz mono utterances and silently reject unreliable output."""

    def __init__(
        self,
        model: str,
        compute: str,
        language: str,
        asr_device: str = "cpu",
        *,
        _model_factory: ModelFactory | None = None,
    ) -> None:
        factory = _load_model if _model_factory is None else _model_factory
        self._model = factory(model, compute, asr_device)
        self._language = language

    def transcribe(self, audio: npt.ArrayLike) -> str | None:
        """Return accepted text, or ``None`` for silence and unreliable candidates."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(samples) / _SAMPLE_RATE < _MIN_AUDIO_SECONDS:
            return None
        raw_segments, _info = self._model.transcribe(
            samples,
            language=self._language,
            vad_filter=True,
        )
        segments = list(raw_segments)
        text = "".join(segment.text for segment in segments).strip()
        if not text:
            return None
        if any(segment.no_speech_prob > _MAX_NO_SPEECH_PROB for segment in segments):
            return None
        if any(segment.avg_logprob < _MIN_AVG_LOGPROB for segment in segments):
            return None
        if text.casefold() in _HALLUCINATIONS:
            return None
        return text
