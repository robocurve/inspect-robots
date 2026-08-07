"""Whisper rejection gauntlet against canned model output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from inspect_robots_voice._transcriber import _HALLUCINATIONS, WhisperTranscriber


@dataclass
class _Segment:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


class _Model:
    def __init__(self, segments: list[_Segment]) -> None:
        self.segments = segments
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self, audio: np.ndarray, *, language: str, vad_filter: bool
    ) -> tuple[list[_Segment], object]:
        self.calls.append({"samples": len(audio), "language": language, "vad_filter": vad_filter})
        return self.segments, object()


def _transcriber(segments: list[_Segment]) -> tuple[WhisperTranscriber, _Model]:
    model = _Model(segments)
    transcriber = WhisperTranscriber(
        "small",
        "auto",
        "en",
        _model_factory=lambda _name, _compute: model,
    )
    return transcriber, model


def _audio(seconds: float = 0.5) -> np.ndarray:
    return np.zeros(round(16_000 * seconds), dtype=np.float32)


def test_normal_sentence_is_accepted_with_vad_enabled() -> None:
    transcriber, model = _transcriber([_Segment(" Keep the wrist level. ")])

    assert transcriber.transcribe(_audio()) == "Keep the wrist level."
    assert model.calls == [{"samples": 8_000, "language": "en", "vad_filter": True}]


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_or_whitespace_text_is_rejected(text: str) -> None:
    transcriber, _model = _transcriber([_Segment(text)])
    assert transcriber.transcribe(_audio()) is None


def test_no_speech_probability_is_rejected() -> None:
    transcriber, _model = _transcriber([_Segment("possibly speech", no_speech_prob=0.61)])
    assert transcriber.transcribe(_audio()) is None


def test_low_average_log_probability_is_rejected() -> None:
    transcriber, _model = _transcriber([_Segment("uncertain", avg_logprob=-1.01)])
    assert transcriber.transcribe(_audio()) is None


def test_audio_shorter_than_four_tenths_is_rejected_before_model_call() -> None:
    transcriber, model = _transcriber([_Segment("too short")])

    assert transcriber.transcribe(_audio(0.399)) is None
    assert model.calls == []


@pytest.mark.parametrize("phrase", sorted(_HALLUCINATIONS))
def test_entire_hallucination_phrase_is_rejected_case_insensitively(phrase: str) -> None:
    transcriber, _model = _transcriber([_Segment(phrase.upper())])
    assert transcriber.transcribe(_audio()) is None


def test_hallucination_words_inside_real_sentence_are_accepted() -> None:
    transcriber, _model = _transcriber([_Segment("I said thank you. Now move left.")])
    assert transcriber.transcribe(_audio()) == "I said thank you. Now move left."


def test_multiple_segments_join_before_filtering() -> None:
    transcriber, _model = _transcriber([_Segment("move "), _Segment("left")])
    assert transcriber.transcribe(_audio()) == "move left"


def test_empty_segment_sequence_is_rejected() -> None:
    transcriber, _model = _transcriber([])
    assert transcriber.transcribe(_audio()) is None
