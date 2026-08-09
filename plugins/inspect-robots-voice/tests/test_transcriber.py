"""Backend selection and Whisper rejection against canned model output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

import inspect_robots_voice._parakeet as parakeet_module
import inspect_robots_voice._transcriber as transcriber_module
from inspect_robots_voice._transcriber import (
    _HALLUCINATIONS,
    WhisperTranscriber,
    _classify_model,
    resolve_transcriber,
)


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
        _model_factory=lambda _name, _compute, _asr: model,
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


def test_model_loads_on_cpu_by_default_and_honors_asr_device() -> None:
    """CUDA auto-detection must never be the default: a GPU-visible machine without the
    CUDA runtime libraries would crash at first transcription (observed on the omen rig)."""
    seen: list[str] = []

    def factory(_name: str, _compute: str, asr_device: str) -> _Model:
        seen.append(asr_device)
        return _Model([])

    WhisperTranscriber("small", "auto", "en", _model_factory=factory)
    WhisperTranscriber("small", "auto", "en", "cuda", _model_factory=factory)

    assert seen == ["cpu", "cuda"]


def test_whisper_none_language_resolves_to_english_and_explicit_language_passes_through() -> None:
    model = _Model([_Segment("accepted")])
    default_language = WhisperTranscriber(
        "small", "auto", None, _model_factory=lambda _name, _compute, _asr: model
    )
    explicit_language = WhisperTranscriber(
        "small", "auto", "fr", _model_factory=lambda _name, _compute, _asr: model
    )

    assert default_language.transcribe(_audio()) == "accepted"
    assert explicit_language.transcribe(_audio()) == "accepted"
    assert [call["language"] for call in model.calls] == ["en", "fr"]


@pytest.mark.parametrize(
    ("model", "canonical"),
    [
        ("parakeet", "nemo-parakeet-tdt-0.6b-v3"),
        ("PARAKEET", "nemo-parakeet-tdt-0.6b-v3"),
        ("parakeet-tdt-0.6b-v3", "nemo-parakeet-tdt-0.6b-v3"),
        ("Parakeet-TDT-0.6B-V3", "nemo-parakeet-tdt-0.6b-v3"),
        ("nemo-parakeet-tdt-0.6b-v3", "nemo-parakeet-tdt-0.6b-v3"),
        ("NeMo-PaRaKeEt-CuStOm", "nemo-parakeet-custom"),
    ],
)
def test_classifier_canonicalizes_parakeet_names(model: str, canonical: str) -> None:
    assert _classify_model(model) == canonical


@pytest.mark.parametrize(
    "model",
    ["small", "distil-small.en", "/models/parakeet-ct2", r"C:\models\parakeet-ct2"],
)
def test_classifier_selects_whisper_names_and_paths(model: str) -> None:
    assert _classify_model(model) is None


def test_classifier_rejects_unknown_parakeetish_name_with_supported_names() -> None:
    with pytest.raises(TypeError) as exc_info:
        resolve_transcriber("parakeet-tdt-1.1b", "auto", None, "cpu")

    message = str(exc_info.value)
    assert "parakeet" in message
    assert "parakeet-tdt-0.6b-v3" in message
    assert "nemo-" in message


@pytest.mark.parametrize(
    ("model", "canonical"),
    [
        ("parakeet", "nemo-parakeet-tdt-0.6b-v3"),
        ("PARAKEET", "nemo-parakeet-tdt-0.6b-v3"),
        ("parakeet-tdt-0.6b-v3", "nemo-parakeet-tdt-0.6b-v3"),
        ("PARAKEET-TDT-0.6B-V3", "nemo-parakeet-tdt-0.6b-v3"),
        ("nemo-parakeet-tdt-0.6b-v3", "nemo-parakeet-tdt-0.6b-v3"),
        ("NeMo-PaRaKeEt-CuStOm", "nemo-parakeet-custom"),
    ],
)
def test_resolver_constructs_parakeet_with_canonical_name(
    monkeypatch: MonkeyPatch, model: str, canonical: str
) -> None:
    constructed: list[str] = []
    sentinel = object()

    def fake_parakeet(name: str) -> object:
        constructed.append(name)
        return sentinel

    monkeypatch.setattr(parakeet_module, "ParakeetTranscriber", fake_parakeet)

    assert resolve_transcriber(model, "int8", "fr", "cuda") is sentinel
    assert constructed == [canonical]


@pytest.mark.parametrize("model", ["small", "distil-small.en", "/models/parakeet-ct2"])
def test_resolver_constructs_whisper_with_exact_arguments(
    monkeypatch: MonkeyPatch, model: str
) -> None:
    calls: list[tuple[str, str, str | None, str]] = []
    sentinel = object()

    def fake_whisper(name: str, compute: str, language: str | None, asr_device: str) -> object:
        calls.append((name, compute, language, asr_device))
        return sentinel

    monkeypatch.setattr(transcriber_module, "WhisperTranscriber", fake_whisper)

    assert resolve_transcriber(model, "int8", "fr", "cuda") is sentinel
    assert calls == [(model, "int8", "fr", "cuda")]
