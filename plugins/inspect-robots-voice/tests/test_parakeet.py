"""Parakeet rejection gauntlet against timestamped fake model output."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots_voice._parakeet import ParakeetTranscriber
from inspect_robots_voice._transcriber import _HALLUCINATIONS


@dataclass
class _Result:
    text: str
    logprobs: list[float] | None = None


class _Model:
    def __init__(self, result: _Result) -> None:
        self.result = result
        self.calls: list[tuple[npt.NDArray[np.float32], int]] = []

    def recognize(self, samples: npt.NDArray[np.float32], *, sample_rate: int) -> _Result:
        self.calls.append((samples, sample_rate))
        return self.result


def _transcriber(result: _Result) -> tuple[ParakeetTranscriber, _Model, list[str]]:
    model = _Model(result)
    names: list[str] = []

    def factory(name: str) -> _Model:
        names.append(name)
        return model

    transcriber = ParakeetTranscriber("nemo-parakeet-tdt-0.6b-v3", _model_factory=factory)
    return transcriber, model, names


def _audio(seconds: float = 0.5) -> npt.NDArray[np.float64]:
    return np.zeros((1, round(16_000 * seconds)), dtype=np.float64)


def test_normal_sentence_is_accepted_with_mono_float32_audio_at_16_khz() -> None:
    transcriber, model, names = _transcriber(_Result(" Keep the wrist level. ", [-0.3, -0.5]))

    assert transcriber.transcribe(_audio()) == "Keep the wrist level."
    assert names == ["nemo-parakeet-tdt-0.6b-v3"]
    assert len(model.calls) == 1
    samples, sample_rate = model.calls[0]
    assert samples.shape == (8_000,)
    assert samples.dtype == np.float32
    assert sample_rate == 16_000


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_or_whitespace_text_is_rejected(text: str) -> None:
    transcriber, _model, _names = _transcriber(_Result(text, [-0.5]))
    assert transcriber.transcribe(_audio()) is None


@pytest.mark.parametrize("phrase", sorted(_HALLUCINATIONS))
def test_entire_hallucination_phrase_is_rejected_case_insensitively(phrase: str) -> None:
    transcriber, _model, _names = _transcriber(_Result(phrase.upper(), [-0.5]))
    assert transcriber.transcribe(_audio()) is None


def test_audio_shorter_than_four_tenths_is_rejected_before_model_call() -> None:
    transcriber, model, _names = _transcriber(_Result("too short", [-0.5]))

    assert transcriber.transcribe(_audio(0.399)) is None
    assert model.calls == []


def test_mean_log_probability_below_threshold_is_rejected() -> None:
    transcriber, _model, _names = _transcriber(_Result("uncertain", [-2.4, -2.7]))
    assert transcriber.transcribe(_audio()) is None


@pytest.mark.parametrize("logprobs", [None, []])
def test_missing_or_empty_logprobs_fail_open(logprobs: list[float] | None) -> None:
    transcriber, _model, _names = _transcriber(_Result("accepted", logprobs))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert transcriber.transcribe(_audio()) == "accepted"
