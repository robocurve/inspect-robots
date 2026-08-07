"""Package exports and CLI-coerced factory validation."""

from __future__ import annotations

import pytest

import inspect_robots_voice
from inspect_robots_voice import VoiceInput, voice_input


def test_package_exports_and_version() -> None:
    assert inspect_robots_voice.__version__ == "0.3.0"
    assert inspect_robots_voice.__all__ == ["VoiceInput", "voice_input"]


@pytest.mark.parametrize("device", [None, 4, "USB microphone"])
def test_factory_accepts_supported_device_forms(device: str | int | None) -> None:
    voice = voice_input(
        model="medium",
        device=device,
        language="fr",
        compute="int8",
        asr_device="cuda",
    )

    assert isinstance(voice, VoiceInput)
    assert voice.model == "medium"
    assert voice.device == device
    assert voice.language == "fr"
    assert voice.compute == "int8"
    assert voice.asr_device == "cuda"


def test_factory_defaults() -> None:
    voice = voice_input()
    assert (voice.model, voice.device, voice.language, voice.compute, voice.asr_device) == (
        "parakeet-tdt-0.6b-v3",
        None,
        None,
        "auto",
        "cpu",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown": "x"}, "unexpected voice argument"),
        ({"model": 1}, "model must be a string"),
        ({"device": True}, "device must be"),
        ({"device": 1.5}, "device must be"),
        ({"language": 1}, "language must be a string or none"),
        ({"compute": False}, "compute must be a string"),
        ({"asr_device": 1}, "asr_device must be a string"),
    ],
)
def test_factory_rejects_unknown_or_mistyped_values(
    kwargs: dict[str, str | int | float | bool | None], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        voice_input(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": None},
        {"asr_device": "cpu"},
        {"compute": "auto"},
        {"language": None, "asr_device": "cpu", "compute": "auto"},
    ],
)
def test_factory_accepts_backend_neutral_parakeet_values(
    kwargs: dict[str, str | None],
) -> None:
    voice = voice_input(**kwargs)
    assert voice.model == "parakeet-tdt-0.6b-v3"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"language": "fr"},
            "parakeet models auto-detect language; drop -V language or pick a whisper model",
        ),
        (
            {"asr_device": "cuda"},
            "the parakeet backend runs on the CPU; use a whisper model for -V asr_device=cuda",
        ),
        (
            {"compute": "int8"},
            "compute applies to whisper models; the parakeet backend is fixed to int8",
        ),
    ],
)
def test_factory_rejects_whisper_only_values_for_parakeet(
    kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(TypeError) as exc_info:
        voice_input(**kwargs)
    assert str(exc_info.value) == message


def test_factory_accepts_whisper_only_values_for_parakeet_named_path() -> None:
    voice = voice_input(
        model="/models/parakeet-ct2",
        language="fr",
        compute="int8",
        asr_device="cuda",
    )

    assert voice.model == "/models/parakeet-ct2"
    assert voice.language == "fr"
    assert voice.compute == "int8"
    assert voice.asr_device == "cuda"


def test_unknown_parakeet_name_error_wins_over_backend_option_error() -> None:
    with pytest.raises(TypeError) as exc_info:
        voice_input(model="parakeet-tdt-1.1b", language="fr")

    message = str(exc_info.value)
    assert "unsupported parakeet model" in message
    assert "auto-detect language" not in message
