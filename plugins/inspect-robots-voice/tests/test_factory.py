"""Package exports and CLI-coerced factory validation."""

from __future__ import annotations

import pytest

import inspect_robots_voice
from inspect_robots_voice import VoiceInput, voice_input


def test_package_exports_and_version() -> None:
    assert inspect_robots_voice.__version__ == "0.1.0"
    assert inspect_robots_voice.__all__ == ["VoiceInput", "voice_input"]


@pytest.mark.parametrize("device", [None, 4, "USB microphone"])
def test_factory_accepts_supported_device_forms(device: str | int | None) -> None:
    voice = voice_input(
        model="medium",
        device=device,
        language="fr",
        compute="int8",
    )

    assert isinstance(voice, VoiceInput)
    assert voice.model == "medium"
    assert voice.device == device
    assert voice.language == "fr"
    assert voice.compute == "int8"


def test_factory_defaults() -> None:
    voice = voice_input()
    assert (voice.model, voice.device, voice.language, voice.compute) == (
        "small",
        None,
        "en",
        "auto",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown": "x"}, "unexpected voice argument"),
        ({"model": 1}, "model must be a string"),
        ({"device": True}, "device must be"),
        ({"device": 1.5}, "device must be"),
        ({"language": None}, "language must be a string"),
        ({"compute": False}, "compute must be a string"),
    ],
)
def test_factory_rejects_unknown_or_mistyped_values(
    kwargs: dict[str, str | int | float | bool | None], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        voice_input(**kwargs)
