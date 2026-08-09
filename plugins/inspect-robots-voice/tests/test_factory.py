"""Package exports and CLI-coerced factory validation."""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

import inspect_robots_voice
from inspect_robots_voice import SpeakerSink, VoiceInput, speaker_sink, voice_input


def test_package_exports_and_version() -> None:
    assert inspect_robots_voice.__version__ == "0.5.1"
    assert inspect_robots_voice.__all__ == [
        "SpeakerSink",
        "VoiceInput",
        "speaker_sink",
        "voice_input",
    ]


def test_speaker_factory_defaults() -> None:
    speaker = speaker_sink()

    assert isinstance(speaker, SpeakerSink)
    assert (
        speaker.voice,
        speaker.speed,
        speaker.volume,
        speaker.device,
        speaker.lang,
        speaker.model,
        speaker.voices,
        speaker.mode,
    ) == ("af_sarah", 1.0, 1.0, None, "en-us", None, None, "interrupt")


@pytest.mark.parametrize("mode", ["blocking", "interrupt", "queue"])
def test_speaker_factory_accepts_speech_modes(mode: str) -> None:
    speaker = speaker_sink(mode=mode)

    assert speaker.mode == mode


@pytest.mark.parametrize("device", [None, 4, "USB speaker"])
def test_speaker_factory_accepts_options_and_device_forms(device: str | int | None) -> None:
    speaker = speaker_sink(
        voice="bf_emma",
        speed=2,
        volume=0,
        device=device,
        lang="en-gb",
        model="/models/kokoro.onnx",
        voices="/models/voices.bin",
    )

    assert speaker.voice == "bf_emma"
    assert speaker.speed == 2.0
    assert speaker.volume == 0.0
    assert speaker.device == device
    assert speaker.lang == "en-gb"
    assert speaker.model == "/models/kokoro.onnx"
    assert speaker.voices == "/models/voices.bin"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown": "x"}, "unexpected speaker argument"),
        ({"voice": 1}, "voice must be a string"),
        ({"speed": "fast"}, "speed must be a number"),
        ({"speed": True}, "speed must be a number"),
        ({"volume": "loud"}, "volume must be a number"),
        ({"volume": False}, "volume must be a number"),
        ({"device": True}, "device must be"),
        ({"device": 1.5}, "device must be"),
        ({"lang": None}, "lang must be a string"),
        ({"model": 1}, "model must be a string or none"),
        ({"voices": False}, "voices must be a string or none"),
        ({"mode": 1}, "mode must be a string and one of: blocking, interrupt, queue"),
        ({"mode": "later"}, "mode must be a string and one of: blocking, interrupt, queue"),
    ],
)
def test_speaker_factory_rejects_unknown_or_mistyped_values(
    kwargs: dict[str, str | int | float | bool | None], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        speaker_sink(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speed": 0}, "speed must be positive"),
        ({"speed": -0.1}, "speed must be positive"),
        ({"volume": -0.1}, "volume must be between 0 and 1"),
        ({"volume": 1.1}, "volume must be between 0 and 1"),
    ],
)
def test_speaker_factory_rejects_numeric_range_violations(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        speaker_sink(**kwargs)


def test_speaker_entry_point_resolves_from_dev_install() -> None:
    matches = [
        entry
        for entry in entry_points(group="inspect_robots.sinks")
        if entry.name == "speaker" and entry.value == "inspect_robots_voice:speaker_sink"
    ]

    assert len(matches) == 1
    assert matches[0].load() is speaker_sink


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
