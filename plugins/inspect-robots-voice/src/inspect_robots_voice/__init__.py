"""Local spoken input and policy narration for Inspect Robots evaluations.

The ``voice`` operator-input entry point captures microphone audio, transcribes accepted
utterances locally, and returns them through the core operator-message channel. The ``speaker``
sink entry point narrates agent policy notes through local text-to-speech. Heavy audio and model
dependencies are loaded only when their component starts.
"""

from __future__ import annotations

from inspect_robots_voice._input import VoiceInput
from inspect_robots_voice._speaker import SpeakerSink
from inspect_robots_voice._transcriber import _classify_model

__all__ = ["SpeakerSink", "VoiceInput", "speaker_sink", "voice_input"]

__version__ = "0.4.0"

ScalarValue = str | int | float | bool | None


def speaker_sink(**kwargs: ScalarValue) -> SpeakerSink:
    """Build a speaker sink from validated ``-S`` options.

    Supported keys are ``voice`` (string, default ``"af_sarah"``), ``speed`` (positive
    number, default ``1.0``), ``volume`` (number from 0 through 1, default ``1.0``),
    ``device`` (string or integer, default system output), ``lang`` (string, default
    ``"en-us"``), and optional ``model`` and ``voices`` file paths. Unknown, incompatible,
    or incorrectly typed values raise :class:`TypeError`.
    """
    allowed = {"voice", "speed", "volume", "device", "lang", "model", "voices"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"unexpected speaker argument(s): {', '.join(unknown)}")

    voice = kwargs.get("voice", "af_sarah")
    speed = kwargs.get("speed", 1.0)
    volume = kwargs.get("volume", 1.0)
    device = kwargs.get("device")
    lang = kwargs.get("lang", "en-us")
    model = kwargs.get("model")
    voices = kwargs.get("voices")
    if not isinstance(voice, str):
        raise TypeError("voice must be a string")
    if not isinstance(speed, (int, float)) or isinstance(speed, bool):
        raise TypeError("speed must be a number")
    if speed <= 0:
        raise TypeError("speed must be positive")
    if not isinstance(volume, (int, float)) or isinstance(volume, bool):
        raise TypeError("volume must be a number")
    if not 0 <= volume <= 1:
        raise TypeError("volume must be between 0 and 1")
    if not (
        device is None
        or isinstance(device, str)
        or (isinstance(device, int) and not isinstance(device, bool))
    ):
        raise TypeError("device must be a string, integer, or none")
    if not isinstance(lang, str):
        raise TypeError("lang must be a string")
    if model is not None and not isinstance(model, str):
        raise TypeError("model must be a string or none")
    if voices is not None and not isinstance(voices, str):
        raise TypeError("voices must be a string or none")
    return SpeakerSink(
        voice=voice,
        speed=float(speed),
        volume=float(volume),
        device=device,
        lang=lang,
        model=model,
        voices=voices,
    )


def voice_input(**kwargs: ScalarValue) -> VoiceInput:
    """Build voice input from validated ``-V`` options.

    Supported keys are ``model`` (string, default ``"parakeet-tdt-0.6b-v3"``), ``device``
    (string or integer, default system input), ``language`` (string or ``None``, default
    ``None``), ``compute`` (string, default ``"auto"``), and ``asr_device`` (string, default
    ``"cpu"``). Unknown, incompatible, or incorrectly typed values raise :class:`TypeError`.
    """
    allowed = {"model", "device", "language", "compute", "asr_device"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"unexpected voice argument(s): {', '.join(unknown)}")

    model = kwargs.get("model", "parakeet-tdt-0.6b-v3")
    device = kwargs.get("device")
    language = kwargs.get("language")
    compute = kwargs.get("compute", "auto")
    asr_device = kwargs.get("asr_device", "cpu")
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if not (
        device is None
        or isinstance(device, str)
        or (isinstance(device, int) and not isinstance(device, bool))
    ):
        raise TypeError("device must be a string, integer, or none")
    if language is not None and not isinstance(language, str):
        raise TypeError("language must be a string or none")
    if not isinstance(compute, str):
        raise TypeError("compute must be a string")
    if not isinstance(asr_device, str):
        raise TypeError("asr_device must be a string")
    parakeet_model = _classify_model(model)
    if parakeet_model is not None:
        if language is not None:
            raise TypeError(
                "parakeet models auto-detect language; drop -V language or pick a whisper model"
            )
        if asr_device != "cpu":
            raise TypeError(
                "the parakeet backend runs on the CPU; use a whisper model for -V asr_device=cuda"
            )
        if compute != "auto":
            raise TypeError(
                "compute applies to whisper models; the parakeet backend is fixed to int8"
            )
    return VoiceInput(
        model=model, device=device, language=language, compute=compute, asr_device=asr_device
    )
