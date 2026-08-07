"""Local voice feedback for attended Inspect Robots evaluations.

The ``voice`` operator-input entry point captures microphone audio, transcribes accepted
utterances locally, and returns them through the core operator-message channel. Heavy audio and
model dependencies are loaded only when voice input starts.
"""

from __future__ import annotations

from inspect_robots_voice._input import VoiceInput

__all__ = ["VoiceInput", "voice_input"]

__version__ = "0.2.0"

ScalarValue = str | int | float | bool | None


def voice_input(**kwargs: ScalarValue) -> VoiceInput:
    """Build voice input from validated ``-V`` options.

    Supported keys are ``model`` (string, default ``"small"``), ``device`` (string or integer,
    default system input), ``language`` (string, default ``"en"``), ``compute`` (string,
    default ``"auto"``), and ``asr_device`` (string, default ``"cpu"``; pass ``"cuda"`` or
    ``"auto"`` to run Whisper on a GPU, which needs the CUDA runtime libraries installed).
    Unknown or incorrectly typed keys raise :class:`TypeError`.
    """
    allowed = {"model", "device", "language", "compute", "asr_device"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"unexpected voice argument(s): {', '.join(unknown)}")

    model = kwargs.get("model", "small")
    device = kwargs.get("device")
    language = kwargs.get("language", "en")
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
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    if not isinstance(compute, str):
        raise TypeError("compute must be a string")
    if not isinstance(asr_device, str):
        raise TypeError("asr_device must be a string")
    return VoiceInput(
        model=model, device=device, language=language, compute=compute, asr_device=asr_device
    )
