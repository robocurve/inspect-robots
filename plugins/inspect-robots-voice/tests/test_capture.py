"""Microphone selection and callback backpressure with a fake sounddevice module."""

from __future__ import annotations

import builtins
import queue
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inspect_robots_voice._capture import MicrophoneCapture


class _Stream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _SoundDevice:
    def __init__(self, devices: list[dict[str, object]], default: object = (0, -1)) -> None:
        self.devices = devices
        self.default = SimpleNamespace(device=default)
        self.streams: list[_Stream] = []

    def query_devices(self) -> list[dict[str, object]]:
        return self.devices

    def InputStream(self, **kwargs: Any) -> _Stream:
        stream = _Stream(**kwargs)
        self.streams.append(stream)
        return stream


_DEVICES = [
    {"name": "Built-in Output", "max_input_channels": 0},
    {"name": "USB Conference Mic", "max_input_channels": 1},
    {"name": "USB Headset Mic", "max_input_channels": 2},
]


@pytest.mark.parametrize(
    ("requested", "default", "expected_index", "expected_name"),
    [
        (1, (2, -1), 1, "USB Conference Mic"),
        ("conference", (2, -1), 1, "USB Conference Mic"),
        ("HEADSET", (1, -1), 2, "USB Headset Mic"),
        (None, (2, -1), 2, "USB Headset Mic"),
        (None, 1, 1, "USB Conference Mic"),
        (None, (-1, -1), None, "system default"),
    ],
)
def test_device_resolution(
    requested: str | int | None,
    default: object,
    expected_index: int | None,
    expected_name: str,
) -> None:
    sounddevice = _SoundDevice(_DEVICES, default)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)

    capture = MicrophoneCapture(
        requested,
        16_000,
        audio_queue,
        _sounddevice=sounddevice,
    )

    assert capture.device_name == expected_name
    assert sounddevice.streams[0].kwargs["device"] == expected_index
    assert sounddevice.streams[0].kwargs["samplerate"] == 16_000


@pytest.mark.parametrize("requested", ["usb", "missing", 0, 99])
def test_ambiguous_or_missing_device_lists_available_inputs(
    requested: str | int,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        MicrophoneCapture(
            requested,
            16_000,
            queue.Queue(maxsize=2),
            _sounddevice=_SoundDevice(_DEVICES),
        )

    message = str(excinfo.value)
    assert "Available input devices:" in message
    assert "1: USB Conference Mic" in message
    assert "2: USB Headset Mic" in message
    assert "Built-in Output" not in message


def test_empty_device_table_is_rendered_in_error() -> None:
    with pytest.raises(ValueError, match=r"\(none\)"):
        MicrophoneCapture(
            "missing",
            16_000,
            queue.Queue(maxsize=1),
            _sounddevice=_SoundDevice([]),
        )


def test_start_and_close_delegate_to_stream() -> None:
    sounddevice = _SoundDevice(_DEVICES)
    capture = MicrophoneCapture(
        1,
        16_000,
        queue.Queue(maxsize=1),
        _sounddevice=sounddevice,
    )

    capture.start()
    capture.close()

    stream = sounddevice.streams[0]
    assert stream.start_calls == stream.stop_calls == stream.close_calls == 1


def test_full_queue_drops_oldest_and_warns_only_once() -> None:
    sounddevice = _SoundDevice(_DEVICES)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    MicrophoneCapture(1, 16_000, audio_queue, _sounddevice=sounddevice)
    callback = sounddevice.streams[0].kwargs["callback"]
    audio_queue.put_nowait(np.array([1.0], dtype=np.float32))

    with pytest.warns(RuntimeWarning, match="dropping oldest audio") as warning_records:
        callback(np.array([[2.0], [3.0]], dtype=np.float32), 2, object(), object())
        callback(np.array([[4.0]], dtype=np.float32), 1, object(), object())

    assert len(warning_records) == 1
    assert np.array_equal(audio_queue.get_nowait(), np.array([4.0], dtype=np.float32))


def test_callback_enqueues_without_warning_when_capacity_remains() -> None:
    sounddevice = _SoundDevice(_DEVICES)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
    capture = MicrophoneCapture(1, 16_000, audio_queue, _sounddevice=sounddevice)

    capture._callback(np.array([[1.0], [2.0]]), 2, object(), object())

    assert np.array_equal(audio_queue.get_nowait(), np.array([1.0, 2.0], dtype=np.float32))


def test_missing_portaudio_error_carries_install_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PortAudio load failure names the per-OS package commands, not just the symptom."""
    real_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    with pytest.raises(OSError, match="libportaudio2"):
        MicrophoneCapture(None, 16_000, audio_queue)
