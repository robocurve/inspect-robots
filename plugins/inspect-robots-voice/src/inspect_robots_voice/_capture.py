"""Lazy sounddevice microphone capture with bounded callback backpressure."""

from __future__ import annotations

import queue as queue_module
import threading
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any, cast

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]
Device = str | int | None


_speakers_lock = threading.Lock()
_active_speakers: set[object] = set()


class MicrophoneCapture:
    """Capture 16-bit-equivalent mono float blocks without blocking PortAudio's callback."""

    def __init__(
        self,
        device: Device,
        sample_rate: int,
        queue: queue_module.Queue[AudioArray],
        *,
        _sounddevice: Any = None,
        blocksize: int = 1_600,
    ) -> None:
        if _sounddevice is None:
            try:
                import sounddevice
            except OSError as exc:
                # sounddevice loads the PortAudio shared library at import time;
                # pip installs the Python binding but not the library on Linux.
                raise OSError(
                    f"{exc}. Voice mode needs the PortAudio system library: "
                    "install it with 'sudo apt install libportaudio2' (Debian/Ubuntu), "
                    "'sudo dnf install portaudio' (Fedora), or 'brew install portaudio' "
                    "(macOS), then re-run. See docs/guide/voice-mode for details."
                ) from exc

            _sounddevice = sounddevice
        self._sounddevice = _sounddevice
        self._queue = queue
        self._warned_full = False
        devices = cast(Sequence[Mapping[str, Any]], self._sounddevice.query_devices())
        self._device_index, self.device_name = self._resolve_device(device, devices)
        self._stream = self._sounddevice.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=self._device_index,
            callback=self._callback,
        )

    def start(self) -> None:
        """Start delivering microphone blocks to the supplied bounded queue."""
        self._stream.start()

    def close(self) -> None:
        """Stop and close the PortAudio stream."""
        self._stream.stop()
        self._stream.close()

    def _resolve_device(
        self, requested: Device, devices: Sequence[Mapping[str, Any]]
    ) -> tuple[int | None, str]:
        input_devices = [
            (index, device)
            for index, device in enumerate(devices)
            if int(device.get("max_input_channels", 0)) > 0
        ]
        if requested is None:
            default = self._sounddevice.default.device
            try:
                raw_index = default[0]
            except (TypeError, IndexError):
                raw_index = default
            if isinstance(raw_index, int) and raw_index >= 0:
                return self._validated_index(raw_index, input_devices, devices)
            return None, "system default"
        if isinstance(requested, int) and not isinstance(requested, bool):
            return self._validated_index(requested, input_devices, devices)

        assert isinstance(requested, str)
        needle = requested.casefold()
        matches = [
            (index, device)
            for index, device in input_devices
            if needle in str(device.get("name", "")).casefold()
        ]
        if len(matches) == 1:
            index, match = matches[0]
            return index, str(match.get("name", f"device {index}"))
        reason = "ambiguous" if matches else "not found"
        raise ValueError(
            f"input device {requested!r} {reason}. Available input devices:\n"
            f"{self._device_table(input_devices)}"
        )

    def _validated_index(
        self,
        index: int,
        input_devices: Sequence[tuple[int, Mapping[str, Any]]],
        all_devices: Sequence[Mapping[str, Any]],
    ) -> tuple[int, str]:
        if 0 <= index < len(all_devices) and any(i == index for i, _device in input_devices):
            return index, str(all_devices[index].get("name", f"device {index}"))
        raise ValueError(
            f"input device index {index} not found. Available input devices:\n"
            f"{self._device_table(input_devices)}"
        )

    @staticmethod
    def _device_table(input_devices: Sequence[tuple[int, Mapping[str, Any]]]) -> str:
        if not input_devices:
            return "  (none)"
        return "\n".join(
            f"  {index}: {device.get('name', f'device {index}')}" for index, device in input_devices
        )

    def _callback(
        self,
        indata: npt.ArrayLike,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info, status
        block = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        with _speakers_lock:
            active = bool(_active_speakers)
        if active:
            block.fill(0)
        try:
            self._queue.put_nowait(block)
            return
        except queue_module.Full:
            pass
        with suppress(queue_module.Empty):
            self._queue.get_nowait()
        self._queue.put_nowait(block)
        if not self._warned_full:
            self._warned_full = True
            warnings.warn(
                "voice capture queue full; dropping oldest audio",
                RuntimeWarning,
                stacklevel=2,
            )
