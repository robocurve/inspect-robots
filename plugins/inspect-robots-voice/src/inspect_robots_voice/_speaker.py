"""Non-blocking policy-note speech with bounded stale-work loss."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots.logging.sink import NullSink
from inspect_robots_voice._tts import KokoroEngine, TtsEngine, resolve_model_files

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog

PlaybackDevice = str | int | None
EngineFactory = Callable[[], TtsEngine]
_QUEUE_SIZE = 4
_DRAIN_TIMEOUT = 15.0
_JOIN_TIMEOUT = 5.0
_CHUNK_SECONDS = 0.1


class _Playback(Protocol):
    def write(self, samples: npt.NDArray[np.float32], sample_rate: int) -> None: ...

    def close(self) -> None: ...


PlaybackFactory = Callable[[], _Playback]


class _SoundDevicePlayback:
    """Write mono chunks through a lazily imported sounddevice output stream."""

    def __init__(self, device: PlaybackDevice) -> None:
        import sounddevice

        self._sounddevice: Any = sounddevice
        self._device = device
        self._stream: Any | None = None
        self._sample_rate: int | None = None

    def write(self, samples: npt.NDArray[np.float32], sample_rate: int) -> None:
        if self._stream is None or self._sample_rate != sample_rate:
            self.close()
            self._stream = self._sounddevice.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
            )
            self._stream.start()
            self._sample_rate = sample_rate
        self._stream.write(samples)

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        self._sample_rate = None
        if stream is not None:
            stream.close()


def extract_speech(messages: Sequence[Any]) -> list[str]:
    """Extract operator narration from defensively parsed assistant tool calls."""
    spoken: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(arguments, dict):
                continue
            field = "summary" if name == "done" else "reason" if name == "give_up" else "note"
            text = arguments.get(field)
            if isinstance(text, str) and (text := text.strip()):
                spoken.append(text)
    return spoken


class SpeakerSink(NullSink):
    """Speak policy narration asynchronously without delaying the control loop."""

    def __init__(
        self,
        *,
        voice: str = "af_sarah",
        speed: float = 1.0,
        volume: float = 1.0,
        device: PlaybackDevice = None,
        lang: str = "en-us",
        model: str | None = None,
        voices: str | None = None,
        engine_factory: EngineFactory | None = None,
        playback_factory: PlaybackFactory | None = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.volume = volume
        self.device = device
        self.lang = lang
        self.model = model
        self.voices = voices

        def build_engine() -> TtsEngine:
            model_path, voices_path = resolve_model_files(self.model, self.voices)
            return KokoroEngine(
                model_path,
                voices_path,
                voice=self.voice,
                speed=self.speed,
                lang=self.lang,
            )

        self._engine_factory = engine_factory or build_engine
        self._playback_factory = playback_factory or (lambda: _SoundDevicePlayback(self.device))
        self._queue: deque[str] = deque()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._playback: _Playback | None = None
        self._inflight = False
        self._disabled = False
        self._closed = False
        self._warned = False
        self._dropped = 0

    def start(self) -> None:
        """Build audio resources and start exactly one daemon narration worker."""
        with self._condition:
            if self._disabled or (self._thread is not None and self._thread.is_alive()):
                return
        engine = self._engine_factory()
        playback = self._playback_factory()
        self._stop.clear()
        self._closed = False
        self._playback = playback
        thread = threading.Thread(
            target=self._worker,
            args=(engine, playback),
            name="inspect-robots-speaker",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        """Enqueue spoken fields from one transcript delta without blocking on audio."""
        del t
        texts = extract_speech(messages)
        if not texts:
            return
        with self._condition:
            thread = self._thread
            if thread is None or not thread.is_alive() or self._disabled or self._stop.is_set():
                return
            for text in texts:
                if len(self._queue) >= _QUEUE_SIZE:
                    self._queue.popleft()
                    self._dropped += 1
                self._queue.append(text)
            self._condition.notify()

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Discard queued prior-trial notes and report accumulated overflow once."""
        del scene_id, epoch
        with self._condition:
            self._queue.clear()
            dropped = self._dropped
            self._dropped = 0
        if dropped:
            suffix = "" if dropped == 1 else "s"
            print(f"speaker: dropped {dropped} stale note{suffix}", file=sys.stderr)

    def on_eval_end(self, log: EvalLog) -> None:
        """Drain successful narration within a bound, or abort every other terminal status."""
        if log.status == "success":
            self._drain(_DRAIN_TIMEOUT)
        self.close()

    def close(self) -> None:
        """Hard-abort queued playback and close audio resources idempotently."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            self._queue.clear()
            self._condition.notify_all()
            thread = self._thread
            playback = self._playback
            self._thread = None
            self._playback = None
        if thread is not None:
            with suppress(BaseException):
                thread.join(timeout=_JOIN_TIMEOUT)
        if playback is not None:
            with suppress(BaseException):
                playback.close()

    def _drain(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._queue or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(timeout=remaining)

    def _worker(self, engine: TtsEngine, playback: _Playback) -> None:
        while not self._stop.is_set():
            with self._condition:
                while not self._queue and not self._stop.is_set():
                    self._condition.wait(timeout=0.1)
                if self._stop.is_set():
                    return
                text = self._queue.popleft()
                self._inflight = True
            try:
                if self._stop.is_set():
                    return
                samples, sample_rate = engine.synthesize(text)
                if self._stop.is_set():
                    return
                gained = np.asarray(samples * np.float32(self.volume), dtype=np.float32)
                chunk_size = max(1, int(sample_rate * _CHUNK_SECONDS))
                for start in range(0, len(gained), chunk_size):
                    if self._stop.is_set():
                        return
                    playback.write(gained[start : start + chunk_size], sample_rate)
            except Exception as exc:
                with self._condition:
                    self._disabled = True
                    self._stop.set()
                    self._queue.clear()
                    self._condition.notify_all()
                if not self._warned:
                    self._warned = True
                    print(
                        f"speaker: disabled after {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                return
            finally:
                with self._condition:
                    self._inflight = False
                    self._condition.notify_all()
