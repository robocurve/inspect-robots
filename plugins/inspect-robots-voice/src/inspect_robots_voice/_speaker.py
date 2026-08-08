"""Off-thread policy-note audio with selectable delivery and bounded waiting.

Operator-ended trials cut narration in every delivery mode.
"""

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
from inspect_robots.types import OPERATOR_END
from inspect_robots_voice._tts import KokoroEngine, TtsEngine, resolve_model_files

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog
    from inspect_robots.rollout import TrialRecord

PlaybackDevice = str | int | None
EngineFactory = Callable[[], TtsEngine]
_QUEUE_SIZE = 4
_BLOCK_TIMEOUT = 15.0
_DRAIN_TIMEOUT = 15.0
_JOIN_TIMEOUT = 5.0
_CHUNK_SECONDS = 0.1
_MODES = ("blocking", "interrupt", "queue")
_DEFAULT_MODE = "interrupt"


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
    """Play narration off-thread; blocking waits are bounded and fail-open."""

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
        mode: str = _DEFAULT_MODE,
        engine_factory: EngineFactory | None = None,
        playback_factory: PlaybackFactory | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of: {', '.join(_MODES)}")
        self.voice = voice
        self.speed = speed
        self.volume = volume
        self.device = device
        self.lang = lang
        self.model = model
        self.voices = voices
        self.mode = mode

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
        self._speech_gen = 0
        self._block_degraded = False

    def start(self) -> None:
        """Build audio resources and start exactly one daemon narration worker."""
        with self._condition:
            if (
                self._disabled
                or self._closed
                or (self._thread is not None and self._thread.is_alive())
            ):
                return
        engine = self._engine_factory()
        playback = self._playback_factory()
        with self._condition:
            if not self._closed:
                self._stop.clear()
                self._playback = playback
                thread = threading.Thread(
                    target=self._worker,
                    args=(engine, playback),
                    name="inspect-robots-speaker",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
                return
        # close() won the race during the (possibly long) engine build; a closed
        # sink must stay closed rather than resurrect a worker.
        with suppress(BaseException):
            playback.close()

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        """Queue one delta without audio work, waiting boundedly only in blocking mode."""
        del t
        texts = extract_speech(messages)
        if not texts:
            return
        blocking_timed_out = False
        with self._condition:
            thread = self._thread
            if thread is None or not thread.is_alive() or self._disabled or self._stop.is_set():
                return
            if self.mode == "interrupt":
                self._speech_gen += 1
                self._queue.clear()
            elif self.mode == "blocking" and not self._block_degraded:
                deadline = time.monotonic() + _BLOCK_TIMEOUT
                while self._queue or self._inflight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._block_degraded = True
                        blocking_timed_out = True
                        break
                    self._condition.wait(timeout=remaining)
                    thread = self._thread
                    if (
                        thread is None
                        or not thread.is_alive()
                        or self._disabled
                        or self._stop.is_set()
                    ):
                        return
            for text in texts:
                if len(self._queue) >= _QUEUE_SIZE:
                    self._queue.popleft()
                    self._dropped += 1
                self._queue.append(text)
            self._condition.notify()
        if blocking_timed_out:
            print("speaker: blocking wait timed out; narration may lag", file=sys.stderr)

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Discard queued prior-trial notes and report accumulated overflow once."""
        del scene_id, epoch
        with self._condition:
            self._queue.clear()
            dropped = self._dropped
            self._dropped = 0
            self._condition.notify_all()
        if dropped:
            suffix = "" if dropped == 1 else "s"
            print(f"speaker: dropped {dropped} stale note{suffix}", file=sys.stderr)

    def on_trial_end(self, record: TrialRecord) -> None:
        """Silence operator-ended narration in every mode while natural endings play out."""
        if record.termination_reason != OPERATOR_END:
            return
        with self._condition:
            self._speech_gen += 1
            self._queue.clear()
            self._condition.notify_all()

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
            thread = self._thread
            playback = self._playback
            self._thread = None
            self._playback = None
            self._condition.notify_all()
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
                gen = self._speech_gen
                self._inflight = True
                self._condition.notify_all()
            try:
                if self._stop.is_set():
                    return
                if self._speech_gen != gen:
                    continue
                samples, sample_rate = engine.synthesize(text)
                if self._stop.is_set():
                    return
                if self._speech_gen != gen:
                    continue
                gained = np.asarray(samples * np.float32(self.volume), dtype=np.float32)
                chunk_size = max(1, int(sample_rate * _CHUNK_SECONDS))
                from inspect_robots_voice._capture import _active_speakers, _speakers_lock

                with _speakers_lock:
                    _active_speakers.add(self)
                try:
                    for start in range(0, len(gained), chunk_size):
                        if self._stop.is_set():
                            return
                        if self._speech_gen != gen:
                            break
                        playback.write(gained[start : start + chunk_size], sample_rate)
                finally:
                    with _speakers_lock:
                        _active_speakers.discard(self)
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
