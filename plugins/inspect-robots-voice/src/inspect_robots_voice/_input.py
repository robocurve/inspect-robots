"""Threaded voice-input orchestration around injectable capture and transcription seams."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots.console import ConsolePoll
from inspect_robots_voice._capture import Device, MicrophoneCapture
from inspect_robots_voice._segmenter import EnergyGate
from inspect_robots_voice._transcriber import WhisperTranscriber

AudioArray = npt.NDArray[np.float32]

_SAMPLE_RATE = 16_000
_BLOCK_SAMPLES = 1_600
_CAPTURE_SECONDS = 30
_QUEUE_BLOCKS = _CAPTURE_SECONDS * _SAMPLE_RATE // _BLOCK_SAMPLES


class _Capture(Protocol):
    device_name: str

    def start(self) -> None: ...

    def close(self) -> None: ...


class _Transcriber(Protocol):
    def transcribe(self, audio: npt.ArrayLike) -> str | None: ...


class _Segmenter(Protocol):
    @property
    def is_open(self) -> bool: ...

    def push(self, block: npt.ArrayLike) -> AudioArray | None: ...

    def reset(self) -> None: ...


CaptureFactory = Callable[[Device, int, queue.Queue[AudioArray]], _Capture]
TranscriberFactory = Callable[[str, str, str, str], _Transcriber]
SegmenterFactory = Callable[[], _Segmenter]


def _capture_factory(
    device: Device, sample_rate: int, audio_queue: queue.Queue[AudioArray]
) -> _Capture:
    return MicrophoneCapture(device, sample_rate, audio_queue, blocksize=_BLOCK_SAMPLES)


def _transcriber_factory(model: str, compute: str, language: str, asr_device: str) -> _Transcriber:
    return WhisperTranscriber(model, compute, language, asr_device)


class VoiceInput:
    """Stream accepted microphone utterances into non-blocking operator polls."""

    def __init__(
        self,
        *,
        model: str = "small",
        device: Device = None,
        language: str = "en",
        compute: str = "auto",
        asr_device: str = "cpu",
        _capture_factory_override: CaptureFactory | None = None,
        _transcriber_factory_override: TranscriberFactory | None = None,
        _segmenter_factory_override: SegmenterFactory | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.language = language
        self.compute = compute
        self.asr_device = asr_device
        self._capture_factory = _capture_factory_override or _capture_factory
        self._transcriber_factory = _transcriber_factory_override or _transcriber_factory
        self._segmenter_factory = _segmenter_factory_override or EnergyGate
        self._audio_queue: queue.Queue[AudioArray] = queue.Queue(maxsize=_QUEUE_BLOCKS)
        self._output: deque[str] = deque()
        self._lock = threading.Lock()
        self._generation = 0
        self._last_generation = 0
        self._worker_error: Exception | None = None
        self._stop = threading.Event()
        self._capture: _Capture | None = None
        self._transcriber: _Transcriber | None = None
        self._segmenter: _Segmenter = self._segmenter_factory()
        self._thread: threading.Thread | None = None
        self._open_generation: int | None = None

    def start(self) -> str:
        """Load transcription, open capture, and start the worker once for this run."""
        if self._thread is not None and self._thread.is_alive():
            assert self._capture is not None
            return f"listening on {self._capture.device_name} (model={self.model})"
        self._stop.clear()
        self._transcriber = self._transcriber_factory(
            self.model, self.compute, self.language, self.asr_device
        )
        capture = self._capture_factory(self.device, _SAMPLE_RATE, self._audio_queue)
        self._capture = capture
        thread = threading.Thread(target=self._worker, name="inspect-robots-voice", daemon=True)
        self._thread = thread
        thread.start()
        try:
            capture.start()
        except BaseException:
            self.close()
            raise
        return f"listening on {capture.device_name} (model={self.model})"

    def poll(self) -> ConsolePoll:
        """Drain accepted utterances or re-raise the worker's first pending failure."""
        with self._lock:
            if self._worker_error is not None:
                error = self._worker_error
                self._worker_error = None
                raise error
            messages = tuple(self._output)
            self._output.clear()
        return ConsolePoll(messages=messages, end=None, sources=("voice",) * len(messages))

    def begin_trial(self) -> None:
        """Atomically invalidate prior utterances, then drain queued capture blocks."""
        with self._lock:
            self._generation += 1
            self._output.clear()
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """Stop capture and join the worker idempotently without raising."""
        capture = self._capture
        thread = self._thread
        self._capture = None
        self._thread = None
        self._stop.set()
        if capture is not None:
            with suppress(BaseException):
                capture.close()
        if thread is not None:
            with suppress(BaseException):
                thread.join()

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    block = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._process_block(block)
        except Exception as exc:
            # Exception, not BaseException: the session's detach path catches only
            # Exception, and anything wider must not escape into the rollout loop.
            with self._lock:
                self._worker_error = exc
            # A dead pipeline must not keep the microphone streaming: the callback
            # would fill the bounded queue and warn from PortAudio's thread, tearing
            # the operator's status line long after voice stopped mattering.
            capture = self._capture
            self._capture = None
            if capture is not None:
                with suppress(BaseException):
                    capture.close()

    def _process_block(self, block: npt.ArrayLike) -> None:
        with self._lock:
            generation = self._generation
        if generation != self._last_generation:
            # Any generation change wipes segmenter state wholesale: not just an open
            # segment, but pre-roll and sub-min-open candidates too, so up to ~400 ms
            # of pre-reset audio can never be prepended to the new trial's first
            # utterance. The worker owns the segmenter, so the reset happens here,
            # never in begin_trial().
            self._segmenter.reset()
            self._open_generation = None
            self._last_generation = generation

        was_open = self._segmenter.is_open
        utterance = self._segmenter.push(block)
        is_open = self._segmenter.is_open
        if not was_open and (is_open or utterance is not None):
            self._open_generation = generation
        if utterance is None:
            return

        open_generation = self._open_generation
        self._open_generation = None
        if open_generation is None:
            open_generation = generation
        with self._lock:
            if open_generation != self._generation:
                self._segmenter.reset()
                return
        assert self._transcriber is not None
        text = self._transcriber.transcribe(utterance)
        if text is not None:
            self._accept(text, open_generation)

    def _accept(self, text: str, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._output.append(text)
