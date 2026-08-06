"""VoiceInput worker orchestration and trial-generation race handling."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from inspect_robots.console import ConsolePoll
from inspect_robots_voice._input import AudioArray, VoiceInput

_AUDIO = np.ones(8_000, dtype=np.float32)
_BLOCK = np.ones(1_600, dtype=np.float32)


class _ScriptedSegmenter:
    def __init__(
        self,
        steps: list[tuple[bool, AudioArray | None] | BaseException],
        *,
        pushed: threading.Event | None = None,
    ) -> None:
        self.steps = list(steps)
        self._is_open = False
        self.reset_calls = 0
        self.pushed = pushed

    @property
    def is_open(self) -> bool:
        return self._is_open

    def push(self, block: object) -> AudioArray | None:
        del block
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            if self.pushed is not None:
                self.pushed.set()
            raise step
        self._is_open, utterance = step
        return utterance

    def reset(self) -> None:
        self.reset_calls += 1
        self._is_open = False


class _Transcriber:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.calls: list[object] = []

    def transcribe(self, audio: object) -> str | None:
        self.calls.append(audio)
        return self.text


class _Capture:
    def __init__(
        self,
        audio_queue: queue.Queue[AudioArray],
        blocks: tuple[AudioArray, ...] = (),
    ) -> None:
        self.audio_queue = audio_queue
        self.blocks = blocks
        self.device_name = "test microphone"
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        for block in self.blocks:
            self.audio_queue.put_nowait(block)

    def close(self) -> None:
        self.close_calls += 1


def _voice(
    segmenter: _ScriptedSegmenter,
    transcriber: _Transcriber,
    *,
    capture_factory: Callable[[object, int, queue.Queue[AudioArray]], _Capture] | None = None,
) -> VoiceInput:
    voice = VoiceInput(
        _segmenter_factory_override=lambda: segmenter,
        _transcriber_factory_override=lambda _model, _compute, _language: transcriber,
        _capture_factory_override=capture_factory,
    )
    voice._transcriber = transcriber
    return voice


def test_accepted_utterance_flows_to_poll_with_voice_sources() -> None:
    segmenter = _ScriptedSegmenter([(True, None), (False, _AUDIO)])
    voice = _voice(segmenter, _Transcriber("move left"))

    voice._process_block(_BLOCK)
    voice._process_block(_BLOCK)

    assert voice.poll() == ConsolePoll(
        messages=("move left",),
        end=None,
        sources=("voice",),
    )
    assert voice.poll() == ConsolePoll()


def test_silence_and_rejected_transcription_send_nothing() -> None:
    silence = _voice(_ScriptedSegmenter([(False, None)]), _Transcriber("unused"))
    silence._process_block(_BLOCK)
    assert silence.poll() == ConsolePoll()

    rejected = _voice(_ScriptedSegmenter([(False, _AUDIO)]), _Transcriber(None))
    rejected._process_block(_BLOCK)
    assert rejected.poll() == ConsolePoll()


def test_begin_trial_atomically_clears_output_and_drains_capture_queue() -> None:
    voice = _voice(_ScriptedSegmenter([]), _Transcriber("unused"))
    voice._accept("queued before trial", 0)
    voice._audio_queue.put_nowait(_BLOCK)

    voice.begin_trial()

    assert voice.poll() == ConsolePoll()
    with pytest.raises(queue.Empty):
        voice._audio_queue.get_nowait()


class _BlockingCloseSegmenter(_ScriptedSegmenter):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__([])
        self.entered = entered
        self.release = release
        self.calls = 0

    def push(self, block: object) -> AudioArray | None:
        del block
        self.calls += 1
        if self.calls == 1:
            self._is_open = True
            return None
        self.entered.set()
        assert self.release.wait(timeout=2)
        self._is_open = False
        return _AUDIO


def test_utterance_closing_after_generation_bump_is_discarded() -> None:
    entered = threading.Event()
    release = threading.Event()
    segmenter = _BlockingCloseSegmenter(entered, release)
    transcriber = _Transcriber("stale")
    voice = _voice(segmenter, transcriber)
    voice._process_block(_BLOCK)
    worker = threading.Thread(target=voice._process_block, args=(_BLOCK,))
    worker.start()
    assert entered.wait(timeout=2)

    voice.begin_trial()
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert transcriber.calls == []
    assert segmenter.reset_calls == 1
    assert voice.poll() == ConsolePoll()


class _BlockingTranscriber(_Transcriber):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__("stale transcription")
        self.entered = entered
        self.release = release

    def transcribe(self, audio: object) -> str | None:
        self.calls.append(audio)
        self.entered.set()
        assert self.release.wait(timeout=2)
        return self.text


def test_transcription_finishing_after_generation_bump_is_discarded() -> None:
    entered = threading.Event()
    release = threading.Event()
    transcriber = _BlockingTranscriber(entered, release)
    voice = _voice(_ScriptedSegmenter([(False, _AUDIO)]), transcriber)
    worker = threading.Thread(target=voice._process_block, args=(_BLOCK,))
    worker.start()
    assert entered.wait(timeout=2)

    voice.begin_trial()
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(transcriber.calls) == 1
    assert voice.poll() == ConsolePoll()


def test_stale_open_utterance_is_worker_reset_before_next_push() -> None:
    segmenter = _ScriptedSegmenter([(True, None), (False, None)])
    voice = _voice(segmenter, _Transcriber("unused"))
    voice._process_block(_BLOCK)
    voice.begin_trial()

    voice._process_block(_BLOCK)

    assert segmenter.reset_calls == 1
    assert voice._open_generation is None


class _CheckingDeque(deque[str]):
    def __init__(
        self,
        lock: threading.Lock,
        appended_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.lock = lock
        self.appended_event = appended_event
        self.appended_while_locked = False
        self.cleared_while_locked = False

    def append(self, value: str) -> None:
        self.appended_while_locked = self.lock.locked()
        super().append(value)
        if self.appended_event is not None:
            self.appended_event.set()

    def clear(self) -> None:
        self.cleared_while_locked = self.lock.locked()
        super().clear()


def test_final_generation_check_and_append_share_the_begin_trial_lock() -> None:
    voice = _voice(_ScriptedSegmenter([]), _Transcriber("unused"))
    output = _CheckingDeque(voice._lock)
    voice._output = cast(deque[str], output)

    voice._accept("accepted", 0)

    assert output.appended_while_locked
    assert tuple(output) == ("accepted",)

    voice.begin_trial()

    assert output.cleared_while_locked
    assert tuple(output) == ()


def test_worker_drains_capture_and_close_is_idempotent_and_joins() -> None:
    accepted = threading.Event()
    segmenter = _ScriptedSegmenter([(True, None), (False, _AUDIO)])
    transcriber = _Transcriber("worker message")
    captures: list[_Capture] = []

    def capture_factory(
        _device: object, _sample_rate: int, audio_queue: queue.Queue[AudioArray]
    ) -> _Capture:
        capture = _Capture(audio_queue, (_BLOCK, _BLOCK))
        captures.append(capture)
        return capture

    voice = _voice(segmenter, transcriber, capture_factory=capture_factory)
    output = _CheckingDeque(voice._lock, accepted)
    voice._output = cast(deque[str], output)

    assert voice.start() == "listening on test microphone (model=small)"
    thread = voice._thread
    assert thread is not None
    assert accepted.wait(timeout=2)
    assert voice.poll().messages == ("worker message",)
    voice.close()
    voice.close()

    assert not thread.is_alive()
    assert captures[0].start_calls == 1
    assert captures[0].close_calls == 1


def test_second_start_reuses_live_capture() -> None:
    segmenter = _ScriptedSegmenter([])
    transcriber = _Transcriber(None)
    captures: list[_Capture] = []

    def capture_factory(
        _device: object, _sample_rate: int, audio_queue: queue.Queue[AudioArray]
    ) -> _Capture:
        capture = _Capture(audio_queue)
        captures.append(capture)
        return capture

    voice = _voice(segmenter, transcriber, capture_factory=capture_factory)
    try:
        first = voice.start()
        second = voice.start()
    finally:
        voice.close()

    assert first == second == "listening on test microphone (model=small)"
    assert len(captures) == 1


def test_worker_error_is_reraised_from_next_poll_only() -> None:
    failed = threading.Event()
    segmenter = _ScriptedSegmenter([RuntimeError("worker failed")], pushed=failed)

    def capture_factory(
        _device: object, _sample_rate: int, audio_queue: queue.Queue[AudioArray]
    ) -> _Capture:
        return _Capture(audio_queue, (_BLOCK,))

    voice = _voice(segmenter, _Transcriber(None), capture_factory=capture_factory)
    voice.start()
    assert failed.wait(timeout=2)
    thread = voice._thread
    assert thread is not None
    thread.join(timeout=2)
    assert not thread.is_alive()
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            voice.poll()
        assert voice.poll() == ConsolePoll()
    finally:
        voice.close()


def test_start_failure_closes_partial_capture_and_propagates() -> None:
    class FailingCapture(_Capture):
        def start(self) -> None:
            raise RuntimeError("cannot open microphone")

    captures: list[FailingCapture] = []

    def capture_factory(
        _device: object, _sample_rate: int, audio_queue: queue.Queue[AudioArray]
    ) -> FailingCapture:
        capture = FailingCapture(audio_queue)
        captures.append(capture)
        return capture

    voice = _voice(
        _ScriptedSegmenter([]),
        _Transcriber(None),
        capture_factory=capture_factory,
    )

    with pytest.raises(RuntimeError, match="cannot open microphone"):
        voice.start()

    assert captures[0].close_calls == 1
