"""Policy-note extraction and fake-only speaker lifecycle tests."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import inspect_robots_voice._speaker as speaker_module
from inspect_robots.log import EvalLog
from inspect_robots.rollout import TrialRecord
from inspect_robots_voice._speaker import SpeakerSink, extract_speech


def _tool_call(name: str, arguments: object) -> dict[str, object]:
    return {"function": {"name": name, "arguments": arguments}}


def _assistant(*calls: dict[str, object]) -> dict[str, object]:
    return {"role": "assistant", "tool_calls": list(calls)}


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([_assistant(_tool_call("move", '{"note": "  moving left  "}'))], ["moving left"]),
        (
            [
                _assistant(
                    _tool_call("move", {"note": "first"}),
                    _tool_call("capture", json.dumps({"note": "second"})),
                )
            ],
            ["first", "second"],
        ),
        (
            [_assistant(_tool_call("done", {"summary": "finished", "hindsight": "retry"}))],
            ["finished"],
        ),
        (
            [_assistant(_tool_call("give_up", {"reason": "blocked", "hindsight": "later"}))],
            ["blocked"],
        ),
        ([_assistant(_tool_call("move", "{bad json"))], []),
        ([_assistant(_tool_call("move", "[]"))], []),
        ([_assistant(_tool_call("move", {"note": "  "}))], []),
        ([_assistant(_tool_call("done", {"hindsight": "not spoken"}))], []),
        ([{"role": "user", "tool_calls": [_tool_call("move", {"note": "no"})]}], []),
        (["not a message"], []),
        ([_assistant({"function": "not a mapping"}, {"not_function": True})], []),
        ([{"role": "assistant", "tool_calls": "not a list"}], []),
    ],
)
def test_extract_speech_table(messages: list[object], expected: list[str]) -> None:
    assert extract_speech(messages) == expected


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
        self.calls.append(text)
        return np.ones(3, dtype=np.float32), 10


class _FakePlayback:
    def __init__(self) -> None:
        self.writes: list[tuple[npt.NDArray[np.float32], int]] = []
        self.close_calls = 0

    def write(self, samples: npt.NDArray[np.float32], sample_rate: int) -> None:
        self.writes.append((samples.copy(), sample_rate))

    def close(self) -> None:
        self.close_calls += 1


class _BlockingEngine(_FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
        self.calls.append(text)
        self.entered.set()
        self.release.wait(timeout=2.0)
        return np.ones(3, dtype=np.float32), 10


class _TaggedEngine(_FakeEngine):
    def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
        self.calls.append(text)
        value = float(self.calls.index(text) + 1)
        return np.full(3, value, dtype=np.float32), 10


class _GatedPlayback(_FakePlayback):
    def __init__(self, gated_writes: int = 0) -> None:
        super().__init__()
        self.entered = [threading.Event() for _ in range(gated_writes)]
        self.release = [threading.Event() for _ in range(gated_writes)]
        self._writes_condition = threading.Condition()

    def write(self, samples: npt.NDArray[np.float32], sample_rate: int) -> None:
        with self._writes_condition:
            super().write(samples, sample_rate)
            index = len(self.writes) - 1
            self._writes_condition.notify_all()
        if index < len(self.entered):
            self.entered[index].set()
            self.release[index].wait(timeout=2.0)

    def wait_for_writes(self, count: int, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._writes_condition:
            while len(self.writes) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._writes_condition.wait(timeout=remaining)
        return True


def _sink(
    engine: _FakeEngine,
    playback: _FakePlayback,
    *,
    volume: float = 1.0,
    mode: str = "interrupt",
) -> SpeakerSink:
    return SpeakerSink(
        volume=volume,
        mode=mode,
        engine_factory=lambda: engine,
        playback_factory=lambda: playback,
    )


def _wait_until(predicate: object, timeout: float = 1.0) -> None:
    assert callable(predicate)
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for speaker worker")
        time.sleep(0.005)


def _log(status: str) -> EvalLog:
    return cast(EvalLog, SimpleNamespace(status=status))


def test_enqueue_is_spoken_in_order_with_volume_gain() -> None:
    engine = _FakeEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback, volume=0.25, mode="queue")
    sink.start()

    sink.log_policy_messages(
        3,
        [
            _assistant(
                _tool_call("move", {"note": "one"}),
                _tool_call("capture", {"note": "two"}),
                _tool_call("done", {"summary": "three"}),
            )
        ],
    )
    _wait_until(lambda: len(engine.calls) == 3 and len(playback.writes) == 9)
    sink.close()

    assert engine.calls == ["one", "two", "three"]
    assert all(sample_rate == 10 for _, sample_rate in playback.writes)
    assert all(
        np.array_equal(chunk, np.array([0.25], dtype=np.float32)) for chunk, _ in playback.writes
    )


def test_overflow_drops_oldest_and_reports_once_at_next_trial(
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _BlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback, mode="queue")
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "inflight"}))])
    assert engine.entered.wait(timeout=1.0)

    for text in ["oldest", "two", "three", "four", "newest"]:
        sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": text}))])
    engine.release.set()
    _wait_until(lambda: len(engine.calls) == 5)

    sink.on_trial_start("next", 1)
    sink.on_trial_start("later", 2)
    sink.close()

    assert engine.calls == ["inflight", "two", "three", "four", "newest"]
    assert capsys.readouterr().err == "speaker: dropped 1 stale note\n"


def test_trial_start_clears_queued_prior_trial_notes() -> None:
    engine = _BlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback, mode="queue")
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "inflight"}))])
    assert engine.entered.wait(timeout=1.0)
    sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "stale"}))])

    sink.on_trial_start("new-scene", 0)
    engine.release.set()
    _wait_until(lambda: len(playback.writes) == 3)
    sink.close()

    assert engine.calls == ["inflight"]


def test_trial_end_does_not_clear_just_enqueued_summary() -> None:
    engine = _FakeEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(4, [_assistant(_tool_call("done", {"summary": "all done"}))])

    sink.on_trial_end(cast(TrialRecord, object()))
    _wait_until(lambda: engine.calls == ["all done"] and len(playback.writes) == 3)
    sink.close()


def test_successful_eval_end_drains_inflight_utterance_before_close() -> None:
    engine = _BlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(4, [_assistant(_tool_call("done", {"summary": "all done"}))])
    assert engine.entered.wait(timeout=1.0)

    finished = threading.Event()

    def end_eval() -> None:
        sink.on_eval_end(_log("success"))
        finished.set()

    end_thread = threading.Thread(target=end_eval)
    end_thread.start()
    assert not finished.wait(timeout=0.02)
    assert playback.close_calls == 0
    engine.release.set()
    end_thread.join(timeout=1.0)

    assert finished.is_set()
    assert len(playback.writes) == 3
    assert playback.close_calls == 1


def test_successful_eval_end_drain_and_join_are_time_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _BlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)
    monkeypatch.setattr(speaker_module, "_DRAIN_TIMEOUT", 0.02)
    monkeypatch.setattr(speaker_module, "_JOIN_TIMEOUT", 0.02)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("done", {"summary": "stalled"}))])
    assert engine.entered.wait(timeout=1.0)

    started = time.monotonic()
    sink.on_eval_end(_log("success"))
    elapsed = time.monotonic() - started
    engine.release.set()

    assert elapsed < 0.2
    assert playback.close_calls == 1


@pytest.mark.parametrize("status", ["cancelled", "error"])
def test_non_successful_eval_end_aborts_without_draining(
    status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _BlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)
    monkeypatch.setattr(speaker_module, "_JOIN_TIMEOUT", 0.02)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "stop now"}))])
    assert engine.entered.wait(timeout=1.0)

    sink.on_eval_end(_log(status))
    engine.release.set()

    assert playback.writes == []
    assert playback.close_calls == 1


def test_worker_exception_warns_once_and_sink_stays_inert(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingEngine(_FakeEngine):
        def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
            self.calls.append(text)
            raise RuntimeError("synthesis failed")

    engine = FailingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "first"}))])
    _wait_until(lambda: sink._disabled)

    sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "second"}))])
    sink.on_trial_start("next", 0)
    sink.on_eval_end(_log("error"))

    assert engine.calls == ["first"]
    assert capsys.readouterr().err == "speaker: disabled after RuntimeError: synthesis failed\n"
    assert playback.close_calls == 1


def test_bare_close_is_idempotent_and_stops_between_playback_chunks() -> None:
    first_write = threading.Event()

    class LongEngine(_FakeEngine):
        def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
            self.calls.append(text)
            return np.ones(100, dtype=np.float32), 10

    class SlowPlayback(_FakePlayback):
        def write(self, samples: npt.NDArray[np.float32], sample_rate: int) -> None:
            super().write(samples, sample_rate)
            first_write.set()
            time.sleep(0.02)

    engine = LongEngine()
    playback = SlowPlayback()
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "long"}))])
    assert first_write.wait(timeout=1.0)

    sink.close()
    sink.close()

    assert 0 < len(playback.writes) < 100
    assert playback.close_calls == 1


def test_log_policy_messages_before_start_is_safe_noop() -> None:
    engine = _FakeEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)

    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "ignored"}))])
    sink.close()

    assert engine.calls == []
    assert playback.writes == []
    assert playback.close_calls == 0


@pytest.mark.parametrize("which", ["engine", "playback"])
def test_startup_factory_errors_propagate(which: str) -> None:
    def fail() -> _FakeEngine:
        raise RuntimeError(f"{which} unavailable")

    engine = _FakeEngine()
    playback = _FakePlayback()
    sink = SpeakerSink(
        engine_factory=fail if which == "engine" else lambda: engine,
        playback_factory=fail if which == "playback" else lambda: playback,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match=f"{which} unavailable"):
        sink.start()


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of: blocking, interrupt, queue"):
        SpeakerSink(mode="unknown")


def test_interrupt_cuts_inflight_within_one_chunk_and_plays_new_note() -> None:
    engine = _TaggedEngine()
    playback = _GatedPlayback(gated_writes=1)
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "old"}))])
    assert playback.entered[0].wait(timeout=1.0)

    sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "new"}))])
    with sink._condition:
        assert list(sink._queue) == ["new"]
        assert sink._dropped == 0
    playback.release[0].set()
    assert playback.wait_for_writes(4)
    sink.close()

    assert engine.calls == ["old", "new"]
    assert [float(chunk[0]) for chunk, _ in playback.writes] == [1.0, 2.0, 2.0, 2.0]
    assert sink._thread is None


def test_interrupt_discards_stale_synthesis_result_and_worker_survives() -> None:
    engine = _BlockingEngine()
    playback = _GatedPlayback()
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "stale"}))])
    assert engine.entered.wait(timeout=1.0)

    sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "current"}))])
    engine.release.set()
    assert playback.wait_for_writes(3)
    sink.close()

    assert engine.calls == ["stale", "current"]
    assert len(playback.writes) == 3


def test_interrupt_multi_text_delta_plays_all_texts_in_order() -> None:
    engine = _FakeEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback)
    sink.start()

    sink.log_policy_messages(
        0,
        [
            _assistant(
                _tool_call("move", {"note": "move now"}),
                _tool_call("done", {"summary": "finished"}),
            )
        ],
    )
    sink.on_eval_end(_log("success"))

    assert engine.calls == ["move now", "finished"]
    assert len(playback.writes) == 6


def test_interrupt_trial_start_preserves_inflight_until_next_delta() -> None:
    engine = _TaggedEngine()
    playback = _GatedPlayback(gated_writes=2)
    sink = _sink(engine, playback)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("done", {"summary": "old"}))])
    assert playback.entered[0].wait(timeout=1.0)

    sink.on_trial_start("next", 1)
    assert sink._speech_gen == 1
    playback.release[0].set()
    assert playback.entered[1].wait(timeout=1.0)

    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "new"}))])
    playback.release[1].set()
    assert playback.wait_for_writes(5)
    sink.close()

    assert engine.calls == ["old", "new"]
    assert [float(chunk[0]) for chunk, _ in playback.writes] == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_blocking_waits_for_inflight_then_enqueues_and_plays() -> None:
    engine = _TaggedEngine()
    playback = _GatedPlayback(gated_writes=1)
    sink = _sink(engine, playback, mode="blocking")
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "first"}))])
    assert playback.entered[0].wait(timeout=1.0)
    returned = threading.Event()

    def enqueue_second() -> None:
        sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "second"}))])
        returned.set()

    hook_thread = threading.Thread(target=enqueue_second)
    hook_thread.start()
    assert not returned.wait(timeout=0.02)
    playback.release[0].set()
    assert returned.wait(timeout=1.0)
    assert playback.wait_for_writes(6)
    hook_thread.join(timeout=1.0)
    sink.close()

    assert engine.calls == ["first", "second"]
    assert [float(chunk[0]) for chunk, _ in playback.writes] == [1.0] * 3 + [2.0] * 3


def test_blocking_skips_gate_for_deltas_without_speakable_text() -> None:
    engine = _TaggedEngine()
    playback = _GatedPlayback(gated_writes=1)
    sink = _sink(engine, playback, mode="blocking")
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "first"}))])
    assert playback.entered[0].wait(timeout=1.0)
    returned = threading.Event()

    def deliver_noteless_delta() -> None:
        sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "   "}))])
        returned.set()

    hook_thread = threading.Thread(target=deliver_noteless_delta)
    hook_thread.start()
    assert returned.wait(timeout=0.5)
    hook_thread.join(timeout=1.0)
    with sink._condition:
        assert list(sink._queue) == []
    playback.release[0].set()
    sink.close()

    assert engine.calls == ["first"]


def test_blocking_fails_open_when_worker_dies() -> None:
    class FailingBlockingEngine(_BlockingEngine):
        def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
            self.calls.append(text)
            self.entered.set()
            self.release.wait(timeout=2.0)
            raise RuntimeError("failed while hook waited")

    engine = FailingBlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback, mode="blocking")
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "first"}))])
    assert engine.entered.wait(timeout=1.0)
    returned = threading.Event()

    def enqueue_second() -> None:
        sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "second"}))])
        returned.set()

    hook_thread = threading.Thread(target=enqueue_second)
    hook_thread.start()
    assert not returned.wait(timeout=0.02)
    engine.release.set()
    assert returned.wait(timeout=1.0)
    hook_thread.join(timeout=1.0)
    sink.close()

    assert engine.calls == ["first"]
    assert list(sink._queue) == []


def test_blocking_fails_open_when_close_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _BlockingEngine()
    playback = _FakePlayback()
    sink = _sink(engine, playback, mode="blocking")
    monkeypatch.setattr(speaker_module, "_JOIN_TIMEOUT", 1.0)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "first"}))])
    assert engine.entered.wait(timeout=1.0)
    returned = threading.Event()

    def enqueue_second() -> None:
        sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "second"}))])
        returned.set()

    hook_thread = threading.Thread(target=enqueue_second)
    close_thread = threading.Thread(target=sink.close)
    hook_thread.start()
    assert not returned.wait(timeout=0.02)
    close_thread.start()
    assert returned.wait(timeout=1.0)
    engine.release.set()
    hook_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert engine.calls == ["first"]
    assert list(sink._queue) == []


def test_blocking_timeout_degrades_once_and_keeps_drop_accounting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _FakeEngine()
    playback = _GatedPlayback(gated_writes=1)
    sink = _sink(engine, playback, mode="blocking")
    monkeypatch.setattr(speaker_module, "_BLOCK_TIMEOUT", 0.02)
    sink.start()
    sink.log_policy_messages(0, [_assistant(_tool_call("move", {"note": "inflight"}))])
    assert playback.entered[0].wait(timeout=1.0)

    sink.log_policy_messages(1, [_assistant(_tool_call("move", {"note": "timed out"}))])
    with sink._condition:
        assert list(sink._queue) == ["timed out"]
        assert sink._dropped == 0
    assert capsys.readouterr().err == "speaker: blocking wait timed out; narration may lag\n"

    monkeypatch.setattr(speaker_module, "_BLOCK_TIMEOUT", 10.0)
    returned = threading.Event()

    def enqueue_degraded() -> None:
        sink.log_policy_messages(2, [_assistant(_tool_call("move", {"note": "degraded"}))])
        returned.set()

    hook_thread = threading.Thread(target=enqueue_degraded)
    hook_thread.start()
    assert returned.wait(timeout=0.2)
    hook_thread.join(timeout=1.0)
    for text in ["three", "four", "five"]:
        sink.log_policy_messages(3, [_assistant(_tool_call("move", {"note": text}))])
    with sink._condition:
        assert list(sink._queue) == ["degraded", "three", "four", "five"]
        assert sink._dropped == 1
    assert capsys.readouterr().err == ""

    playback.release[0].set()
    sink.close()
