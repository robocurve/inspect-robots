"""Live policy transcript deltas follow inference events without affecting rollout."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from inspect_robots.approver import AutoApprover
from inspect_robots.controller import DefaultController
from inspect_robots.eval import _Broadcast
from inspect_robots.logging.sink import LogSink, NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import TrialRecord, rollout
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

_SCENE = Scene(id="transcript", instruction="reach")


class _DeltaPolicy(ScriptedPolicy):
    """Return configured transcript deltas after successive inferences."""

    def __init__(self, deltas: list[Iterable[Any] | None]) -> None:
        super().__init__(chunk_size=1)
        self._deltas = list(deltas)
        self.delta_calls = 0

    def transcript_delta(self) -> Iterable[Any] | None:
        """Return the next configured delta and count hook invocations."""
        self.delta_calls += 1
        return self._deltas.pop(0) if self._deltas else None


class _RaisingDeltaPolicy(ScriptedPolicy):
    """Raise from every attempted transcript delta collection."""

    def __init__(self) -> None:
        super().__init__(chunk_size=1)
        self.delta_calls = 0

    def transcript_delta(self) -> Iterable[Any] | None:
        """Raise after recording the attempted live-stream collection."""
        self.delta_calls += 1
        raise RuntimeError("delta exploded")


class _RecordingSink(NullSink):
    """Capture transcript and step call order, optionally raising on transcript rows."""

    def __init__(self, *, raise_messages: bool = False) -> None:
        self.raise_messages = raise_messages
        self.message_calls: list[tuple[int, list[Any]]] = []
        self.message_batches_were_lists: list[bool] = []
        self.order: list[tuple[str, int]] = []

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        """Record a defensive copy of the delivered batch."""
        self.message_batches_were_lists.append(isinstance(messages, list))
        self.message_calls.append((t, list(messages)))
        self.order.append(("messages", t))
        if self.raise_messages:
            raise RuntimeError("sink exploded")

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Record each normal step after any same-step transcript batch."""
        self.order.append(("step", t))


def _run(policy: ScriptedPolicy, sink: LogSink, *, max_steps: int = 3) -> TrialRecord:
    return rollout(
        policy,
        CubePickEmbodiment(),
        _SCENE,
        max_steps=max_steps,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=sink,
    )


def test_delta_is_delivered_at_inference_step_before_log_step() -> None:
    message = {"role": "assistant", "content": "moving"}
    policy = _DeltaPolicy([[message], None])
    sink = _RecordingSink()

    _run(policy, sink, max_steps=2)

    assert sink.message_calls == [(0, [message])]
    assert sink.order[:3] == [("messages", 0), ("step", 0), ("step", 1)]


def test_policy_without_delta_hook_never_calls_sink_hook() -> None:
    sink = _RecordingSink()

    _run(ScriptedPolicy(chunk_size=1), sink)

    assert sink.message_calls == []


def test_sink_without_message_hook_never_collects_delta() -> None:
    policy = _DeltaPolicy([[{"role": "assistant", "content": "unused"}]])

    _run(policy, NullSink())

    assert policy.delta_calls == 0


def test_none_and_empty_deltas_do_not_call_sink_hook() -> None:
    policy = _DeltaPolicy([None, []])
    sink = _RecordingSink()

    _run(policy, sink, max_steps=2)

    assert policy.delta_calls == 2
    assert sink.message_calls == []


def test_raising_delta_hook_warns_once_and_latches_off() -> None:
    policy = _RaisingDeltaPolicy()
    sink = _RecordingSink()

    with pytest.warns(RuntimeWarning, match="RuntimeError: delta exploded") as caught:
        record = _run(policy, sink)

    assert len(caught) == 1
    assert policy.delta_calls == 1
    assert sink.message_calls == []
    assert len(record.steps) == 3
    assert record.status == "success"
    assert record.truncated and record.termination_reason == "max_steps"


def test_raising_sink_hook_warns_once_and_latches_off() -> None:
    policy = _DeltaPolicy([[1], [2], [3]])
    sink = _RecordingSink(raise_messages=True)

    with pytest.warns(RuntimeWarning, match="RuntimeError: sink exploded") as caught:
        record = _run(policy, sink)

    assert len(caught) == 1
    assert policy.delta_calls == 1
    assert sink.message_calls == [(0, [1])]
    assert len(record.steps) == 3
    assert record.status == "success"


def test_iterable_delta_is_materialized_and_empty_generator_is_skipped() -> None:
    def _entries() -> Iterable[dict[str, str]]:
        yield {"role": "tool", "content": "done"}

    sink = _RecordingSink()
    empty_source: list[dict[str, str]] = []
    policy = _DeltaPolicy([_entries(), (item for item in empty_source)])

    _run(policy, sink, max_steps=2)

    assert sink.message_calls == [(0, [{"role": "tool", "content": "done"}])]
    assert sink.message_batches_were_lists == [True]


def test_broadcast_fans_messages_only_to_implementing_sinks() -> None:
    first = _RecordingSink()
    second = NullSink()
    third = _RecordingSink()
    broadcast = _Broadcast([first, second, third])
    hook = getattr(broadcast, "log_policy_messages", None)

    assert callable(hook)
    hook(4, ["row"])

    assert first.message_calls == [(4, ["row"])]
    assert third.message_calls == [(4, ["row"])]


def test_broadcast_without_message_sinks_keeps_rollout_gate_closed() -> None:
    broadcast = _Broadcast([NullSink(), NullSink()])
    policy = _DeltaPolicy([["unused"]])

    assert not callable(getattr(broadcast, "log_policy_messages", None))
    _run(policy, broadcast)

    assert policy.delta_calls == 0


def test_null_sink_deliberately_omits_policy_message_hook() -> None:
    assert getattr(NullSink(), "log_policy_messages", None) is None
