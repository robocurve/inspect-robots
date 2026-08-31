"""Policy-facing rollout observations carry a reserved environment step."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inspect_robots.approver import AutoApprover
from inspect_robots.console import ConsolePoll, EndRequest, OperatorInput
from inspect_robots.controller import DefaultController
from inspect_robots.errors import EmbodimentFault, _CancelledTrial
from inspect_robots.logging.sink import LogSink, NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.rollout import TrialRecord, rollout
from inspect_robots.scene import Scene
from inspect_robots.session import OperatorSession
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.transcript import judgement_source, operator_event
from inspect_robots.types import OPERATOR_END, Action, ActionChunk, Observation, StepResult

_SCENE = Scene(id="step-probe", instruction="hold still")
_ACTION_SPACE = Box(
    shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world")
)


class _ProbePolicy:
    """Record every policy-facing observation and emit one action chunk per call."""

    def __init__(self, *, chunk_size: int = 1) -> None:
        self.info = PolicyInfo(name="step-probe", action_space=_ACTION_SPACE)
        self.config = PolicyConfig(action_horizon=chunk_size)
        self.chunk_size = chunk_size
        self.observations: list[Observation] = []

    def reset(self, scene: Scene) -> None:
        """Clear observations before the trial."""
        self.observations.clear()

    def act(self, observation: Observation) -> ActionChunk:
        """Capture the observation and return a hold-still action chunk."""
        self.observations.append(observation)
        return ActionChunk(actions=[Action(data=np.zeros(2)) for _ in range(self.chunk_size)])


class FakeOperatorInput:
    """Return scripted console polls and expose lifecycle call counts."""

    def __init__(
        self,
        polls: list[ConsolePoll | BaseException],
        *,
        begin_error: Exception | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.polls = list(polls)
        self.begin_error = begin_error
        self.order = order
        self.poll_calls = 0
        self.begin_calls = 0

    def poll(self) -> ConsolePoll:
        """Return the next scripted result or raise its scripted exception."""
        self.poll_calls += 1
        item = self.polls.pop(0) if self.polls else ConsolePoll()
        if isinstance(item, BaseException):
            raise item
        return item

    def begin_trial(self) -> None:
        """Record trial initialization and raise the configured failure, if any."""
        self.begin_calls += 1
        if self.order is not None:
            self.order.append("begin")
        if self.begin_error is not None:
            raise self.begin_error


class _EndingOperatorInput(FakeOperatorInput):
    """Add a duck-typed trial-finalization hook to the scripted operator input."""

    def __init__(
        self,
        polls: list[ConsolePoll | BaseException],
        *,
        end_error: Exception | None = None,
    ) -> None:
        super().__init__(polls)
        self.end_error = end_error
        self.end_calls = 0

    def end_trial(self) -> None:
        """Record trial finalization and raise the configured failure, if any."""
        self.end_calls += 1
        if self.end_error is not None:
            raise self.end_error


class _RaisingFooterSession(OperatorSession):
    """Open a real footer, then fail at one selected rollout console lifecycle site."""

    def __init__(self, output: list[str], *, fail_at: str) -> None:
        super().__init__(
            write=output.append,
            fd_readable=lambda: False,
            width_fn=lambda: 200,
            isatty_fn=lambda: True,
            raw_mode_fn=object,
            restore_fn=lambda _state: None,
        )
        self.fail_at = fail_at
        self.end_calls = 0
        self.enable_footer(label="sent")

    def begin_trial(self) -> None:
        """Open the footer before raising a configured begin failure."""
        super().begin_trial()
        if self.fail_at == "begin":
            raise RuntimeError("begin failed")

    def poll(self) -> ConsolePoll:
        """Raise a configured poll failure before the normal console poll."""
        if self.fail_at == "poll":
            raise RuntimeError("poll failed")
        return super().poll()

    def end_trial(self) -> None:
        """Count every teardown request before idempotently closing the real footer."""
        self.end_calls += 1
        super().end_trial()


class _TrackingEmbodiment(CubePickEmbodiment):
    """Track reset ordering and how many actions reached the embodiment."""

    def __init__(self, order: list[str] | None = None) -> None:
        super().__init__()
        self.order = order
        self.step_calls = 0

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Record only after the underlying reset has returned."""
        observation = super().reset(scene, seed=seed)
        if self.order is not None:
            self.order.append("reset")
        return observation

    def step(self, action: Action) -> StepResult:
        """Count each action before delegating to the mock world."""
        self.step_calls += 1
        return super().step(action)


class _StatusTrackingEmbodiment(CubePickEmbodiment):
    """Emit one session status from each step to expose degraded plain rendering."""

    def __init__(self, session: OperatorSession) -> None:
        super().__init__()
        self.session = session

    def step(self, action: Action) -> StepResult:
        """Render a post-disable status before applying the action."""
        self.session.status("later status")
        return super().step(action)


class _FailingStepEmbodiment(CubePickEmbodiment):
    """Raise a trial-ending embodiment failure from the first action."""

    def step(self, action: Action) -> StepResult:
        """Fail every attempted action so rollout exercises its exception path."""
        del action
        raise RuntimeError("motion failed")


class _ObservationSink(NullSink):
    """Capture observations delivered to ``log_step``."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Record the logged observation."""
        self.observations.append(observation)


class _ExtraEmbodiment(CubePickEmbodiment):
    """Attach colliding and unrelated extras to every produced observation."""

    def __init__(self) -> None:
        super().__init__()
        self.observations: list[Observation] = []

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Return and record a reset observation carrying embodiment extras."""
        self.observations.clear()
        observation = replace(
            super().reset(scene, seed=seed),
            extra={"env_step": "theirs", "unrelated": 1},
        )
        self.observations.append(observation)
        return observation

    def step(self, action: Action) -> StepResult:
        """Return and record a step observation carrying embodiment extras."""
        result = super().step(action)
        observation = replace(
            result.observation,
            extra={"env_step": "theirs", "unrelated": 1},
        )
        self.observations.append(observation)
        return replace(result, observation=observation)


def _run(
    policy: _ProbePolicy,
    embodiment: CubePickEmbodiment,
    *,
    max_steps: int,
    sink: LogSink | None = None,
    operator_input: OperatorInput | None = None,
) -> TrialRecord:
    return rollout(
        policy,
        embodiment,
        _SCENE,
        max_steps=max_steps,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=sink or NullSink(),
        operator_input=operator_input,
    )


def test_rollout_injects_step_only_into_policy_observations() -> None:
    policy = _ProbePolicy()
    sink = _ObservationSink()

    record = _run(policy, CubePickEmbodiment(), max_steps=4, sink=sink)

    assert [observation.extra["env_step"] for observation in policy.observations] == [0, 1, 2, 3]
    assert len(record.steps) == len(sink.observations) == 4
    assert all("env_step" not in step.observation.extra for step in record.steps)
    assert all("env_step" not in observation.extra for observation in sink.observations)


def test_rollout_step_merge_preserves_extras_and_overwrites_collision() -> None:
    policy = _ProbePolicy()

    _run(policy, _ExtraEmbodiment(), max_steps=3)

    assert [observation.extra["env_step"] for observation in policy.observations] == [0, 1, 2]
    assert all(observation.extra["unrelated"] == 1 for observation in policy.observations)


def test_rollout_step_injection_shares_the_images_mapping() -> None:
    policy = _ProbePolicy()
    embodiment = _ExtraEmbodiment()

    _run(policy, embodiment, max_steps=3)

    assert len(policy.observations) == 3
    assert all(
        policy_observation.images is embodiment_observation.images
        for policy_observation, embodiment_observation in zip(
            policy.observations, embodiment.observations[:3], strict=True
        )
    )


def test_operator_message_reaches_only_the_next_chunked_inference() -> None:
    policy = _ProbePolicy(chunk_size=2)
    operator_input = FakeOperatorInput([ConsolePoll(messages=("keep the wrist level",))])

    _run(policy, CubePickEmbodiment(), max_steps=4, operator_input=operator_input)

    assert [observation.extra["operator_messages"] for observation in policy.observations] == [
        [{"t": 0, "text": "keep the wrist level", "source": "console"}],
        [],
    ]


def test_operator_messages_across_steps_keep_their_drain_step() -> None:
    policy = _ProbePolicy(chunk_size=2)
    operator_input = FakeOperatorInput(
        [
            ConsolePoll(messages=("first",)),
            ConsolePoll(messages=("second",)),
            ConsolePoll(),
        ]
    )

    _run(policy, CubePickEmbodiment(), max_steps=3, operator_input=operator_input)

    assert [observation.extra["operator_messages"] for observation in policy.observations] == [
        [{"t": 0, "text": "first", "source": "console"}],
        [{"t": 1, "text": "second", "source": "console"}],
    ]


def test_verdictless_console_end_stops_before_inference_or_motion() -> None:
    policy = _ProbePolicy()
    embodiment = _TrackingEmbodiment()
    operator_input = FakeOperatorInput([ConsolePoll(), ConsolePoll(end=EndRequest())])

    record = _run(policy, embodiment, max_steps=3, operator_input=operator_input)

    assert record.terminated is True
    assert record.termination_reason == OPERATOR_END
    assert record.operator_judgement is None
    assert len(record.steps) == embodiment.step_calls == 1
    assert len(policy.observations) == 1


def test_verdict_console_end_records_judgement_note_and_operator_event() -> None:
    policy = _ProbePolicy()
    operator_input = FakeOperatorInput(
        [ConsolePoll(end=EndRequest(verdict="y", note="clean pickup"))]
    )

    record = _run(
        policy,
        _TrackingEmbodiment(),
        max_steps=3,
        operator_input=operator_input,
    )

    assert record.terminated is True
    assert record.termination_reason == OPERATOR_END
    assert record.operator_judgement == "y"
    assert record.operator_note == "clean pickup"
    operator_events = [event for event in record.events if event.kind == "operator"]
    assert len(operator_events) == 1
    assert operator_events[0].t == 0
    assert operator_events[0].data == {
        "verdict": "y",
        "source": "console",
        "note": "clean pickup",
    }
    record.events.insert(0, operator_event(-1, "n", source="stale"))
    assert judgement_source(record) == "console"


def test_messages_in_an_ending_poll_are_recorded_as_transcript_events() -> None:
    policy = _ProbePolicy()
    embodiment = _TrackingEmbodiment()
    operator_input = FakeOperatorInput(
        [
            ConsolePoll(
                messages=("gripper slipped",),
                end=EndRequest(),
                sources=("voice",),
            )
        ]
    )

    record = _run(policy, embodiment, max_steps=3, operator_input=operator_input)

    message_events = [event for event in record.events if event.kind == "operator_message"]
    assert len(message_events) == 1
    assert message_events[0].t == 0
    assert message_events[0].data == {"text": "gripper slipped", "source": "voice"}
    assert embodiment.step_calls == 0
    assert policy.observations == []


def test_message_sources_are_parallel_with_console_fallback() -> None:
    policy = _ProbePolicy()
    operator_input = FakeOperatorInput(
        [ConsolePoll(messages=("spoken", "typed"), sources=("voice",))]
    )

    record = _run(policy, CubePickEmbodiment(), max_steps=1, operator_input=operator_input)

    assert policy.observations[0].extra["operator_messages"] == [
        {"t": 0, "text": "spoken", "source": "voice"},
        {"t": 0, "text": "typed", "source": "console"},
    ]
    assert [
        event.data["source"] for event in record.events if event.kind == "operator_message"
    ] == [
        "voice",
        "console",
    ]


def test_begin_trial_runs_once_after_embodiment_reset() -> None:
    order: list[str] = []
    operator_input = FakeOperatorInput([], order=order)

    _run(
        _ProbePolicy(),
        _TrackingEmbodiment(order),
        max_steps=1,
        operator_input=operator_input,
    )

    assert operator_input.begin_calls == 1
    assert order == ["reset", "begin"]


def test_end_trial_runs_on_success_path() -> None:
    operator_input = _EndingOperatorInput([])

    record = _run(
        _ProbePolicy(),
        CubePickEmbodiment(),
        max_steps=1,
        operator_input=operator_input,
    )

    assert record.status == "success"
    assert operator_input.end_calls == 1


def test_end_trial_runs_on_trial_ending_exception_path() -> None:
    operator_input = _EndingOperatorInput([])

    with pytest.raises(EmbodimentFault, match="motion failed"):
        _run(
            _ProbePolicy(),
            _FailingStepEmbodiment(),
            max_steps=1,
            operator_input=operator_input,
        )

    assert operator_input.end_calls == 1


def test_operator_input_without_end_trial_hook_is_skipped_without_error() -> None:
    operator_input = FakeOperatorInput([])

    record = _run(
        _ProbePolicy(),
        CubePickEmbodiment(),
        max_steps=1,
        operator_input=operator_input,
    )

    assert record.status == "success"


def test_raising_end_trial_warns_without_masking_successful_outcome() -> None:
    operator_input = _EndingOperatorInput([], end_error=RuntimeError("restore failed"))

    with pytest.warns(
        RuntimeWarning,
        match="Operator console disabled for this trial after RuntimeError: restore failed",
    ) as warning_records:
        record = _run(
            _ProbePolicy(),
            CubePickEmbodiment(),
            max_steps=1,
            operator_input=operator_input,
        )

    assert len(warning_records) == 1
    assert record.status == "success"
    assert operator_input.end_calls == 1


def test_raising_end_trial_warns_without_masking_trial_exception() -> None:
    operator_input = _EndingOperatorInput([], end_error=RuntimeError("restore failed"))

    with (
        pytest.warns(
            RuntimeWarning,
            match="Operator console disabled for this trial after RuntimeError: restore failed",
        ) as warning_records,
        pytest.raises(EmbodimentFault, match="motion failed"),
    ):
        _run(
            _ProbePolicy(),
            _FailingStepEmbodiment(),
            max_steps=1,
            operator_input=operator_input,
        )

    assert len(warning_records) == 1
    assert operator_input.end_calls == 1


def test_no_operator_input_keeps_reserved_key_out_of_policy_observations() -> None:
    policy = _ProbePolicy(chunk_size=2)

    _run(policy, CubePickEmbodiment(), max_steps=3)

    assert all("operator_messages" not in observation.extra for observation in policy.observations)


def test_raising_poll_warns_once_disables_channel_and_preserves_record() -> None:
    policy = _ProbePolicy()
    operator_input = FakeOperatorInput(
        [RuntimeError("poll failed"), ConsolePoll(messages=("must not surface",))]
    )

    with pytest.warns(
        RuntimeWarning,
        match="Operator console disabled for this trial after RuntimeError: poll failed",
    ) as warning_records:
        record = _run(
            policy,
            CubePickEmbodiment(),
            max_steps=3,
            operator_input=operator_input,
        )

    assert operator_input.poll_calls == 1
    assert len(warning_records) == 1
    assert len(record.steps) == 3
    assert record.status == "success"
    assert record.truncated is True
    assert record.termination_reason == "max_steps"


def test_end_trial_still_runs_after_raising_poll_disables_channel() -> None:
    operator_input = _EndingOperatorInput([RuntimeError("poll failed")])

    with pytest.warns(
        RuntimeWarning,
        match="Operator console disabled for this trial after RuntimeError: poll failed",
    ) as warning_records:
        record = _run(
            _ProbePolicy(),
            CubePickEmbodiment(),
            max_steps=2,
            operator_input=operator_input,
        )

    assert len(warning_records) == 1
    assert record.status == "success"
    assert operator_input.poll_calls == 1
    assert operator_input.end_calls == 2


def test_raising_poll_closes_footer_at_disable_time_before_later_statuses() -> None:
    output: list[str] = []
    operator_input = _RaisingFooterSession(output, fail_at="poll")

    with pytest.warns(
        RuntimeWarning,
        match="Operator console disabled for this trial after RuntimeError: poll failed",
    ) as warning_records:
        record = _run(
            _ProbePolicy(),
            _StatusTrackingEmbodiment(operator_input),
            max_steps=1,
            operator_input=operator_input,
        )

    assert len(warning_records) == 1
    assert record.status == "success"
    assert operator_input.end_calls == 2
    assert output == ["\r\x1b[K> ", "\r\x1b[K", "\r  later status   "]
    assert "Esc ends the episode" not in "".join(output)


def test_raising_begin_trial_warns_once_and_disables_channel() -> None:
    policy = _ProbePolicy()
    operator_input = FakeOperatorInput([], begin_error=RuntimeError("drain failed"))

    with pytest.warns(
        RuntimeWarning,
        match="Operator console disabled for this trial after RuntimeError: drain failed",
    ) as warning_records:
        record = _run(
            policy,
            CubePickEmbodiment(),
            max_steps=3,
            operator_input=operator_input,
        )

    assert operator_input.begin_calls == 1
    assert operator_input.poll_calls == 0
    assert len(warning_records) == 1
    assert len(record.steps) == 3
    assert record.status == "success"


def test_raising_begin_trial_closes_footer_at_disable_time_before_later_statuses() -> None:
    output: list[str] = []
    operator_input = _RaisingFooterSession(output, fail_at="begin")

    with pytest.warns(
        RuntimeWarning,
        match="Operator console disabled for this trial after RuntimeError: begin failed",
    ) as warning_records:
        record = _run(
            _ProbePolicy(),
            _StatusTrackingEmbodiment(operator_input),
            max_steps=1,
            operator_input=operator_input,
        )

    assert len(warning_records) == 1
    assert record.status == "success"
    assert operator_input.end_calls == 2
    assert output == ["\r\x1b[K> ", "\r\x1b[K", "\r  later status   "]
    assert "Esc ends the episode" not in "".join(output)


def test_keyboard_interrupt_from_poll_uses_cancelled_trial_path() -> None:
    original = KeyboardInterrupt("operator interrupt")
    operator_input = FakeOperatorInput([original])

    with pytest.raises(_CancelledTrial) as excinfo:
        _run(
            _ProbePolicy(),
            _TrackingEmbodiment(),
            max_steps=3,
            operator_input=operator_input,
        )

    record = excinfo.value.record
    assert record.status == "cancelled"
    assert record.error == "cancelled by user (KeyboardInterrupt)"
    assert record.steps == []
    assert excinfo.value.__cause__ is original


def test_end_trial_runs_on_keyboard_interrupt_cancelled_trial_path() -> None:
    original = KeyboardInterrupt("operator interrupt")
    operator_input = _EndingOperatorInput([original])

    with pytest.raises(_CancelledTrial) as excinfo:
        _run(
            _ProbePolicy(),
            _TrackingEmbodiment(),
            max_steps=3,
            operator_input=operator_input,
        )

    assert excinfo.value.__cause__ is original
    assert excinfo.value.record.status == "cancelled"
    assert operator_input.end_calls == 1
