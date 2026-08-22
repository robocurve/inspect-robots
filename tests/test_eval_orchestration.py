"""eval() orchestration hardening: error-log survival, error-trial scoring,
partial-record delivery, fail_on_error timing, embodiment lifecycle, seeding."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval, eval_set, read_eval_log
from inspect_robots.console import ConsolePoll, EndRequest
from inspect_robots.errors import (
    CompatibilityError,
    ConfigError,
    EmbodimentFault,
    PolicyError,
    SafetyAbort,
    _CancelledTrial,
)
from inspect_robots.eval import _git_commit
from inspect_robots.log import EvalLog, EvalSpec
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import embodiment as embodiment_decorator
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import Score, min_distance_to_goal, operator_scorer, success_at_end
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.task import Epochs, Task, TaskEnvelope
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

_BOX = Box(shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"))


def _task(*, epochs: int | Epochs = 1, max_steps: int = 60, scorer: object = None) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=scorer or success_at_end(),  # type: ignore[arg-type]
        max_steps=max_steps,
        epochs=epochs,
    )


class _RecordingSink(NullSink):
    """Collects the records delivered via on_trial_end."""

    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.records.append(record)


class _FaultAfterEpochsEmbodiment(CubePickEmbodiment):
    """Runs ``good_epochs`` full trials, then faults on the next step."""

    def __init__(self, good_epochs: int) -> None:
        super().__init__()
        self.good_epochs = good_epochs
        self._resets = 0

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._resets += 1
        return super().reset(scene, seed=seed)

    def step(self, action: Action) -> StepResult:
        if self._resets > self.good_epochs:
            raise EmbodimentFault("motor stalled")
        return super().step(action)


class _BoomOnSecondEpochPolicy(ScriptedPolicy):
    """Behaves normally on the first epoch, explodes on later epochs."""

    def __init__(self) -> None:
        super().__init__()
        self._resets = 0

    def reset(self, scene: Scene) -> None:
        self._resets += 1
        super().reset(scene)

    def act(self, observation: Observation) -> ActionChunk:
        if self._resets > 1:
            raise RuntimeError("inference exploded")
        return super().act(observation)


class _BoomPolicy:
    def __init__(self) -> None:
        self.info = PolicyInfo(name="boom", action_space=_BOX)
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        raise RuntimeError("inference exploded")


class _InterruptingPolicy(ScriptedPolicy):
    """Raise one specific Ctrl-C during inference and expose an audit record."""

    def __init__(self, interrupt: KeyboardInterrupt, *, interrupt_on_call: int = 2) -> None:
        super().__init__(chunk_size=1)
        self.interrupt = interrupt
        self.interrupt_on_call = interrupt_on_call
        self.act_calls = 0

    def reset(self, scene: Scene) -> None:
        self.act_calls = 0
        super().reset(scene)

    def act(self, observation: Observation) -> ActionChunk:
        self.act_calls += 1
        if self.act_calls == self.interrupt_on_call:
            raise self.interrupt
        return super().act(observation)

    def transcript(self) -> object:
        return {"act_calls": self.act_calls}


class _CountingScorer:
    """Count every trajectory presented for scoring."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        del record, target
        self.calls += 1
        return Score(value=1.0)


class _CaptureGrader:
    """Record every trial presented to the pre-scoring grader seam.

    ``parked_at_grade`` snapshots ``parked_observation`` at call time, so a
    regression that populated the field only after ``before_scoring`` returned
    cannot slip past assertions made on the shared mutable record.
    """

    name = "capture"

    def __init__(self) -> None:
        self.records: list[TrialRecord] = []
        self.parked_at_grade: list[object] = []

    def grade(self, record: TrialRecord, scene: Scene) -> None:
        del scene
        self.records.append(record)
        self.parked_at_grade.append(record.parked_observation)


class _ParkedEmbodiment(CubePickEmbodiment):
    """Return or raise one configured result from the optional park hook."""

    def __init__(self, park_result: object) -> None:
        super().__init__()
        self.park_result = park_result
        self.park_calls = 0

    def observe_parked(self) -> object:
        self.park_calls += 1
        if isinstance(self.park_result, BaseException):
            raise self.park_result
        return self.park_result


class _ScriptedOperatorInput:
    """Expose per-trial polls and model input typed between trial boundaries."""

    def __init__(self, trials: list[list[ConsolePoll]]) -> None:
        self.trials = trials
        self.active: list[ConsolePoll] = []
        self.pending_messages: list[str] = []
        self.discarded_messages: list[str] = []
        self.begin_calls = 0

    def poll(self) -> ConsolePoll:
        """Merge pending input with the next scripted poll for the active trial."""
        scripted = self.active.pop(0) if self.active else ConsolePoll()
        pending = tuple(self.pending_messages)
        messages = (*pending, *scripted.messages)
        sources = (
            *("console" for _ in pending),
            *(
                scripted.sources[i] if i < len(scripted.sources) else "console"
                for i in range(len(scripted.messages))
            ),
        )
        self.pending_messages.clear()
        return ConsolePoll(messages=messages, end=scripted.end, sources=sources)

    def begin_trial(self) -> None:
        """Discard inter-trial input and activate the next trial's poll script."""
        self.begin_calls += 1
        self.discarded_messages.extend(self.pending_messages)
        self.pending_messages.clear()
        index = self.begin_calls - 1
        self.active = list(self.trials[index]) if index < len(self.trials) else []


# --------------------------------------------------------------------------- #
# Ctrl-C during rollout writes a partial cancelled log before propagating.
# --------------------------------------------------------------------------- #
def test_cancelled_eval_writes_partial_log_with_forensic_data(tmp_path: Path) -> None:
    policy = _InterruptingPolicy(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        eval(
            _task(),
            policy,
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            store_frames=True,
        )

    (written,) = tmp_path.glob("*.json")
    log = read_eval_log(str(written))
    scene = log.samples[0]
    assert log.status == "cancelled"
    assert scene.status == "cancelled"
    assert scene.policy_transcripts == ({"act_calls": 2},)
    assert scene.termination_reasons == (None,)
    assert scene.epochs == ({},)
    assert scene.reduced == {}
    assert log.results.metrics == {}
    assert log.results.errored_trials == 0
    assert log.error == "cancelled by user (KeyboardInterrupt)"
    assert log.stats.frames_dir is not None
    assert list(Path(log.stats.frames_dir).rglob("*.npy"))


def test_cancelled_eval_reraises_typed_exception_with_original_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from inspect_robots.rollout import rollout as actual_rollout

    original = KeyboardInterrupt("operator interrupt")
    raised: list[_CancelledTrial] = []

    def capturing_rollout(*args: object, **kwargs: object) -> TrialRecord:
        try:
            return actual_rollout(*args, **kwargs)  # type: ignore[arg-type]
        except _CancelledTrial as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(sys.modules["inspect_robots.eval"], "rollout", capturing_rollout)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        eval(
            _task(),
            _InterruptingPolicy(original),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
        )

    assert isinstance(excinfo.value, _CancelledTrial)
    assert excinfo.value is raised[0]
    assert excinfo.value.__cause__ is original


def test_cancelled_policy_reset_records_t_minus_one_and_zero_steps(tmp_path: Path) -> None:
    class _ResetInterruptPolicy(ScriptedPolicy):
        def reset(self, scene: Scene) -> None:
            raise KeyboardInterrupt

    records = _RecordingSink()
    json_sink = JsonLogSink(str(tmp_path))

    with pytest.raises(KeyboardInterrupt):
        eval(
            _task(),
            _ResetInterruptPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            sinks=[records, json_sink],
        )

    (record,) = records.records
    assert record.status == "cancelled"
    assert record.steps == []
    assert record.events[-1].kind == "error"
    assert record.events[-1].t == -1
    assert json_sink.path is not None and json_sink.path.exists()
    assert read_eval_log(str(json_sink.path)).status == "cancelled"


def test_cancelled_first_scene_halts_before_second_scene(tmp_path: Path) -> None:
    task = Task(
        name="two-scenes",
        scenes=[
            Scene(id="s0", instruction="reach", init_seed=0),
            Scene(id="s1", instruction="reach", init_seed=1),
        ],
        scorer=success_at_end(),
        max_steps=60,
    )

    with pytest.raises(KeyboardInterrupt):
        eval(
            task,
            _InterruptingPolicy(KeyboardInterrupt(), interrupt_on_call=1),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
        )

    (written,) = tmp_path.glob("*.json")
    log = read_eval_log(str(written))
    assert log.results.total_scenes == 1
    assert tuple(scene.scene_id for scene in log.samples) == ("s0",)


def test_cancelled_trial_is_never_scored() -> None:
    scorer = _CountingScorer()

    with pytest.raises(KeyboardInterrupt):
        eval(
            _task(scorer=scorer),
            _InterruptingPolicy(KeyboardInterrupt(), interrupt_on_call=1),
            CubePickEmbodiment(),
            sinks=[NullSink()],
            store_actions=False,
        )

    assert scorer.calls == 0


def test_all_errored_guard_does_not_rewrite_cancelled_status(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        eval(
            _task(),
            _InterruptingPolicy(KeyboardInterrupt(), interrupt_on_call=1),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
        )

    (written,) = tmp_path.glob("*.json")
    log = read_eval_log(str(written))
    assert log.status == "cancelled"
    assert log.error == "cancelled by user (KeyboardInterrupt)"


def test_errored_then_cancelled_epochs_preserve_both_records(tmp_path: Path) -> None:
    class _ErrorThenCancelPolicy(_InterruptingPolicy):
        def __init__(self) -> None:
            super().__init__(KeyboardInterrupt(), interrupt_on_call=1)
            self.resets = 0

        def reset(self, scene: Scene) -> None:
            self.resets += 1
            if self.resets == 1:
                raise RuntimeError("first epoch failed")
            super().reset(scene)

    records = _RecordingSink()
    json_sink = JsonLogSink(str(tmp_path))

    with pytest.raises(KeyboardInterrupt):
        eval(
            _task(epochs=2),
            _ErrorThenCancelPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            sinks=[records, json_sink],
        )

    assert json_sink.path is not None
    log = read_eval_log(str(json_sink.path))
    assert log.status == "cancelled"
    assert log.results.errored_trials == 1
    assert log.results.total_trials == 2
    assert log.samples[0].epochs == ({}, {})
    assert [record.status for record in records.records] == ["error", "cancelled"]


# --------------------------------------------------------------------------- #
# 1. A halted eval must still produce an error log, whatever the reducer.
# --------------------------------------------------------------------------- #
def test_halted_eval_with_pass_at_k_reducer_still_writes_log(tmp_path: Path) -> None:
    # Fault at epoch 2 of 5 leaves fewer scores than k; pass_at_5 would raise.
    task = _task(epochs=Epochs(count=5, reducer="pass_at_5"))
    (log,) = eval(
        task, ScriptedPolicy(), _FaultAfterEpochsEmbodiment(good_epochs=2), log_dir=str(tmp_path)
    )
    assert isinstance(log, EvalLog)
    assert log.status == "error"
    assert log.error is not None and "motor stalled" in log.error
    assert log.samples[0].error is not None and "reducer" in log.samples[0].error
    # The halt path keeps the parallel tuples aligned: the faulted trial gets
    # a None reason next to its empty epoch entry.
    assert log.samples[0].termination_reasons == ("success", "success", None)
    assert len(log.samples[0].termination_reasons) == len(log.samples[0].epochs)
    assert list(tmp_path.glob("*.json"))  # the log reached disk


def test_categorical_scorer_with_mean_reducer_degrades_to_error_log(tmp_path: Path) -> None:
    class _CategoricalScorer:
        name = "direction"

        def __call__(self, record: TrialRecord, target: object) -> object:
            from inspect_robots.scorer import Score

            return Score(value="left")

    task = _task(epochs=2, scorer=_CategoricalScorer())
    (log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.error is not None and "reducer 'mean' failed" in log.error
    assert log.results.metrics == {}  # the failed reducer contributes no metric


# --------------------------------------------------------------------------- #
# 2. Errored trials are never scored and cannot poison metrics.
# --------------------------------------------------------------------------- #
def test_errored_trials_are_not_scored(tmp_path: Path) -> None:
    task = _task(epochs=2, scorer=min_distance_to_goal())
    (log,) = eval(task, _BoomOnSecondEpochPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    scene = log.samples[0]
    assert scene.status == "error"  # the failed epoch is visible...
    assert scene.epochs[1] == {}  # ...as an empty (unscored) epoch entry
    # ...but the metric comes from the good epoch only: finite, not inf.
    assert np.isfinite(log.results.metrics["min_distance_to_goal"])
    assert log.status == "success"  # data survived: partials stay tolerated
    assert log.results.errored_trials == 1
    assert log.results.total_trials == 2
    assert scene.termination_reasons == ("success", None)
    assert len(scene.termination_reasons) == len(scene.epochs)


def test_step_limit_reason_and_horizon_are_recorded(tmp_path: Path) -> None:
    (log,) = eval(_task(max_steps=1), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.samples[0].termination_reasons == ("max_steps",)
    assert log.eval.max_steps == 1


def test_all_trials_errored_degrades_to_error_status(tmp_path: Path) -> None:
    # Issue #73: a run that scored nothing must not report success.
    (log,) = eval(_task(), _BoomPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.error == "all 1 trial(s) errored; nothing was scored"
    assert log.results.errored_trials == log.results.total_trials == 1
    assert log.results.metrics == {}


def test_halt_error_message_is_not_overwritten_by_all_errored(tmp_path: Path) -> None:
    # A SafetyAbort halt already sets status/error; the all-errored degrade
    # must not clobber the more specific message.
    class _AbortPolicy(ScriptedPolicy):
        def reset(self, scene: Scene) -> None:
            raise SafetyAbort("operator hit the e-stop")

    (log,) = eval(_task(), _AbortPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.error == "SafetyAbort: operator hit the e-stop"  # not the all-errored message


# --------------------------------------------------------------------------- #
# 3. Partial records reach the sinks (forensics survive errors).
# --------------------------------------------------------------------------- #
def test_policy_error_partial_record_reaches_sinks() -> None:
    class _BoomLaterPolicy(_BoomPolicy):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        def act(self, observation: Observation) -> ActionChunk:
            self._calls += 1
            if self._calls > 1:
                raise RuntimeError("inference exploded later")
            return ActionChunk(actions=[Action(data=np.zeros(2)) for _ in range(4)])

    sink = _RecordingSink()
    (log,) = eval(
        _task(),
        _BoomLaterPolicy(),
        CubePickEmbodiment(),
        sinks=[sink],
        store_actions=False,
    )
    assert log.status == "error"  # its only trial errored (issue #73)
    (record,) = sink.records
    assert record.status == "error"
    assert len(record.steps) == 4  # the steps walked before the failure survive
    assert log.stats.total_steps == 4


def test_halt_without_attached_record_still_produces_error_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Defensive path: a halting error raised without rollout's record attachment
    # (e.g. from third-party middleware) must still yield an error log.
    def fake_rollout(*args: object, **kwargs: object) -> TrialRecord:
        raise EmbodimentFault("fault with no record")

    import sys

    # inspect_robots.eval the *module* is shadowed by the eval() function on the
    # package, so fetch it from sys.modules.
    monkeypatch.setattr(sys.modules["inspect_robots.eval"], "rollout", fake_rollout)
    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.results.total_trials == 0  # nothing to count or deliver


def test_halt_delivers_partial_record_and_counts_trial() -> None:
    sink = _RecordingSink()
    (log,) = eval(
        _task(),
        ScriptedPolicy(),
        _FaultAfterEpochsEmbodiment(good_epochs=0),
        sinks=[sink],
        store_actions=False,
    )
    assert log.status == "error"
    assert log.results.total_trials == 1  # the aborted trial is counted...
    (record,) = sink.records  # ...and its record delivered to sinks
    assert record.status == "error"
    assert record.error is not None and "motor stalled" in record.error


# --------------------------------------------------------------------------- #
# 4. fail_on_error is evaluated after every trial, not per scene.
# --------------------------------------------------------------------------- #
def test_fail_on_error_true_stops_at_first_error(tmp_path: Path) -> None:
    task = _task(epochs=3)
    (log,) = eval(
        task, _BoomPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), fail_on_error=True
    )
    assert log.status == "error"
    assert log.results.total_trials == 1  # stopped immediately, not after 3 epochs


# --------------------------------------------------------------------------- #
# 5. Embodiment lifecycle: eval closes what it resolves, and only that.
# --------------------------------------------------------------------------- #
_CLOSED: list[str] = []


class _ClosableEmbodiment(CubePickEmbodiment):
    def close(self) -> None:
        _CLOSED.append("closed")


embodiment_decorator("closable-cubepick")(_ClosableEmbodiment)


class _TrackedClosablePolicy(ScriptedPolicy):
    """A policy whose optional cleanup hook records the lifecycle boundary."""

    def __init__(
        self,
        events: list[str],
        *,
        bind_error: Exception | None = None,
        close_error: Exception | None = None,
    ):
        super().__init__()
        self.bind_error = bind_error
        self.close_error = close_error
        self.close_calls = 0
        self.events = events

    def bind(self, embodiment_info: object) -> None:
        del embodiment_info
        if self.bind_error is not None:
            raise self.bind_error

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("policy")
        if self.close_error is not None:
            raise self.close_error


class _TrackedClosableEmbodiment(CubePickEmbodiment):
    """A real toy embodiment with observable cleanup for eval ownership tests."""

    def __init__(self, events: list[str], *, close_error: Exception | None = None):
        super().__init__()
        self.close_calls = 0
        self.events = events
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("embodiment")
        if self.close_error is not None:
            raise self.close_error


def _register_owned_eval_components(
    monkeypatch: pytest.MonkeyPatch,
    policy: _TrackedClosablePolicy,
    embodiment: _TrackedClosableEmbodiment,
) -> None:
    """Install per-test factories while leaving eval's real registry resolution intact."""
    from inspect_robots import registry

    monkeypatch.setitem(registry._FACTORIES["policy"], "closable-policy", lambda: policy)
    monkeypatch.setitem(
        registry._FACTORIES["embodiment"], "closable-embodiment", lambda: embodiment
    )


def test_eval_closes_registry_resolved_policy_and_embodiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(events)
    embodiment = _TrackedClosableEmbodiment(events)
    _register_owned_eval_components(monkeypatch, policy, embodiment)

    eval(_task(max_steps=1), "closable-policy", "closable-embodiment", log_dir=str(tmp_path))

    assert policy.close_calls == 1
    assert embodiment.close_calls == 1
    assert events == ["embodiment", "policy"]


def test_eval_closes_registry_resolved_policy_when_embodiment_factory_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from inspect_robots import registry

    events: list[str] = []
    policy = _TrackedClosablePolicy(events, close_error=RuntimeError("policy close exploded"))

    class _NotedFactoryError(RuntimeError):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.notes: list[str] = []

        def add_note(self, note: str) -> None:
            self.notes.append(note)

    factory_error = _NotedFactoryError("embodiment factory exploded")

    def raise_from_embodiment_factory() -> _TrackedClosableEmbodiment:
        raise factory_error

    monkeypatch.setitem(registry._FACTORIES["policy"], "closable-policy", lambda: policy)
    monkeypatch.setitem(
        registry._FACTORIES["embodiment"],
        "failing-embodiment",
        raise_from_embodiment_factory,
    )

    with pytest.raises(RuntimeError, match="embodiment factory exploded") as excinfo:
        eval(_task(max_steps=1), "closable-policy", "failing-embodiment", log_dir=str(tmp_path))

    assert excinfo.value is factory_error
    assert policy.close_calls == 1
    assert events == ["policy"]
    assert any("policy close exploded" in note for note in factory_error.notes)


def test_eval_does_not_close_caller_constructed_policy(tmp_path: Path) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(events)
    embodiment = _TrackedClosableEmbodiment(events)

    eval(_task(max_steps=1), policy, embodiment, log_dir=str(tmp_path))

    assert policy.close_calls == 0
    assert embodiment.close_calls == 0


def test_eval_closes_embodiment_before_reraising_owned_policy_close_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(events, close_error=RuntimeError("policy close exploded"))
    embodiment = _TrackedClosableEmbodiment(events)
    _register_owned_eval_components(monkeypatch, policy, embodiment)

    with pytest.raises(RuntimeError, match="policy close exploded"):
        eval(_task(max_steps=1), "closable-policy", "closable-embodiment", log_dir=str(tmp_path))

    assert policy.close_calls == 1
    assert embodiment.close_calls == 1
    assert events == ["embodiment", "policy"]


def test_successful_eval_reraises_owned_policy_close_error_inside_outer_except(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(events, close_error=RuntimeError("policy close exploded"))
    embodiment = _TrackedClosableEmbodiment(events)
    _register_owned_eval_components(monkeypatch, policy, embodiment)
    outer_error = RuntimeError("outer handler")

    try:
        raise outer_error
    except RuntimeError:
        with pytest.raises(RuntimeError, match="policy close exploded"):
            eval(
                _task(max_steps=1),
                "closable-policy",
                "closable-embodiment",
                log_dir=str(tmp_path),
            )

    assert getattr(outer_error, "__notes__", ()) == ()
    assert events == ["embodiment", "policy"]


def test_eval_attempts_owned_policy_cleanup_after_embodiment_close_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(events)
    embodiment = _TrackedClosableEmbodiment(
        events, close_error=RuntimeError("embodiment close exploded")
    )
    _register_owned_eval_components(monkeypatch, policy, embodiment)

    with pytest.raises(RuntimeError, match="embodiment close exploded"):
        eval(_task(max_steps=1), "closable-policy", "closable-embodiment", log_dir=str(tmp_path))

    assert policy.close_calls == 1
    assert embodiment.close_calls == 1
    assert events == ["embodiment", "policy"]


def test_eval_preserves_first_cleanup_error_when_both_owned_closes_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(events, close_error=RuntimeError("policy close exploded"))
    embodiment = _TrackedClosableEmbodiment(
        events, close_error=RuntimeError("embodiment close exploded")
    )
    _register_owned_eval_components(monkeypatch, policy, embodiment)

    with pytest.raises(RuntimeError, match="embodiment close exploded"):
        eval(_task(max_steps=1), "closable-policy", "closable-embodiment", log_dir=str(tmp_path))

    assert policy.close_calls == 1
    assert embodiment.close_calls == 1
    assert events == ["embodiment", "policy"]


def test_eval_preserves_active_error_when_owned_policy_close_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    policy = _TrackedClosablePolicy(
        events,
        bind_error=RuntimeError("evaluation exploded"),
        close_error=RuntimeError("policy close exploded"),
    )
    embodiment = _TrackedClosableEmbodiment(events)
    _register_owned_eval_components(monkeypatch, policy, embodiment)

    with pytest.raises(RuntimeError, match="evaluation exploded") as excinfo:
        eval(_task(max_steps=1), "closable-policy", "closable-embodiment", log_dir=str(tmp_path))

    assert policy.close_calls == 1
    assert embodiment.close_calls == 1
    assert events == ["embodiment", "policy"]
    if hasattr(excinfo.value, "add_note"):
        notes = getattr(excinfo.value, "__notes__", ())
        assert any("policy close exploded" in note for note in notes)


def test_eval_binds_adaptive_policy_before_compat(tmp_path: Path) -> None:
    """A bind() hook runs after resolution and before compat (plan 0008 §3c).

    The policy starts with a deliberately incompatible action space; only the
    bind() call adopting the embodiment's spaces lets compat pass, so a green
    eval proves the ordering.
    """
    from inspect_robots.embodiment import EmbodimentInfo

    class _AdaptivePolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.bound_names: list[str] = []
            self.info = PolicyInfo(
                name="adaptive",
                action_space=Box(shape=(9,), semantics=ActionSemantics("joint_pos")),
            )

        def bind(self, embodiment_info: EmbodimentInfo) -> None:
            self.bound_names.append(embodiment_info.name)
            self.info = PolicyInfo(
                name="adaptive",
                action_space=embodiment_info.action_space,
                observation_space=embodiment_info.observation_space,
            )

    adaptive = _AdaptivePolicy()
    logs = eval(_task(max_steps=60), adaptive, CubePickEmbodiment(), log_dir=str(tmp_path))
    assert adaptive.bound_names == ["cubepick"]
    assert logs[0].status == "success"


def test_eval_binds_spaces_frames_and_scenes_to_duck_typed_sinks_before_start(
    tmp_path: Path,
) -> None:
    """Offer optional sink bindings in order and skip absent or non-callable hooks."""

    class _SpaceAware(NullSink):
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
            self.calls.append(("bind_spaces", action_space, observation_space))

        def bind_frames_dir(self, frames_dir: str | None) -> None:
            self.calls.append(("bind_frames_dir", frames_dir))

        def bind_scenes(self, scenes: Sequence[Scene]) -> None:
            """Record the offered scenes for ordering assertions."""
            self.calls.append(("bind_scenes", scenes))

        def on_eval_start(self, spec: EvalSpec) -> None:
            del spec
            self.calls.append(("on_eval_start",))

    class _OddAttr(NullSink):
        bind_spaces = "not a hook"
        bind_frames_dir = "not a hook"
        bind_scenes = "not a hook"

    embodiment = CubePickEmbodiment()
    aware = _SpaceAware()
    no_hook = NullSink()
    odd = _OddAttr()

    task = _task(max_steps=1)
    (log,) = eval(
        task,
        ScriptedPolicy(),
        embodiment,
        sinks=[aware, no_hook, odd],
        log_dir=str(tmp_path),
    )

    assert log.status == "success"
    assert aware.calls == [
        (
            "bind_spaces",
            embodiment.info.action_space,
            embodiment.info.observation_space,
        ),
        ("bind_frames_dir", None),
        ("bind_scenes", task.scenes),
        ("on_eval_start",),
    ]
    assert getattr(no_hook, "bind_spaces", None) is None
    assert odd.bind_spaces == "not a hook"
    assert getattr(no_hook, "bind_frames_dir", None) is None
    assert odd.bind_frames_dir == "not a hook"
    assert getattr(no_hook, "bind_scenes", None) is None
    assert odd.bind_scenes == "not a hook"


def test_eval_binds_the_exact_stored_frames_directory(tmp_path: Path) -> None:
    """The optional hook receives the same string persisted on the final log."""

    class _FrameAware(NullSink):
        def __init__(self) -> None:
            self.frames_dir: str | None = None

        def bind_frames_dir(self, frames_dir: str | None) -> None:
            self.frames_dir = frames_dir

    sink = _FrameAware()
    (log,) = eval(
        _task(max_steps=1),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        sinks=[sink],
        log_dir=str(tmp_path),
        store_frames=True,
    )

    assert sink.frames_dir == log.stats.frames_dir
    assert sink.frames_dir is not None


def test_eval_binds_task_envelope_before_reset(tmp_path: Path) -> None:
    """bind_task fires once per eval with the task's envelope, before any reset."""

    class _HorizonAware(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[object] = []

        def bind_task(self, envelope: TaskEnvelope) -> None:
            self.calls.append(envelope)

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self.calls.append("reset")
            return super().reset(scene, seed=seed)

    two_scenes = Task(
        name="t",
        scenes=[
            Scene(id="s0", instruction="reach", init_seed=0),
            Scene(id="s1", instruction="reach", init_seed=1),
        ],
        scorer=success_at_end(),
        max_steps=7,
        epochs=2,
    )
    aware = _HorizonAware()
    (log,) = eval(two_scenes, ScriptedPolicy(), aware, log_dir=str(tmp_path))
    assert log.status == "success"
    assert aware.calls[0] == TaskEnvelope(name="t", max_steps=7)
    # Exactly one bind per eval — not per scene or per epoch; resets follow it.
    assert aware.calls.count("reset") == 4
    assert [c for c in aware.calls if c != "reset"] == [TaskEnvelope(name="t", max_steps=7)]


def test_eval_resolves_seconds_horizon_into_envelope_rollout_and_log(tmp_path: Path) -> None:
    class _HorizonAware(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[TaskEnvelope] = []

        def bind_task(self, envelope: TaskEnvelope) -> None:
            self.envelopes.append(envelope)

    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_seconds=0.1,
    )
    aware = _HorizonAware()
    (log,) = eval(task, ScriptedPolicy(), aware, log_dir=str(tmp_path))

    assert aware.envelopes == [TaskEnvelope(name="timed", max_steps=1)]
    assert log.samples[0].termination_reasons == ("max_steps",)
    assert log.eval.max_seconds == 0.1
    assert log.eval.max_steps == 1


def test_bind_task_rebinds_per_eval_latest_wins(tmp_path: Path) -> None:
    class _HorizonAware(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[TaskEnvelope] = []

        def bind_task(self, envelope: TaskEnvelope) -> None:
            self.envelopes.append(envelope)

    aware = _HorizonAware()
    eval(_task(max_steps=5), ScriptedPolicy(), aware, log_dir=str(tmp_path))
    eval(_task(max_steps=9), ScriptedPolicy(), aware, log_dir=str(tmp_path))
    assert [e.max_steps for e in aware.envelopes] == [5, 9]


def test_eval_ignores_non_callable_bind_task(tmp_path: Path) -> None:
    class _OddAttr(CubePickEmbodiment):
        bind_task = "not a hook"

    (log,) = eval(_task(max_steps=5), ScriptedPolicy(), _OddAttr(), log_dir=str(tmp_path))
    assert log.status == "success"
    assert _OddAttr.bind_task == "not a hook"


class _BindRaisesEmbodiment(_ClosableEmbodiment):
    def bind_task(self, envelope: TaskEnvelope) -> None:
        raise RuntimeError("refusing this task")


embodiment_decorator("bind-raises-cubepick")(_BindRaisesEmbodiment)


def test_raising_bind_task_aborts_before_any_rollout(tmp_path: Path) -> None:
    _CLOSED.clear()
    with pytest.raises(RuntimeError, match="refusing this task"):
        eval(_task(max_steps=5), ScriptedPolicy(), "bind-raises-cubepick", log_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.json"))  # no rollout started, no log written
    assert _CLOSED == ["closed"]  # registry-owned embodiment still released


def test_compatibility_check_runs_before_bind_task(tmp_path: Path) -> None:
    """An incompatible pair fails before the embodiment receives an envelope."""

    class _WidePolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.info = PolicyInfo(
                name="wide",
                action_space=Box(shape=(9,), semantics=ActionSemantics("joint_pos")),
            )

    with pytest.raises(CompatibilityError, match="action_dim"):
        eval(_task(max_steps=5), _WidePolicy(), "bind-raises-cubepick", log_dir=str(tmp_path))


def test_invalid_seconds_rate_fails_before_bind_task(tmp_path: Path) -> None:
    from dataclasses import replace

    class _RateMissing(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, control_hz=None)
            self.envelopes: list[TaskEnvelope] = []

        def bind_task(self, envelope: TaskEnvelope) -> None:
            self.envelopes.append(envelope)

    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_seconds=120.0,
    )
    embodiment = _RateMissing()
    with pytest.raises(CompatibilityError, match="task_horizon_control_rate"):
        eval(task, ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert embodiment.envelopes == []


def test_embodiment_base_bind_task_is_a_noop() -> None:
    from inspect_robots.embodiment import EmbodimentBase

    class _Minimal(EmbodimentBase):
        info = CubePickEmbodiment().info

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            return CubePickEmbodiment().reset(scene, seed=seed)

        def step(self, action: Action) -> StepResult:
            raise NotImplementedError

    _Minimal().bind_task(TaskEnvelope(name="t", max_steps=1))  # must exist and do nothing


def test_policy_base_bind_is_a_noop() -> None:
    from inspect_robots.embodiment import EmbodimentInfo
    from inspect_robots.policy import PolicyBase

    class _Minimal(PolicyBase):
        info = PolicyInfo(name="m", action_space=_BOX)

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    obs_space = CubePickEmbodiment().info.observation_space
    info = EmbodimentInfo(name="e", action_space=_BOX, observation_space=obs_space)
    _Minimal().bind(info)  # must exist and do nothing


def test_eval_closes_string_resolved_embodiment(tmp_path: Path) -> None:
    _CLOSED.clear()
    eval(_task(max_steps=5), ScriptedPolicy(), "closable-cubepick", log_dir=str(tmp_path))
    assert _CLOSED == ["closed"]


def test_eval_does_not_close_caller_owned_embodiment(tmp_path: Path) -> None:
    _CLOSED.clear()
    eval(_task(max_steps=5), ScriptedPolicy(), _ClosableEmbodiment(), log_dir=str(tmp_path))
    assert _CLOSED == []  # the caller owns the object's lifecycle


def test_eval_closes_resolved_embodiment_even_on_failure(tmp_path: Path) -> None:
    _CLOSED.clear()
    wide_policy_info = PolicyInfo(
        name="wide",
        action_space=Box(shape=(7,), semantics=ActionSemantics("eef_delta_pos", frame="world")),
    )

    class _WidePolicy:
        info = wide_policy_info
        config = PolicyConfig()

        def reset(self, scene: Scene) -> None:
            return None

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.zeros(7))])

    from inspect_robots.errors import CompatibilityError

    with pytest.raises(CompatibilityError):
        eval(_task(), _WidePolicy(), "closable-cubepick", log_dir=str(tmp_path))
    assert _CLOSED == ["closed"]  # released even though the run failed fast


# --------------------------------------------------------------------------- #
# 6. seed=None draws recorded OS entropy; bad reducers fail fast.
# --------------------------------------------------------------------------- #
def test_seed_none_draws_recorded_entropy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("os.urandom", lambda n: b"\x2a" + b"\x00" * (n - 1))
    (log,) = eval(
        _task(max_steps=5), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=None
    )
    assert log.eval.seed == 42  # the drawn seed is recorded, not None


def test_unknown_reducer_fails_fast_as_config_error(tmp_path: Path) -> None:
    task = _task(epochs=Epochs(count=2, reducer="bogus"))
    with pytest.raises(ConfigError, match="unknown epoch reducer"):
        eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))


def test_policy_error_without_attached_record_synthesizes_one(tmp_path: Path) -> None:
    # A PolicyError raised outside the rollout internals (no record attached)
    # still yields a scored-as-error trial rather than a crash.
    class _EagerErrorController:
        def next_action(self, policy: object, obs: object, t: int, store: object) -> Action:
            raise PolicyError("controller-level failure")

    sink = _RecordingSink()
    (log,) = eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
        controller=_EagerErrorController(),
    )
    assert log.status == "error"  # its only trial errored (issue #73)
    (record,) = sink.records
    assert record.status == "error"


# --------------------------------------------------------------------------- #
# 7. _git_commit: dirty suffix, deterministic via a fake git.
# --------------------------------------------------------------------------- #
class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_git_commit_appends_dirty_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "rev-parse" in cmd:
            return _FakeCompleted("abc123\n")
        return _FakeCompleted(" M file.py\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _git_commit() == "abc123-dirty"


def test_git_commit_clean_tree_has_no_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "rev-parse" in cmd:
            return _FakeCompleted("abc123\n")
        return _FakeCompleted("")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _git_commit() == "abc123"


# --------------------------------------------------------------------------- #
# 8. before_scoring hook: the R6 seam for capturing operator judgements.
# --------------------------------------------------------------------------- #
def test_before_scoring_runs_before_scorers_and_persists_judgement(tmp_path: Path) -> None:
    task = _task(scorer=operator_scorer())
    seen: list[tuple[str, int]] = []

    def judge(record: TrialRecord, scene: Scene) -> None:
        seen.append((scene.id, record.epoch))
        record.operator_judgement = "yes"

    (log,) = eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), before_scoring=judge
    )
    assert seen == [("s0", 0)]
    # The operator scorer read the verdict, so the hook ran before scoring.
    assert log.results.metrics["operator"] == 1.0
    assert log.samples[0].operator_judgements == ("yes",)
    assert log.samples[0].instruction == "reach"


def test_before_scoring_skipped_for_errored_trials(tmp_path: Path) -> None:
    # Epoch 0 succeeds and is judged; epoch 1 errors (PolicyError) and must
    # neither be scored nor prompt the hook — its judgement slot stays None,
    # parallel to the empty epochs entry.
    task = _task(epochs=2, scorer=operator_scorer())
    calls: list[int] = []

    def judge(record: TrialRecord, scene: Scene) -> None:
        calls.append(record.epoch)
        record.operator_judgement = "yes"
        record.operator_note = "clean pickup"

    (log,) = eval(
        task,
        _BoomOnSecondEpochPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        before_scoring=judge,
    )
    assert calls == [0]
    scene = log.samples[0]
    assert scene.epochs == ({"operator": 1.0}, {})
    assert scene.operator_judgements == ("yes", None)
    assert scene.operator_notes == ("clean pickup", None)


def test_before_scoring_exception_propagates(tmp_path: Path) -> None:
    def bad_hook(record: TrialRecord, scene: Scene) -> None:
        raise RuntimeError("hook exploded")

    with pytest.raises(RuntimeError, match="hook exploded"):
        eval(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            before_scoring=bad_hook,
        )


def test_before_scoring_default_none_records_no_judgements(tmp_path: Path) -> None:
    (log,) = eval(_task(epochs=2), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.samples[0].operator_judgements == (None, None)
    assert log.samples[0].operator_messages == ((), ())


def test_eval_set_forwards_before_scoring(tmp_path: Path) -> None:
    def judge(record: TrialRecord, scene: Scene) -> None:
        record.operator_judgement = "pass"

    success, logs = eval_set(
        _task(scorer=operator_scorer()),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        before_scoring=judge,
    )
    assert success
    assert logs[0].samples[0].operator_judgements == ("pass",)
    assert logs[0].results.metrics["operator"] == 1.0


def test_eval_observes_parked_once_before_grading(tmp_path: Path) -> None:
    parked = Observation(images={"parked": np.zeros((2, 2, 3), dtype=np.uint8)})
    embodiment = _ParkedEmbodiment(parked)
    grader = _CaptureGrader()

    (log,) = eval(
        _task(max_steps=1),
        ScriptedPolicy(),
        embodiment,
        grader=grader,
        log_dir=str(tmp_path),
    )

    assert log.status == "success"
    assert embodiment.park_calls == 1
    assert len(grader.records) == 1
    assert grader.parked_at_grade[0] is parked


def test_eval_does_not_observe_parked_without_a_grader(tmp_path: Path) -> None:
    embodiment = _ParkedEmbodiment(Observation())

    (log,) = eval(_task(max_steps=1), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))

    assert log.status == "success"
    assert embodiment.park_calls == 0


def test_eval_does_not_observe_parked_after_operator_judgement(tmp_path: Path) -> None:
    embodiment = _ParkedEmbodiment(Observation())
    grader = _CaptureGrader()
    operator_input = _ScriptedOperatorInput([[ConsolePoll(end=EndRequest(verdict="y"))]])

    eval(
        _task(max_steps=1),
        ScriptedPolicy(),
        embodiment,
        grader=grader,
        operator_input=operator_input,
        log_dir=str(tmp_path),
    )

    assert embodiment.park_calls == 0
    assert grader.records[0].operator_judgement == "y"


def test_eval_does_not_observe_parked_after_definitive_termination(tmp_path: Path) -> None:
    embodiment = _ParkedEmbodiment(Observation())
    embodiment.goal_radius = 2.0
    grader = _CaptureGrader()

    eval(
        _task(max_steps=1),
        ScriptedPolicy(),
        embodiment,
        grader=grader,
        log_dir=str(tmp_path),
    )

    assert embodiment.park_calls == 0
    assert grader.records[0].termination_reason == "success"


def test_eval_degrades_when_observe_parked_raises(tmp_path: Path) -> None:
    embodiment = _ParkedEmbodiment(RuntimeError("park jammed"))
    grader = _CaptureGrader()

    with pytest.warns(
        RuntimeWarning,
        match=r"observe_parked\(\) failed with RuntimeError: park jammed",
    ):
        (log,) = eval(
            _task(max_steps=1),
            ScriptedPolicy(),
            embodiment,
            grader=grader,
            log_dir=str(tmp_path),
        )

    assert grader.records[0].parked_observation is None
    assert log.samples[0].epochs[0] == {"success_at_end": 0.0}


@pytest.mark.parametrize(
    "error",
    [SafetyAbort("park unsafe"), EmbodimentFault("park faulted")],
)
def test_eval_halts_when_observe_parked_raises_a_halt_error(
    error: Exception, tmp_path: Path
) -> None:
    embodiment = _ParkedEmbodiment(error)

    with pytest.raises(type(error), match=str(error)):
        eval(
            _task(max_steps=1),
            ScriptedPolicy(),
            embodiment,
            grader=_CaptureGrader(),
            log_dir=str(tmp_path),
        )


def test_eval_accepts_observe_parked_declining_without_warning(tmp_path: Path) -> None:
    embodiment = _ParkedEmbodiment(None)
    grader = _CaptureGrader()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eval(
            _task(max_steps=1),
            ScriptedPolicy(),
            embodiment,
            grader=grader,
            log_dir=str(tmp_path),
        )

    assert grader.records[0].parked_observation is None


def test_eval_rejects_non_observation_parked_result(tmp_path: Path) -> None:
    embodiment = _ParkedEmbodiment("not an observation")
    grader = _CaptureGrader()

    with pytest.warns(
        RuntimeWarning,
        match=(
            "observe_parked\\(\\) returned str; expected Observation or None; "
            "grading from last-step frames"
        ),
    ):
        eval(
            _task(max_steps=1),
            ScriptedPolicy(),
            embodiment,
            grader=grader,
            log_dir=str(tmp_path),
        )

    assert grader.records[0].parked_observation is None


def test_eval_ignores_non_callable_observe_parked(tmp_path: Path) -> None:
    class _OddAttr(CubePickEmbodiment):
        observe_parked = "not a hook"

    grader = _CaptureGrader()
    embodiment = _OddAttr()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eval(
            _task(max_steps=1),
            ScriptedPolicy(),
            embodiment,
            grader=grader,
            log_dir=str(tmp_path),
        )

    assert grader.records[0].parked_observation is None
    assert _OddAttr.observe_parked == "not a hook"


# --------------------------------------------------------------------------- #
# 9. Attended operator input is trial-scoped and persisted beside each epoch.
# --------------------------------------------------------------------------- #
def test_eval_begins_console_each_trial_and_discards_scoring_window_input(
    tmp_path: Path,
) -> None:
    operator_input = _ScriptedOperatorInput(
        [
            [ConsolePoll(messages=("trial one feedback",), end=EndRequest())],
            [ConsolePoll(end=EndRequest())],
        ]
    )

    def type_during_scoring(record: TrialRecord, scene: Scene) -> None:
        del scene
        if record.epoch == 0:
            operator_input.pending_messages.append("typed during scoring")

    (log,) = eval(
        _task(epochs=2, scorer=_CountingScorer()),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        operator_input=operator_input,
        before_scoring=type_during_scoring,
    )

    assert operator_input.begin_calls == 2
    assert operator_input.discarded_messages == ["typed during scoring"]
    assert log.samples[0].operator_messages == (
        ({"t": 0, "text": "trial one feedback", "source": "console"},),
        (),
    )


def test_eval_operator_message_source_round_trips_through_written_log(tmp_path: Path) -> None:
    operator_input = _ScriptedOperatorInput(
        [
            [
                ConsolePoll(
                    messages=("persist this feedback",),
                    end=EndRequest(),
                    sources=("voice",),
                )
            ]
        ]
    )

    (log,) = eval(
        _task(scorer=_CountingScorer()),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        operator_input=operator_input,
    )

    expected = (({"t": 0, "text": "persist this feedback", "source": "voice"},),)
    assert log.samples[0].operator_messages == expected
    (path,) = tmp_path.glob("*.json")
    restored_messages = read_eval_log(str(path)).samples[0].operator_messages
    assert isinstance(restored_messages, tuple)
    assert isinstance(restored_messages[0], tuple)
    assert restored_messages == expected


# --------------------------------------------------------------------------- #
# 10. Policy trial lifecycle hooks: setup and artifact persistence.
# --------------------------------------------------------------------------- #
def test_on_trial_start_runs_before_each_trials_first_act(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []

    class _StartHookPolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = -1

        def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
            self.epoch = epoch
            events.append(("start", scene_id, epoch, log_dir, run_id))

        def act(self, observation: Observation) -> ActionChunk:
            events.append(("act", self.epoch))
            return super().act(observation)

    eval(_task(epochs=2), _StartHookPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))

    starts = [event for event in events if event[0] == "start"]
    assert [(event[1], event[2], event[3]) for event in starts] == [
        ("s0", 0, str(tmp_path)),
        ("s0", 1, str(tmp_path)),
    ]
    assert starts[0][4] == starts[1][4]
    for epoch in range(2):
        start_index = next(
            i for i, event in enumerate(events) if event[:3] == ("start", "s0", epoch)
        )
        act_index = next(i for i, event in enumerate(events) if event == ("act", epoch))
        assert start_index < act_index


def test_raising_on_trial_start_skips_rollout_but_keeps_results_parallel(
    tmp_path: Path,
) -> None:
    class _LifecycleSink(_RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.starts: list[tuple[str, int]] = []

        def on_trial_start(self, scene_id: str, epoch: int) -> None:
            self.starts.append((scene_id, epoch))

    class _StartFailurePolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.current_epoch = -1
            self.resets: list[int] = []
            self.acts: list[int] = []
            self.ends: list[int] = []

        def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
            self.current_epoch = epoch
            if epoch == 0:
                raise RuntimeError("capture setup exploded")

        def reset(self, scene: Scene) -> None:
            self.resets.append(self.current_epoch)
            super().reset(scene)

        def act(self, observation: Observation) -> ActionChunk:
            self.acts.append(self.current_epoch)
            return super().act(observation)

        def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
            self.ends.append(record.epoch)

    policy = _StartFailurePolicy()
    sink = _LifecycleSink()
    (log,) = eval(
        _task(epochs=2),
        policy,
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink, JsonLogSink(str(tmp_path))],
    )

    scene = log.samples[0]
    parallel = (
        scene.epochs,
        scene.operator_judgements,
        scene.operator_notes,
        scene.operator_messages,
        scene.trial_metadata,
        scene.termination_reasons,
        scene.policy_transcripts,
    )
    assert {len(values) for values in parallel} == {2}
    assert scene.epochs[0] == {}
    assert scene.error == "policy.on_trial_start failed: capture setup exploded"
    assert log.results.total_trials == 2
    assert log.results.errored_trials == 1
    assert policy.resets == [1]
    assert policy.acts and set(policy.acts) == {1}
    assert policy.ends == [1]
    assert sink.starts == [("s0", 0), ("s0", 1)]
    assert [record.epoch for record in sink.records] == [0, 1]
    assert sink.records[0].status == "error"
    assert sink.records[0].error == scene.error
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_raising_on_trial_start_counts_toward_fail_on_error(tmp_path: Path) -> None:
    class _SecondStartFails(ScriptedPolicy):
        def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
            if epoch == 1:
                raise RuntimeError("second setup exploded")

    (log,) = eval(
        _task(epochs=3),
        _SecondStartFails(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        fail_on_error=True,
    )

    assert log.status == "error"
    assert log.error == "fail_on_error threshold exceeded (1 errors)"
    assert log.results.total_trials == 2
    assert log.results.errored_trials == 1
    assert len(log.samples[0].epochs) == 2
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_on_trial_end_hook_persists_metadata_and_recovers_from_errors(tmp_path: Path) -> None:
    task = _task(epochs=2)
    seen_ids: list[str] = []

    class _HookPolicy(ScriptedPolicy):
        def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
            # First epoch works, second raises exception
            seen_ids.append(run_id)
            if record.epoch == 0:
                record.metadata["test_key"] = "test_val"
            else:
                raise RuntimeError("hook exploded")

    (log,) = eval(task, _HookPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))

    # Eval returns an error log (rather than crashing) when a hook throws.
    assert log.status == "error"
    assert log.error is not None and "hook exploded" in log.error

    scene = log.samples[0]
    # The first epoch succeeded so its metadata is retained.
    # The second epoch failed in the hook, so only framework metadata follows
    # it into the log; both delivered trials receive action side-car pointers.
    first_metadata, second_metadata = scene.trial_metadata
    assert first_metadata["test_key"] == "test_val"
    assert set(first_metadata) == {"test_key", "actions"}
    assert set(second_metadata) == {"actions"}
    assert len(seen_ids) == 2
    assert seen_ids[0] == seen_ids[1]


def test_operator_fields_are_captured_before_the_on_trial_end_hook(tmp_path: Path) -> None:
    # The verdict and the note are documented as strictly parallel, so they must
    # be read off the record at the same instant. A hook that sets either one
    # runs too late for both: if one append ever drifts past on_trial_end, this
    # sees that tuple populated while the other is still (None,), and fails.
    class _LateJudgePolicy(ScriptedPolicy):
        def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
            record.operator_judgement = "y"
            record.operator_note = "written too late to be recorded"

    (log,) = eval(_task(), _LateJudgePolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))

    scene = log.samples[0]
    assert scene.operator_judgements == (None,)
    assert scene.operator_notes == (None,)


def test_on_trial_end_hook_error_on_already_errored_trial(tmp_path: Path) -> None:
    # A test to cover the branch where status is ALREADY "error" when the hook throws.
    # This happens if a previous epoch's hook already failed, setting global status="error".
    class _DoubleCrashPolicy(ScriptedPolicy):
        def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
            raise RuntimeError("hook exploded")

    # We run 2 epochs. Epoch 0 throws in hook, setting status="error".
    # Epoch 1 throws in hook, covering the `if status == "success"` False branch.
    (log,) = eval(
        _task(epochs=2),
        _DoubleCrashPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    assert log.status == "error"
    scene = log.samples[0]
    assert scene.status == "error"
    assert scene.error is not None
    # Both epochs ran the hook and failed
    assert "hook exploded" in scene.error


def test_policy_transcripts_parallel_scored_and_errored_epochs(tmp_path: Path) -> None:
    class _TranscriptBoomOnSecondEpoch(_BoomOnSecondEpochPolicy):
        def transcript(self) -> object:
            return {"reset": self._resets}

    (log,) = eval(
        _task(epochs=2),
        _TranscriptBoomOnSecondEpoch(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    scene = log.samples[0]
    assert scene.epochs[0] and scene.epochs[1] == {}
    assert scene.policy_transcripts == ({"reset": 1}, {"reset": 2})
    assert len(scene.policy_transcripts) == len(scene.epochs)


def test_halted_trial_transcript_reaches_the_persisted_log(tmp_path: Path) -> None:
    class _TranscriptScripted(ScriptedPolicy):
        def transcript(self) -> object:
            return [{"role": "assistant", "content": "walking to the cube"}]

    (log,) = eval(
        _task(epochs=2),
        _TranscriptScripted(),
        _FaultAfterEpochsEmbodiment(good_epochs=1),
        log_dir=str(tmp_path),
    )
    scene = log.samples[0]
    assert log.status == "error"
    assert len(scene.policy_transcripts) == len(scene.epochs) == 2
    # The faulted second trial keeps its transcript: forensics matter most
    # exactly when the trial died.
    assert scene.policy_transcripts[1] == [{"role": "assistant", "content": "walking to the cube"}]


def test_hookless_policy_yields_all_none_transcripts(tmp_path: Path) -> None:
    class _HooklessPolicy:
        def __init__(self) -> None:
            self._delegate = ScriptedPolicy()
            self.info = self._delegate.info
            self.config = self._delegate.config

        def reset(self, scene: Scene) -> None:
            self._delegate.reset(scene)

        def act(self, observation: Observation) -> ActionChunk:
            return self._delegate.act(observation)

    (log,) = eval(_task(epochs=2), _HooklessPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.samples[0].policy_transcripts == (None, None)
