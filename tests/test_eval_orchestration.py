"""eval() orchestration hardening: error-log survival, error-trial scoring,
partial-record delivery, fail_on_error timing, embodiment lifecycle, seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval, eval_set
from inspect_robots.errors import ConfigError, EmbodimentFault, PolicyError, SafetyAbort
from inspect_robots.eval import _git_commit
from inspect_robots.log import EvalLog
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import embodiment as embodiment_decorator
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import min_distance_to_goal, operator_scorer, success_at_end
from inspect_robots.spaces import ActionSemantics, Box
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
    (log,) = eval(_task(), _BoomLaterPolicy(), CubePickEmbodiment(), sinks=[sink])
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
        _task(), ScriptedPolicy(), _FaultAfterEpochsEmbodiment(good_epochs=0), sinks=[sink]
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

    aware = _HorizonAware()
    (log,) = eval(_task(epochs=2, max_steps=7), ScriptedPolicy(), aware, log_dir=str(tmp_path))
    assert log.status == "success"
    assert aware.calls[0] == TaskEnvelope(name="t", max_steps=7)
    # Exactly one bind per eval, even across multiple epochs; resets follow it.
    assert aware.calls.count("reset") == 2
    assert [c for c in aware.calls if c != "reset"] == [TaskEnvelope(name="t", max_steps=7)]


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
