"""eval() orchestration hardening: error-log survival, error-trial scoring,
partial-record delivery, fail_on_error timing, embodiment lifecycle, seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.errors import ConfigError, EmbodimentFault, PolicyError
from inspect_robots.eval import _git_commit
from inspect_robots.log import EvalLog
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import embodiment as embodiment_decorator
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import min_distance_to_goal, success_at_end
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.task import Epochs, Task
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
    assert log.status == "success"  # fail_on_error=False: the eval itself ran
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
    assert log.status == "success"  # fail_on_error=False
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
