"""Tests for the Grader component: protocol, builtin operator grader, eval wiring."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import inspect_robots.registry as reg
from inspect_robots import eval as ir_eval
from inspect_robots import eval_set
from inspect_robots.errors import ConfigError
from inspect_robots.grader import Grader, operator_grader
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import Scorer, success_at_end
from inspect_robots.session import OperatorSession
from inspect_robots.task import Task
from inspect_robots.types import ActionChunk, Observation


class _BoomPolicy:
    """Fail inference so every trial errors and grading must be skipped."""

    def __init__(self) -> None:
        self.info = PolicyInfo(name="boom", action_space=CubePickEmbodiment().info.action_space)
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        """Accept any scene; failure happens at act()."""
        return None

    def act(self, observation: Observation) -> ActionChunk:
        """Raise so the trial records an error."""
        raise RuntimeError("inference exploded")


class _RecordingGrader:
    """A protocol-conforming grader that logs each grade() call."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def grade(self, record: TrialRecord, scene: Scene) -> None:
        """Record the trial identity and stamp a judgement."""
        del scene
        self.calls.append((record.scene_id, record.epoch))
        record.operator_judgement = "y"


def _task(*, epochs: int = 1, scorer: Scorer | None = None) -> Task:
    return Task(
        name="grader-test",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=scorer if scorer is not None else success_at_end(),
        max_steps=60,
        epochs=epochs,
    )


def _scripted_session(answers: list[str]) -> tuple[OperatorSession, list[str]]:
    iterator = iter(answers)
    output: list[str] = []
    return OperatorSession(input_fn=lambda _p: next(iterator), write=output.append), output


def test_operator_grader_conforms_and_resolves_from_registry() -> None:
    grader = resolve("grader", "operator")
    assert isinstance(grader, Grader)
    assert grader.name == "operator"
    assert isinstance(operator_grader(), Grader)


def test_operator_grader_delegates_to_prompt_verdict() -> None:
    session, _output = _scripted_session(["partial", "slipped near the goal"])
    grader = operator_grader(session)
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    grader.grade(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "partial"
    assert record.operator_note == "slipped near the goal"


def test_connect_session_rebinds_the_prompt_seams() -> None:
    first, _ = _scripted_session([])
    second, _ = _scripted_session(["n", ""])
    grader = operator_grader(first)
    grader.connect_session(second)
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    grader.grade(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "n"


def test_operator_grader_builds_a_lazy_default_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(["y", ""])
    monkeypatch.setattr(builtins, "input", lambda _p: next(answers))
    grader = operator_grader()
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    grader.grade(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "y"


def test_eval_calls_grader_once_per_scored_trial(tmp_path: Path) -> None:
    grader = _RecordingGrader()
    logs = ir_eval(
        _task(epochs=2),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        grader=grader,
    )
    assert grader.calls == [("s0", 0), ("s0", 1)]
    assert logs[0].samples[0].operator_judgements == ("y", "y")


def test_eval_never_grades_errored_trials(tmp_path: Path) -> None:
    grader = _RecordingGrader()
    ir_eval(
        _task(),
        _BoomPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        grader=grader,
    )
    assert grader.calls == []


def test_eval_resolves_grader_registry_names(tmp_path: Path) -> None:
    grader = _RecordingGrader()
    reg.grader("recording-for-test")(lambda: grader)
    try:
        ir_eval(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            grader="recording-for-test",
        )
    finally:
        reg._FACTORIES["grader"].pop("recording-for-test", None)
    assert grader.calls == [("s0", 0)]


def test_eval_rejects_grader_and_before_scoring_together(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not both"):
        ir_eval(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            grader=_RecordingGrader(),
            before_scoring=lambda record, scene: None,
        )


def test_eval_rejects_non_conforming_grader(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Grader protocol"):
        ir_eval(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            grader=object(),  # type: ignore[arg-type]
        )


def test_eval_set_shares_one_grader_across_tasks(tmp_path: Path) -> None:
    grader = _RecordingGrader()
    success, logs = eval_set(
        [_task(), _task()],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        grader=grader,
    )
    assert success
    assert len(logs) == 2
    assert grader.calls == [("s0", 0), ("s0", 0)]
