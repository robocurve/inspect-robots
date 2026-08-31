"""Durable per-trial action side-cars from eval orchestration."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from inspect_robots import eval, eval_set
from inspect_robots.approver import ClampApprover
from inspect_robots.console import ConsolePoll, EndRequest
from inspect_robots.eval import _write_action_log
from inspect_robots.frames import _safe
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation, StepResult


class _RecordingSink(NullSink):
    """Collect records after every trial lifecycle hook has run."""

    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.records.append(record)


class FakeOperatorInput:
    """Return scripted console polls to the eval-level operator seam."""

    def __init__(self, polls: list[ConsolePoll]) -> None:
        self.polls = list(polls)

    def begin_trial(self) -> None:
        return None

    def poll(self) -> ConsolePoll:
        return self.polls.pop(0) if self.polls else ConsolePoll()


def _task(*, scene_id: str = "s0", epochs: int = 1, max_steps: int = 40) -> Task:
    return Task(
        name="action-log",
        scenes=[Scene(id=scene_id, instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
        epochs=epochs,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_eval_writes_complete_action_log_and_pointer(tmp_path: Path) -> None:
    sink = _RecordingSink()

    (log,) = eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    (record,) = sink.records
    pointer = record.metadata["actions"]
    assert isinstance(pointer, str)
    path = tmp_path / pointer
    assert path.exists()
    assert log.samples[0].trial_metadata == ({"actions": pointer},)

    lines = _read_jsonl(path)
    header, steps = lines[0], lines[1:]
    run_stamp = Path(pointer).parts[1]
    assert header == {
        "kind": "header",
        "run_id": run_stamp,
        "scene_id": "s0",
        "epoch": 0,
        "action_dim": 2,
        "labels": ["dx", "dy"],
    }
    assert [step["t"] for step in steps] == list(range(len(record.steps)))
    assert len(steps) == len(record.steps) > 0
    for line, step in zip(steps, record.steps, strict=True):
        np.testing.assert_array_equal(line["action"], np.asarray(step.action.data).ravel())


def test_action_log_records_executed_clamped_action(tmp_path: Path) -> None:
    class _OutOfBoundsPolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__(chunk_size=1)

        def act(self, observation: Observation) -> ActionChunk:
            del observation
            return ActionChunk(actions=[Action(data=np.array([0.5, -0.5]))])

    embodiment = CubePickEmbodiment()
    sink = _RecordingSink()
    eval(
        _task(max_steps=1),
        _OutOfBoundsPolicy(),
        embodiment,
        log_dir=str(tmp_path),
        sinks=[sink],
        approver=ClampApprover(embodiment.info.action_space),
    )

    (record,) = sink.records
    (step_line,) = _read_jsonl(tmp_path / record.metadata["actions"])[1:]
    np.testing.assert_array_equal(step_line["action"], [0.1, -0.1])
    assert step_line["action"] != [0.5, -0.5]


def test_errored_trial_writes_partial_action_log(tmp_path: Path) -> None:
    class _BoomAfterOneStepPolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__(chunk_size=1)

        def act(self, observation: Observation) -> ActionChunk:
            if self.num_inferences:
                raise RuntimeError("inference exploded")
            return super().act(observation)

    sink = _RecordingSink()
    (log,) = eval(
        _task(),
        _BoomAfterOneStepPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    (record,) = sink.records
    assert log.status == record.status == "error"
    assert len(record.steps) == 1
    lines = _read_jsonl(tmp_path / record.metadata["actions"])
    assert len(lines) == 2
    np.testing.assert_array_equal(lines[1]["action"], record.steps[0].action.data)


def test_cancelled_trial_writes_action_log_before_reraise(tmp_path: Path) -> None:
    class _InterruptAfterOneStepPolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__(chunk_size=1)

        def act(self, observation: Observation) -> ActionChunk:
            if self.num_inferences:
                raise KeyboardInterrupt("operator interrupt")
            return super().act(observation)

    sink = _RecordingSink()
    with pytest.raises(KeyboardInterrupt):
        eval(
            _task(),
            _InterruptAfterOneStepPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            sinks=[sink],
        )

    (record,) = sink.records
    assert record.status == "cancelled"
    assert len(record.steps) == 1
    pointer = record.metadata["actions"]
    assert (tmp_path / pointer).exists()
    assert len(_read_jsonl(tmp_path / pointer)) == 2


def test_store_actions_false_creates_no_directory_or_pointer(tmp_path: Path) -> None:
    sink = _RecordingSink()
    eval(
        _task(max_steps=1),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
        store_actions=False,
    )

    assert not (tmp_path / "actions").exists()
    assert "actions" not in sink.records[0].metadata


def test_zero_step_trial_writes_header_only_action_log(tmp_path: Path) -> None:
    sink = _RecordingSink()
    operator_input = FakeOperatorInput([ConsolePoll(end=EndRequest())])

    eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
        operator_input=operator_input,
    )

    (record,) = sink.records
    assert record.steps == []
    assert len(_read_jsonl(tmp_path / record.metadata["actions"])) == 1


def test_start_failed_trial_writes_header_only_action_log(tmp_path: Path) -> None:
    class _StartFailurePolicy(ScriptedPolicy):
        def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
            raise RuntimeError("capture setup exploded")

    sink = _RecordingSink()
    (log,) = eval(
        _task(),
        _StartFailurePolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    (record,) = sink.records
    assert log.status == record.status == "error"
    assert record.steps == []
    pointer = record.metadata["actions"]
    assert len(_read_jsonl(tmp_path / pointer)) == 1


def test_epochs_write_distinct_action_files_in_one_run_directory(tmp_path: Path) -> None:
    sink = _RecordingSink()
    eval(
        _task(epochs=2, max_steps=1),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    pointers = [record.metadata["actions"] for record in sink.records]
    assert Path(pointers[0]).parent == Path(pointers[1]).parent
    assert [Path(pointer).name for pointer in pointers] == ["s0-e0.jsonl", "s0-e1.jsonl"]
    assert all((tmp_path / pointer).exists() for pointer in pointers)


def test_scene_id_sanitization_matches_frames_safe(tmp_path: Path) -> None:
    scene_id = "../hostile/scene"
    sink = _RecordingSink()
    eval(
        _task(scene_id=scene_id, max_steps=1),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    pointer = sink.records[0].metadata["actions"]
    assert Path(pointer).name == f"{_safe(scene_id)}-e0.jsonl"
    assert (tmp_path / pointer).resolve().is_relative_to(tmp_path.resolve())
    assert _read_jsonl(tmp_path / pointer)[0]["scene_id"] == scene_id


@pytest.mark.parametrize(
    ("semantics", "expected"),
    [
        (None, None),
        (ActionSemantics(control_mode="eef_delta_pos", frame="world"), None),
        (
            ActionSemantics(control_mode="eef_delta_pos", frame="world", dim_labels=("x", "y")),
            ["x", "y"],
        ),
    ],
)
def test_action_log_labels_follow_action_semantics(
    tmp_path: Path,
    semantics: ActionSemantics | None,
    expected: list[str] | None,
) -> None:
    embodiment = CubePickEmbodiment()
    action_space = Box(
        shape=(2,),
        low=np.array([-0.1, -0.1]),
        high=np.array([0.1, 0.1]),
        semantics=semantics,
    )
    embodiment.info = replace(embodiment.info, action_space=action_space)
    sink = _RecordingSink()

    eval(
        _task(max_steps=1),
        ScriptedPolicy(),
        embodiment,
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    header = _read_jsonl(tmp_path / sink.records[0].metadata["actions"])[0]
    assert header["labels"] == expected


def test_non_finite_policy_action_errors_with_header_only_log(tmp_path: Path) -> None:
    class _NaNPolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__(chunk_size=1)

        def act(self, observation: Observation) -> ActionChunk:
            del observation
            return ActionChunk(actions=[Action(data=np.full(2, np.nan))])

    sink = _RecordingSink()
    embodiment = CubePickEmbodiment()
    step = Mock(wraps=embodiment.step)
    with (
        patch.object(embodiment, "step", step),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        (log,) = eval(
            _task(),
            _NaNPolicy(),
            embodiment,
            log_dir=str(tmp_path),
            sinks=[sink],
        )

    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert log.status == "error"
    assert log.error == "all 1 trial(s) errored; nothing was scored"
    sample = log.samples[0]
    assert sample.error is not None and sample.error.startswith("PolicyError:")
    assert "non-finite" in sample.error
    (record,) = sink.records
    assert record.steps == []
    pointer = record.metadata["actions"]
    assert len(_read_jsonl(tmp_path / pointer)) == 1
    step.assert_not_called()


def test_write_action_log_rejects_nan_before_touching_disk(tmp_path: Path) -> None:
    record = TrialRecord(
        scene_id="nan",
        epoch=0,
        seed=0,
        steps=[
            StepRecord(
                t=0,
                observation=Observation(),
                action=Action(data=np.array([np.nan, 0.0])),
                result=StepResult(observation=Observation()),
            )
        ],
    )

    with pytest.warns(RuntimeWarning, match="Action log disabled for this trial"):
        pointer = _write_action_log(record, str(tmp_path), "run", Box(shape=(2,)))

    assert pointer is None
    assert not (tmp_path / "actions").exists()


def test_action_log_io_error_degrades_without_affecting_eval(tmp_path: Path) -> None:
    (tmp_path / "actions").write_text("blocks the directory", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="Action log disabled for this trial"):
        (log,) = eval(
            _task(max_steps=1),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
        )

    assert log.status == "success"
    assert log.samples[0].trial_metadata == ({},)
    assert list(tmp_path.glob("*.json"))


def test_eval_set_forwards_store_actions_behaviorally(tmp_path: Path) -> None:
    success, logs = eval_set(
        _task(max_steps=1),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        store_actions=False,
    )

    assert success
    assert logs[0].samples[0].trial_metadata == ({},)
    assert not (tmp_path / "actions").exists()
