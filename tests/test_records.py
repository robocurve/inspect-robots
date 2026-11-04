"""Records: each trial's identity, produced chunks and events, on disk and in the log."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import DONE, Termination
from inspect_robots import eval as ir_eval
from inspect_robots.action_manager import DefaultActionManager, EnsemblingActionManager
from inspect_robots.errors import PolicyError
from inspect_robots.eval import _describe
from inspect_robots.log import EvalLog, read_eval_log
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation


def _task(epochs: int = 1, max_steps: int = 40) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
        epochs=epochs,
    )


def _record_files(log_dir: Path, record_dir: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = log_dir / record_dir
    document = json.loads((root / "record.json").read_text())
    lines = [json.loads(line) for line in (root / "chunks.jsonl").read_text().splitlines()]
    return document, lines


class _Finisher:
    """Moves twice, then ends with a stated reason; the chunk is longer than the replan interval."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="finisher",
            action_space=CubePickEmbodiment().info.action_space,
            terminal_actions=frozenset({DONE}),
        )
        self.config = PolicyConfig()
        self.calls = 0

    def reset(self, scene: Scene) -> None:
        self.calls = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.calls += 1
        moves = [Action(data=np.full(2, 0.01 * i)) for i in range(3)]
        if self.calls == 1:
            return ActionChunk(actions=moves, inference_latency_s=0.25)
        return ActionChunk(actions=[], terminal=Termination(DONE, "reached it"))


def test_every_trial_gets_a_unique_record_and_directory(tmp_path: Path) -> None:
    (log,) = ir_eval(_task(epochs=2), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    (sample,) = log.samples
    assert len(set(sample.record_ids)) == 2
    assert all(record_id and len(record_id) == 32 for record_id in sample.record_ids)
    assert log.stats.records_dir is not None
    for record_id, record_dir in zip(sample.record_ids, sample.record_dirs, strict=True):
        assert record_dir is not None and record_dir.startswith("records/")
        document, lines = _record_files(tmp_path, record_dir)
        assert document["record_id"] == record_id
        assert lines[0]["kind"] == "header" and lines[0]["record_id"] == record_id
    assert log.version == EvalLog.SCHEMA_VERSION == 3


def test_chunks_are_recorded_as_produced_not_as_executed(tmp_path: Path) -> None:
    (log,) = ir_eval(
        _task(),
        _Finisher(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        action_manager=DefaultActionManager(replan_interval=2),
    )
    (sample,) = log.samples
    (record_dir,) = sample.record_dirs
    assert record_dir is not None
    document, lines = _record_files(tmp_path, record_dir)
    header, first, last = lines
    assert header["action_dim"] == 2
    # The policy produced three actions; the action manager executed two of them.
    assert first == {
        "t": 0,
        "actions": [[0.0, 0.0], [0.01, 0.01], [0.02, 0.02]],
        "terminal": None,
        "latency_s": 0.25,
        "wall_s": first["wall_s"],
    }
    assert first["wall_s"] >= 0
    assert last["t"] == 2 and last["actions"] == []
    assert last["terminal"] == {"kind": "done", "reason": "reached it"}
    assert document["steps"] == 2 and document["chunks"] == 2
    assert document["termination"] == {
        "reason": "done",
        "detail": "reached it",
        "terminated": False,
        "truncated": True,
    }
    assert document["actions"] == sample.trial_metadata[0]["actions"]


def test_the_policy_terminal_event_reaches_the_log(tmp_path: Path) -> None:
    (log,) = ir_eval(_task(), _Finisher(), CubePickEmbodiment(), log_dir=str(tmp_path))
    (log_file,) = tmp_path.glob("*.json")
    reread = read_eval_log(str(log_file))
    (events,) = reread.samples[0].events
    (terminal,) = [event for event in events if event["kind"] == "policy_terminal"]
    assert terminal == {
        "kind": "policy_terminal",
        "t": 3,
        "data": {"kind": "done", "reason": "reached it"},
    }
    assert events[0]["kind"] == "reset"
    assert reread.samples[0].events == log.samples[0].events
    document, _ = _record_files(tmp_path, str(log.samples[0].record_dirs[0]))
    assert document["events"] == list(events)


def test_the_configuration_is_resolved_per_trial_with_credentials_redacted(
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class _KeyedConfig(PolicyConfig):
        api_key: str | None = "sk-live-123"
        api_key_env: str = "PROVIDER_KEY"
        max_tokens: int = 512

    policy = ScriptedPolicy()
    policy.config = _KeyedConfig()
    (log,) = ir_eval(
        _task(epochs=2),
        policy,
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        seed=7,
        action_manager=DefaultActionManager(replan_interval=3),
    )
    sample = log.samples[0]
    first, _ = _record_files(tmp_path, str(sample.record_dirs[0]))
    second, _ = _record_files(tmp_path, str(sample.record_dirs[1]))
    config = first["configuration"]
    assert config["policy"]["config"]["api_key"] == "<redacted>"
    assert config["policy"]["config"]["api_key_env"] == "PROVIDER_KEY"
    assert config["policy"]["config"]["max_tokens"] == 512
    assert log.eval.policy_config["api_key"] == "<redacted>"
    assert "sk-live-123" not in json.dumps(log.to_dict())
    assert config["action_manager"] == {
        "type": "inspect_robots.action_manager.DefaultActionManager",
        "replan_interval": 3,
        "triggers": [],
    }
    assert config["guard"]["type"] == "inspect_robots.guard.PassThroughGuard"
    assert config["scene"]["instruction"] == "reach the cube"
    assert config["rubrics"][0]["name"] == "success_at_end"
    assert config["seed"] == 7
    assert config["trial_seed"] != second["configuration"]["trial_seed"]


def test_every_chunk_of_a_replanning_manager_is_recorded(tmp_path: Path) -> None:
    embodiment = CubePickEmbodiment()
    (log,) = ir_eval(
        _task(max_steps=5),
        ScriptedPolicy(),
        embodiment,
        log_dir=str(tmp_path),
        action_manager=EnsemblingActionManager(embodiment.info.action_space),
    )
    _, lines = _record_files(tmp_path, str(log.samples[0].record_dirs[0]))
    assert [line["t"] for line in lines[1:]] == [0, 1, 2, 3, 4]


def test_action_managers_see_the_policy_through_the_recorder(tmp_path: Path) -> None:
    seen: list[str] = []

    class _Naming(DefaultActionManager):
        def next_action(self, policy: Any, observation: Any, t: int, store: Any) -> Any:
            seen.append(policy.info.name)
            return super().next_action(policy, observation, t, store)

    ir_eval(
        _task(max_steps=2),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        action_manager=_Naming(),
    )
    assert seen == ["scripted", "scripted"]


def test_a_failed_trial_still_records_the_chunk_that_failed_it(tmp_path: Path) -> None:
    class _NaNPolicy(ScriptedPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.array([np.nan, -np.inf]))])

    (log,) = ir_eval(_task(), _NaNPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    document, lines = _record_files(tmp_path, str(log.samples[0].record_dirs[0]))
    assert document["status"] == "error"
    assert "non-finite" in document["error"]
    assert lines[1]["actions"] == [["nan", "-inf"]]
    assert document["events"][-1]["kind"] == "error"


def test_records_can_be_turned_off(tmp_path: Path) -> None:
    (log,) = ir_eval(
        _task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), store_records=False
    )
    assert log.samples[0].record_dirs == (None,)
    assert log.samples[0].record_ids[0] is not None
    assert log.stats.records_dir is None
    assert not (tmp_path / "records").exists()


def test_a_record_write_failure_degrades_to_a_warning(tmp_path: Path) -> None:
    (tmp_path / "records").write_text("not a directory")
    with pytest.warns(RuntimeWarning, match="Record storage disabled for this trial"):
        (log,) = ir_eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.samples[0].record_dirs == (None,)


def test_older_logs_read_with_empty_record_fields(tmp_path: Path) -> None:
    (log,) = ir_eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    data = log.to_dict()
    data["version"] = 2
    del data["stats"]["records_dir"]
    for sample in data["samples"]:
        for key in ("record_ids", "record_dirs", "events"):
            del sample[key]
    old = EvalLog.from_dict(data)
    assert old.version == 2
    assert old.stats.records_dir is None
    assert (old.samples[0].record_ids, old.samples[0].events) == ((), ())


def test_policy_errors_before_any_chunk_leave_an_empty_chunk_file(tmp_path: Path) -> None:
    class _Broken(ScriptedPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            raise PolicyError("model unreachable")

    (log,) = ir_eval(_task(), _Broken(), CubePickEmbodiment(), log_dir=str(tmp_path))
    document, lines = _record_files(tmp_path, str(log.samples[0].record_dirs[0]))
    assert document["chunks"] == 0 and len(lines) == 1


# -- describing configuration -----------------------------------------------


def test_describe_keeps_records_json_safe() -> None:
    class _Inner:
        def __init__(self) -> None:
            self.limit = np.float32(0.5)
            self._private = "hidden"

    class _Outer:
        def __init__(self) -> None:
            self.inner = _Inner()
            self.labels = frozenset({"b", "a"})
            self.gain = np.arange(3)
            self.big = np.zeros((10, 10))
            self.bad = float("nan")
            self.slots = (1, None, True)

    described = _describe(_Outer())
    assert described["inner"]["type"].endswith("<locals>._Inner")
    assert described["inner"]["limit"] == 0.5
    assert "_private" not in described["inner"]
    assert described["labels"] == ["a", "b"]
    assert described["gain"] == [0, 1, 2]
    assert described["big"] == {"type": "ndarray", "shape": [10, 10], "dtype": "float64"}
    assert described["bad"] == "nan"
    assert described["slots"] == [1, None, True]
    assert _describe({"a": {"b": {"c": {"d": {"e": 1}}}}}) == {"a": {"b": {"c": {"d": "<dict>"}}}}
    assert _describe({"hf_token": "x", "password": None, "auth": object()})["hf_token"] == (
        "<redacted>"
    )
    assert _describe({"password": None}) == {"password": None}
    assert _describe(slice(1, 2)) == {"type": "builtins.slice"}


class _BufferReuser:
    """Returns the same mutable buffer on every call, filled with the call count minus one."""

    def __init__(self) -> None:
        self.info = PolicyInfo(name="reuser", action_space=CubePickEmbodiment().info.action_space)
        self.config = PolicyConfig()
        self.buffer = np.zeros(2)
        self.calls = 0

    def reset(self, scene: Scene) -> None:
        self.calls = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.buffer[:] = 0.01 * self.calls
        self.calls += 1
        return ActionChunk(actions=[Action(data=self.buffer)])


def test_chunks_copy_actions_from_a_reused_policy_buffer(tmp_path: Path) -> None:
    (log,) = ir_eval(
        _task(max_steps=2), _BufferReuser(), CubePickEmbodiment(), log_dir=str(tmp_path)
    )
    (record_dir,) = log.samples[0].record_dirs
    assert record_dir is not None
    _, lines = _record_files(tmp_path, record_dir)
    assert [line["actions"] for line in lines[1:]] == [[[0.0, 0.0]], [[0.01, 0.01]]]


class _NumpyTimer:
    """Reports its inference latency as a numpy scalar (or NaN), as numpy-timed VLAs do."""

    def __init__(self, latency: Any) -> None:
        self.info = PolicyInfo(name="timer", action_space=CubePickEmbodiment().info.action_space)
        self.config = PolicyConfig()
        self.latency = latency

    def reset(self, scene: Scene) -> None:
        pass

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(2))], inference_latency_s=self.latency)


@pytest.mark.parametrize(
    ("latency", "stored"),
    [(np.float32(0.25), 0.25), (float("nan"), "nan")],
)
def test_numpy_and_nan_latencies_are_recorded_as_strict_json(
    tmp_path: Path, latency: Any, stored: Any
) -> None:
    (log,) = ir_eval(
        _task(max_steps=1), _NumpyTimer(latency), CubePickEmbodiment(), log_dir=str(tmp_path)
    )
    assert log.status == "success"
    (record_dir,) = log.samples[0].record_dirs
    assert record_dir is not None
    _, lines = _record_files(tmp_path, record_dir)
    assert lines[1]["latency_s"] == stored


def test_numpy_rubric_metadata_does_not_escape_eval(tmp_path: Path) -> None:
    from inspect_robots import Rubric

    task = Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        max_steps=2,
        rubrics=[Rubric("reached", success_at_end(), metadata={"n": np.int64(3)})],
    )
    (log,) = ir_eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "success"
    (record_dir,) = log.samples[0].record_dirs
    assert record_dir is not None
    document, _ = _record_files(tmp_path, record_dir)
    assert document["configuration"]["rubrics"][0]["metadata"] == {"n": 3}


def test_unencodable_rubric_metadata_still_writes_a_log(tmp_path: Path) -> None:
    from inspect_robots import Rubric

    rubric = Rubric(
        "reached",
        success_at_end(),
        metadata={"weights": np.array([1.0, 2.0]), "source": Path("a.md"), "n": np.int64(3)},
    )
    task = Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        max_steps=2,
        rubrics=[rubric],
    )
    (log,) = ir_eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "success"
    (log_file,) = tmp_path.glob("*.json")
    reread = read_eval_log(str(log_file))
    assert reread.eval.rubrics[0]["metadata"] == {"n": 3}
    assert reread.samples[0].rubrics[0]["metadata"] == {"n": 3}
