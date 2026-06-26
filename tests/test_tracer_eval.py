"""End-to-end tracer: one scene -> chunked rollout -> score -> EvalLog.

This is the load-bearing integration test. It exercises the whole vertical
through every architectural seam (Controller, Approver, LogSink) and must stay
green for the life of the project.
"""

from __future__ import annotations

from pathlib import Path

from robolens import EvalLog, eval, read_eval_log
from robolens.logging import JsonLogSink
from robolens.mock import CubePickEmbodiment, RandomPolicy, ScriptedPolicy
from robolens.scene import Scene
from robolens.scorer import success_at_end
from robolens.task import Task


def _cubepick_task() -> Task:
    return Task(
        name="cubepick-reach",
        scenes=[
            Scene(id="center", instruction="reach the cube", init_seed=0),
            Scene(id="corner", instruction="reach the cube", init_seed=1),
        ],
        scorer=success_at_end(),
        max_steps=80,
    )


def test_scripted_policy_succeeds(tmp_path: Path) -> None:
    logs = eval(
        _cubepick_task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    assert isinstance(logs, list) and len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    # A competent scripted policy reaches the cube on every scene.
    assert log.results.metrics["success_at_end"] == 1.0
    assert log.results.total_scenes == 2


def test_random_policy_mostly_fails(tmp_path: Path) -> None:
    logs = eval(
        _cubepick_task(),
        RandomPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        seed=123,
    )
    log = logs[0]
    assert log.status == "success"  # the eval ran fine; the policy just performs poorly
    assert log.results.metrics["success_at_end"] < 1.0


def test_chunked_open_loop_execution(tmp_path: Path) -> None:
    # The scripted policy emits action chunks of length > 1 that are played
    # open-loop. Confirm more environment steps occurred than policy inferences.
    embodiment = CubePickEmbodiment()
    policy = ScriptedPolicy(chunk_size=6)
    eval(_cubepick_task(), policy, embodiment, log_dir=str(tmp_path))
    assert policy.num_inferences > 0
    assert embodiment.num_steps > policy.num_inferences  # H>1 open-loop


def test_eval_log_round_trips(tmp_path: Path) -> None:
    eval(
        _cubepick_task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    log = read_eval_log(str(written[0]))
    assert isinstance(log, EvalLog)
    assert log.version == EvalLog.SCHEMA_VERSION
    assert log.results.metrics["success_at_end"] == 1.0


def test_sink_receives_lifecycle(tmp_path: Path) -> None:
    events: list[str] = []

    class RecordingSink(JsonLogSink):
        def on_eval_start(self, spec: object) -> None:
            events.append("eval_start")
            super().on_eval_start(spec)  # type: ignore[arg-type]

        def on_trial_start(self, scene_id: str, epoch: int) -> None:
            events.append("trial_start")

        def on_trial_end(self, record: object) -> None:
            events.append("trial_end")

        def on_eval_end(self, log: EvalLog) -> None:
            events.append("eval_end")
            super().on_eval_end(log)

    eval(
        _cubepick_task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        sinks=[RecordingSink(str(tmp_path))],
    )
    assert events[0] == "eval_start"
    assert events[-1] == "eval_end"
    assert events.count("trial_start") == events.count("trial_end") == 2
