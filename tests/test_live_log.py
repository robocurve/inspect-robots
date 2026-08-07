"""Lifecycle, isolation, and eval integration tests for running JSON snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from inspect_robots import eval, eval_set, read_eval_log
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats
from inspect_robots.logging import LiveLogSink
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots.transcript import operator_message_event
from inspect_robots.types import Action, ActionChunk, Observation, StepResult


class _Clock:
    """Expose deterministic monotonic readings to throttle tests."""

    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def _spec(task: str = "Pick / Cube") -> EvalSpec:
    return EvalSpec(
        task=task,
        policy="agent",
        embodiment="arm",
        created="2026-08-06T12:00:00+00:00",
        inspect_robots_version="test",
    )


def _transition(t: int = 0) -> StepRecord:
    observation = Observation()
    action = Action(np.zeros(1))
    result = StepResult(observation)
    return StepRecord(t, observation, action, result)


def _record(
    *,
    epoch: int = 0,
    status: str = "success",
    transcript: object = None,
) -> TrialRecord:
    return TrialRecord(
        scene_id="scene-a",
        epoch=epoch,
        seed=7,
        steps=[_transition()],
        termination_reason=None if status == "error" else "done",
        status=status,
        error="policy failed" if status == "error" else None,
        inference_latencies=[0.25],
        operator_judgement="y" if status == "success" else None,
        operator_note="steady" if status == "success" else None,
        events=[operator_message_event(0, "slower", "voice")],
        metadata={"temperature": math.inf},
        policy_transcript=transcript,
    )


def _final_log() -> EvalLog:
    return EvalLog(
        version=1,
        status="success",
        eval=_spec(),
        results=EvalResults(total_scenes=0, total_trials=0),
        stats=EvalStats("start", "end", 1.0, 0),
    )


def _parallel_lengths(path: Path) -> set[int]:
    sample = read_eval_log(str(path)).samples[0]
    return {
        len(sample.epochs),
        len(sample.operator_judgements),
        len(sample.operator_notes),
        len(sample.operator_messages),
        len(sample.trial_metadata),
        len(sample.termination_reasons),
        len(sample.policy_transcripts),
    }


def test_lifecycle_throttle_parallel_epochs_replacement_and_reuse(tmp_path: Path) -> None:
    clock = _Clock()
    sink = LiveLogSink(str(tmp_path), min_write_interval_s=1.0, clock=clock)

    sink.on_eval_start(_spec())
    assert sink.path is not None
    first_path = sink.path
    assert first_path.name.startswith("pick-cube_")
    assert first_path.name.endswith(".live.json")
    started = read_eval_log(str(first_path))
    assert started.status == "started"
    assert started.samples == ()

    sink.on_trial_start("scene-a", 0)
    assert _parallel_lengths(first_path) == {1}
    trial_started = read_eval_log(str(first_path))
    assert trial_started.samples[0].trial_metadata[0]["live"]["step"] == 0
    before_step = first_path.read_bytes()
    transition = _transition()
    sink.log_step(4, transition.observation, transition.action, transition.result)
    assert first_path.read_bytes() == before_step

    sink.log_policy_messages(4, [{"role": "assistant", "content": "first"}, "marker"])
    assert read_eval_log(str(first_path)).samples[0].policy_transcripts[0] == []
    clock.now += 1.0
    first = {"role": "assistant", "content": "second"}
    sink.log_policy_messages(5, [first])
    first["content"] = "mutated later"
    visible = read_eval_log(str(first_path))
    assert visible.samples[0].policy_transcripts[0] == [
        {"role": "assistant", "content": "first"},
        "marker",
        {"role": "assistant", "content": "second"},
    ]
    assert visible.samples[0].trial_metadata[0]["live"]["step"] == 5
    assert visible.stats.total_steps == 5

    sink.on_trial_end(_record(transcript=[{"role": "assistant", "content": "final"}]))
    completed = read_eval_log(str(first_path))
    sample = completed.samples[0]
    assert completed.results.total_trials == 1
    assert completed.stats.total_steps == 1
    assert completed.stats.mean_inference_latency_s == 0.25
    assert sample.policy_transcripts[0] == [{"role": "assistant", "content": "final"}]
    assert sample.operator_messages[0] == ({"t": 0, "text": "slower", "source": "voice"},)
    assert sample.trial_metadata[0]["temperature"] is None
    assert sample.operator_judgements == ("y",)
    assert sample.operator_notes == ("steady",)
    assert sample.termination_reasons == ("done",)

    sink.on_trial_start("scene-a", 1)
    assert len(read_eval_log(str(first_path)).samples) == 1
    assert _parallel_lengths(first_path) == {2}
    sink.log_policy_messages(0, ["fallback transcript"])
    sink.on_trial_end(_record(epoch=1, status="error"))
    errored = read_eval_log(str(first_path))
    assert errored.samples[0].status == "error"
    assert errored.samples[0].error == "policy failed"
    assert errored.samples[0].policy_transcripts[1] == ["fallback transcript"]
    assert errored.results.errored_trials == 1

    sink.on_eval_end(_final_log())
    assert not first_path.exists()
    sink.on_eval_end(_final_log())

    sink.on_eval_start(_spec("Second task"))
    assert sink.path is not None and sink.path != first_path
    assert read_eval_log(str(sink.path)).samples == ()
    sink.on_eval_end(_final_log())


class _BadInt:
    """Fail conversion so the log-step hook's isolation path executes."""

    def __int__(self) -> int:
        raise ValueError("bad step")


class _BadList(list[Any]):
    """Fail delta accumulation so the policy-message isolation path executes."""

    def extend(self, values: Any) -> None:
        del values
        raise ValueError("bad delta")


@pytest.mark.parametrize(
    "hook",
    ["eval_start", "trial_start", "log_step", "policy_messages", "trial_end"],
)
def test_every_mutating_hook_failure_warns_once_and_disables(
    hook: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = _Clock()
    sink = LiveLogSink(str(tmp_path / hook), min_write_interval_s=0, clock=clock)
    if hook == "eval_start":
        sink._clock = lambda: (_ for _ in ()).throw(RuntimeError("clock failed"))
        sink.on_eval_start(_spec())
    else:
        sink.on_eval_start(_spec())
        if hook == "trial_start":
            sink._clock = lambda: (_ for _ in ()).throw(RuntimeError("clock failed"))
            sink.on_trial_start("scene-a", 0)
        elif hook == "log_step":
            transition = _transition()
            sink.log_step(
                cast(int, _BadInt()),
                transition.observation,
                transition.action,
                transition.result,
            )
        elif hook == "policy_messages":
            sink._current_transcript = _BadList()
            sink.log_policy_messages(0, [{}])
        else:
            sink.on_trial_end(_record())

    transition = _transition()
    sink.on_trial_start("ignored", 0)
    sink.log_step(0, transition.observation, transition.action, transition.result)
    sink.log_policy_messages(0, [])
    sink.on_trial_end(_record())
    assert capsys.readouterr().err.count("live JSON logging disabled") == 1


def test_write_and_unlink_failures_are_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sink = LiveLogSink(str(tmp_path))
    monkeypatch.setattr(
        "inspect_robots.logging.live_log.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("disk")),
    )
    sink.on_eval_start(_spec())
    assert "OSError: disk" in capsys.readouterr().err

    monkeypatch.undo()
    sink = LiveLogSink(str(tmp_path / "unlink"))
    sink.on_eval_start(_spec())
    assert sink.path is not None
    monkeypatch.setattr(
        Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy"))
    )
    sink.on_eval_end(_final_log())
    assert "OSError: busy" in capsys.readouterr().err


def test_cancelled_trial_prestart_write_guard_and_repeated_disable(tmp_path: Path) -> None:
    sink = LiveLogSink(str(tmp_path), min_write_interval_s=0)
    with pytest.raises(RuntimeError, match="has not started"):
        sink._write(0.0, force=True)
    sink.on_eval_end(_final_log())

    sink.on_eval_start(_spec())
    sink.on_trial_start("scene-a", 0)
    sink.on_trial_end(_record(status="cancelled"))
    assert sink.path is not None
    cancelled = read_eval_log(str(sink.path))
    assert cancelled.samples[0].status == "cancelled"
    assert cancelled.samples[0].error is None

    sink.on_trial_start("scene-a", 1)
    sink.on_trial_end(_record(epoch=1, status="error"))
    sink.on_trial_start("scene-a", 2)
    sink.on_trial_end(_record(epoch=2, status="cancelled"))
    assert read_eval_log(str(sink.path)).samples[0].status == "error"

    sink._disable(RuntimeError("first"))
    sink._disable(RuntimeError("second"))


class _DeltaPolicy(ScriptedPolicy):
    """Expose one live transcript delta after every scripted inference."""

    def __init__(self) -> None:
        super().__init__(chunk_size=1)
        self._pending: list[dict[str, str]] = []
        self._transcript: list[dict[str, str]] = []

    def act(self, observation: Observation) -> ActionChunk:
        action = super().act(observation)
        row = {"role": "assistant", "content": f"turn {len(self._transcript)}"}
        self._pending.append(row)
        self._transcript.append(row)
        return action

    def transcript_delta(self) -> list[dict[str, str]]:
        rows, self._pending = self._pending, []
        return rows

    def transcript(self) -> list[dict[str, str]]:
        return self._transcript


class _ProbeSink(NullSink):
    """Read snapshots synchronously after the preceding live sink hook."""

    def __init__(self, live_sink: LiveLogSink) -> None:
        self.live_sink = live_sink
        self.start_paths: list[Path] = []
        self.message_logs: list[EvalLog] = []
        self.trial_logs: list[EvalLog] = []
        self.removed_on_end = False

    def on_eval_start(self, spec: EvalSpec) -> None:
        del spec
        assert self.live_sink.path is not None
        self.start_paths.append(self.live_sink.path)

    def _read(self) -> EvalLog:
        assert self.live_sink.path is not None
        return read_eval_log(str(self.live_sink.path))

    def log_policy_messages(self, t: int, messages: list[Any]) -> None:
        del t, messages
        self.message_logs.append(self._read())

    def on_trial_end(self, record: TrialRecord) -> None:
        del record
        self.trial_logs.append(self._read())

    def on_eval_end(self, log: EvalLog) -> None:
        del log
        assert self.live_sink.path is not None
        self.removed_on_end = not self.live_sink.path.exists()


def test_eval_integration_probe_observes_valid_mid_run_snapshots(tmp_path: Path) -> None:
    live_sink = LiveLogSink(str(tmp_path), min_write_interval_s=0)
    probe = _ProbeSink(live_sink)
    task = Task(
        name="live-integration",
        scenes=[Scene(id="scene-0", instruction="reach")],
        scorer=success_at_end(),
        max_steps=2,
    )

    logs = eval(
        task,
        _DeltaPolicy(),
        CubePickEmbodiment(),
        sinks=[live_sink, probe],
        log_dir=str(tmp_path),
    )

    assert logs[0].status == "success"
    assert probe.message_logs
    assert all(log.status == "started" for log in probe.message_logs)
    assert probe.trial_logs[0].results.total_trials == 1
    assert probe.trial_logs[0].samples[0].policy_transcripts[0]
    assert probe.removed_on_end


def test_eval_set_threads_and_reuses_caller_supplied_live_sink(tmp_path: Path) -> None:
    live_sink = LiveLogSink(str(tmp_path), min_write_interval_s=0)
    probe = _ProbeSink(live_sink)
    tasks = [
        Task(
            name=f"set-{index}",
            scenes=[Scene(id=f"scene-{index}", instruction="reach")],
            scorer=success_at_end(),
            max_steps=1,
        )
        for index in range(2)
    ]

    success, logs = eval_set(
        tasks,
        _DeltaPolicy(),
        CubePickEmbodiment(),
        sinks=[live_sink, probe],
        log_dir=str(tmp_path),
    )

    assert success
    assert len(logs) == 2
    assert len(set(probe.start_paths)) == 2
    assert all(not path.exists() for path in probe.start_paths)
