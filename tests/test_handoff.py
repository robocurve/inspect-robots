"""Handoff trials: an agent process commands the robot itself and the framework grades."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.cli import main as cli_main
from inspect_robots.errors import ConfigError, EmbodimentFault, PolicyError
from inspect_robots.handoff import (
    AGENT_EXIT,
    DEFAULT_TIME_LIMIT_S,
    ENV_DIR,
    FINISHED,
    GRADING,
    RUNNING,
    TIMED_OUT,
    HandoffAgent,
    HandoffChannel,
    build_prompt,
    handoff,
    supports_handoff,
)
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.mock.handoff_cubepick import (
    STATE_FILE,
    CubePickDriver,
    CubeTouchedGrader,
    HandoffCubePick,
)
from inspect_robots.rollout import Record
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

# A well-behaved agent: touch the cube, park, report done, wait for grading, exit.
_TOUCH = f"""
import os, subprocess, sys
from inspect_robots.handoff import HandoffChannel
from inspect_robots.mock.handoff_cubepick import CubePickDriver
robot = CubePickDriver("{STATE_FILE}")
cube = robot.state()["cube"]
robot.move_to(cube[0], cube[1])
robot.park()
channel = HandoffChannel(os.environ["{ENV_DIR}"])
channel.signal_done("touched and parked")
assert channel.wait(30) == "{FINISHED}"
"""
# Touches the cube, then exits where it stands without reporting anything.
_WALK_OFF = f"""
from inspect_robots.mock.handoff_cubepick import CubePickDriver
robot = CubePickDriver("{STATE_FILE}")
cube = robot.state()["cube"]
robot.move_to(cube[0], cube[1])
"""
_IDLE = "import time; time.sleep(600)"
# Ignores the polite stop signal, so stopping it needs the second one.
_STUBBORN = (
    "import pathlib, signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "pathlib.Path('ready').write_text('')\n"
    "time.sleep(600)\n"
)


def _agent(script: str, **kwargs: Any) -> HandoffAgent:
    kwargs.setdefault("exit_grace_s", 5.0)
    kwargs.setdefault("poll_s", 0.01)
    return HandoffAgent([sys.executable, "-c", script], **kwargs)


def _scene() -> Scene:
    return Scene(id="s", instruction="Touch the cube, then park.", init_seed=3)


def _run(agent: HandoffAgent, embodiment: Any, tmp_path: Path, **kwargs: Any) -> Record:
    kwargs.setdefault("time_limit_s", 20.0)
    return handoff(agent, embodiment, _scene(), seed=3, epoch=0, workdir=tmp_path, **kwargs)


def _past_deadline() -> tuple[Any, Any]:
    """A clock that jumps past any deadline at its second reading, and a no-op sleep."""
    readings = iter([0.0])
    return (lambda: next(readings, 1e9)), (lambda seconds: None)


# -- the control channel ------------------------------------------------------


def test_channel_reports_phase_and_done(tmp_path: Path) -> None:
    channel = HandoffChannel(tmp_path / "c")
    assert channel.phase() is None
    assert channel.done() is None
    channel.open()
    assert channel.phase() == RUNNING
    channel.signal_done("all good")
    assert channel.done() == "all good"
    channel.open()  # a new trial clears the previous done
    assert channel.done() is None
    channel.signal_done()
    assert channel.done() == ""


def test_channel_tolerates_broken_files(tmp_path: Path) -> None:
    channel = HandoffChannel(tmp_path)
    (tmp_path / "status.json").write_text("{not json")
    assert channel.phase() is None
    (tmp_path / "status.json").write_text("[1]")
    assert channel.phase() is None
    (tmp_path / "status.json").write_text('{"phase": 3}')
    assert channel.phase() is None


def test_channel_wait_returns_on_a_final_phase_or_the_timeout(tmp_path: Path) -> None:
    channel = HandoffChannel(tmp_path)
    channel.open()
    naps: list[float] = []
    clock = iter([0.0, 0.0, 5.0])
    assert channel.wait(1.0, clock=lambda: next(clock), sleep=naps.append) == RUNNING
    assert naps == [0.2]
    channel.set_phase(FINISHED)
    assert channel.wait(1.0) == FINISHED


# -- the agent process --------------------------------------------------------


def test_agent_rejects_bad_configuration() -> None:
    with pytest.raises(ConfigError, match="non-empty command"):
        HandoffAgent([])
    with pytest.raises(ConfigError, match="time_limit_s"):
        HandoffAgent(["x"], time_limit_s=0)


def test_agent_config_names_env_keys_only() -> None:
    agent = HandoffAgent(["x", "{prompt}"], time_limit_s=20, env={"API_KEY": "secret"})
    assert agent.config() == {
        "command": ["x", "{prompt}"],
        "time_limit_s": 20,
        "exit_grace_s": 10.0,
        "env_keys": ["API_KEY"],
    }
    assert "secret" not in json.dumps(agent.config())


def test_agent_is_inert_before_it_starts() -> None:
    agent = HandoffAgent(["x"])
    assert agent.exit_code() is None
    agent.stop()
    record = Record(scene_id="s", epoch=0, seed=0)
    agent.on_trial_end(record, "logs", "run")
    assert record.metadata == {}


def test_agent_substitutes_prompt_and_workdir(tmp_path: Path) -> None:
    channel = HandoffChannel(tmp_path / "handoff")
    channel.open()
    script = "import sys, pathlib; pathlib.Path(sys.argv[2], 'seen').write_text(sys.argv[1])"
    agent = HandoffAgent([sys.executable, "-c", script, "{prompt}", "{workdir}"])
    agent.start(tmp_path, "the task", channel)
    record = Record(scene_id="s", epoch=0, seed=0)  # no handoff details to fill in
    agent.on_trial_end(record, "logs", "run")
    assert (tmp_path / "seen").read_text() == "the task"
    assert channel.phase() == FINISHED
    assert record.metadata == {}


def test_stop_kills_an_agent_that_ignores_the_first_signal(tmp_path: Path) -> None:
    channel = HandoffChannel(tmp_path / "handoff")
    channel.open()
    agent = _agent(_STUBBORN, exit_grace_s=0.3)
    agent.start(tmp_path, "", channel)
    deadline = time.monotonic() + 10
    while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    agent.stop()
    assert agent.exit_code() is not None
    agent.stop()  # already gone: nothing to signal


def test_prompt_states_the_task_the_robot_and_the_ending() -> None:
    prompt = build_prompt(
        _scene(),
        robot_notes="Use the bus.",
        embodiment_docs=" +x is right ",
        time_limit_s=20,
        cli="C",
    )
    assert "Touch the cube, then park." in prompt
    assert "Use the bus.\n\n+x is right\n" in prompt
    assert "You have 20 seconds" in prompt
    assert '`C done --reason "what you did"`' in prompt and "`C wait`" in prompt
    bare = build_prompt(
        _scene(), robot_notes="Use the bus.", embodiment_docs=None, time_limit_s=1, cli="C"
    )
    assert "Use the bus.\n\n# Ending" in bare


# -- one trial ----------------------------------------------------------------


def test_done_ends_the_trial_and_the_agent_waits_for_grading(tmp_path: Path) -> None:
    embodiment = HandoffCubePick()
    agent = _agent(_TOUCH)
    record = _run(agent, embodiment, tmp_path)

    assert record.status == "success" and record.steps == []
    assert record.termination_reason == "done"
    assert record.termination_detail == "touched and parked"
    assert [event.kind for event in record.events] == ["reset", "policy_terminal"]
    assert record.initial_observation is not None and record.unstepped_observation is not None
    assert float(record.initial_observation.state["cube_touched"][0]) == 0.0
    assert float(record.unstepped_observation.state["cube_touched"][0]) == 1.0
    details = record.metadata["handoff"]
    assert details["agent_parked"] is True and details["forced_park"] is False
    assert details["time_limit_s"] == 20.0
    assert "Touch the cube, then park." in (tmp_path / "TASK.md").read_text()

    # The agent is still alive, waiting to hear that grading is over.
    channel = HandoffChannel(tmp_path / "handoff")
    assert channel.phase() == GRADING and agent.exit_code() is None
    agent.on_trial_end(record, "logs", "run")
    assert channel.phase() == FINISHED
    assert details["exit_code"] == 0 and details["exited_by_itself"] is True
    assert (tmp_path / "agent.stderr.log").read_text() == ""


def test_an_agent_that_exits_without_done_is_recorded_as_such(tmp_path: Path) -> None:
    embodiment = HandoffCubePick()
    agent = _agent(_WALK_OFF)
    record = _run(agent, embodiment, tmp_path)
    assert record.termination_reason == AGENT_EXIT
    assert record.termination_detail == "the agent exited with code 0 without reporting done"
    details = record.metadata["handoff"]
    # It left the robot at the cube, and nothing parks it but a time-out.
    assert details["agent_parked"] is False and details["forced_park"] is False
    agent.on_trial_end(record, "logs", "run")
    assert details["exited_by_itself"] is True


def test_the_time_limit_stops_the_agent_and_parks_the_robot(tmp_path: Path) -> None:
    embodiment = HandoffCubePick()
    agent = _agent(_IDLE, exit_grace_s=2.0)
    clock, sleep = _past_deadline()
    record = _run(agent, embodiment, tmp_path, clock=clock, sleep=sleep)
    assert record.termination_reason == "max_seconds" and record.termination_detail is None
    assert record.metadata["handoff"]["forced_park"] is True
    assert agent.exit_code() is not None
    channel = HandoffChannel(tmp_path / "handoff")
    assert channel.phase() == TIMED_OUT
    agent.on_trial_end(record, "logs", "run")
    assert channel.phase() == TIMED_OUT  # a time-out is not rewritten as finished
    assert record.metadata["handoff"]["exited_by_itself"] is True


class _Bare:
    """The smallest embodiment a handoff trial accepts: no park, no parked check."""

    def __init__(self, final: Any = None) -> None:
        self.info = CubePickEmbodiment().info
        self.final = Observation() if final is None else final
        self.retaken = 0

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        return Observation()

    def release(self, workdir: str) -> str:
        return "notes"

    def retake(self) -> Any:
        self.retaken += 1
        if isinstance(self.final, Exception):
            raise self.final
        return self.final

    def step(self, action: Any) -> Any:
        raise AssertionError("a handoff trial never steps")

    def close(self) -> None:
        return None


def test_a_time_out_without_park_hooks_parks_nothing(tmp_path: Path) -> None:
    clock, sleep = _past_deadline()
    record = _run(_agent(_IDLE, exit_grace_s=2.0), _Bare(), tmp_path, clock=clock, sleep=sleep)
    details = record.metadata["handoff"]
    assert details["agent_parked"] is None and details["forced_park"] is False


def test_an_embodiment_without_the_hooks_is_refused(tmp_path: Path) -> None:
    assert supports_handoff(HandoffCubePick()) and not supports_handoff(CubePickEmbodiment())
    with pytest.raises(ConfigError, match=r"release\(workdir\) and retake\(\)"):
        _run(_agent(_IDLE), CubePickEmbodiment(), tmp_path)


def test_a_failed_reset_is_an_embodiment_fault_with_the_record(tmp_path: Path) -> None:
    class _NoReset(_Bare):
        def __init__(self, exc: Exception) -> None:
            super().__init__()
            self.exc = exc

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            raise self.exc

    with pytest.raises(EmbodimentFault, match="no bus") as generic:
        _run(_agent(_IDLE), _NoReset(RuntimeError("no bus")), tmp_path)
    assert generic.value.record is not None and generic.value.record.status == "error"
    with pytest.raises(EmbodimentFault, match="typed") as typed:
        _run(_agent(_IDLE), _NoReset(EmbodimentFault("typed")), tmp_path)
    assert typed.value.record is not None


def test_an_agent_that_cannot_start_is_a_policy_error_and_the_robot_comes_back(
    tmp_path: Path,
) -> None:
    embodiment = _Bare()
    agent = HandoffAgent([str(tmp_path / "no-such-agent")])
    with pytest.raises(PolicyError, match="could not start the handoff agent") as raised:
        _run(agent, embodiment, tmp_path)
    assert raised.value.record is not None and embodiment.retaken == 1


def test_an_interrupt_stops_the_agent_and_takes_the_robot_back(tmp_path: Path) -> None:
    def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    embodiment = _Bare(final=RuntimeError("bus busy"))  # a failing retake must not mask it
    agent = _agent(_IDLE, exit_grace_s=2.0)
    with pytest.raises(KeyboardInterrupt):
        _run(agent, embodiment, tmp_path, sleep=interrupted)
    assert agent.exit_code() is not None and embodiment.retaken == 1


@pytest.mark.parametrize(
    ("final", "message"),
    [
        (RuntimeError("bus busy"), "bus busy"),
        (EmbodimentFault("motor fault"), "motor fault"),
        ("not an observation", "must return an Observation"),
    ],
)
def test_a_failed_retake_is_an_embodiment_fault(tmp_path: Path, final: Any, message: str) -> None:
    agent = _agent(_WALK_OFF.replace("robot.move_to", "pass #"))
    (tmp_path / STATE_FILE).write_text(json.dumps({"cube": [0.5, 0.5]}))
    with pytest.raises(EmbodimentFault, match=message) as raised:
        _run(agent, _Bare(final=final), tmp_path)
    assert raised.value.record is not None and raised.value.record.status == "error"


# -- the mock world -----------------------------------------------------------


def test_driver_moves_are_bounded_and_touching_sticks(tmp_path: Path) -> None:
    embodiment = HandoffCubePick()
    initial = embodiment.reset(_scene(), seed=3)
    notes = embodiment.release(str(tmp_path))
    assert str(tmp_path / STATE_FILE) in notes
    robot = CubePickDriver(tmp_path / STATE_FILE)
    state = robot.move(5.0, -5.0)
    assert state["eef"] == pytest.approx([0.2, 0.0]) and state["touched"] is False
    assert robot.move_to(-1.0, 0.0)["eef"] == pytest.approx([0.0, 0.0])  # stops at the wall
    cube = state["cube"]
    assert robot.move_to(cube[0], cube[1])["touched"] is True
    assert robot.park()["eef"] == pytest.approx([0.1, 0.1])
    assert robot.state()["touched"] is True

    final = embodiment.retake()
    assert embodiment.is_parked()
    assert float(final.state["cube_touched"][0]) == 1.0
    assert (initial.images["top"] == (0, 200, 0)).all(axis=-1).any()
    assert (final.images["top"] == (0, 0, 200)).all(axis=-1).any()
    assert not (final.images["top"] == (0, 200, 0)).all(axis=-1).any()
    with pytest.raises(RuntimeError, match="without a release"):
        embodiment.retake()


def test_park_returns_the_effector_to_its_start(tmp_path: Path) -> None:
    embodiment = HandoffCubePick()
    embodiment.reset(_scene(), seed=3)
    embodiment.release(str(tmp_path))
    CubePickDriver(tmp_path / STATE_FILE).move(0.1, 0.1)
    embodiment.retake()
    assert not embodiment.is_parked()
    embodiment.park()
    assert embodiment.is_parked()


def test_grader_judges_the_final_state() -> None:
    grader = CubeTouchedGrader()
    unjudged = Record(scene_id="s", epoch=0, seed=0)
    grader.grade(unjudged, _scene())
    assert unjudged.operator_judgement is None
    stateless = Record(scene_id="s", epoch=0, seed=0, unstepped_observation=Observation())
    grader.grade(stateless, _scene())
    assert stateless.operator_judgement is None
    for touched, verdict in ((1.0, "success"), (0.0, "failure")):
        record = Record(
            scene_id="s",
            epoch=0,
            seed=0,
            unstepped_observation=Observation(state={"cube_touched": np.array([touched])}),
        )
        grader.grade(record, _scene())
        assert record.operator_judgement == verdict
        assert record.events[-1].data["source"] == "state"


# -- through eval() -----------------------------------------------------------


def test_eval_runs_and_grades_a_handoff_task(tmp_path: Path) -> None:
    agent = _agent(_TOUCH, name="scripted", time_limit_s=20)
    (log,) = ir_eval(
        "cubepick-touch",
        agent,
        "cubepick-handoff",
        grader="cubepick-touched",
        log_dir=str(tmp_path),
    )
    assert log.status == "success"
    assert log.eval.policy == "scripted"
    assert log.eval.max_steps is None and log.eval.max_seconds == 20
    assert log.eval.policy_config["time_limit_s"] == 20
    assert log.results.metrics["operator"] == 1.0
    (sample,) = log.samples
    assert sample.termination_reasons == ("done",)
    assert sample.operator_judgements == ("success",)
    details = sample.trial_metadata[0]["handoff"]
    assert details["agent_parked"] is True and details["exit_code"] == 0
    assert Path(details["workdir"]).is_relative_to(tmp_path.resolve() / "handoff")
    assert HandoffChannel(Path(details["workdir"]) / "handoff").phase() == FINISHED


def test_eval_scores_a_timed_out_trial_as_a_failure(tmp_path: Path) -> None:
    agent = _agent(_IDLE, time_limit_s=0.2, exit_grace_s=2.0)
    (log,) = ir_eval(
        "cubepick-touch", agent, HandoffCubePick(), grader="cubepick-touched", log_dir=str(tmp_path)
    )
    assert log.status == "success"
    assert log.results.metrics["operator"] == 0.0
    (sample,) = log.samples
    assert sample.termination_reasons == ("max_seconds",)
    assert sample.trial_metadata[0]["handoff"]["forced_park"] is True


def test_eval_takes_the_limit_from_the_task_then_the_default(tmp_path: Path) -> None:
    from inspect_robots.registry import resolve

    task = resolve("task", "cubepick-touch")
    (log,) = ir_eval(task, _agent(_TOUCH), HandoffCubePick(), log_dir=str(tmp_path))
    assert log.eval.max_seconds == DEFAULT_TIME_LIMIT_S == 480.0


def test_eval_refuses_an_embodiment_that_cannot_be_handed_over(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot run handoff trials"):
        ir_eval("cubepick-touch", _agent(_IDLE), "cubepick", log_dir=str(tmp_path))
    assert not (tmp_path / "handoff").exists()


# -- the agent-side commands --------------------------------------------------


def test_cli_needs_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV_DIR, raising=False)
    assert cli_main(["handoff", "status"]) == 2
    assert "no handoff trial here" in capsys.readouterr().err
    assert cli_main(["handoff", "status", "--dir", str(tmp_path)]) == 2
    assert f"no handoff trial at {tmp_path}" in capsys.readouterr().err


def test_cli_reports_done_status_and_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    channel = HandoffChannel(tmp_path)
    channel.open()
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    assert cli_main(["handoff", "status"]) == 0
    assert capsys.readouterr().out.strip() == RUNNING
    assert cli_main(["handoff", "done", "--reason", "parked"]) == 0
    assert "no longer yours" in capsys.readouterr().out
    assert channel.done() == "parked"
    assert cli_main(["handoff", "wait", "--timeout", "0"]) == 1
    assert capsys.readouterr().out.strip() == RUNNING
    channel.set_phase(FINISHED)
    assert cli_main(["handoff", "wait"]) == 0
    assert capsys.readouterr().out.strip() == FINISHED
