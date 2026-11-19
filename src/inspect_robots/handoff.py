"""Handoff trials: an outside agent process controls the robot directly.

In a rollout the framework owns the loop: it asks a policy for actions, reviews
them and steps the embodiment. A handoff trial inverts that. The embodiment
releases the robot, an agent process (for example a coding agent with full
computer access) commands it however it likes, and the framework takes the
robot back to grade what the agent left behind. No actions pass through the
framework, so no action manager or guard applies and the record holds no steps.

One trial, as [`handoff`][inspect_robots.handoff.handoff] runs it:

1. ``embodiment.reset()`` prepares the scene and returns the initial observation.
2. ``embodiment.release(workdir)`` stops the embodiment's own control and
   returns markdown notes telling the agent how to command the robot.
3. The agent process starts in ``workdir`` with ``TASK.md`` (the instruction,
   those notes and the rules below) and ``INSPECT_ROBOTS_HANDOFF_DIR`` set.
4. The agent works. When it has finished it parks the robot, stops commanding
   it and reports ``done`` on the control channel.
5. On ``done``, on the agent's exit, or at the time limit (the agent's
   processes are then stopped), ``embodiment.retake()`` takes the robot back
   and returns the final observation. After a time-out the robot is parked.
6. Grading runs. The agent may wait on the control channel for the
   ``finished`` phase before it exits.

The control channel is a directory holding two small JSON files, so an agent
needs nothing but a shell: ``status.json`` (the trial's phase, written by the
framework) and ``done.json`` (written by the agent). ``inspect-robots handoff
done|wait|status`` are the agent-side commands.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from inspect_robots.errors import ConfigError, EmbodimentFault, InspectRobotsError, PolicyError
from inspect_robots.rollout import Record, _record_failure
from inspect_robots.scene import Scene
from inspect_robots.transcript import reset_event, terminal_event
from inspect_robots.types import Observation

if TYPE_CHECKING:
    from inspect_robots.embodiment import Embodiment

#: Environment variable naming the control-channel directory for the agent.
ENV_DIR = "INSPECT_ROBOTS_HANDOFF_DIR"
#: The time limit of a handoff trial when neither the agent nor the task sets one.
DEFAULT_TIME_LIMIT_S = 480.0

#: The agent has the robot.
RUNNING = "running"
#: The agent reported done or exited; the framework is taking the robot back.
RETAKING = "retaking"
#: The framework has the robot and is grading the trial.
GRADING = "grading"
#: Grading is over; an agent waiting on the channel should exit.
FINISHED = "finished"
#: The time limit passed; the agent's processes were stopped.
TIMED_OUT = "timed_out"

#: Phases after which nothing more will happen on the channel.
_FINAL_PHASES = frozenset({FINISHED, TIMED_OUT})

#: Termination reason of a trial whose agent exited without reporting done.
AGENT_EXIT = "agent_exit"
_DONE = "done"
_MAX_SECONDS = "max_seconds"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace ``path`` atomically so a reader never sees a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload)), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object at ``path``, or ``None`` when absent or not an object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class HandoffChannel:
    """The control channel between the framework and the agent, as a directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def open(self) -> None:
        """Create the directory, clear any earlier ``done`` and enter ``running``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "done.json").unlink(missing_ok=True)
        self.set_phase(RUNNING)

    def set_phase(self, phase: str) -> None:
        """Publish the trial's phase to the agent."""
        _write_json(self.directory / "status.json", {"phase": phase})

    def phase(self) -> str | None:
        """The published phase, or ``None`` when no trial has opened the channel."""
        status = _read_json(self.directory / "status.json")
        phase = None if status is None else status.get("phase")
        return phase if isinstance(phase, str) else None

    def signal_done(self, reason: str = "") -> None:
        """Agent side: report that the robot is parked and the agent has finished."""
        _write_json(self.directory / "done.json", {"reason": reason})

    def done(self) -> str | None:
        """The agent's stated reason once it reported done, else ``None``."""
        payload = _read_json(self.directory / "done.json")
        if payload is None:
            return None
        return str(payload.get("reason") or "")

    def wait(
        self,
        timeout_s: float,
        *,
        poll_s: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> str | None:
        """Agent side: block until grading is over; return the last phase seen."""
        deadline = clock() + timeout_s
        while True:
            phase = self.phase()
            if phase in _FINAL_PHASES or clock() >= deadline:
                return phase
            sleep(poll_s)


def _stop_process(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """Stop ``proc`` and every process in its group: terminate, then kill."""
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        if hasattr(os, "killpg"):
            # The agent may have left background processes commanding the
            # robot; the group outlives its leader, so signal it either way.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, sig)
        elif proc.poll() is None:  # pragma: no cover - Windows has no process groups
            proc.terminate() if sig == signal.SIGTERM else proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace_s)


@dataclass
class HandoffAgent:
    """An agent process that is handed the robot for one trial at a time.

    ``command`` is the argv to run. In each argument ``{prompt}`` is replaced
    by the trial's full instructions (the same text as ``TASK.md``) and
    ``{workdir}`` by the trial's working directory, which is also the
    process's current directory. The process runs in its own process group
    with ``INSPECT_ROBOTS_HANDOFF_DIR`` set; its output goes to
    ``agent.stdout.log`` and ``agent.stderr.log`` in the working directory.

    ``time_limit_s`` overrides the task's ``max_seconds``; with neither, the
    limit is ``DEFAULT_TIME_LIMIT_S``. ``exit_grace_s`` is how long the agent
    gets to exit by itself once grading is over, and how long each stop signal
    is given to work.

    Pass a ``HandoffAgent`` to [`eval`][inspect_robots.eval.eval] where a policy
    would go to run a task as handoff trials.
    """

    command: Sequence[str]
    name: str = "handoff"
    time_limit_s: float | None = None
    exit_grace_s: float = 10.0
    poll_s: float = 0.1
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an empty command and non-positive limits before any trial."""
        if not self.command:
            raise ConfigError("HandoffAgent needs a non-empty command")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ConfigError("HandoffAgent.time_limit_s must be positive")
        self._proc: subprocess.Popen[bytes] | None = None
        self._logs: list[IO[bytes]] = []
        self._channel: HandoffChannel | None = None

    def config(self) -> dict[str, Any]:
        """What governed the agent, for the eval log (environment values excluded)."""
        return {
            "command": list(self.command),
            "time_limit_s": self.time_limit_s,
            "exit_grace_s": self.exit_grace_s,
            "env_keys": sorted(self.env),
        }

    def start(self, workdir: Path, prompt: str, channel: HandoffChannel) -> None:
        """Start the agent process in ``workdir`` for one trial."""
        argv = [
            arg.replace("{prompt}", prompt).replace("{workdir}", str(workdir))
            for arg in self.command
        ]
        self._channel = channel
        self._logs = [
            (workdir / "agent.stdout.log").open("wb"),
            (workdir / "agent.stderr.log").open("wb"),
        ]
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=workdir,
                env={**os.environ, **self.env, ENV_DIR: str(channel.directory)},
                stdin=subprocess.DEVNULL,
                stdout=self._logs[0],
                stderr=self._logs[1],
                start_new_session=True,
            )
        except OSError:
            self._close_logs()
            raise

    def exit_code(self) -> int | None:
        """The process's exit code, or ``None`` while it runs or before it started."""
        return None if self._proc is None else self._proc.poll()

    def stop(self) -> None:
        """Stop the agent and everything it started. Safe to call repeatedly."""
        if self._proc is not None:
            _stop_process(self._proc, self.exit_grace_s)
        self._close_logs()

    def _close_logs(self) -> None:
        for log in self._logs:
            log.close()
        self._logs = []

    def on_trial_end(self, record: Record, log_dir: str, run_id: str) -> None:
        """Announce that grading is over, let the agent exit, then stop what is left."""
        proc, channel = self._proc, self._channel
        if proc is None or channel is None:
            return
        if channel.phase() != TIMED_OUT:
            channel.set_phase(FINISHED)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=self.exit_grace_s)
        exited_by_itself = proc.poll() is not None
        self.stop()
        details = record.metadata.get("handoff")
        if isinstance(details, dict):
            details["exit_code"] = proc.poll()
            details["exited_by_itself"] = exited_by_itself
        self._proc = None
        self._channel = None


def build_prompt(
    scene: Scene,
    *,
    robot_notes: str,
    embodiment_docs: str | None,
    time_limit_s: float,
    cli: str,
) -> str:
    """The agent's instructions for one trial: the task, the robot and how to end."""
    docs = (embodiment_docs or "").strip()
    robot = robot_notes.strip() + (f"\n\n{docs}" if docs else "")
    return (
        f"# Task\n\n{scene.instruction}\n\n"
        "You control the robot directly for this trial. Nothing reviews or limits "
        "your commands, and you may use this computer however you like.\n\n"
        f"# The robot\n\n{robot}\n\n"
        "# Ending the trial\n\n"
        f"You have {time_limit_s:g} seconds. When they run out your processes are "
        "stopped and the robot is parked for you.\n\n"
        "When you have finished, or have decided to give up:\n\n"
        "1. Park the robot and stop every process of yours that commands it.\n"
        f'2. Report it: `{cli} done --reason "what you did"`\n'
        f"3. Wait for grading to end, then exit: `{cli} wait`\n\n"
        "After you report done the robot is no longer yours: do not command it again.\n"
    )


def _agent_cli() -> str:
    """The agent-side command prefix, runnable from any directory on this machine."""
    return f"{sys.executable} -m inspect_robots.cli handoff"


def _hook(embodiment: object, name: str) -> Callable[..., Any] | None:
    hook = getattr(embodiment, name, None)
    return hook if callable(hook) else None


def supports_handoff(embodiment: object) -> bool:
    """Whether ``embodiment`` offers both ``release(workdir)`` and ``retake()``."""
    return _hook(embodiment, "release") is not None and _hook(embodiment, "retake") is not None


def handoff(
    agent: HandoffAgent,
    embodiment: Embodiment,
    scene: Scene,
    *,
    time_limit_s: float,
    seed: int | None,
    epoch: int,
    workdir: str | Path,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Record:
    """Run one handoff trial and return its record.

    The record has no steps. ``initial_observation`` is what ``reset()``
    returned and ``unstepped_observation`` is what ``retake()`` returned, so a
    grader has the frames from before and after the agent acted.
    ``termination_reason`` is ``"done"`` (the agent reported it),
    ``"agent_exit"`` (the process ended without reporting) or
    ``"max_seconds"``; ``metadata["handoff"]`` holds the working directory, the
    agent's log paths, how long the agent had the robot, whether the robot was
    parked when taken back (``None`` if the embodiment cannot tell) and whether
    the framework parked it after a time-out.

    The agent process is still running when this returns after ``done``: call
    ``agent.on_trial_end(record, ...)`` once grading is over (``eval()`` does).

    The embodiment must offer ``release(workdir) -> str`` and
    ``retake() -> Observation``. A failure to start the agent is a
    ``PolicyError``; a failure to release or take back the robot is an
    ``EmbodimentFault``. Every error carries the partial record.
    """
    release, retake = _hook(embodiment, "release"), _hook(embodiment, "retake")
    if release is None or retake is None:
        raise ConfigError(
            f"embodiment {embodiment.info.name!r} cannot run a handoff trial: it needs "
            "release(workdir) and retake() hooks"
        )
    record = Record(scene_id=scene.id, epoch=epoch, seed=seed)
    record.events.append(reset_event(seed))
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    channel = HandoffChannel(work / "handoff")
    try:
        initial = embodiment.reset(scene, seed=seed)
        channel.open()
        robot_notes = str(release(str(work)))
    except InspectRobotsError as exc:
        raise _record_failure(record, exc, -1) from None
    except Exception as exc:
        raise _record_failure(record, EmbodimentFault(str(exc)), -1) from exc
    record.initial_observation = initial

    prompt = build_prompt(
        scene,
        robot_notes=robot_notes,
        embodiment_docs=embodiment.info.docs,
        time_limit_s=time_limit_s,
        cli=_agent_cli(),
    )
    (work / "TASK.md").write_text(prompt, encoding="utf-8")

    started = clock()
    try:
        try:
            agent.start(work, prompt, channel)
        except OSError as exc:
            raise _record_failure(
                record, PolicyError(f"could not start the handoff agent: {exc}"), 0
            ) from exc
        reason, detail = _await_agent(agent, channel, started + time_limit_s, clock, sleep)
    except BaseException:
        # Whatever went wrong, nothing of the agent's may keep commanding the
        # robot, and the embodiment needs it back to park or close it.
        agent.stop()
        with contextlib.suppress(Exception):
            retake()
        raise
    held_s = clock() - started

    if reason == _MAX_SECONDS:
        channel.set_phase(TIMED_OUT)
        agent.stop()
    else:
        channel.set_phase(RETAKING)
    try:
        final = retake()
        is_parked = _hook(embodiment, "is_parked")
        agent_parked = None if is_parked is None else bool(is_parked())
        forced_park = False
        park = _hook(embodiment, "park")
        if reason == _MAX_SECONDS and park is not None:
            park()
            forced_park = True
    except InspectRobotsError as exc:
        agent.stop()
        raise _record_failure(record, exc, 0) from None
    except Exception as exc:
        agent.stop()
        raise _record_failure(record, EmbodimentFault(str(exc)), 0) from exc
    if not isinstance(final, Observation):
        agent.stop()
        raise _record_failure(
            record,
            EmbodimentFault(
                f"embodiment {embodiment.info.name!r} retake() must return an Observation"
            ),
            0,
        )
    if reason != _MAX_SECONDS:
        channel.set_phase(GRADING)

    record.unstepped_observation = final
    record.truncated = True
    record.termination_reason = reason
    record.termination_detail = detail or None
    record.events.append(terminal_event(0, reason, detail))
    record.metadata["handoff"] = {
        "agent": agent.name,
        "workdir": str(work),
        "stdout": str(work / "agent.stdout.log"),
        "stderr": str(work / "agent.stderr.log"),
        "time_limit_s": time_limit_s,
        "held_s": round(held_s, 3),
        "agent_parked": agent_parked,
        "forced_park": forced_park,
    }
    return record


def _await_agent(
    agent: HandoffAgent,
    channel: HandoffChannel,
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[str, str]:
    """Wait for done, the agent's exit or the deadline; return the reason and detail."""
    while True:
        # Read the exit code first: an agent that reports done and exits at
        # once must count as done, not as an unexplained exit.
        code = agent.exit_code()
        done = channel.done()
        if done is not None:
            return _DONE, done
        if code is not None:
            return AGENT_EXIT, f"the agent exited with code {code} without reporting done"
        if clock() >= deadline:
            return _MAX_SECONDS, ""
        sleep(agent.poll_s)
