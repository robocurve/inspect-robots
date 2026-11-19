# 0090: Handoff trials

Status: built on the mock world; not yet run with a real agent or a real robot.

## Problem

Every trial runs through `rollout()`: the framework asks a policy for an action
chunk, passes each action through the guards and steps the embodiment. That
fits a policy that is a function of the observation. It does not fit an agent
that should be in charge: a coding agent with full computer access that
commands the robot itself, with its own tools, for as long as it likes.

## Change

A second kind of trial, `handoff()` in `handoff.py`, beside `rollout()`. The
framework prepares the scene, hands the robot to an agent process, takes it
back, and grades what the agent left. No action passes through the framework,
so there are no chunks, no action manager and no guards, and the record has no
steps.

`eval(task, HandoffAgent(command), embodiment)` runs a task this way: a
`HandoffAgent` goes where a policy would. Tasks, rubrics, graders, scorers,
logs, annotations and the embodiment's evaluation interface (`health`,
`prepare_trial`, `home`, `park`) are used unchanged.

### One trial

1. `embodiment.reset()` prepares the scene and returns the initial observation.
2. `embodiment.release(workdir)` stops the embodiment's own control and returns
   markdown notes on how to command the robot.
3. The agent process starts in `workdir`, in its own process group, with
   `TASK.md` (the instruction, those notes, the embodiment's `docs` and the
   ending rules) and `INSPECT_ROBOTS_HANDOFF_DIR` set.
4. The agent decides when it has finished. It parks the robot, stops commanding
   it, and runs `inspect-robots handoff done`.
5. The trial ends on `done`, on the agent's exit, or at the time limit. At the
   limit the agent's process group is stopped (terminate, then kill).
6. `embodiment.retake()` takes the robot back and returns the final
   observation. After a time-out the framework calls `park()`.
7. Grading runs. An agent that ran `inspect-robots handoff wait` is released
   when the channel reaches `finished`, and anything still running is stopped.

The time limit is `HandoffAgent.time_limit_s`, else the task's `max_seconds`,
else 480 seconds.

### Embodiment hooks

Duck-typed, like the rest of the evaluation interface:

- `release(workdir: str) -> str` (required)
- `retake() -> Observation` (required)
- `is_parked() -> bool` (optional): recorded as `agent_parked`
- `park()` (optional): called after a time-out, and by `eval()` after every trial

### Control channel

A directory with two JSON files: `status.json` (`running`, `retaking`,
`grading`, `finished`, `timed_out`; written by the framework) and `done.json`
(written by the agent). The agent is told the phase, never the verdict.

### Record

`initial_observation` and `unstepped_observation` hold the observations from
before and after the agent acted; the `vlm` grader now takes its initial frames
from the former when a record has no steps. `termination_reason` is `done`,
`agent_exit` or `max_seconds`. `metadata["handoff"]` has the working directory,
the agent's stdout and stderr logs, how long it held the robot, `agent_parked`,
`forced_park` and the exit code.

### Mock world

`cubepick-handoff` (`mock/handoff_cubepick.py`) releases CubePick as one JSON
state file with a `CubePickDriver` over it. The `cubepick-touch` task asks the
agent to touch the cube and park; touching leaves a mark that outlasts parking,
which the `cubepick-touched` grader reads from the final observation.

## Not done

- `inspect-robots run` cannot start a handoff trial; only the Python API can.
- No real embodiment offers `release`/`retake`.
- Nothing limits what the agent does to the robot, and nothing stops it reading
  the rubric or the logs if they are on the same computer.
- The agent's event stream is kept as a log file, not as a transcript in the
  eval log.
- Frames from during the trial are not recorded; only before and after.
