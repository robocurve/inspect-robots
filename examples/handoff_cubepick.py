"""A handoff trial on the CubePick mock world: an agent process drives the robot itself.

    python examples/handoff_cubepick.py                  # a scripted agent
    python examples/handoff_cubepick.py --agent idle     # never finishes: shows the time limit
    python examples/handoff_cubepick.py --agent codex    # the Codex CLI, if installed

The scripted agent is the smallest well-behaved one: it reads the robot's
state, touches the cube, parks, reports done, waits for grading and exits.
"""

from __future__ import annotations

import argparse
import sys

from inspect_robots import eval
from inspect_robots.handoff import HandoffAgent

_SCRIPTED = """
import glob, subprocess, sys
from inspect_robots.mock.handoff_cubepick import CubePickDriver
robot = CubePickDriver(glob.glob("cubepick_state.json")[0])
cube = robot.state()["cube"]
robot.move_to(cube[0], cube[1])
robot.park()
cli = [sys.executable, "-m", "inspect_robots.cli", "handoff"]
subprocess.run([*cli, "done", "--reason", "touched the cube and parked"], check=True)
subprocess.run([*cli, "wait"], check=True)
"""

_AGENTS = {
    "scripted": [sys.executable, "-c", _SCRIPTED],
    "idle": [sys.executable, "-c", "import time; time.sleep(3600)"],
    # Codex reads the prompt from argv and writes its event stream to stdout.
    # This sandbox keeps its writes inside the trial directory; swap in
    # --dangerously-bypass-approvals-and-sandbox to give it the whole computer.
    "codex": [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "{prompt}",
    ],
}


def main() -> int:
    """Run one trial and print how it ended."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(_AGENTS), default="scripted")
    parser.add_argument("--time-limit", type=float, default=20.0, help="seconds")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    agent = HandoffAgent(_AGENTS[args.agent], name=args.agent, time_limit_s=args.time_limit)
    (log,) = eval(
        "cubepick-touch",
        agent,
        "cubepick-handoff",
        grader="cubepick-touched",
        log_dir=args.log_dir,
    )
    scene = log.samples[0]
    details = scene.trial_metadata[0]["handoff"]
    print(f"status:        {log.status}")
    print(f"ended by:      {scene.termination_reasons[0]}")
    print(f"judgement:     {scene.operator_judgements[0]}")
    print(f"agent parked:  {details['agent_parked']}")
    print(f"forced park:   {details['forced_park']}")
    print(f"held the robot {details['held_s']} s; agent logs in {details['workdir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
