**Core:** handoff trials (plan 0090). `eval(task, HandoffAgent(command), embodiment)`
hands the robot to an agent process that commands it directly, with no action
chunks, action manager or guards, then takes it back and grades the result. The
agent reports `done` with `inspect-robots handoff done` and may wait for grading
with `inspect-robots handoff wait`; at the time limit (the agent's, else the
task's `max_seconds`, else 480 s) its processes are stopped and the robot is
parked. Embodiments opt in with `release(workdir)` and `retake()` hooks. The
`cubepick-handoff` embodiment, `cubepick-touch` task and `cubepick-touched`
grader exercise it on the mock world (`examples/handoff_cubepick.py`).
