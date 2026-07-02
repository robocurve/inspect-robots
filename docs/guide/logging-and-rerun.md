# Logging & Rerun

## The eval log

Every run produces an immutable [`EvalLog`][inspect_robots.log.EvalLog] — the canonical,
reproducible record. It mirrors Inspect AI: `version`, `status`, an `eval` spec
(task/policy/embodiment, created time, git revision, package versions), `results`
(aggregate metrics), `stats` (timing, inference latency), per-scene `samples`, and
a structured `error`.

```python
from inspect_robots import eval, read_eval_log

(log,) = eval("cubepick-reach", "scripted", "cubepick", log_dir="logs")
again = read_eval_log("logs/cubepick-reach_xxxx.json")   # always re-readable
```

Logs are written **atomically** (temp file + rename), schema-versioned, and carry
a read-back guarantee: a newer Inspect Robots always reads an older log.

## Sinks

A [`LogSink`][inspect_robots.logging.LogSink] observes the run lifecycle
(`on_eval_start` → per trial `on_trial_start`/`log_step`/`on_trial_end` →
`on_eval_end`). Builtins:

- [`JsonLogSink`][inspect_robots.logging.JsonLogSink] — always on; the canonical JSON record.
- [`RerunSink`][inspect_robots.logging.RerunSink] — optional, lazily imported.

```python
from inspect_robots.logging import JsonLogSink, RerunSink

eval(task, policy, embodiment, sinks=[JsonLogSink("logs"), RerunSink("run.rrd")])
```

## Rerun visualization

`RerunSink` streams camera images, proprioception, action vectors, reward, and
termination markers to a [Rerun](https://github.com/rerun-io/rerun) recording. It
imports `rerun-sdk` lazily — if it isn't installed, the sink warns once and
no-ops, so core never depends on it. Install with `pip install "inspect-robots[rerun]"`.

## Frame side-cars

Camera frames are large. With `store_frames=True`, the rollout streams frames to
`<log_dir>/frames` through a [`FrameStore`][inspect_robots.frames.FrameStore] and the
`TrialRecord` keeps lightweight [`FrameRef`][inspect_robots.frames.FrameRef] handles — so
long, multi-camera episodes stay memory-safe and remain scorable from disk.

```python
eval(task, policy, embodiment, log_dir="logs", store_frames=True)
```
