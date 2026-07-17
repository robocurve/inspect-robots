# Logging & Rerun

## The eval log

Every run produces an immutable [`EvalLog`][inspect_robots.log.EvalLog]: the canonical,
reproducible record. It mirrors Inspect AI: `version`, `status`, an `eval` spec
(task/policy/embodiment, created time, git revision, package versions), `results`
(aggregate metrics), `stats` (timing, inference latency), per-scene `samples`, and
a structured `error`.

```python
from inspect_robots import eval, read_eval_log

(log,) = eval("cubepick-reach", "scripted", "cubepick", log_dir="logs")
again = read_eval_log("logs/cubepick-reach_xxxx.json")   # always re-readable
```

Logs are written atomically (temp file + rename), schema-versioned, and carry
a read-back guarantee: a newer Inspect Robots always reads an older log.

Pressing Ctrl-C during a rollout writes a log with `status: "cancelled"` and
everything gathered so far, including the partial trial record and transcript.

## Sinks

A [`LogSink`][inspect_robots.logging.LogSink] observes the run lifecycle
(`on_eval_start` → per trial `on_trial_start`/`log_step`/`on_trial_end` →
`on_eval_end`). Builtins:

- [`JsonLogSink`][inspect_robots.logging.JsonLogSink]: the default; the canonical JSON record.
- [`RerunSink`][inspect_robots.logging.RerunSink]: optional, lazily imported.

Passing `sinks=` replaces the default `JsonLogSink`, it does not add to it.
Include one in the list if you still want the JSON log:

```python
from inspect_robots.logging import JsonLogSink, RerunSink

eval(task, policy, embodiment, sinks=[JsonLogSink("logs"), RerunSink("run.rrd")])
```

## Rerun visualization

`RerunSink` streams camera images, proprioception, action vectors, reward, and
termination markers to a [Rerun](https://github.com/rerun-io/rerun) recording. It
imports `rerun-sdk` lazily: if it isn't installed, the sink warns once and
no-ops, so core never depends on it. Install with `pip install "inspect-robots[rerun]"`.

Logging is non-blocking. `log_step` snapshots each transition and a background
worker hands it to the SDK, so a slow or stalled viewer connection never delays
the control loop (on real hardware, a blocked viewer used to stall the robot
mid-episode). Under sustained backpressure the sink degrades visualization
instead of control: camera frames are dropped first, so scalar plots stay
complete, then whole steps, and the totals are reported as a `RuntimeWarning`
when the eval ends. The queue is drained at every trial boundary (bounded by
`flush_timeout`), so an eval that aborts mid-run loses at most the current
trial's queued tail; the JSON eval log is synchronous and never affected.

Camera frames are JPEG-compressed by default (`jpeg_quality=75`), which cuts
viewer bandwidth by an order of magnitude. Pass `jpeg_quality=None` for
pixel-exact frames. Compression needs pillow (the `rerun` extra includes it);
without it the sink warns once and logs raw frames. Frames of record are never
at stake either way: scoring reads from the `FrameStore` side-car, not from
Rerun.

```python
RerunSink("run.rrd")                   # record to a file, view later
RerunSink(spawn=True)                  # live viewer on this machine (CLI: --rerun)
RerunSink(connect_url="rerun+http://127.0.0.1:9876/proxy")  # stream to a running viewer
RerunSink(spawn=True, jpeg_quality=None, queue_size=128)  # lossless, deeper buffer
```

The three modes are mutually exclusive: rerun's `save`/`spawn`/`connect_grpc`
calls each replace the SDK's global sink, so combining them raises `ValueError`
rather than silently dropping a stream.

### Live transcript in the viewer

Policies that support transcript streaming automatically add conversation rows
at `trial/<scene>/e<epoch>/llm`. In the Rerun viewer, add a TextLog view and
select that entity path. Tool results use the DEBUG level and system prompts use
TRACE, so enable both levels in the view's log-level filter to see the whole
conversation.

Scrubbing the `step` timeline highlights the transcript rows emitted for that
control step alongside its camera and state data. This live stream is a
best-effort visualization and transcript updates may be dropped under
backpressure. The transcript persisted in the eval log is collected separately
at trial end and remains the complete audit record.

On a headless robot box, `spawn=True` has nowhere to open a window. Run the
viewer on your own machine instead and stream to it: `rerun` on your laptop,
`ssh -R 9876:localhost:9876 <robot>` for the tunnel, then
`inspect-robots run ... --rerun-connect` (a bare `--rerun-connect` targets the
tunnel's localhost URL above; pass a URL to reach a viewer elsewhere). Viewer
and SDK versions must match for live connections.

## Frame side-cars

Camera frames are large. With `store_frames=True`, the rollout streams frames to
a per-run subdirectory of `<log_dir>/frames` through a
[`FrameStore`][inspect_robots.frames.FrameStore] and the `TrialRecord` keeps lightweight
[`FrameRef`][inspect_robots.frames.FrameRef] handles, so long, multi-camera episodes stay
memory-safe and remain scorable from disk. Trial ids repeat across runs, so
each eval gets its own directory; read the exact path from the log's
`stats.frames_dir` rather than globbing `<log_dir>/frames` directly.

```python
eval(task, policy, embodiment, log_dir="logs", store_frames=True)
```

Stored frames are raw `.npy` arrays, not a video. To watch an episode after
the fact, render them with the [`video` subcommand](cli.md#inspect-robots-video):

```bash
inspect-robots video logs/adhoc_xxxx.json
```

`inspect-robots inspect` prints the frames directory and this command as a
hint whenever a log has stored frames.

## Policy transcripts

Policies can persist a per-trial audit record in the eval log; read it with
`inspect-robots inspect LOG.json --transcript`, or render a self-contained
conversation page with [`inspect-robots view`](cli.md#inspect-robots-view):

```bash
inspect-robots view LOG.json
```

The agent policy stores its conversation, with streamed image bytes replaced by
`[image omitted: streamed camera frame]`. The preceding label, such as
`camera 'top_cam' (step 480):`, is emitted whether or not frames are stored,
and when they are (`store_frames=True`) it provides the step join key from a
transcript observation to the stored frame.

`FrameStore` sanitizes trial and camera names before building
`{trial}_{camera}_{t:06d}.npy`. When the sanitizer rewrites a name, use
`StepRecord.image_refs` and `FrameRef.path` as the authoritative mapping instead
of assembling the path from the transcript label.
