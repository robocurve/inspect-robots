# Logging & Rerun

## The eval log

Every run produces an immutable [`EvalLog`](/api/#inspect_robots.log.EvalLog): the canonical,
auditable record. It mirrors Inspect AI: `version`, `status`, an `eval` spec
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

To follow completed trials and the current policy conversation in a browser,
see [Watching a run live](live-view.md).

## Sinks

A [`LogSink`](/api/#inspect_robots.logging.sink.LogSink) observes the run lifecycle
(`on_eval_start` → per trial `on_trial_start`/`log_step`/`on_trial_end` →
`on_eval_end`). Builtins:

- [`JsonLogSink`](/api/#inspect_robots.logging.json_log.JsonLogSink): the default; the canonical JSON record.
- [`LiveLogSink`](/api/#inspect_robots.logging.live_log.LiveLogSink): a transient, schema-valid running snapshot.
- [`RerunSink`](/api/#inspect_robots.logging.rerun_sink.RerunSink): optional, lazily imported.

Passing `sinks=` replaces the default `JsonLogSink`, it does not add to it.
Include one in the list if you still want the JSON log:

```python
from inspect_robots.logging import JsonLogSink, RerunSink

eval(task, policy, embodiment, sinks=[JsonLogSink("logs"), RerunSink("run.rrd")])
```

`eval_set(..., sinks=...)` reuses the same sink instances across its sequential
task runs. Caller-supplied sinks must reset their run state in `on_eval_start`.

## Rerun visualization

`RerunSink` streams camera images, proprioception, action vectors, reward, and
termination markers to a [Rerun](https://github.com/rerun-io/rerun) recording. It
imports `rerun-sdk` lazily: if it isn't installed, the sink warns once and
no-ops, so core never depends on it. Install with `pip install "inspect-robots[rerun]"`.

The sink lays out two rows: camera views in left/top/right name order across
the top, then a tabbed text panel (latest LLM message up front, the full
transcript and the reward series behind tabs) beside one plot per arm, with
commanded `action/*` and measured `state/*` together for each side.
Embodiments without `dim_labels` get one combined joints plot; runs without
cameras send only the second row.
The layout is re-sent at each trial boundary so it follows the live trial,
which resets viewer tweaks then; a single-trial `run` sends it exactly once.
The layout is built from the declared spaces, so entities logged outside them
(an undeclared state key, or a camera whose runtime name differs from its
declared name) are not shown by the sent layout and must be added in the
viewer manually.

Logging is non-blocking. `log_step` snapshots each transition and a background
worker hands it to the SDK, so a slow or stalled viewer connection never delays
the control loop (on real hardware, a blocked viewer used to stall the robot
mid-episode). Under sustained backpressure the sink degrades visualization
instead of control: camera frames are dropped first, so scalar plots stay
complete, then whole steps, and the totals are reported as a `RuntimeWarning`
when the eval ends. The queue is drained at every trial boundary (bounded by
`flush_timeout`), so an eval that aborts mid-run loses at most the current
trial's queued tail. A live viewer and its teed `.rrd` receive the same worker
stream, so viewer-paced shedding reaches the file too. The `.rrd` records what
the viewer received, not a guaranteed-complete record. The JSON eval log is
synchronous and never affected.

Camera frames are JPEG-compressed by default (`jpeg_quality=75`), which cuts
viewer bandwidth by an order of magnitude. Pass `jpeg_quality=None` for
pixel-exact frames. Compression needs pillow (the `rerun` extra includes it);
without it the sink warns once and logs raw frames. Frames of record are never
at stake either way: scoring reads from the `FrameStore` side-car, not from
Rerun.

```python
RerunSink("run.rrd")                              # record one fixed file
RerunSink(recording_dir="logs")                  # fresh task_slug_xxxxxxxx.rrd per eval
RerunSink(spawn=True, recording_dir="logs")      # local viewer plus file
RerunSink(spawn=True, spawn_port=9877, recording_dir="logs")
RerunSink(connect_url="rerun+http://127.0.0.1:9876/proxy", recording_dir="logs")
RerunSink(spawn=True, jpeg_quality=None, queue_size=128)  # live only, lossless
```

File recording combines with either live mode through `set_sinks` in rerun-sdk
0.24 or newer. With rerun-sdk 0.20 through 0.23, the sink warns once, continues
the live view, and skips the teed file. Only `spawn=True` with `connect_url`, or
`recording_path` with `recording_dir`, raises `ValueError`.

For `inspect-robots run`, a live viewer or `--rerun-connect` saves a `.rrd` in
the log directory by default. Replay it later with:

```bash
rerun logs/task_slug_xxxxxxxx.rrd
```

Use `--no-rerun-save` for a live-only run, or set `rerun_save = false` under
`[defaults]` to make that the rig default. Use explicit `--rerun-save` without
a live-view option to record only; `--rerun-save --no-rerun` also selects this
headless mode. The CLI prints the resolved `.rrd` path after a completed eval.

### Live transcript in the viewer

Policies that support transcript streaming automatically add conversation rows
at `trial/<scene>/e<epoch>/llm`. In the Rerun viewer, add a TextLog view and
select that entity path. Tool results use the DEBUG level and system prompts use
TRACE, so enable both levels in the view's log-level filter to see the whole
conversation. The most recent assistant message is also available as markdown
at `trial/<scene>/e<epoch>/llm/latest`; add a Text Document view for a wrapped
reading pane that stays synchronized with the timeline cursor and always shows
the policy's latest decision.

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
and SDK versions must match for live connections. For a replayable file without
a live connection, use `--rerun-save`. Hosts driving two rigs give each config
its own `rerun_port` so each run spawns its own viewer window.

## Frame side-cars

Camera frames are large. With `store_frames=True`, the rollout streams frames to
a per-run subdirectory of `<log_dir>/frames` through a
[`FrameStore`](/api/#inspect_robots.frames.FrameStore) and the `TrialRecord` keeps lightweight
[`FrameRef`](/api/#inspect_robots.frames.FrameRef) handles, so long, multi-camera episodes stay
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
transcript observation to the stored frame. `inspect-robots view` performs
this step join internally, embedding only an exact match and otherwise leaving
the placeholder in place.

`FrameStore` sanitizes trial and camera names before building
`{trial}_{camera}_{t:06d}.npy`. When the sanitizer rewrites a name, use
`StepRecord.image_refs` and `FrameRef.path` as the authoritative mapping instead
of assembling the path from the transcript label. That remains the right advice
for programmatic consumers. The `view` command performs this join internally
with the same sanitizer and an exact-match-or-degrade contract.

## Wire capture

The agent policy records **exactly what each LLM call sent and received** —
by default, for every run. The saved transcript alone is not that object:
outgoing requests carry the tool schemas, only the newest `image_horizon`
frames (older ones become elision stubs), rendered depth composites, and, on
the Messages wire, `cache_control` breakpoints — none of which survive into
`policy_transcripts`. Wire capture stores the real thing, per attempt,
including retries and the failed calls a run died on.

Layout, under the log directory:

```
wire/<run_id>/<trial_id>/calls.jsonl   one JSON row per HTTP attempt
wire/<run_id>/blobs/<sha256>.png       content-addressed image bytes
```

Each row records `call`, `attempt`, `endpoint`, `t`, `duration_s`,
`request`, `status`, `response`, and (failed attempts only) `error`. Image
payloads inside `request` are replaced by `$blob:<sha256>` sentinels — the
sha of the decoded PNG stored once in `blobs/` — with every other part key,
including data-URL prefixes and `cache_control`, preserved verbatim.
Restoring a byte-faithful request is a string substitution: replace each
sentinel with the base64 of its blob file. The trial's record points at its
capture via `metadata["wire_capture"]`.

Browse it with either viewer:

- `inspect-robots view <log>` — each trial gains a collapsible **Wire**
  section: per-call status, params, delta-rendered messages as sent, and
  the frames the model could actually see, deduplicated against the
  report's embedded-media budget.
- `inspect-robots inspect <log> --wire` — a per-trial call table;
  `--wire N` (with `--trial <scene>-e<epoch>` when the log has several
  trials) dumps every attempt row of call `N`.

Capture is best-effort and can never fail an eval: any sink error prints
one warning and disables capture for that trial. Disable it entirely with
`-P wire_capture=false`. Budget for roughly `store_frames`-scale disk usage
(a 100-call three-camera trial with depth rendering writes on the order of
100 MB, dominated by blobs).
