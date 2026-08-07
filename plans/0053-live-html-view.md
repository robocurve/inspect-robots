# Live HTML view of a running eval + prominent live-view tip for `policy=agent`

> Status: PLANNED — issue #329. Implements turn-by-turn browser viewing of an
> in-progress run and the bright CLI tip pointing operators at it.

## Problem

The HTML viewer is strictly post-hoc. `JsonLogSink` writes the `EvalLog` once,
atomically, in `on_eval_end`; while a run executes there is no log on disk, so
`inspect-robots view --serve` shows nothing until the run finishes. Watching a
run live (each agent turn, its notes, operator/voice interjections,
observations) currently requires the Rerun viewer. Operators running
`policy=agent` on a rig want a browser tab that updates as the run progresses,
plus an unmissable pointer to it when the run starts.

## Goals

1. While an eval runs, a browser page served by `inspect-robots view LOG_DIR
   --serve` shows the run so far — completed trials, and the in-progress
   trial's policy transcript (agent turns, notes, folded-in operator/voice
   feedback lines) — updating within a few seconds of each new turn.
2. Zero new dependencies; core stays NumPy+stdlib only.
3. The run process and the viewer stay decoupled processes (sink writes files;
   `view --serve` renders them), mirroring Inspect AI's log-file split.
4. When a `run`/`eval-set` invocation resolves the `agent` policy, the CLI
   prints a prominent brightly colored tip with the exact live-view command.
5. The atomic-final-log invariant is untouched: an interrupted run still leaves
   either a complete valid final log or none.

## Non-goals

- No websocket/live server inside the run process.
- No live *scores* (scoring happens after `on_trial_end`, outside sinks; live
  pages show scores as pending).
- No change to the Rerun sink.
- No frame streaming changes: live pages reuse the existing stored-frame
  correlation (`FrameStore` already streams PNGs to disk during the rollout).

## Design

### 1. `LiveLogSink` — `src/inspect_robots/logging/live_log.py`

A new sink that maintains a **valid schema-v1 `EvalLog` with
`status="started"`** on disk while the run executes. Because the file is
schema-valid, `read_eval_log`, `view`, and the directory renderer consume it
unchanged.

- **Filename:** `<slug(task)>_<uuid8>.live.json` in `log_dir` — matched by the
  directory renderer's existing `*.json` glob; the `.live.json` suffix is the
  cheap marker the serve loop uses to detect a run in progress.
- **Construction:** `LiveLogSink(log_dir, *, min_write_interval_s=1.0,
  clock=time.monotonic)` — injectable clock for deterministic throttle tests.
- **Hooks:**
  - `on_eval_start(spec)` — snapshot the spec, write immediately (the page
    appears as soon as the run starts).
  - `on_trial_start(scene_id, epoch)` — open an in-progress trial entry;
    forced write.
  - `log_step(t, ...)` — record the current step index only (an int); a write
    happens only if the throttle interval elapsed. High-rate control steps
    must never turn into high-rate disk writes.
  - `log_policy_messages(t, messages)` (duck-typed extension, same contract as
    `RerunSink`) — append a shallow snapshot (`[dict(m) for m in messages]`,
    defensive against non-dict rows) of the delta to the in-progress trial's
    transcript; throttled write. This is the turn-by-turn channel: for
    `policy=agent`, deltas carry the assistant turns, notes, and the
    `operator feedback (step N): ...` user lines that voice/console input
    becomes (`plugins/inspect-robots-agent/.../policy.py:_operator_lines`).
  - `on_trial_end(record)` — replace the in-progress entry with the record's
    data: `termination_reason`, `operator_judgement`/`operator_note`,
    delivered operator messages, the normalized `record.policy_transcript`
    when present (falling back to the accumulated deltas), and
    `status="error"` propagation for errored trials; forced write.
  - `on_eval_end(log)` — `unlink(missing_ok=True)` the live file. The
    canonical `JsonLogSink` final log replaces it; the live page disappears
    from the index and the final page takes over.
- **Log assembly invariants:** all `SceneResult` parallel tuples stay strictly
  parallel. The in-progress trial contributes one entry to every parallel
  field (`epochs` gets `{}`, `operator_judgements`/`termination_reasons` get
  `None`, `policy_transcripts` gets the accumulated transcript).
  `trial_metadata` for the in-progress trial carries
  `{"live": {"step": t, "updated_at": <iso>}}` so the page can show progress.
  `EvalResults`/`EvalStats` are filled with the counts so far
  (`completed_at=""` → set to the last-write time, `duration_s` so far).
- **Write path:** identical discipline to `JsonLogSink` — reuse its
  `_sanitize` (import it; it is module-level) and temp-file + `os.replace`.
  Each write is atomic so the renderer can never read a torn file. No fsync on
  the throttled path (a lost live update is worthless; final log durability is
  `JsonLogSink`'s job).
- **Failure isolation:** the sink must never kill a run. Wrap the disk write
  in try/except; on failure warn once to stderr and disable further writes.
- **Crash staleness:** a hard-killed run leaves a `*.live.json` behind. The
  index row still renders (valid log, status `running`), and the page shows
  its `updated_at`. Documented; no reaper in this plan.
- **Exports:** `inspect_robots.logging.__all__` gains `LiveLogSink`;
  registered as `sink("live-json")` in `_builtins.py` next to the existing
  `sink("json")`. Update `tests/test_api_snapshot.py` if the snapshot covers
  it.

### 2. Renderer — `_html.py`, `_html_index.py`

- `_STATUS_DISPLAY` gains `"started": "running"`; `_status_class` /
  `_status_badge` give `running` a distinct bright badge (amber), reused by
  the index row.
- `render_html(...)` gains `refresh_seconds: int | None = None`. When set, the
  page gets `<meta http-equiv="refresh" content="N">` plus a prominent banner
  under the header: `RUNNING — refreshes every Ns · last update HH:MM:SS`
  (from the live `updated_at` metadata when present). Interpolations escaped
  once, same as everything else in the module.
- Pending scores: a running log's epochs are `{}` — render `pending` in the
  score cells rather than blank, only when the log status is `started`.

### 3. Serve loop — `cli.py`

- New module constant `_SERVE_LIVE_RERENDER_SECONDS = 2` beside
  `_SERVE_RERENDER_SECONDS = 60`.
- `_serve_view_directory` ticks every `_SERVE_LIVE_RERENDER_SECONDS`. Each
  tick: if `log_dir.glob("*.live.json")` is non-empty, run the incremental
  re-render (mtime-gated, so only pages whose log changed are re-rendered);
  otherwise only re-render when 60s have accumulated (preserving today's
  cadence and cost when nothing is live). `_serve_sleep` stays the injection
  point for tests.
- `_render_log_page` / `_render_view_directory` pass
  `refresh_seconds=_SERVE_LIVE_RERENDER_SECONDS` down to `render_html` only
  for logs with `status == "started"` **and** only in serve mode (a static
  `view` of a stale live log must not meta-refresh forever).
- Accepted cost: while a live log exists, the directory pass re-parses every
  log JSON every 2s. Mtime gating keeps re-rendering (the expensive part)
  incremental; typical log dirs make the parse pass cheap. Called out in the
  plan so it is a deliberate tradeoff, revisit if dirs grow.

### 4. CLI wiring + the bright tip — `cli.py`

- `run` (registered-task, ad-hoc) and `eval-set` append
  `LiveLogSink(args.log_dir)` to the sink list by default. New `--no-live-log`
  flag on both commands opts out. The flag exists because a tmpfs-averse or
  NFS-hostile rig may want zero mid-run writes.
- **The tip:** printed with the other run announcements (next to the `rerun:`
  line), when the resolved policy registry name is `agent`:

  ```
  live: watch this run in your browser →  inspect-robots view logs --serve --open
        each agent turn, notes, and operator/voice input, updating live
  ```

  First line: new `_BRIGHT_MAGENTA = "95"` + `_BOLD` (via nested `_styled`,
  matching the existing helper), so it is unmissable on a dark or light
  terminal. Second line dim. `logs` is the actual `args.log_dir` value. The
  existing `_styled` NO_COLOR/tty handling applies unchanged. Suppressed when
  `--no-live-log` was passed (never advertise a view that will not update).
- Predicate: the *resolved* policy name (`resolved.policy.info.name ==
  "agent"`), not the CLI string, so config-default agent runs get the tip too.

### 5. Docs, changelog, module map

- New docs page `docs/live-view.md` ("Watching a run live"): the two-terminal
  flow, what appears live vs. only in the final log (scores), staleness note,
  Rerun as the high-rate alternative. Follows the public-writing style rule.
- README: one bullet in the viewing section.
- `CHANGELOG.md` entry; `src/inspect_robots/CLAUDE.md` module map rows
  (`logging/` and `cli.py`/`_html.py` rows updated).

## Testing (100% coverage, no hardware)

- **Sink unit tests:** lifecycle round-trip via `read_eval_log` after every
  hook; throttle honored/bypassed (fake clock); parallel-tuple invariants;
  transcript delta accumulation and `on_trial_end` replacement; errored-trial
  propagation; unlink on end (and `missing_ok` when already gone); write
  failure disables sink without raising; sanitize reuse (non-finite floats).
- **Integration:** `CubePick` + scripted policy with a `LiveLogSink` — assert
  a mid-run snapshot parses as a valid `EvalLog` with `status="started"` and
  the final state removes the live file.
- **Renderer:** refresh meta + banner emitted only when `refresh_seconds`
  set; `running` badge; pending score cells; escaping.
- **Serve loop:** with `_serve_sleep` stubbed — fast tick re-renders when a
  live file exists, 60s cadence preserved otherwise; refresh_seconds passed
  only for started logs in serve mode.
- **CLI:** tip printed exactly when the resolved policy is `agent` (and
  suppressed with `--no-live-log`); sink wired by default; flag removes it;
  eval-set parity.

## Implementation order

1. `LiveLogSink` + tests (pure core, no CLI surface).
2. Renderer changes + tests.
3. Serve-loop cadence + tests.
4. CLI wiring, `--no-live-log`, the bright tip + tests.
5. Docs, README, CHANGELOG, module map, API snapshot.

## Rejected alternatives

- **HTTP server inside the run process:** couples the control loop to a web
  server, violates the sink model, and dies with the run — the decoupled
  file-based split survives crashes and matches Inspect AI.
- **Incremental append-format (`.eval`-style) log:** a second on-disk format
  is the long-term answer for very long runs, but a rewritten-JSON snapshot
  is dramatically simpler, keeps every consumer working unchanged, and the
  payloads at robot-eval scale (tens of trials, bounded transcripts) rewrite
  in milliseconds.
- **WebSocket push to the browser:** meta-refresh at 2s is within the latency
  the user asked for ("each new turn") at a fraction of the complexity; no JS
  state to keep alive across refreshes because pages are self-contained.
