# Live HTML view of a running eval + prominent live-view tip for `policy=agent`

> Status: PLANNED — issue #329. Implements turn-by-turn browser viewing of an
> in-progress run and the bright CLI tip pointing operators at it.
> CHANGELOG entry links this file as `[plan 0054](plans/0054-live-html-view.md)`
> per the established convention (renumbered from 0053, taken on main).

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
  - `log_step(t, ...)` — record the current step index only (an int). **Never
    writes to disk** (R1 critique): `log_step` runs synchronously on the
    control thread, and a full-log JSON rewrite (up to megabytes of transcript
    on a long agent run) must not inject serialization + disk latency into
    the control period. The step count rides along on the next turn's write.
  - `log_policy_messages(t, messages)` (duck-typed extension, same contract as
    `RerunSink`) — append a shallow snapshot (`[dict(m) if isinstance(m,
    dict) else m for m in messages]` — R3: `dict()` on a non-dict row raises;
    sink.py warns core does not enforce the shape on this live path) of the
    delta to the in-progress trial's transcript; throttled write. This is the turn-by-turn channel: for
    `policy=agent`, deltas carry the assistant turns, notes, and the
    `operator feedback (step N): ...` user lines that voice/console input
    becomes (`plugins/inspect-robots-agent/.../policy.py:_operator_lines`).
    For slow-turn policies (agent, capx — the only in-tree
    `transcript_delta` implementors) a write here is invisible to control:
    the hook fires right after an inference that just took seconds. The
    throttle is what keeps this safe in general: a future high-rate policy
    defining `transcript_delta` would call this at control rate, and the
    1s cap bounds full-log rewrites (R2). Staleness bound, stated so nobody
    "fixes" it with a flush thread: a throttle-suppressed delta is never
    lost (deltas accumulate) but becomes visible only at the next write —
    at worst the trial's final turn waits for `on_trial_end`'s forced
    write. With 1.0s intervals and multi-second turns this is rare and
    acceptable (R2).
  - `on_trial_end(record)` — replace the in-progress entry with the record's
    data: `termination_reason`, `operator_judgement`/`operator_note`,
    delivered operator messages, the normalized `record.policy_transcript`
    when present (falling back to the accumulated deltas), and
    `status="error"` propagation for errored trials; forced write.
  - `on_eval_end(log)` — `unlink(missing_ok=True)` the live file. The
    canonical `JsonLogSink` final log replaces it in the index.
- **`on_eval_end` is not guaranteed** (R1): `bus.on_eval_end` is not in a
  `finally` — an uncaught scorer failure, a grader exception, or Ctrl-C at
  the operator verdict prompt (the *target scenario* for this feature) skips
  it and would leak a phantom "running" log. Therefore: the sink exposes
  `path` (like `JsonLogSink.path`), and `_cmd_run`/`_cmd_eval_set` unlink it
  in their existing `finally` blocks. **Binding discipline** (R2): the sink
  variable is pre-bound to `None` *before* the `try` (mirroring
  `voice_input` at cli.py:1441) — sink construction happens inside the
  `try`, so a `finally` referencing a name bound there would raise
  `UnboundLocalError` on early failures (`--epochs`, guardrail build, voice
  start, grader build) and mask the original exception. Guard shape:
  `if live_sink is not None and live_sink.path is not None:
  live_sink.path.unlink(missing_ok=True)` — `missing_ok` makes the
  post-success double-unlink a no-op and the guard covers the
  pre-`on_eval_start` `path is None` case. **Sink order is
  pinned and tested:** `LiveLogSink` is appended *after* `JsonLogSink`, so
  `_Broadcast.on_eval_end` writes the final log before the live file is
  removed — there is never a moment with neither file on disk.
- **Reusable across sequential runs** (R1): `on_eval_start` fully resets
  state — new filename, cleared trials — so one instance serves a whole
  eval-set. This invariant is stated in the class docstring and tested.
- **Log assembly invariants:** all `SceneResult` parallel tuples stay strictly
  parallel. Samples are keyed by `scene_id` in first-seen order (R3): a
  repeat `scene_id` on `on_trial_start` (multi-epoch) appends one
  in-progress slot to that scene's existing parallel tuples — never a
  duplicate `SceneResult` — and `on_trial_end` replaces the last slot. The
  in-progress slot contributes one entry to every parallel field (`epochs`
  gets `{}`, `operator_judgements`/`termination_reasons` get `None`,
  `policy_transcripts` gets the accumulated transcript). The in-progress
  scene's `status` is `"success"`-so-far, switching to `"error"` if an
  earlier epoch errored (R3: stays within log.py's documented sample-status
  vocabulary). `trial_metadata` for the in-progress trial carries
  `{"live": {"step": t, "updated_at": <iso>}}` so the page can show
  progress. `EvalResults`/`EvalStats` are filled with the counts so far:
  `total_trials` = trials completed so far (matching the final log's
  completed-not-planned semantics), `completed_at` = the last-write time,
  `duration_s` so far (R3: one reading, not two).
- **Write path:** identical discipline to `JsonLogSink` — reuse its
  `_sanitize` (import it; it is module-level) and temp-file + `os.replace`.
  Each write is atomic so the renderer can never read a torn file. No fsync on
  the throttled path (a lost live update is worthless; final log durability is
  `JsonLogSink`'s job).
- **Failure isolation:** the sink must never kill a run. Every hook body —
  not just the disk write (R3: a malformed delta row would otherwise raise
  in the hook and kill the run) — is wrapped; on failure warn once to
  stderr and disable the sink.
- **Crash staleness:** a hard-killed run leaves a `*.live.json` behind. The
  index row still renders (valid log, status `running`), and the page shows
  its `updated_at`. Documented; no reaper in this plan.
- **Exports:** `inspect_robots.logging.__all__` gains `LiveLogSink`;
  registered as `sink("live-json")` in `_builtins.py` next to the existing
  `sink("json")`. The top-level API snapshot does not cover
  `inspect_robots.logging` (verified R1), so only `logging/__init__.py`
  changes. The class docstring notes that a sinks list containing *only* a
  `LiveLogSink` ends the run with no log on disk (the final write is
  `JsonLogSink`'s job).

### 2. Renderer — `_html.py`, `_html_index.py`

- `_STATUS_DISPLAY` gains `"started": "running"`; `_status_class` /
  `_status_badge` give `running` a distinct bright badge (amber). The badge
  CSS must land in **both** duplicated stylesheets — `_html.py`'s `_STYLES`
  and `_html_index.py`'s — or the index badge silently renders unstyled (R1).
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
- **Stateless cadence rule** (R1, restated per R2 so the two off-by-one-tick
  behaviors cannot be implemented inconsistently): every render pass
  computes `live_set = {p.name for p in log_dir.glob("*.live.json")}` once
  (R3: `bool(<generator>)` is always truthy — the set, not the generator, is
  the predicate) and uses it for both decisions; the index (and running
  pages) get `refresh_seconds = _SERVE_LIVE_RERENDER_SECONDS if live_set
  else _SERVE_RERENDER_SECONDS`. The **initial** render before serving uses
  the same rule (today it hardcodes 60s — a run already live at serve
  startup must not hand `--open` a 60s-refresh index). The loop sleeps
  `_serve_sleep(_SERVE_LIVE_RERENDER_SECONDS)` every iteration and
  *accumulates* toward the 60s baseline (R3); each tick decides only
  *whether* to run a pass: the live set differs from the previous pass's
  (baseline: empty set, so an already-live run renders fast on the first
  tick), the live set is non-empty, or 60s have accumulated. `_serve_sleep`
  stays the injection point for tests.
- `_render_log_page` / `_render_view_directory` pass
  `refresh_seconds=_SERVE_LIVE_RERENDER_SECONDS` down to `render_html` only
  for logs with `status == "started"` **and** only in serve mode (a static
  `view` of a stale live log must not meta-refresh forever).
- **Orphaned live pages become redirects, derived from the filesystem** (R1,
  mechanism corrected R2): every pass, `out_dir.glob("*.live.html")` minus
  pages backed by a current `log_dir/*.live.json` are rewritten as
  zero-delay redirect stubs to `index.html` — an operator's auto-refreshing
  tab lands on the index (where the final page now is) instead of a dead
  "running" page frozen forever (deleting would 404 the refreshing tab,
  which is worse). Filesystem-derived, not tracked in loop memory: a serve
  *restart* after a crash must still stub the orphans it never saw vanish.
  Idempotence mechanism pinned (R3): the stub's first line is the fixed
  sentinel comment `<!-- inspect-robots live redirect stub -->`; the orphan
  pass reads only the first line and skips files that already carry it.
- **Started logs re-render every pass by design** (R3, simplifying R2's
  `<=` gate, which combined with pinning was equivalent anyway):
  `render_page = force or log.status == "started" or <existing mtime
  gate>`. The mtime gate — with ns pinning, `os.utime(page,
  ns=(st.st_atime_ns, st.st_mtime_ns))` from a single pre-read `stat`
  (float-seconds pinning reintroduces the coarse-granularity hole) —
  applies to *completed* logs only, closing the write-during-render race
  for them. The accepted-cost paragraph below covers the started-log churn
  honestly: typically one live log, and temp+replace makes the rewrite
  browser-safe.
- **Torn-read hardening** (R2; widened R3): live pages, redirect stubs,
  **and `index.html`** — the page every tab meta-refreshes and the one
  rewritten every pass — are written via temp file + `os.replace` (matching
  the sink's discipline); a 2s writer racing a 2s browser poll makes
  half-written reads plausible, and Windows readers can make in-place
  overwrites misbehave.
- **Vanishing live logs are not errors** (R2): `on_eval_end` unlinks the
  live file between the directory glob and `read_eval_log`/`stat`; a
  `FileNotFoundError` on a `*.live.json` is skipped silently (that file's
  normal end of life), never surfaced as a red "unreadable" index row.
- **Quiet re-renders** (R2): the `[i/n] rendering ...` stderr line is gated
  by the existing `quiet` flag — a 2s cadence must not scroll the serve
  terminal thirty lines a minute.
- **Empty or missing log dir tolerated under `--serve`** (R2): today an
  empty dir is `SystemExit` and the initial render is outside the loop's
  try/except — the advertised tip command would *fail on a fresh rig* if
  the operator copy-pastes it before the first live write lands (session +
  voice/ASR startup can take 10+ s). Under `--serve`, render a "no logs
  yet" index and keep ticking; the fast tick picks the live file up.
  Precisely (R3): a nonexistent LOG_DIR under `--serve` is treated as a
  logs directory and created (`_cmd_view`'s is_dir dispatch at
  cli.py:2111-2114 currently routes it to the single-file branch — that
  dispatch changes for `--serve`; a typo'd path serves an empty "no logs
  yet" index, same as an empty dir). `_render_view_directory` conditions
  its no-logs `SystemExit` on serve mode and still returns a
  `_DirectoryRenderResult` — the HTTP handler needs `out_dir`, which is
  computable with zero logs.
- Accepted cost: while a live log exists, the directory pass re-parses every
  log JSON every 2s. Mtime gating keeps re-rendering (the expensive part)
  incremental; typical log dirs make the parse pass cheap. Called out in the
  plan so it is a deliberate tradeoff, revisit if dirs grow.

### 4. CLI wiring + the bright tip — `cli.py`

- `run` (registered-task, ad-hoc) appends `LiveLogSink(args.log_dir)` to its
  sink list, **after** `JsonLogSink` (ordering invariant above).
- **`eval_set()` gains `sinks: list[LogSink] | None = None`** threaded to
  each `eval()` call (R1: today `eval_set` has no sinks parameter at all, so
  the CLI wiring was unimplementable as written). Public-API signature
  addition: docstring, docs, CHANGELOG. `_cmd_eval_set` passes
  `[JsonLogSink, LiveLogSink]`, relying on the sink's documented
  reusability across the set's sequential runs. R2 verified: `eval_set`
  passes one shared `log_dir` to every task (no per-task dirs), and
  `JsonLogSink` is already reuse-safe (fresh filename + `path` per
  `on_eval_end`) — but this changes eval-set from a fresh `JsonLogSink` per
  task to shared instances, so the `eval_set` docstring must state the
  reuse contract for caller-supplied sinks. `eval_set` catches nothing per
  task, so the CLI `finally` leak guard is load-bearing there too.
- New `--no-live-log` flag on both commands opts out (a tmpfs-averse or
  NFS-hostile rig may want zero mid-run writes).
- Both commands unlink `sink.path` in their existing `finally` blocks when
  the file still exists (the leak guard from §1).
- **The tip:** printed with the other run announcements (after
  `_build_and_announce_guardrails` — `_cmd_eval_set` has no `rerun:` line to
  anchor on; R2), when the resolved policy registry name is `agent`:

  ```
  live: watch this run in your browser →  inspect-robots view logs --serve --open
        each agent turn, notes, and operator/voice input, updating live
  ```

  First line: new `_BOLD_BRIGHT_MAGENTA = "1;95"` (one code string, matching
  the single-code `_styled` helper shape), so it is unmissable on a dark or
  light terminal. Second line dim. `logs` is the actual `args.log_dir`
  value, `shlex.quote`d (R3: the tip sells copy-paste-ability; a log dir
  with a space must not break the advertised command).
  The existing `_styled` NO_COLOR/tty handling applies unchanged. Suppressed
  when `--no-live-log` was passed (never advertise a view that will not
  update). Printed with the other pre-session announcements, so it cannot
  collide with the OperatorSession footer.
- Predicate: `resolved.policy_name == "agent"` (the resolved registry name on
  `_ResolvedComponents`, verified to be the literal `"agent"` for the plugin
  pre- and post-bind), so config-default agent runs get the tip too.

### 5. Docs, changelog, module map

- New docs page `docs/guide/live-view.md` ("Watching a run live"): the
  two-terminal flow, what appears live vs. only in the final log (scores),
  staleness note, Rerun as the high-rate alternative. Registered in
  `website/sidebars.ts` (R1: unregistered pages never appear on the site) and
  cross-linked from `docs/guide/logging-and-rerun.md`. Follows the
  public-writing style rule.
- README: one bullet in the viewing section.
- `CHANGELOG.md` entry; `src/inspect_robots/CLAUDE.md` module map rows
  (`logging/` and `cli.py`/`_html.py` rows updated).

## Testing (100% coverage, no hardware)

- **Sink unit tests:** lifecycle round-trip via `read_eval_log` after every
  hook; throttle honored/bypassed (fake clock); parallel-tuple invariants
  including the multi-epoch same-`scene_id` append path (R3);
  transcript delta accumulation and `on_trial_end` replacement; non-dict
  delta rows survive; errored-trial
  propagation; unlink on end (and `missing_ok` when already gone); write
  failure disables sink without raising; sanitize reuse (non-finite floats).
- **Integration:** `CubePick` with a `LiveLogSink` — mechanism specified
  (R2), since no `mock/` policy defines `transcript_delta` (only agent/capx
  do) and "mid-run" needs a deterministic observation point: use a small
  test policy (or wrapper) exposing `transcript_delta()`, and append a probe
  sink *after* the `LiveLogSink` whose `on_trial_end`/`log_policy_messages`
  call `read_eval_log(live_sink.path)` and stash the parsed log —
  `_Broadcast` fans out synchronously in list order, so the probe always
  observes the state right after the live sink's forced write. No
  sleep-based races. Assert the snapshot is a valid `status="started"` log
  and the final state removes the live file.
- **Renderer:** refresh meta + banner emitted only when `refresh_seconds`
  set; `running` badge; pending score cells; escaping.
- **Serve loop:** with `_serve_sleep` stubbed — fast tick re-renders when a
  live file exists; re-render fires on live-set *disappearance* (final page +
  index within one tick); 60s cadence preserved otherwise; index refresh
  follows the stateless live rule (including the initial render with a live
  file already present); orphan stubs derived from the filesystem (including
  the serve-restart case: a pre-existing `.live.html` with no backing log is
  stubbed on the first pass); stub idempotence; `FileNotFoundError` on a
  vanished live log skipped silently; `[i/n] rendering` line gated by
  `quiet`; empty/missing log dir under `--serve` serves a "no logs yet"
  index instead of exiting; temp+replace page writes; `os.utime` ns pinning
  and the `<=` gate for started logs (fake clock).
- **CLI:** tip printed exactly when the resolved policy is `agent` (and
  suppressed with `--no-live-log`); sink wired by default *after*
  `JsonLogSink` (ordering asserted); flag removes it; `finally` unlinks a
  leaked live file on a simulated grader-prompt Ctrl-C; eval-set parity via
  the new `eval_set(sinks=...)` parameter (threading + reuse across runs).

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
