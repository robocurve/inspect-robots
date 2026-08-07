# Live frames in running snapshots + headless-aware live-view tip

> Status: PLANNED — issue #337 items 1 and 7 (rig-1 shakedown, 2026-08-07).
> Follow-up to plan 0055. The report-layout batch (items 2-6, 8) is a separate
> plan/PR that builds on this one.

## Problem

1. Live pages render transcript-only. Camera frames stream to disk during the
   rollout (`FrameStore`, every control step), but `stats.frames_dir` is only
   stamped onto the final log by `eval()` (eval.py:591). `LiveLogSink` never
   learns the frames directory, so its snapshots carry `frames_dir=None` and
   `_render_log_page` skips frame resolution entirely.
2. The bright `policy=agent` tip advertises `view LOGS --serve --open`, which
   on a rig (SSH, headless) prints "could not open browser" and leaves the
   operator to discover `--host 0.0.0.0` themselves — observed verbatim on
   rig-1.

## Design

### 1. `bind_frames_dir` — frames plumbing to interested sinks

- New duck-typed sink extension, same pattern and lifecycle as `bind_spaces`:
  `bind_frames_dir(frames_dir: str)` is called by `eval()` at most once per
  run, between `bind_spaces` and `on_eval_start`, and only when frame storage
  is enabled (`frame_store` was constructed). `_Broadcast` forwards it to
  sinks that define a callable `bind_frames_dir`, like it does `bind_spaces`
  and `log_policy_messages`. Documented in `logging/sink.py`'s module
  docstring next to the other two extensions (the name is claimed by the eval
  loop; deliberately off `LogSink`/`NullSink`, matching the precedent).
- `LiveLogSink` stores the bound value as pending next-run state;
  `on_eval_start` consumes it into the current run and clears the pending
  slot (each `eval()` call re-binds, so eval-set reuse stays correct; a run
  without frame storage gets no call and the snapshot carries
  `frames_dir=None`).
- Every snapshot's `EvalStats` then carries `frames_dir`, and the existing
  `_render_log_page` → `resolve_frames_dir` → `_FrameContext` path renders
  frames on live pages with zero renderer changes to the correlation logic.

### 2. Bounded live-page frame cost

A live page re-renders every 2s, and frame embedding encodes `.npy` files to
PNG data URLs at render time. A full 50 MB default budget re-encoded per tick
is unacceptable CPU (~740 steps × 3 cams on a real rig trial). Therefore:

- `_render_view_directory` passes a dedicated live budget for logs with
  `status == "started"`: module constant `_LIVE_FRAMES_BUDGET_MB = 5.0`,
  overridable with a new `--live-frames-budget` option on `view` (0 disables
  live frames entirely; the existing `--frames-budget` continues to govern
  completed pages).
- **Selection order flips to newest-first for live pages.** The budget
  consumer currently walks the transcript in document order and stops when
  the budget is spent — on a growing log that keeps the run's *oldest* frames
  and drops the newest, exactly backwards for an operator watching now. The
  budget pass for `started` logs must allocate from the newest transcript
  turn backwards (rendering still displays them in document order; turns
  whose frames lost the allocation render their existing textual
  placeholder).
- Accepted cost, stated: a live tick re-encodes at most the budgeted ~5 MB of
  PNGs; at 2s cadence that is measurable but bounded, and it only happens
  while a live log exists. No on-disk PNG cache in this plan (revisit if rigs
  report tick overruns).

### 3. Headless-aware tip

- New helper in `cli.py`: `_headless_session(env: Mapping[str, str]) -> bool`
  — True when `SSH_CONNECTION` is set, else on Linux when neither `DISPLAY`
  nor `WAYLAND_DISPLAY` is set, else False. SSH wins over a forwarded
  `DISPLAY` (X-forwarded browsing is never the intent). Pure function of the
  passed env, unit-tested with plain dicts.
- `_announce_live_view` picks the variant:
  - headless: command line advertises
    `inspect-robots view <logdir> --serve --host 0.0.0.0`, and the dim second
    line appends the URL it will produce,
    `then open http://<socket.gethostname()>:8300/`, keeping the copy-paste
    affordance for the laptop browser.
  - local display: the current `--serve --open` text, unchanged.
- `_setup.py`'s existing headless check (rerun-viewer warning,
  `_setup.py:1515`) is deliberately NOT unified: it answers "can a viewer
  window open here" (DISPLAY-only, X-forwarding counts), while the tip
  answers "is the human somewhere else" (SSH wins). One sentence comment in
  the helper pointing at the distinction so nobody "deduplicates" them later.

## Testing (100% coverage, no hardware)

- `_Broadcast.bind_frames_dir` forwarding (defined/undefined sinks), call
  ordering relative to `bind_spaces`/`on_eval_start`, and no-call when frame
  storage is off.
- `LiveLogSink`: pending-bind consumed by `on_eval_start`, cleared after,
  `frames_dir` present in every snapshot of a bound run and absent otherwise;
  eval-set reuse re-binds per run (probe-sink pattern from plan 0055).
- Renderer/CLI: live pages embed frames under the live budget with
  newest-first allocation (synthetic frames on disk via `FrameStore`);
  `--live-frames-budget 0` disables; completed pages unaffected by the live
  budget; the 2s serve tick still renders (no regression to cadence tests).
- `_headless_session`: parametrized env/platform matrix (SSH set, DISPLAY
  set, both, neither, darwin/win32 short-circuit via monkeypatched
  `sys.platform`).
- Tip: both variants asserted (env-injected), `shlex.quote` preserved,
  suppression rules from plan 0055 unchanged.
- Docs: `docs/guide/live-view.md` frames paragraph updated (live pages show
  recent frames, final report the full set), CHANGELOG entry
  (`[plan 0057](plans/0057-live-frames-and-headless-tip.md)`,
  [#337](https://github.com/robocurve/inspect-robots/issues/337)), module-map
  rows (`logging/`, `eval.py`, `cli.py`).

## Implementation order

1. Sink extension + `_Broadcast` + `LiveLogSink` binding + tests.
2. Live frame budget + newest-first allocation + tests.
3. `_headless_session` + tip variants + tests.
4. Docs, CHANGELOG, module map.

## Non-goals

- No change to the final report's frame behavior or budget.
- No PNG cache; no MP4/flipbook (plan for items 2-6, 8 covers the video).
- No change to `_setup.py`'s rerun headless warning.
