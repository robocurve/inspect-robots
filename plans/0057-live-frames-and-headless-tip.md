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
  `bind_frames_dir(frames_dir: str | None)` is called by `eval()`
  **unconditionally, exactly once per run**, between `bind_spaces` and
  `on_eval_start` — with `str(frame_store.root)` (i.e.
  `log_dir/frames/<run_stamp>`, byte-identical to what `eval()` stamps on the
  final log at eval.py:591, so `resolve_frames_dir`'s candidate-2
  reconstruction works) when frame storage is on, and `None` when off (R1:
  the unconditional call makes a plain overwrite safe — no pending/consume
  slot, no "no call means None" asymmetry, matching the
  `RerunSink.bind_spaces` overwrite precedent, which is likewise safe only
  because it is re-called every run). `_Broadcast` forwards it to sinks that
  define a callable `bind_frames_dir`. Documented in `logging/sink.py`'s
  module docstring next to the other two extensions (the name is claimed by
  the eval loop; deliberately off `LogSink`/`NullSink`, matching precedent).
- `LiveLogSink.bind_frames_dir` stores the value with the same
  try/`_disable` failure-isolation discipline as every other hook (R1);
  `on_eval_start`'s reset exempts this one field — safe precisely because
  the unconditional per-run re-bind fires before every `on_eval_start`, so a
  frame-less run 2 of an eval-set overwrites run 1's value with `None` and
  can never leak run-1 frames under a repeated trial prefix.
- Every snapshot's `EvalStats` then carries `frames_dir`, and the existing
  `_render_log_page` → `resolve_frames_dir` → `_FrameContext` path renders
  frames on live pages with zero renderer changes to the correlation logic.

### 2. Bounded live-page frame cost

A live page re-renders every 2s, and frame embedding encodes `.npy` files to
PNG data URLs at render time. A full 50 MB default budget re-encoded per tick
is unacceptable CPU (~740 steps × 3 cams on a real rig trial). Therefore:

- **Budget matrix, pinned** (R1): the live budget applies exactly when
  `args.serve` **and** `log.status == "started"` (directory mode) — the CPU
  rationale exists only under the 2s tick, mirroring 0055's
  `refresh_seconds` gate. A *static* directory view of a live log and the
  single-file `view foo.live.json` path both keep the full `--frames-budget`
  and no refresh. All three cells tested.
- New `view` option `--live-frames-budget`, default
  `_LIVE_FRAMES_BUDGET_MB = 8.0`, validated finite and ≥ 0 exactly like its
  sibling, and with the **same zero semantics as the sibling** (`0` =
  unlimited; R1: two budget flags with opposite zero meanings is a footgun).
  Help text warns that unlimited is not recommended while serving. Turning
  live frames fully off is `--no-frames` (which already precedes frame
  resolution and governs both page kinds).
- What the default buys, measured with the repo's encoder (R1): a 640x480
  rig frame downscales to ~222 KB base64, ~4 ms encode, so 8 MB ≈ 36 frames
  ≈ the last dozen control steps across a 3-camera rig (several agent
  turns), at ~150 ms of encode per 2s tick. A visible product choice, not an
  implementation accident.
- **Newest-first needs a two-pass design** (R1: there is no standalone
  budget pass to "flip" — `_frame_image` charges the shared `_FrameBudget`
  only after encoding, interleaved with single-pass HTML generation inside
  `_render_frame_parts`):
  1. a shared enumeration helper extracts the frame references
     (`_FRAME_LABEL_RE` + placeholder matching) from the transcript so the
     pre-pass and the render pass cannot drift;
  2. a reverse pre-pass walks references newest-to-oldest, encoding and
     charging the live budget until spent, filling a data-URL cache — each
     winning frame is encoded exactly once (the cache is what keeps the
     two-pass design from doubling the CPU the budget exists to bound);
  3. the render pass displays document order, consuming the cache; losing
     frames keep their existing textual placeholder lines (transcript text,
     already rendered today when `_frame_image` returns `None`).
  Signaling: `render_html` gains a `live_frames_budget_bytes: int | None`
  parameter — `None` (default) keeps today's document-order behavior;
  set, it activates the newest-first pre-pass. `cli.py` passes it per the
  budget matrix above; the renderer never sniffs serve-ness itself.
- **Wire-capture blobs are elided on started pages** (R1: `_render_wire_blob`
  charges the same shared budget in document order and would starve or race
  the newest-first frames on every tick). Started pages always render the
  existing `[blob elided: media budget]` placeholder for wire media; the
  full wire render belongs to the final page. Tested.
- Accepted cost, stated: a live tick re-encodes at most the budgeted ~8 MB
  of PNGs; at 2s cadence that is measurable but bounded, and it only happens
  while a live log exists. No on-disk PNG cache in this plan (revisit if
  rigs report tick overruns).

### 3. Headless-aware tip

- New helper in `cli.py`:
  `_headless_session(env: Mapping[str, str], platform: str = sys.platform)
  -> bool` — True when `SSH_CONNECTION` is set, else on Linux when neither
  `DISPLAY` nor `WAYLAND_DISPLAY` is set, else False. SSH wins over a
  forwarded `DISPLAY` (X-forwarded browsing is never the intent). Pure in
  its two arguments (R1: platform injected, not monkeypatched), unit-tested
  with plain dicts. WSL note recorded: WSLg sets `DISPLAY` → non-headless →
  `--open`, the right outcome; pre-WSLg WSL lands headless, where serve's
  own printed `localhost` URL still rescues the operator.
- `_announce_live_view` picks the variant:
  - headless: command line advertises
    `inspect-robots view <logdir> --serve --host 0.0.0.0`, and the dim
    second line appends the URL to open. Host preference (R1:
    `gethostname()` short names often don't resolve from the operator's
    laptop): the server IP from `SSH_CONNECTION`'s third field when present
    (bracketed if IPv6) — it is the address the operator demonstrably
    reached this machine by — falling back to `socket.gethostname()`
    otherwise (which keeps the tip consistent with `_serve_display_urls`).
  - local display: the current `--serve --open` text, unchanged.
- `_setup.py`'s existing headless check (rerun-viewer warning,
  `_setup.py:1515`) is deliberately NOT unified: it answers "can a viewer
  window open here" (DISPLAY-only, X-forwarding counts), while the tip
  answers "is the human somewhere else" (SSH wins). One sentence comment in
  the helper pointing at the distinction so nobody "deduplicates" them later.

## Testing (100% coverage, no hardware)

- `_Broadcast.bind_frames_dir` forwarding (defined/undefined sinks), call
  ordering relative to `bind_spaces`/`on_eval_start`, unconditional call
  with `None` when frame storage is off.
- `LiveLogSink`: bound value survives `on_eval_start`'s reset; run 2 of an
  eval-set with storage off overwrites to `None` (no cross-run frame leak);
  `frames_dir` present in every snapshot of a bound run and absent
  otherwise (probe-sink pattern from plan 0055); hook failure disables the
  sink, never raises.
- Renderer/CLI: live pages embed frames under the live budget with
  newest-first allocation and single-encode cache (synthetic frames on disk
  via `FrameStore`); budget-matrix cells — serve+started = live budget,
  static dir view of a started log = full budget, single-file view of a
  live file = full budget; `--live-frames-budget` validation (negative,
  non-finite rejected; 0 = unlimited like the sibling); `--no-frames` beats
  the live budget; wire blobs elided on started pages; a live snapshot with
  a *relative* stored `frames_dir` resolves via `resolve_frames_dir`'s
  second candidate (renderer running from a different CWD); the 2s serve
  tick still renders (no regression to cadence tests).
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
