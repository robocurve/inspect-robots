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
- `LiveLogSink.bind_frames_dir` is a **bare assignment**
  (`self._frames_dir = frames_dir` as its only statement) — no
  try/`_disable` wrapper (R2: a store cannot raise, and the wrapper would be
  actively harmful: `on_eval_start` begins with `self._disabled = False`,
  so a disable from a failed bind would be undone milliseconds later while
  the reset-exemption preserved the stale value — resurrecting the exact
  cross-run leak this design eliminates. The bare-assignment shape also
  matches the real `RerunSink.bind_spaces` precedent, which has no
  isolation either). `on_eval_start`'s reset exempts this one field — safe
  precisely because the unconditional per-run re-bind fires before every
  `on_eval_start`, so a frame-less run 2 of an eval-set overwrites run 1's
  value with `None` and can never leak run-1 frames under a repeated trial
  prefix.
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
     (`_FRAME_LABEL_RE` + the *stateful* label→placeholder pending match,
     the `_is_chat_transcript`/user-role/list-content gating, and the
     per-trial `_safe(f"{scene_id}-e{trial}")` prefix); **step 1 refactors
     `_render_frame_parts` to consume this helper's matches** (R2: if only
     the pre-pass uses it, the two passes drift and the budget charges for
     frames the render pass never shows);
  2. a reverse pre-pass walks references newest-to-oldest — across scenes
     and trials in reverse document order, the log's tail being the newest
     turn — encoding and charging the live budget until spent, filling a
     data-URL cache keyed `(trial_prefix, camera, step)`. The cache hangs
     off the shared `_FrameBudget` (R2: `_FrameContext` is a fresh frozen
     copy per trial; the budget object is the only state every trial
     shares). Each winning frame is encoded exactly once.
  3. the render pass displays document order, consuming the cache. In live
     mode `_frame_image` consults the cache **before** the
     `budget.truncated` short-circuit and does not re-charge for hits (R2:
     the pre-pass legitimately ends with `truncated` set — without the
     cache-first check every winner would render as a placeholder and the
     page would show zero frames); a cache miss in live mode returns `None`
     without loading or encoding (the frame lost the allocation). Losing
     frames keep their existing textual placeholder lines.
  Signaling: `render_html` gains a `live_frames_budget_bytes: int | None`
  parameter — `None` (default) keeps today's document-order behavior; set,
  it activates the newest-first pre-pass. The effective live limit is
  `min(live_frames_budget_bytes, frames_budget_bytes)` treating 0 as
  unlimited on both sides (R2: the live default must not exceed an explicit
  tighter `--frames-budget`), and the truncation chip prints the
  *effective* budget, not the sibling's (R2: today it formats
  `frames_budget_bytes`, which would read "truncated at 50 MB" on an 8 MB
  live page). `cli.py` passes it per the budget matrix above; the renderer
  never sniffs serve-ness itself.
- **Wire-capture blobs are elided on started pages, keyed on
  `log.status == "started"`** — a new parameter alongside the existing
  `scores_pending` derivation, NOT on `live_frames_budget_bytes` (R2: the
  renderer never sniffs serve-ness, so status is the only coherent key;
  consequence accepted and stated: static and single-file views of a live
  log also elide wire media — a running snapshot's wire section is partial
  anyway, and the full render belongs to the final page). (R1 rationale:
  `_render_wire_blob` charges the same shared budget in document order and
  would starve or race the newest-first frames on every tick.) Tested for
  the serve cell and the static cell.
- Accepted costs, stated (R2: not just CPU): per 2s tick while a live log
  exists — ~150 ms of PNG encode; an ~8 MB atomic page rewrite (~14 GB/h of
  disk writes over a long attended run, on a dir that already holds the
  frames themselves); and an ~8 MB browser re-download per meta-refresh
  (~32 Mbps sustained toward the operator — fine on LAN/tailnet, the
  `--live-frames-budget` knob is the relief valve for thin WAN links). No
  on-disk PNG cache in this plan (revisit if rigs report tick overruns).

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
    and well-formed (four whitespace-separated fields; bracketed if IPv6) —
    it is the address the operator demonstrably reached this machine by —
    falling back to `socket.gethostname()` otherwise (which keeps the tip
    consistent with `_serve_display_urls`). Known caveat, accepted (R2):
    under ProxyJump the third field is the jump-host-facing address; the
    fallback and serve's own printed URLs cover that operator.
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
  otherwise (probe-sink pattern from plan 0055).
- Renderer/CLI: live pages embed frames under the live budget with
  newest-first allocation and single-encode cache (synthetic frames on disk
  via `FrameStore`); budget-matrix cells — serve+started = live budget,
  static dir view of a started log = full budget, single-file view of a
  live file = full budget; `--live-frames-budget` validation (negative,
  non-finite rejected; 0 = unlimited like the sibling); `--no-frames` beats
  the live budget; the effective-min rule (`--frames-budget 2
  --live-frames-budget 8` caps live pages at 2 MB) and the truncation chip
  printing the effective budget; wire blobs elided on started pages (serve
  cell and static cell); a live snapshot with
  a *relative* stored `frames_dir` resolves via `resolve_frames_dir`'s
  second candidate (renderer running from a different CWD); the 2s serve
  tick still renders (no regression to cadence tests).
- `_headless_session`: parametrized env/platform matrix (SSH set, DISPLAY
  set, both, neither; darwin/win32 short-circuit exercised by passing the
  `platform` argument — no monkeypatching; R2).
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
