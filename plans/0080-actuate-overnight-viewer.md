# 0080: Overnight dual-rig campaign viewer

Demo-branch only. The operator sleeps through most of the dual-rig
campaign (plan 0079); this adds a single phone-friendly page showing both
rigs' progress, and pins headless full-data recording so nothing depends
on live viewers.

## Recording pins (rig scripts)

`run_trials_rig1.py` / `run_trials_rig2.py` extend their baked args
(before the user passthrough, so later user flags still win) from
`--max-steps 1200` to:

```
--max-steps 1200 --store-frames --no-rerun --rerun-save
```

- `--no-rerun`: overrides rig-1's `rerun = true` config so no viewer
  windows spawn overnight.
- `--rerun-save`: without a viewer this records the rollout (cameras,
  joint states, actions) to a per-eval `.rrd` in the rig's log dir
  (verified against the CLI help; the morning's logs already carry such
  rrds).
- `--store-frames`: drift-proofs the configs' `store_frames = true`, so
  per-step camera frames land under `logs-rigN/frames/<run_stamp>/`.

Existing storage that needs no change: per-eval EvalLog json, actions
logs (commanded joints per step), and LLM transcripts, all under the
per-rig log dir.

## New: `examples/actuate/overnight.py`

Stdlib HTTP server, `serve.py`'s patterns (ThreadingHTTPServer, no-store
cache headers, silent request log), default `--port 8380`. Booth-editable
constants at top:

```python
TRIALS_PER_RIG = 5     # must match the rig scripts
STALL_AFTER_MIN = 30   # incomplete + no activity this long = stalled
RIGS = {"rig1": ("state-rig1", "logs-rig1"), "rig2": ("state-rig2", "logs-rig2")}
```

Routes:

- `/` serves `overnight.html`.
- `/api/overnight` returns one JSON payload:
  - `generated_at` (iso, server time).
  - per rig (`rig1`, `rig2`): `status` (the rig's status.json passthrough:
    eval_index, started_at, roles, models, cooling), `status_age_s`,
    `results`: per-model `{n, mean}` over scored rows (same non-null
    finite rule as everywhere, roster-seeded), `recent`: last 3 rows
    (eval_index, test_taker, score, task, ts, clips), `last_activity_s`
    (seconds since the newer of status.json mtime and last results row
    ts), `done` (every model at TRIALS_PER_RIG), `stalled` (not done and
    `last_activity_s` > STALL_AFTER_MIN minutes; missing dirs count as
    not-started, never stalled).
  - `pooled`: per-model `{n, mean}` across both rigs plus
    `total_scored` and `target` (16 models-halves x TRIALS_PER_RIG... no:
    `2 * TRIALS_PER_RIG * len(ROSTER-test-takers)` = 80).
  Missing files/dirs degrade to empty sections, never 500s.
- `/media/rig1/<name>.mp4` and `/media/rig2/<name>.mp4` serve each rig's
  `state-rigN/media` clips with the same name-allowlist regex as
  serve.py.

Eligibility comes from `_roster` (`ROSTER`, `ACCENTS`, roles filtering as
in `run._eligible` — import run and reuse `_eligible` to avoid a second
roles interpretation).

## New: `examples/actuate/overnight.html`

Dark, mobile-first single column (the page will be read on a phone over
the tailnet), reusing monitor.html's palette (#0e1116 bg, #151a22 cards,
muted #687280, model accent colors from the JS map, system-ui stack).
Auto-refresh: fetch `/api/overnight` every 30 s; a fetch failure shows a
"server unreachable" banner rather than stale data without warning.
Layout, top to bottom:

1. Header: "Actuate Overnight" + pooled progress line `NN/80 scored` with
   a thin progress bar + "updated HH:MM:SS" (client-rendered from
   generated_at).
2. Per-rig card (rig-1 then rig-2), each with:
   - Rig title row with a state chip: `RUNNING` (green tint) /
     `COOLING x.x C` (blue) / `STALLED Nm` (red) / `DONE` (muted) /
     `NOT STARTED` (muted). Stalled shows minutes since last activity.
   - Current-eval line: `Eval #N - <test-taker> - trial x/5 - running
     M min` (from status + that model's scored count; hide when done).
   - Progress grid: one row per test-taker in roster order: accent-tinted
     name, TRIALS_PER_RIG dot cells (filled = scored), and the rig-local
     mean (or a dash at n=0).
   - Recent strip: the last 3 scored-or-not evals as small cards: eval
     number, model, score badge (or "unscored"), one-line task, and the
     top-cam clip when present (`autoplay muted loop playsinline`, small,
     capped height) via `/media/rigN/...`.
3. Pooled leaderboard table: model, n (of 10), mean, sorted by mean desc
   with n=0 last; accent-colored model names.

No external assets; all CSS inline in the file like monitor.html.

## README

Extend the dual-rig section: start the overnight page alongside the
campaigns (`tmux new -d -s actuate-overnight 'python
examples/actuate/overnight.py'`), reachable from the tailnet at
`http://<rig-host>:8380/`; the rig scripts record everything for morning
review (rrd with joint states, frames, actions, transcripts) with no
live viewers; the monitor pages (8377/8378) stay optional for attended
watching.

## Out of scope

- Serving or rendering rrd/frames content in the page (morning review
  uses `inspect-robots view logs-rigN/ --serve` and the rerun viewer on
  the recorded rrds).
- Push notifications and auth (tailnet-only page).
- Any change to run.py, start.sh, serve.py, monitor.html, _roster.py,
  _thermal.py, run_trials.py.

## Validation

- ruff gates; a scripted check that `/api/overnight` renders correct
  progress/stall/done states from synthetic state-rigN fixtures
  (including missing dirs), and that the media route rejects traversal
  names, plus a live smoke on a spare port.
- Rig scripts: assert the pinned argv now carries the three recording
  flags before user args.
