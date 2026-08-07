# Save a .rrd alongside the live Rerun view by default (tee) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **Critique status:** R0 — not yet critiqued.

**Goal:** Close #340. The live Rerun viewer (`inspect-robots run --rerun`) is the popular
way to watch a rollout, but a live-viewed run persists no `.rrd`: `RerunSink` treats
`recording_path`, `spawn`, and `connect_url` as mutually exclusive, and the CLI never
exposes `recording_path` at all. The exclusivity encodes a pre-0.24 SDK limitation
(`rr.save`/`rr.spawn`/`rr.connect_grpc` each *replace* the global sink). rerun-sdk 0.24
added `rr.set_sinks(GrpcSink(...), FileSink(path))`, which tees one recording stream to a
live viewer and a `.rrd` file simultaneously, and `rr.spawn(connect=False)` launches the
viewer process without grabbing the sink. After this change, every `run` that uses the
live view (or `--rerun-connect`) also writes a `.rrd` into the log dir by default; the
`.rrd` replays in `rerun <file>` exactly what the live view showed, per-trial blueprint
layout included, because it is literally the same stream.

Out of scope: `eval-set` (it has no rerun flags today and grows none here), raising the
`rerun-sdk>=0.20` floor (tee is feature-detected, matching the existing
`set_time`/`Scalars`/`Image.compress` shims), and any change to what is logged. The
`EvalLog` + `FrameStore` remain the source of truth; the sink's drop-shedding (which
protects the control-rate loop) applies to both branches of the tee, so the `.rrd` is
"what the viewer saw", not a guaranteed-complete record — `docs/guide/logging-and-rerun.md`
says so explicitly.

**Architecture:** two seams, both already injectable in tests via the `sink._rr = fake`
pattern in `tests/test_rerun_sink.py`.

1. **`RerunSink` grows a tee** (`src/inspect_robots/logging/rerun_sink.py`).

   - New constructor param `recording_dir: str | None = None` (keyword-only, like the
     other options). Mutually exclusive with `recording_path` (one fixed file vs a
     directory of per-eval files), but *combinable* with `spawn` or `connect_url`.
     `recording_path` also becomes combinable with `spawn`/`connect_url`;
     `spawn`+`connect_url` stays an error. The two removed `ValueError`s and their tests
     go away.
   - Filename derivation: at `on_eval_start`, when `recording_dir` is set, derive
     `{_slug(spec.task)}_{uuid4().hex[:8]}.rrd` inside it — the same naming convention
     `JsonLogSink` uses for the final log (`logging/json_log.py:86`). Import `_slug`
     from `json_log` (same package; do not duplicate the regex). The stems cannot match
     the JSON log exactly because `JsonLogSink` draws its own uuid at `on_eval_end`;
     matching the *convention* is the contract, exact correlation stays with the
     `EvalLog`. Expose the derived path as `self.resolved_recording_path: Path | None`
     (reset to `None` at every `on_eval_start` before deriving) so the CLI can report
     it. `recording_dir` is created with `mkdir(parents=True, exist_ok=True)` before
     use — same guarantee `JsonLogSink` gives the log dir.
   - Startup matrix in `on_eval_start` (all inside the existing single
     try/except-disable envelope; a failure anywhere degrades to the warned no-op
     exactly as today):
     - recording target only (no spawn/connect): `rr.init` → `rr.save(path)` —
       unchanged behavior, now also reachable via `recording_dir`.
     - live mode only (no recording target): `rr.init` → `rr.spawn(...)` /
       `rr.connect_grpc(url)` — byte-for-byte today's behavior, so a
       `--no-rerun-save` run touches no new SDK surface.
     - live mode + recording target (the tee): feature-detect
       `getattr(rr, "set_sinks", None)`, `getattr(rr, "GrpcSink", None)`,
       `getattr(rr, "FileSink", None)`. If all three exist: `rr.init` → (when
       spawning) `rr.spawn(memory_limit=..., port=..., connect=False)` →
       `rr.set_sinks(rr.GrpcSink(url=<grpc url>), rr.FileSink(<path>))`. The gRPC
       url is `connect_url` when connecting, else
       `f"rerun+http://127.0.0.1:{spawn_port}/proxy"` (always explicit; never rely
       on `GrpcSink()`'s default, so a custom `--rerun-port` tees to the right
       viewer). If any of the three attrs is missing (SDK 0.20–0.23): warn once
       ("this rerun-sdk predates set_sinks; the live view continues but the .rrd
       was skipped — upgrade to rerun-sdk>=0.24") and fall back to live mode only.
       `spawn(connect=False)` is only ever called on the set_sinks path, so old
       SDKs never see the kwarg.
   - Shutdown path is untouched: `_probe_recording_flush` already flushes the global
     recording, which flushes every teed sink; the final file finalization on process
     exit is the same mechanism the existing `recording_path` mode relies on.
   - Module docstring: rewrite the second sentence ("can write a ``.rrd`` recording,
     spawn a local viewer, **or** connect...") to describe the tee and its 0.24 gate,
     and update the constructor-comment rationale at the exclusivity check (it
     currently states the pre-0.24 global-sink replacement rule as timeless truth).

2. **CLI default-on wiring** (`src/inspect_robots/cli.py`, `run` subcommand only) plus
   a per-rig config key (`src/inspect_robots/defaults.py`).

   - New flag on `p_run`: `--rerun-save` with `argparse.BooleanOptionalAction`,
     `default=None`, help: also save the stream as a `.rrd` next to the eval log
     (default: on whenever the rerun viewer is active; `--no-rerun-save` or a
     `rerun_save = false` config line disables; `--rerun-save` alone records without
     opening a viewer).
   - Resolution: `save_wanted = args.rerun_save if args.rerun_save is not None else
     defaults.rerun_save` (config default `True`).
   - Sink construction (the `cli.py:1567-1586` block):
     - connect branch: `RerunSink(connect_url=..., recording_dir=args.log_dir if
       save_wanted else None)`; the status line becomes
       `rerun: connect <url> (+ .rrd)` when teeing.
     - spawn branch (`spawn_wanted` unchanged): pass
       `recording_dir=args.log_dir if save_wanted else None`; status line
       `rerun: live viewer (+ .rrd)` when teeing.
     - new record-only branch: when neither connect nor spawn is wanted but
       `args.rerun_save is True` (explicit flag, not config default — a bare
       `rerun_save = true` config line must not add sinks to viewer-less runs),
       append `RerunSink(recording_dir=args.log_dir)` and print
       `rerun: recording .rrd`.
   - After `eval()` returns, where the log path is reported, also print the `.rrd`
     path if the rerun sink exists and `resolved_recording_path` is set (keep a
     reference to the constructed `RerunSink` the way `live_sink` is kept). No print
     when the sink degraded to a no-op (path stays `None`), so a rig without
     rerun-sdk sees no phantom path.
   - Flag validation: `--no-rerun-save` needs no cross-checks (it is a pure opt-out;
     harmless without a viewer). `--rerun-save --no-rerun` is legal and means
     record-only. No new mutual-exclusion errors.
   - `defaults.py`: add `rerun_save: bool = True` to the dataclass; parse it with the
     same must-be-bool validation as `rerun` (error text
     `[defaults] rerun_save must be true or false, got {raw!r}`); add `"rerun_save"`
     to the known-keys tuple and to the bool-validated set in `_set_default`
     (`defaults.py:245`); add a `("rerun_save", defaults.rerun_save, None)` row to
     `config show` (`cli.py:2562` region).

**Failure-mode note (accepted tradeoff):** the tee lives inside the sink's existing
startup try/except, so an unwritable `recording_dir` disables the *whole* sink including
the live view, with the existing "RerunSink disabled" warning. The dir is `args.log_dir`,
which `JsonLogSink`/`LiveLogSink` already write to, so in the CLI a broken log dir fails
the run before the sink matters. Splitting the envelope to save the live view on
file-only failures is complexity the failure likelihood does not buy.

## Steps

### 1. Sink: constructor + naming

- [ ] `RerunSink.__init__`: add `recording_dir` (keyword-only), drop the
      spawn×recording and connect×recording `ValueError`s, add
      `recording_path`×`recording_dir` exclusivity, keep spawn×connect. Update the
      class and `__init__` docstrings (tee semantics, 0.24 gate, `recording_dir`
      naming convention, `resolved_recording_path` contract).
- [ ] `on_eval_start`: reset + derive `resolved_recording_path` (recording_dir mode
      creates the directory, draws a fresh uuid per eval — a reused sink gets a new
      file per eval); implement the startup matrix above with the set_sinks feature
      gate and the warn-once fallback (`self._set_sinks_warned` guard, reset never —
      one warning per sink instance, matching `_warned`).
- [ ] Tests (`tests/test_rerun_sink.py`): update the two now-deleted exclusivity
      tests; add recording_dir×recording_path exclusivity; extend `_StartupRR` with
      `set_sinks`/`GrpcSink`/`FileSink` capture (record constructor args) and cover:
      spawn tee call order (`init` → `spawn(connect=False, memory_limit, port)` →
      `set_sinks(GrpcSink(url=default-port url), FileSink(derived path))`), custom
      `spawn_port` url, connect tee (`GrpcSink(url=connect_url)`, no `connect_grpc`
      call), recording_dir-only (`save` with derived name matching
      `^{slug}_[0-9a-f]{8}\.rrd$`), fresh name on second `on_eval_start`,
      old-SDK fallback (no `set_sinks` attr → live only + one warning, twice
      to prove warn-once), `resolved_recording_path` None when no recording target
      and None after the sink degrades on startup failure.

### 2. CLI + config

- [ ] `defaults.py`: `rerun_save` field, parse, known-keys, `config set` bool
      validation; tests beside the existing `rerun` parsing tests (true/false/invalid,
      `config set rerun_save`).
- [ ] `cli.py`: `--rerun-save` flag, resolution, the three sink-construction branches,
      status lines, post-eval `.rrd` path report, `config show` row. Tests in the CLI
      suite (fake sink via monkeypatched `RerunSink` or the sink's `_rr` fake, per the
      existing CLI rerun tests' pattern): default-on tee when `--rerun` active,
      `--no-rerun-save` suppresses, explicit `--rerun-save` alone builds the
      record-only sink, config `rerun_save = false` suppresses, config-true does not
      create viewer-less sinks, path report printed/suppressed.

### 3. Docs + changelog

- [ ] `docs/guide/live-view.md` and `docs/guide/logging-and-rerun.md`: the live view
      now leaves a `.rrd` beside the eval log; how to replay (`rerun <file>`); the
      opt-outs; the 0.24 requirement for the tee; the drops caveat (viewer-paced
      shedding reaches the file too). `docs/guide/cli.md`: the new flag. README: one
      line where the live view is pitched. Follow the public-facing writing style
      rules (no em dashes, no mid-sentence bold).
- [ ] CHANGELOG `[Unreleased]` → Added: default `.rrd` capture for live-viewed runs,
      `--rerun-save/--no-rerun-save`, `rerun_save` config key, plan link.
- [ ] Gates: `ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov` at 100%.

## Test inventory (delta)

| Area | New/changed tests |
|------|-------------------|
| sink ctor | recording×spawn and recording×connect become legal (delete 2 tests), recording_path×recording_dir error, spawn×connect still errors |
| sink startup | spawn tee order+args, custom port url, connect tee, dir-mode naming + fresh-per-eval + mkdir, old-SDK warn-once fallback, resolved path lifecycle |
| defaults | rerun_save parse true/false/invalid, config set/show |
| cli | default tee, --no-rerun-save, record-only, config off, config-true-no-viewer, path report |
