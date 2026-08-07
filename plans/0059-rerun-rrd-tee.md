# Save a .rrd alongside the live Rerun view by default (tee) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **Critique status:** R1 (2026-08-07, vs main @ cf2a4dd3) found 5 substantive issues
> (docs retargeted to logging-and-rerun.md with the stale exclusivity paragraph
> rewritten; a real-SDK contract test added beside the fake-based inventory; the
> startup-failure path must actively clear `resolved_recording_path`; the CLI fake
> sink in test_registry_cli.py must grow the new kwarg/attribute) plus 7 nits, folded
> in below. R2 found 4 more (path derivation pinned after the `_ensure_rerun()`
> early return with an SDK-absent lifecycle test, else SDK-less rigs print a phantom
> path; the module docstring's threading-contract sentence joins the edit list; the
> connect branch gets its own tee kwarg assertions; the real-SDK test gains a
> functional FileSink round-trip) plus 7 nits, folded in below. R3 found 2 (the
> old-SDK fallback is a third phantom-path source and must also clear
> `resolved_recording_path`, with a test; §2's leftover causal sentence attributed
> the SDK-absent guarantee to the except handler, which never runs on that path)
> plus 6 nits, folded in below. R4 (vs main @ 6008edb1) found NO design issues; its
> one substantive finding was environmental (plan number 0058 taken on main by
> PR #338, renumbered to 0059 and rebased) and its 6 nits (reused-sink FileSink
> release documented, record-only status line named in the over-promise note,
> blueprint-replay claim softened pending a manual check, help-text quote boundary,
> `_set_sinks_warned` in the constructor list, connect-fallback arc heads-up) are
> folded in. R5 (vs main @ 6008edb1) returned NO SUBSTANTIVE ISSUES; its nits are
> post-rebase line drift only, with every target named by symbol (sink block now
> cli.py:1595-1621, spawn call sites 1617/1619, Ctrl-C 1641-1647, config show
> ~2604, `_FakeRerunSink` at test_registry_cli.py:5880-5897) — navigate by symbol.

**Goal:** Close #340. The live Rerun viewer (`inspect-robots run --rerun`) is the popular
way to watch a rollout, but a live-viewed run persists no `.rrd`: `RerunSink` treats
`recording_path`, `spawn`, and `connect_url` as mutually exclusive, and the CLI never
exposes `recording_path` at all. The exclusivity encodes a pre-0.24 SDK limitation
(`rr.save`/`rr.spawn`/`rr.connect_grpc` each *replace* the global sink). rerun-sdk 0.24
added `rr.set_sinks(GrpcSink(...), FileSink(path))`, which tees one recording stream to a
live viewer and a `.rrd` file simultaneously, and `rr.spawn(connect=False)` launches the
viewer process without grabbing the sink. After this change, every `run` that uses the
live view (or `--rerun-connect`) also writes a `.rrd` into the log dir by default; the
`.rrd` replays in `rerun <file>` what the live view showed, from the same stream. The
per-trial blueprint rides the stream and should replay too; verify that manually on
the rig once during implementation and keep the docs claim no stronger than what that
check shows (the real-SDK test round-trips data, not blueprint application).

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
     `EvalLog`. Expose the target as `self.resolved_recording_path: Path | None`:
     `None` reset at the top of every `on_eval_start`, then set (derived name in
     `recording_dir` mode, `Path(recording_path)` in fixed mode — one unified CLI
     report contract) **only after the `_ensure_rerun()` early return, inside the
     try block**. Placement is normative: on a rig without rerun-sdk,
     `on_eval_start` returns at the early return (`rerun_sink.py:632-634`) before
     any try/except runs, so deriving earlier would leave a phantom path for the
     CLI to print. `recording_dir` is created with
     `mkdir(parents=True, exist_ok=True)` at derivation time (inside the same try;
     an unwritable dir hits the disable envelope) — same guarantee `JsonLogSink`
     gives the log dir.
   - Startup matrix in `on_eval_start` (all inside the existing single
     try/except-disable envelope; a failure anywhere degrades to the warned no-op
     exactly as today, and the except handler must **also reset
     `resolved_recording_path = None`** — the path is derived before the SDK calls
     that can throw, so without the reset the CLI would report a phantom `.rrd`):
     - bare sink (no recording target, no spawn/connect): `rr.init` only —
       unchanged (`test_default_startup_never_spawns_a_viewer` still holds).
     - recording target only (no spawn/connect): `rr.init` → `rr.save(path)` —
       unchanged behavior, now also reachable via `recording_dir`.
     - live mode only (no recording target): `rr.init` → `rr.spawn(...)` /
       `rr.connect_grpc(url)` — byte-for-byte today's behavior, so a
       `--no-rerun-save` run touches no new SDK surface.
     - live mode + recording target (the tee): feature-detect the three attrs in
       ONE combined check (`all(hasattr(rr, name) for name in ("set_sinks",
       "GrpcSink", "FileSink"))` or equivalent single branch, so the 100% branch
       gate needs one missing-attr test, not three). If all three exist: `rr.init` → (when
       spawning) `rr.spawn(memory_limit=..., port=..., connect=False)` →
       `rr.set_sinks(rr.GrpcSink(url=<grpc url>), rr.FileSink(<path>))`. The gRPC
       url is `connect_url` when connecting, else
       `f"rerun+http://127.0.0.1:{spawn_port}/proxy"` (always explicit; never rely
       on `GrpcSink()`'s default, so a custom `--rerun-port` tees to the right
       viewer). If any of the three attrs is missing (SDK 0.20–0.23): warn once
       per sink instance ("this rerun-sdk predates set_sinks; the live view
       continues but the .rrd was skipped; upgrade to rerun-sdk>=0.24" — no em
       dash, repo prose habit), **clear `resolved_recording_path = None`** (this
       fallback is the third phantom-path source: the SDK is present and nothing
       raises, so neither the early-return placement nor the except-handler reset
       covers it — without this clear the CLI would report a `.rrd` that was
       never attached), and fall back to live mode only. The CLI status line
       prints "(+ .rrd)" at construction time and therefore over-promises in this
       fallback and when rerun-sdk is absent entirely, and the record-only
       status line (`rerun: recording .rrd`) over-promises the same way on an
       SDK-less rig; in all cases the sink's own warning is the correction,
       accepted as is (the post-eval path report, by contrast, never lies: all
       three phantom sources clear the path). If the fallback dispatches
       differently for spawn vs connect internally, each arc needs its own test;
       the branch gate will surface this.
   - Shutdown path is untouched: `_probe_recording_flush` already flushes the global
     recording, which flushes every teed sink; the final file finalization on process
     exit is the same mechanism the existing `recording_path` mode relies on. For a
     sink reused across programmatic evals, eval N's `FileSink` is released when
     eval N+1's `rr.init` replaces the global stream, with completeness provided by
     `on_eval_end`'s flush probe; say so in the sink docstring. Fixed
     `recording_path` + tee reused across evals truncates and rewrites the same
     file per eval, identical to today's `rr.save` semantics.
   - Module docstring: rewrite the second sentence ("can write a ``.rrd`` recording,
     spawn a local viewer, **or** connect...") to describe the tee and its 0.24 gate,
     update the constructor-comment rationale at the exclusivity check (it
     currently states the pre-0.24 global-sink replacement rule as timeless truth),
     and **extend the threading-contract sentence** at `rerun_sink.py:24-26`: the
     caller-path startup set grows `set_sinks`/`GrpcSink`/`FileSink` construction
     and the `recording_dir` mkdir, so "All Rerun SDK calls after
     ``init``/``spawn``/``connect_grpc``/``save`` happen on the worker" must list
     the new startup calls or it becomes false. Also retune the drop-report hint at
     `rerun_sink.py:574-578` to "record to a .rrd file only (no live viewer)" since
     a teed .rrd now shares the viewer-paced shedding.

2. **CLI default-on wiring** (`src/inspect_robots/cli.py`, `run` subcommand only) plus
   a per-rig config key (`src/inspect_robots/defaults.py`).

   - New flag on `p_run`: `--rerun-save` with `argparse.BooleanOptionalAction`,
     `default=None`. Help text (ends here, rationale below is NOT part of it):
     "also save the stream as a .rrd next to the eval log (default: on whenever
     the rerun viewer is active; without a viewer, --rerun-save records to a
     .rrd only; --no-rerun-save or a rerun_save config key overrides)".
     Rationale for the "without a viewer" phrasing: on a rig whose config sets
     `rerun = true` the flag tees an already-spawning viewer rather than staying
     viewer-less, so the help must not promise "records without opening a
     viewer" unconditionally.
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
     path when the rerun sink exists and `getattr(sink, "resolved_recording_path",
     None)` is truthy (keep a reference to the constructed `RerunSink` the way
     `live_sink` is kept; `getattr` with a `None` default so monkeypatched fake
     sinks without the attribute stay legal). The report never prints a phantom:
     SDK-less rigs are covered by the top-of-method reset plus derive-after-early-
     return placement (§1), startup failures by the except-handler reset, and the
     old-SDK tee fallback by its own explicit clear. The Ctrl-C path
     (`cli.py:1607-1613`) returns before the summary; a partially written `.rrd`
     is intentionally not reported there.
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

- [ ] `RerunSink.__init__`: add `recording_dir` (keyword-only), initialize
      `self.resolved_recording_path: Path | None = None` and the
      `self._set_sinks_warned = False` guard in the constructor (a
      constructed-but-never-started sink must not raise `AttributeError`), drop
      the spawn×recording and connect×recording `ValueError`s, add
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
      old-SDK fallback (needs a SEPARATE stripped fake missing the tee attrs —
      the extended `_StartupRR` has all three, so the fallback test subclasses or
      deletes them; live only + one warning, twice to prove warn-once),
      `resolved_recording_path` None when no recording target, None after the
      sink degrades on startup failure (the except-handler reset), None after
      the old-SDK tee fallback (the fallback's explicit clear), None when
      `recording_dir` is set but rerun-sdk is absent (SDK-less lifecycle test
      beside the existing `skipif(_RERUN_INSTALLED)` tests at
      tests/test_rerun_sink.py:168-184 — this is the branch main CI actually
      runs, since the dev extra has no rerun-sdk), and `Path(recording_path)` in
      fixed mode.
- [ ] Real-SDK contract tests, following the existing `_RERUN_INSTALLED`-gated
      pattern (`test_real_rerun_spawn_signature_accepts_forwarded_kwargs` and
      friends at tests/test_rerun_sink.py:1550-1584, run by the `test-rerun` CI
      job at .github/workflows/ci.yml:102-119 against the locked SDK, currently
      0.34.0). Two tiers: (a) signatures — `rr.set_sinks`/`rr.GrpcSink`/
      `rr.FileSink` exist, `GrpcSink` accepts `url=`, `FileSink` accepts a path,
      `spawn` accepts `connect=`; (b) a functional viewer-less round-trip in the
      precedent of `test_real_rerun_accepts_the_blueprint`: `rr.init` →
      `rr.set_sinks(rr.FileSink(tmp_path / "t.rrd"))` → log one scalar → flush →
      assert the file exists and is non-empty. A kwarg rename or a call-order
      rejection then fails in CI instead of on a rig.

### 2. CLI + config

- [ ] `defaults.py`: `rerun_save` field, parse, known-keys, `config set` bool
      validation; tests beside the existing `rerun` parsing tests (true/false/invalid,
      `config set rerun_save`).
- [ ] `cli.py`: `--rerun-save` flag, resolution, the three sink-construction branches,
      status lines, post-eval `.rrd` path report, `config show` row. **Extend
      `_FakeRerunSink` in `tests/test_registry_cli.py:5596-5608` first**: its
      `__init__` must accept `recording_dir` (13 tests use the `_fake_rerun`
      fixture and would `TypeError` the moment the CLI passes the new kwarg by
      default) and should carry a `resolved_recording_path` attribute for the
      report tests (the CLI's `getattr` default keeps attribute-less fakes legal).
      New CLI tests (same fake pattern), asserting the **constructor kwarg**, not
      just the printed report (the `recording_dir=... if save_wanted else None`
      ternary produces no coverage arc, so only a kwarg assertion catches a
      forgotten branch): spawn default tee passes `recording_dir=log_dir`, spawn
      + `--no-rerun-save` passes `None`, spawn with `--rerun-port` also carries
      `recording_dir=log_dir` (the port-carrying `RerunSink(spawn=True, ...)`
      call is a separate call site from the portless one, cli.py:1583/1585, and
      a forgotten kwarg there would slip past the portless tests), connect
      default tee passes `recording_dir=log_dir`, connect + `--no-rerun-save`
      passes `None`,
      explicit `--rerun-save` alone builds the record-only sink, config
      `rerun_save = false` suppresses, config-true does not create viewer-less
      sinks, path report printed/suppressed.

### 3. Docs + changelog

- [ ] `docs/guide/logging-and-rerun.md` is the home for the tee content: the live
      view now leaves a `.rrd` beside the eval log; how to replay (`rerun <file>`);
      the opt-outs; the 0.24 requirement; the drops caveat (viewer-paced shedding
      reaches the file too). **Rewrite the now-false paragraph at lines 94-96**
      ("The three modes are mutually exclusive...") and refresh the mode-example
      block at lines 87-91 — only spawn×connect and path×dir raise after this
      change. `docs/guide/live-view.md` documents the HTML browser viewer, NOT the
      Rerun viewer: it gets at most a one-line cross-reference at its existing
      rerun pointer (line 40), no tee content. `docs/guide/cli.md`: the new flag,
      plus `rerun_save` in the config example block (lines 272-273).
      `_setup.py:214-225` headless warning: mention `--rerun-save` as the headless
      way to keep a replayable `.rrd` (its "frames still record with store_frames"
      guidance stays). README: one line where the live view is pitched.
      `src/inspect_robots/CLAUDE.md` module map: refresh the `cli.py`,
      `defaults.py`, and `logging/` rows for the new flag/key/tee, citing THIS
      plan as 0059 and merging with (not replacing) main's updated `logging/`
      row text, which already cites "plans 0055 and 0058" for LiveLogSink live
      frames. Follow the public-facing writing style rules (no em dashes, no
      mid-sentence bold).
- [ ] CHANGELOG `[Unreleased]` → Added: default `.rrd` capture for live-viewed runs,
      `--rerun-save/--no-rerun-save`, `rerun_save` config key, link to plan 0059.
- [ ] Gates: `ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov` at 100%.

## Test inventory (delta)

| Area | New/changed tests |
|------|-------------------|
| sink ctor | recording×spawn and recording×connect become legal (delete 2 tests), recording_path×recording_dir error, spawn×connect still errors |
| sink startup | spawn tee order+args, custom port url, connect tee, dir-mode naming + fresh-per-eval + mkdir, old-SDK warn-once fallback (stripped fake), resolved path lifecycle incl. SDK-absent and fixed-path modes |
| sink real-SDK | `_RERUN_INSTALLED`-gated contract test: set_sinks/GrpcSink(url=)/FileSink(path)/spawn(connect=) signatures |
| defaults | rerun_save parse true/false/invalid, config set/show |
| cli | `_FakeRerunSink` signature extension, default tee, --no-rerun-save, record-only, config off, config-true-no-viewer, path report printed/suppressed |
