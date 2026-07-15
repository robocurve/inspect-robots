# 0017 — Interrupted runs write a cancelled eval log

Issue: #116. Status: draft.

## Problem

`KeyboardInterrupt` is a `BaseException`, and both `rollout()` and `eval()`
guard only `Exception`-derived failures. Ctrl-C therefore propagates straight
out of `eval()` before `bus.on_eval_end(log)` runs: no `EvalLog` is written,
so the policy transcript, timing stats, step records, and the `stats.frames_dir`
pointer are all lost. The FrameStore side-car has already persisted every frame
by that point, but nothing can join to them (`inspect-robots video` takes a log
path). Observed on a real robot run 2026-07-15: a ~1,000-step agent-policy
episode was stopped mid-run and left 3,075 orphaned `.npy` files and no record.

Note `rollout()`'s `finally` block already collects `inference_latencies`
while any exception unwinds, and `policy_transcript` too *provided
`policy.reset` completed* (the `policy_reset_ok` gate at rollout.py:307, whose
`# pragma: no branch` stays valid after this change — do not "fix" it). The
data survives to the `TrialRecord`; only the record itself is dropped, because
`KeyboardInterrupt` carries no `.record` and `eval()` never catches it.

## Design

### New exception: `_CancelledTrial(KeyboardInterrupt)`

In `src/inspect_robots/errors.py`, private to the framework. The leading
underscore is load-bearing: `docs/api/index.md` renders
`::: inspect_robots.errors` and mkdocs filters only `"!^_"`, so a public name
would auto-appear in the published API docs. `_CancelledTrial` stays out of
the docs, out of `inspect_robots.__all__`, and is exempt from ruff D101.

```python
class _CancelledTrial(KeyboardInterrupt):
    """Ctrl-C during a trial, carrying the partial record (mirrors exc.record)."""
    def __init__(self, message: str, record: TrialRecord) -> None: ...
```

Subclassing `KeyboardInterrupt` keeps outer `except KeyboardInterrupt`
handlers (user scripts, pytest, asyncio) working unchanged, while giving
`eval()` a typed carrier for the partial record — the same shape as the
existing `InspectRobotsError.record` pattern, without `setattr` on a stdlib
exception instance (which mypy strict rejects).

The `TYPE_CHECKING` import of `TrialRecord` already exists in errors.py
(lines 21-22, used by `InspectRobotsError.record`) — do not add a duplicate.
`__init__` MUST call `super().__init__(message)` so `str(exc)` carries the
message — `eval()`'s `error = str(exc)` depends on it, and a test pins the
resulting `log.error` text.

### `rollout()`: catch, mark, re-raise typed

Add one handler to the existing outer `try` (the one whose `finally` collects
the transcript):

```python
except KeyboardInterrupt as exc:
    record.status = "cancelled"
    record.error = "cancelled by user (KeyboardInterrupt)"
    record.events.append(error_event(t, "KeyboardInterrupt", "cancelled by user"))
    raise _CancelledTrial(record.error, record) from exc
```

- `t` must be defined on every path: ADD `t = -1` before `policy.reset(scene)`
  (matching the `-1` convention `_record_failure` already uses for pre-loop
  failures). The existing `t = 0` before the `while` (rollout.py:215) STAYS —
  removing it in favor of the `-1` initializer would shift step indices and
  run `max_steps + 1` iterations.
- The `finally` block runs after this handler, so `policy_transcript` and
  `inference_latencies` land on the record exactly as they do today for
  typed errors.
- `TrialRecord.status` docstring/comment gains `"cancelled"` as a value.

### `eval()`: preserve, write, re-raise

At the rollout call site, alongside the existing handlers:

```python
except _CancelledTrial as exc:
    status = "cancelled"
    error = str(exc)
    scene_status = "cancelled"
    scene_error = error
    halted = True
    cancelled_exc = exc
    record = exc.record
```

- `error = str(exc)` — no `exc.args` conditional or fallback: rollout always
  passes a non-empty message, so a fallback arm would be dead code guarding a
  value that cannot occur (simplicity, not the coverage gate, is the reason —
  short-circuit forms would not register as branch arcs anyway).
- `cancelled_exc: _CancelledTrial | None = None` is initialized with the
  other loop state.
- The existing record-preservation block runs with the scoring gate widened:
  `if record.status == "error":` becomes `if record.status != "success":`,
  BUT `errored_trials += 1` moves under a nested `if record.status ==
  "error":` so a cancelled trial is preserved (termination reason,
  transcript, `on_trial_end`) and never scored, yet does NOT count as
  errored — otherwise `EvalResults.errored_trials` and the `inspect` view's
  "trials: N (1 errored)" line would mislabel a cancellation as an error in
  the very forensic view this feature exists to enable. A test pins
  `errored_trials == 0` for a cancelled-only run.
- `halted = True` reuses the existing halt path to break both loops and skip
  remaining scenes/epochs.
- The "all trials errored" guard (issue #73) needs NO change: it fires only
  when `status == "success"`, and status is `"cancelled"` on this path, so it
  cannot overwrite. Do not touch the guard; a test pins the behavior.
- After `bus.on_eval_end(log)` (log written by the JSON sink at this point):

```python
if cancelled_exc is not None:
    raise cancelled_exc
```

  Re-raising the same instance preserves the chained traceback (`__cause__`
  is the original `KeyboardInterrupt`). Callers see Ctrl-C semantics: scripts
  abort, pytest aborts, `eval_set` propagates.
- A Ctrl-C that lands *outside* the rollout call (during scoring, reducers,
  or log assembly — microseconds vs. minutes of rollout wall-clock) keeps
  today's behavior (no log). A *second* Ctrl-C landing inside the new
  handlers before the log write completes also loses the log, but still
  exits 130 cleanly from the CLI (the nested `except KeyboardInterrupt`
  catches it and the `.exists()` guard routes to "no log written"). Both are
  explicitly out of scope; the `eval()` docstring notes them so neither gets
  reported as a bug.
- The cancel handler's unconditional `status = "cancelled"` overwrites an
  earlier top-level `"error"` (e.g. a scene-0 reducer failure followed by
  Ctrl-C in scene 1); the reducer note survives in that scene's
  `scene_error`. Accepted deliberately: it mirrors the existing
  EmbodimentFault handler's unconditional overwrite, and "cancelled" is the
  fact the operator needs first.
- Accepted quirk: if an earlier epoch of the scene scored before the
  cancellation, the reducer can legitimately fail on the partial score set
  (e.g. `pass_at_k` over fewer epochs than k); the existing reducer-failure
  handler then overwrites the *scene* status to `"error"` with its note. The
  top-level `"cancelled"` status survives (that handler only overwrites
  top-level status when it is `"success"`). Acceptable: the reducer note is
  real information, and the top-level status is the source of truth for
  cancellation.

### Log schema

`EvalLog.status` and `SceneResult.status` are plain documented strings; the
comments in `log.py` gain `"cancelled"`. `read_eval_log` performs no status
validation, so the read-back guarantee holds and older readers see a string
they merely don't special-case. `EvalLog.SCHEMA_VERSION` is unchanged (no
structural change).

### CLI

`_cmd_run` wraps the `eval(...)` call with a NEW `try` nested *inside* the
existing `try:` (cli.py ~line 700) and covering ONLY the `eval()` call
(~line 753) — placement matters twice over: `sink` is constructed at ~line
739, so attaching the handler to the outer try would hit an unbound `sink`
for a Ctrl-C before that line; and `return 130` from the nested handler
still runs the outer `finally: embodiment.close()`, which is required for
real hardware.

```python
try:
    logs = eval(...)
except KeyboardInterrupt:
    if sink.path is not None and sink.path.exists():
        _print_degraded(f"cancelled: partial log written to {sink.path}")
        # then a one-line inspect hint (reuse _print_run_summary's hint
        # logic only if cheaply factorable)
    else:
        _print_degraded("cancelled: no log written")
    return 130
```

130 = conventional SIGINT exit code, propagated via the existing
`SystemExit(main())`. The `.exists()` guard covers the window inside
`JsonLogSink.on_eval_end` where `self.path` is assigned before the atomic
rename lands — without it a Ctrl-C in that window prints a path that isn't
there.

### Display

`inspect`'s per-scene failure listing (cli.py:581, `if scene.status ==
"error":`) widens to `if scene.status != "success":` so a cancelled scene
shows its detail line instead of silently disappearing from the very view
this feature feeds; pinned by a test.

### Docs and stale docstrings

- `docs/guide/logging-and-rerun.md`: one short paragraph in "The eval log" —
  Ctrl-C writes a `status: "cancelled"` log with everything gathered so far.
- `eval()` docstring: mention the cancelled path and its re-raise.
- `EvalResults.errored_trials` docstring (log.py:87-89) currently reads
  "Trials recorded but never scored" — after this change cancelled trials are
  also recorded-but-never-scored yet excluded from the count; tighten the
  wording to errored trials specifically.
- `rerun_sink.py:431-432` comment ("``eval()`` does not guarantee
  ``on_eval_end`` on every failure path (scorer/hook exceptions, Ctrl-C)"):
  qualify the Ctrl-C parenthetical — it now reaches `on_eval_end` when the
  interrupt lands inside the rollout window.
- README: not needed (behavioral detail, not a feature surface).

## Testing (100% branch coverage, mypy strict incl. tests)

All tests use the existing `CubePick` mock world; a policy whose `act` raises
`KeyboardInterrupt` at step N simulates Ctrl-C deterministically.

1. `eval()` with the interrupting policy inside `pytest.raises(KeyboardInterrupt)`:
   the sink wrote a file; `read_eval_log` returns `status == "cancelled"`,
   scene status `"cancelled"`, non-empty transcript (policy provides one),
   `termination_reasons` preserved, no metrics for the cancelled trial,
   `results.errored_trials == 0` (cancelled ≠ errored),
   `log.error == "cancelled by user (KeyboardInterrupt)"` (pins the
   `super().__init__(message)` contract behind `str(exc)`),
   `stats.frames_dir` set when `store_frames=True`.
2. Re-raise identity: the caught exception is a `KeyboardInterrupt` (and a
   `_CancelledTrial`), `__cause__` is the original.
3. Interrupt on the very first `policy.reset` (`t == -1` path): record has
   zero steps, log still written.
4. Multi-scene task interrupted in scene 0: scene 1 never runs
   (`total_scenes == 1` in results), matching the halt semantics.
5. Cancelled trials are not scored: scorer call count pinned via a counting
   scorer.
6. The all-errored guard does not rewrite `"cancelled"` to `"error"`.
7. CLI: `_cmd_run` with a monkeypatched `eval` raising `KeyboardInterrupt` —
   three cases: `sink.path` set to an existing file → exit 130 + "partial log
   written" line; `sink.path` left `None` → "no log written", 130; `sink.path`
   set but the file absent (the pre-rename window) → "no log written", 130.
8. `rollout()` unit test: KI from `embodiment.step` mid-loop →
   `_CancelledTrial` raised, `record.status == "cancelled"`, transcript
   collected by `finally`.
9. Mixed outcomes: one errored epoch then a cancelled epoch →
   `errored_trials == 1`, status `"cancelled"`, both trials preserved
   (pins the nested errored-counting branch both ways).
10. `inspect` display: rendering a cancelled log shows the cancelled scene's
    detail line (pins the widened `!= "success"` check in cli.py:581).

## Out of scope

- Ctrl-C outside the rollout window (scoring/reducers/log assembly).
- Signal handling beyond `KeyboardInterrupt` (SIGTERM etc.).
- Resuming cancelled runs (`eval_set` retry_attempts is a separate thread).
- A `video`/`inspect` affordance for logless frame dirs (this fix removes the
  main way they arise).
