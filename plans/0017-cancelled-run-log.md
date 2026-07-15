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

Note `rollout()`'s `finally` block already collects `inference_latencies` and
`policy_transcript` while *any* exception unwinds — the data survives to the
`TrialRecord`; only the record itself is dropped, because `KeyboardInterrupt`
carries no `.record` and `eval()` never catches it.

## Design

### New exception: `CancelledTrial(KeyboardInterrupt)`

In `src/inspect_robots/errors.py`, private to the framework (NOT added to
`inspect_robots.__all__` or the errors module's public docs):

```python
class CancelledTrial(KeyboardInterrupt):
    """Ctrl-C during a trial, carrying the partial record (mirrors exc.record)."""
    def __init__(self, message: str, record: TrialRecord) -> None: ...
```

Subclassing `KeyboardInterrupt` keeps outer `except KeyboardInterrupt`
handlers (user scripts, pytest, asyncio) working unchanged, while giving
`eval()` a typed carrier for the partial record — the same shape as the
existing `InspectRobotsError.record` pattern, without `setattr` on a stdlib
exception instance (which mypy strict rejects).

`TrialRecord` is imported under `TYPE_CHECKING` in errors.py if needed to
avoid an import cycle (errors is imported by rollout).

### `rollout()`: catch, mark, re-raise typed

Add one handler to the existing outer `try` (the one whose `finally` collects
the transcript):

```python
except KeyboardInterrupt as exc:
    record.status = "cancelled"
    record.error = "cancelled by user (KeyboardInterrupt)"
    record.events.append(error_event(t, "KeyboardInterrupt", "cancelled by user"))
    raise CancelledTrial(record.error, record) from exc
```

- `t` must be defined on every path: initialize `t = -1` before
  `policy.reset(scene)` (matching the `-1` convention `_record_failure`
  already uses for pre-loop failures) and let the loop rebind it.
- The `finally` block runs after this handler, so `policy_transcript` and
  `inference_latencies` land on the record exactly as they do today for
  typed errors.
- `TrialRecord.status` docstring/comment gains `"cancelled"` as a value.

### `eval()`: preserve, write, re-raise

At the rollout call site, alongside the existing handlers:

```python
except CancelledTrial as exc:
    status = "cancelled"
    error = exc.args[0] if exc.args else "cancelled by user (KeyboardInterrupt)"
    scene_status = "cancelled"
    scene_error = error
    halted = True
    cancelled_exc = exc
    record = exc.record
```

- `cancelled_exc: KeyboardInterrupt | None = None` is initialized with the
  other loop state.
- The existing record-preservation block runs unchanged *except* the scoring
  gate: `if record.status == "error":` becomes `if record.status != "success":`
  so a cancelled partial trial is preserved (termination reason, transcript,
  `on_trial_end`) but never scored — identical treatment to errored trials,
  same rationale (a half-trial must not masquerade as data).
- `halted = True` reuses the existing halt path to break both loops and skip
  remaining scenes/epochs.
- The "all trials errored" guard (issue #73) must not overwrite the cancelled
  status: change its condition from `status == "success"` — it already only
  fires then, and `status` is `"cancelled"` here, so no change is actually
  needed; a test pins this.
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
  today's behavior (no log). Explicitly out of scope; noted in the docstring.

### Log schema

`EvalLog.status` and `SceneResult.status` are plain documented strings; the
comments in `log.py` gain `"cancelled"`. `read_eval_log` performs no status
validation, so the read-back guarantee holds and older readers see a string
they merely don't special-case. `EvalLog.SCHEMA_VERSION` is unchanged (no
structural change).

### CLI

`_cmd_run` wraps the `eval(...)` call:

```python
try:
    logs = eval(...)
except KeyboardInterrupt:
    if sink.path is not None:
        _print_degraded(f"cancelled: partial log written to {sink.path}")
        # then the existing inspect/video hints, reusing _print_run_summary's
        # hint logic if cheaply factorable, else a one-line inspect hint
    else:
        _print_degraded("cancelled: no log written")
    return 130
```

130 = conventional SIGINT exit code. `sink` is the `JsonLogSink` the CLI
already constructs (`cli.py` ~line 739); its `.path` attribute is set by
`on_eval_end` and stays `None` when the interrupt preceded the write.

### Docs

- `docs/guide/logging-and-rerun.md`: one short paragraph in "The eval log" —
  Ctrl-C writes a `status: "cancelled"` log with everything gathered so far.
- `eval()` docstring: mention the cancelled path and its re-raise.
- README: not needed (behavioral detail, not a feature surface).

## Testing (100% branch coverage, mypy strict incl. tests)

All tests use the existing `CubePick` mock world; a policy whose `act` raises
`KeyboardInterrupt` at step N simulates Ctrl-C deterministically.

1. `eval()` with the interrupting policy inside `pytest.raises(KeyboardInterrupt)`:
   the sink wrote a file; `read_eval_log` returns `status == "cancelled"`,
   scene status `"cancelled"`, non-empty transcript (policy provides one),
   `termination_reasons` preserved, no metrics for the cancelled trial,
   `stats.frames_dir` set when `store_frames=True`.
2. Re-raise identity: the caught exception is a `KeyboardInterrupt` (and a
   `CancelledTrial`), `__cause__` is the original.
3. Interrupt on the very first `policy.reset` (`t == -1` path): record has
   zero steps, log still written.
4. Multi-scene task interrupted in scene 0: scene 1 never runs
   (`total_scenes == 1` in results), matching the halt semantics.
5. Cancelled trials are not scored: scorer call count pinned via a counting
   scorer.
6. The all-errored guard does not rewrite `"cancelled"` to `"error"`.
7. CLI: `_cmd_run` with a monkeypatched `eval` raising `KeyboardInterrupt`
   after setting `sink.path` → exit code 130 + "partial log written" line;
   with `sink.path` left `None` → "no log written" line, code 130.
8. `rollout()` unit test: KI from `embodiment.step` mid-loop → `CancelledTrial`
   raised, `record.status == "cancelled"`, transcript collected by `finally`.

## Out of scope

- Ctrl-C outside the rollout window (scoring/reducers/log assembly).
- Signal handling beyond `KeyboardInterrupt` (SIGTERM etc.).
- Resuming cancelled runs (`eval_set` retry_attempts is a separate thread).
- A `video`/`inspect` affordance for logless frame dirs (this fix removes the
  main way they arise).
