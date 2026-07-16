# 0014 — Bound the spawned Rerun viewer's memory; bound the exit path against a wedged viewer

Issue: [#95](https://github.com/robocurve/inspect-robots/issues/95)

## Problem

`RerunSink(spawn=True)` calls `rr.init(app_id, spawn=True)`, which spawns the
viewer with rerun's default options, notably `--memory-limit=75%`. That is 75%
of *system* RAM (~46 GiB on a 64 GiB workstation). Because `spawn` reuses any
viewer already listening on port 9876, every eval run of a working day stacks
its recording (full camera streams) into the same viewer process. Observed on
the omen rig after ~11 runs: 6.8 GiB RSS, ~73% CPU sustained for 6 hours,
frozen UI, and ingest backpressure (`quota_channel` warnings, ~355 KB stuck in
the client socket send buffer).

Second-order failure: with the viewer wedged, the eval CLI wrote its log and
then hung at interpreter exit for 75+ seconds, until the viewer was killed.

### Where the exit hang actually lives (empirical, rerun-sdk 0.34.1)

Reproduced against a TCP peer that accepts but never reads:

- `rr.log` is non-blocking: 35 calls totaling ~105 MB returned in 0.07 s. The
  data queues inside the SDK client (dropping via `quota_channel`), so the
  sink's worker thread stays fast, `RerunSink.flush()` succeeds, and the
  sink's existing `wedged` detection in `_shutdown()` never fires. The sink
  finishes cleanly; the hang is *not* in the sink.
- The hang is the SDK's own atexit hook (`rerun_shutdown` →
  `bindings.shutdown()`, registered at import), which flushes the stream
  unboundedly.
- `rr.disconnect()` is not a fix: it flushes *before* swapping the sink and
  blocks indefinitely against a wedged peer ("Flushing the gRPC stream has
  taken over 10.0s…", still stuck when killed at 60 s).
- `rr.unregister_shutdown()` returns immediately even on a wedged stream and
  the process then exits cleanly; it is present and exported in both 0.20 and
  0.34.1.
- A second `rr.init` in the same process also blocks forever on a wedged
  stream (`bindings.flush_and_cleanup_orphaned_recordings()` at the top of
  `init`), so a wedged sink instance must never re-init.

## Fix

Two changes, both in `src/inspect_robots/logging/rerun_sink.py`.

### 1. Spawn with a bounded viewer memory limit

- Add a constructor parameter `spawn_memory_limit: str = "2GiB"` to
  `RerunSink`; documented as consulted only when `spawn=True` (no new
  mutual-exclusion rules).
- In `on_eval_start`, replace

  ```python
  rr.init(self.application_id, spawn=self.spawn)
  ```

  with

  ```python
  rr.init(self.application_id)
  if self.spawn:
      rr.spawn(memory_limit=self.spawn_memory_limit)
  ```

  Semantic equivalence verified in both 0.20.0 and 0.34.1 sources:
  `init(spawn=True)` literally calls `spawn(default_blueprint=...)` after
  initializing logging, with the same port 9876, `connect=True` default and
  port-reuse behavior, and the sink passes no blueprint in either form.
  `spawn(memory_limit=...)` exists at the 0.20 floor.
- With the limit bounded, the viewer purges the oldest events instead of
  accumulating a full day of recordings. 2 GiB comfortably holds several
  JPEG-compressed runs (the SDK's own `server_memory_limit` default is
  "1GiB", so the string format is canonical).
- The calls stay inside the existing `try/except` in `on_eval_start`, so any
  spawn failure degrades to the existing warned no-op.
- Known limitation (documented, not solved): the limit applies only to
  viewers *this package spawns*. A viewer already running on port 9876 —
  launched manually or before the upgrade — keeps its old limit; `rr.spawn`
  just connects to it. Release note: "kill any running Rerun viewer once
  after upgrading; the cap applies from the next spawn."

### 2. Bound the exit path: probe with a bounded flush, unregister the SDK atexit hook on wedge

Runs unconditionally in `_shutdown()` (i.e. from `on_eval_end`), on the
caller thread, regardless of the sink-level `wedged` flag (which the incident
shows stays False). It must run *before* `_shutdown()`'s early return for the
no-worker case — the probe is gated on `self._rr is not None` (init happened),
not on a worker having started, so an eval that connected but logged no steps
is still bounded.

1. Obtain the stream: `get_rec = getattr(rr, "get_global_data_recording",
   None)`; `rec = get_rec() if get_rec is not None else None`. If `rec` is
   `None` (never inited, or unknown SDK surface), do nothing.
2. Resolve the flush surface first: `flush = getattr(rec, "flush", None)`.
   `RecordingStream.flush` first appears in rerun-sdk **0.22.0** (as
   `flush(blocking: bool = True)`; the `timeout_sec=` kwarg is newer still) —
   0.20/0.21 expose no Python-accessible flush at all. If `flush` is `None`,
   the probe is *inconclusive*: skip it, keep the atexit hook, do **not**
   disable the sink. On 0.20/0.21 the exit path therefore keeps today's
   behavior (no regression, but no new protection either); the bounded exit
   guarantee requires rerun-sdk >= 0.22.
3. Bounded flush probe: run `flush()` on a fresh daemon thread and
   `join(self.flush_timeout)`. The thread bound (not a `timeout_sec=` kwarg)
   is deliberate: 0.22's `flush` has no timeout parameter, and a daemon
   thread bounds *any* signature uniformly — same disowning pattern the sink
   already uses for its wedged worker. Exceptions raised by an existing
   `flush` are swallowed inside the probe thread and count as a completed
   (healthy) probe. Empirically verified sound on 0.34.1: the native flush
   releases the GIL, `join(timeout)` returns on time against a wedged peer,
   and a leaked probe thread finishing late is harmless.
4. If the flush completed in time: healthy connection. The tail is drained;
   leave the SDK's atexit hook alone (it will be fast at exit). Nothing else
   changes.
5. If the flush timed out: the stream is wedged.
   - `unregister = getattr(rr, "unregister_shutdown", None)`; if present,
     call it inside `try/except Exception: pass` — this removes the SDK's
     unbounded atexit flush so interpreter exit cannot hang.
   - Set `self._disabled = True` so this sink instance never calls `rr.init`
     again (re-init verified to hang; see above).
   - Warn (`RuntimeWarning`, style of the existing shutdown warnings): the
     viewer connection is stalled; visualization is disabled for this sink;
     queued SDK-side data was abandoned.
6. Documented known limitations: a *new* `RerunSink` constructed in the same
   process against the same wedged viewer can still hang inside `rr.init`
   (the SDK offers no bounded init), and the probe only runs on paths that
   reach `on_eval_end` — Ctrl-C or scorer/hook exceptions that skip it keep
   the SDK's unbounded atexit hook (a sink-registered atexit is possible
   future hardening). Single-run CLI usage reaching eval end (the incident
   shape) is fully covered.

Worst-case `on_eval_end` latency becomes ~3× `flush_timeout` (sink flush +
worker join + SDK flush probe) = 30 s at defaults, in exchange for an exit
that — for any eval reaching `on_eval_end` on rerun-sdk >= 0.22 — can never
hang indefinitely.

Threading note: the flush probe and `unregister_shutdown` run off the worker
thread, which the module docstring currently forbids ("all SDK calls after
init happen on the worker"). That invariant exists because SDK *timeline*
state is thread-local; `RecordingStream.flush` is internally synchronized and
`unregister_shutdown` is pure atexit manipulation, so both are safe. Amend
the module docstring to state the shutdown-path exception and why.

## Tests (TDD, extend `tests/test_rerun_sink.py` fakes)

Spawn path:

1. `spawn=True` calls `rr.init` **without** `spawn=` and then `rr.spawn` with
   `memory_limit="2GiB"` (capture kwargs on the fake).
2. `spawn_memory_limit="4GiB"` is forwarded verbatim.
3. `spawn=False` (default) never calls `rr.spawn`.
4. `connect_url=...` path unchanged: `init` then `connect_grpc`, no `spawn`.
5. `rr.spawn` raising disables the sink with the existing "RerunSink
   disabled" warning. Keep the existing init-raise test
   (`test_viewer_failure_disables_sink_instead_of_crashing`) for init-failure
   coverage but fix its name/docstring — after the split, a missing viewer
   binary surfaces from `rr.spawn`, not `rr.init`.

Exit path (fake `rec` object returned by a fake `get_global_data_recording`):

6. Flush probe timeout: construct the sink with a small `flush_timeout`
   (existing wedged tests use 0.05 — the default would cost 10 s of suite
   wall time), and block the fake `flush` on `gate.wait(timeout=30.0)` (the
   existing pattern) so the leaked daemon probe thread eventually unblocks
   instead of outliving the pytest run. Assert: `unregister_shutdown` is
   called, the sink is disabled (`_disabled` / subsequent `on_eval_start` is
   a no-op), and the stall warning is emitted.
7. Healthy flush (returns instantly): `unregister_shutdown` is **not**
   called, sink stays enabled — a healthy run must keep the SDK's atexit
   flush.
8. Shim paths: fake without `get_global_data_recording`; `get_rec` returning
   `None`; `rec` without `flush` (inconclusive: no disable, no unregister,
   hook kept — the 0.20/0.21 posture); wedged path with
   `unregister_shutdown` absent; `unregister_shutdown` raising (swallowed).
   All shut down cleanly.
9. Make the interaction with the existing wedged-worker test explicit: the
   current `_install_fake_rerun` fake has none of the new attributes, so
   `test_wedged_worker_is_disowned_and_backlog_dropped` will exercise the
   attribute-missing shim path — assert that intentionally rather than by
   accident.
10. An exception raised by an *existing* `flush` is swallowed inside the
    probe thread and treated as a completed probe (missing `flush` is the
    separate inconclusive path of test 8, never a "completed" probe).

Integration (non-gating, `pytest.importorskip("rerun")` so it skips in core
CI where rerun-sdk is not a dev dependency; runs on dev machines with
`[all]`): subprocess that inits a real recording, connects to a local TCP
listener that accepts but never reads, logs a payload, and exits — assert the
subprocess terminates within a hard timeout. This is the only test that could
have caught the `disconnect()` blocker; fakes assert wiring, not liveness.

Existing mutual-exclusion and lifecycle tests stay green; 100% coverage holds
(`--cov-fail-under=100`) — every new branch above is reachable with the fake
pattern.

## Out of scope

- No new CLI flag: the bounded default fixes the failure mode; a
  `--rerun-memory-limit` flag can follow if anyone needs a different ceiling.
- No change to `connect_url` mode: a remote viewer's memory policy belongs to
  whoever launched it. (Note `DEFAULT_RERUN_CONNECT_URL` in `cli.py` points
  at the same 127.0.0.1:9876 — a manually-launched unbounded viewer there is
  the user's choice.)
- No attempt to reconfigure an already-running viewer: rerun offers no API
  for that; the limit applies from the next fresh spawn.

## Docs

- Module docstring: bounded-spawn default; amended threading invariant
  (shutdown-path exception, and why it is safe); wedged-exit behavior
  (bounded probe, atexit unregister, sink disables itself).
- `RerunSink` class/param docstrings: `spawn_memory_limit`.
- README rerun section (if it mentions spawning): one sentence noting the
  viewer is spawned with a 2 GiB cap so long sessions stay responsive, plus
  the upgrade note about killing a pre-existing viewer.
