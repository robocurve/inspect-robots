# 0079: eval_set keeps completed logs when a later task fails

Closes #298.

## Problem

`eval_set()` (`src/inspect_robots/eval.py`) runs its tasks in a bare loop.
If `eval()` raises for task N (a `ConfigError` from a bad epoch reducer or
`resolve_envelope`, a `CompatibilityError`, a `bind`/`bind_task` hook
failure, a scorer or `before_scoring` exception), the exception propagates
out of `eval_set`, the `(success, logs)` tuple is never returned, and the
caller loses the already-completed logs of tasks 1..N-1 even though they are
on disk. On the CLI the operator gets a raw traceback instead of the
per-task summary and hints. Reproduced on `main` (`f4b6f03f`) both through
the API (the issue's script) and through `inspect-robots eval-set` with a
registered task whose `Epochs.reducer` is invalid.

## Design

Per-task containment with a synthesized error log, so the documented
contract `success == all(log.status == "success" for log in logs)` stays
literally true and the reason travels in `log.error`:

- Wrap the `eval(...)` call in `try/except`. Two classes must keep
  propagating and are re-raised first, mirroring `eval.py`'s own
  `observe_parked` seam (`except (SafetyAbort, EmbodimentFault): raise`):
  `SafetyAbort` and `EmbodimentFault`. Both are `Exception` subclasses and
  both can escape `eval()` today (from `observe_parked()`, from
  `bind`/`bind_task`/`embodiment.info`, from `embodiment.close()` in
  `eval()`'s `finally`). The repo invariant in `errors.py` is that a safety
  halt never auto-advances to the next scene unattended; containing it here
  would let the next task's `rollout()` reset the arm that just raised. The
  generic handler is `except Exception`. `KeyboardInterrupt` (and
  `_CancelledTrial`, which derives from it) is not an `Exception` and
  propagates unchanged; `_cmd_eval_set` already handles it.
- On a contained failure append one `EvalLog` built by a private helper
  `_error_log_for(task, policy, embodiment, *, seed, exc) -> EvalLog`:
  `status="error"`, `error=f"{type(exc).__name__}: {exc}"`,
  `results=EvalResults(total_scenes=0, total_trials=0)`,
  `stats=EvalStats(started_at=now, completed_at=now, duration_s=0.0,
  total_steps=0)`, `samples=()`, and an `EvalSpec` whose `task`, `policy`,
  and `embodiment` names are the registry strings when strings were passed
  and `task.name` / `policy.info.name` / `embodiment.info.name` when
  objects were passed; `seed=seed`; `max_steps`/`max_seconds` copied from
  the `Task` when one was passed, `None` otherwise (cosmetic: the summary
  row then shows the declared horizon for step-based tasks). Use the existing `_now_iso()`, `__version__`,
  and `_git_commit()` helpers for `created`, `inspect_robots_version`, and
  `git_commit`.
- The synthetic log is **not** pushed through the sinks and **not** written
  to `log_dir`: it never had an `on_eval_start`, and a zero-trial JSON file
  for a task that never ran would show up in `inspect-robots view` as a
  run. In-memory only; the CLI summary renders it.
- **Live-log orphan.** An exception that escapes `_run_eval` between
  `bus.on_eval_start` and `bus.on_eval_end` (a `before_scoring`/grader
  raise is the realistic case) leaves the `LiveLogSink`'s
  `<task>_<id>.live.json` on disk in the "running" state. Today
  `_cmd_eval_set`'s `finally` unlinks it; once `eval_set` continues, the
  next task's `on_eval_start` reassigns `self.path` and the stale page
  would be listed forever as running in `inspect-robots view`. Fix in
  `logging/live_log.py`: `LiveLogSink` tracks whether the current run
  reached `on_eval_end` (a `_finished` flag, set there only after its own
  unlink succeeded, cleared in `on_eval_start`); when `on_eval_start` finds
  a previous `self.path` whose run did not finish, it unlinks that file
  under `with suppress(OSError)` (`missing_ok=True` already covers absence)
  **before** the existing `try:` that opens the new file, so a stale-file
  cleanup failure cannot run through `_disable` and silently kill the new
  run's live log. A failed unlink is simply retried at the next
  `on_eval_start`. Test in `tests/test_live_log.py` next to
  `test_eval_set_threads_and_reuses_caller_supplied_live_sink`.
- The loop continues with the remaining tasks.
- Contract change to state plainly in the docstring and CHANGELOG: after
  this change a `CompatibilityError`, an unknown `policy`/`embodiment`
  registry name, or a task factory's `ConfigError` no longer raises out of
  `eval_set`; each produces one error log and the set continues (a
  set-wide misconfiguration therefore yields N identical error rows, and a
  string hardware embodiment is re-resolved once per task). Only
  `_grading_hook`'s `ConfigError`, raised before the loop, still
  propagates. No existing test asserts that `eval_set` raises.
- Known edge, one sentence in the docstring: with a string embodiment, a
  non-safety exception from `embodiment.close()` in `eval()`'s `finally`
  fires after the JSON log was written, so `eval_set` appends an error row
  for a task whose real log is complete on disk (pre-existing `eval()`
  behaviour; strictly better than today's total loss).
- Docstring: replace the sentence about `success` with the full contract,
  including "a task that raises before or without producing a log
  contributes one `status="error"` log carrying the exception text, the
  remaining tasks still run, and `SafetyAbort`/`EmbodimentFault`/
  `KeyboardInterrupt` still propagate".

CLI: verify, do not change. `_print_eval_set_summary` in `cli.py` already
prints `log.error` as the row detail when a log has no metrics, and the
command already exits 1 when `success` is false. Known cosmetic gap, accepted
for this PR: the failure hint names `{log_dir}/<task>_<id>.json`, which does
not exist for a synthetic log.

## Tests

- `tests/test_eval_log.py`, next to `test_eval_set_runs_multiple_tasks`
  (line 321): `test_eval_set_reports_failed_task_and_keeps_completed_logs`.
  Two `Task`s with `ScriptedPolicy()` / `CubePickEmbodiment()`, the second
  built with `Epochs(count=1, reducer="bogus")`. Assert `success is False`,
  `len(logs) == 2`, `logs[0].status == "success"` with its task name,
  `logs[1].status == "error"`, `"bogus"` in `logs[1].error`,
  `logs[1].eval.task` is the second task's name, `logs[1].eval.max_steps`
  equals the task's, `logs[1].samples == ()`, and exactly one JSON file for
  the first task exists in `tmp_path`.
- Same file: a case with string `task`, `policy`, and `embodiment` names
  where the task fails (an unknown task name raises `KeyError` from the
  registry, or a registered bogus-reducer task), so all three string arms
  of the spec-name ternaries execute (branch coverage is on).
- Same file, propagation: monkeypatch `inspect_robots.eval.eval` (the
  `sys.modules[...]` pattern used in `tests/test_eval_orchestration.py`) to
  raise, parametrized over `SafetyAbort`, `EmbodimentFault`, and
  `KeyboardInterrupt`; assert each propagates out of `eval_set` and no
  log list is returned.
- `tests/test_registry_cli.py`, mirroring
  `test_cli_eval_set_runs_multiple_exact_tasks` (line 931): register a task
  with an invalid reducer, run `eval-set` with a good task first, assert
  `rc == 1`, the good task's `[completed]` row and the bad task's `[error]`
  row with the `ConfigError` text both appear in stdout, and the good
  task's log is on disk.
- `tests/test_live_log.py`: a caller-supplied `LiveLogSink` whose first run
  ends without `on_eval_end` (a `before_scoring` that raises) followed by a
  second task: the first `.live.json` is gone, and the second run started
  and finished, observed via the existing `_ProbeSink` capturing
  `start_paths` (the second file is also unlinked at its own `on_eval_end`,
  as `test_eval_set_threads_and_reuses_caller_supplied_live_sink` shows).

Coverage stays at 100%.

## Docs

- `CHANGELOG.md` under `## [Unreleased]` / `### Fixed`, `**Core:**` prefix,
  referencing #298.
- `docs/guide/logging-and-rerun.md` around line 46 (sink reuse paragraph):
  one sentence that a reused `LiveLogSink` removes a previous run's stale
  "running" snapshot at the next `on_eval_start`, and that on the Python
  API the last task's orphan stays until then.
- `docs/guide/cli.md` around lines 474-476 ("The exit code is `0` iff every
  task's log has `status == "success"`"): add the one-sentence rule above,
  and that safety halts still stop the set.

## Out of scope

Honoring `retry_attempts` / resumption (#136). Containing scorer or
`before_scoring` exceptions inside `_run_eval` so that the failing task's
own partial log survives; that is a separate change to `eval()`'s contract
and is noted on the issue as a follow-up (the live-log orphan fix above
covers the on-disk side effect of that gap in the meantime).

## Addendum (after diff review)

The halt contract is narrower than the Design section's wording implies: only
a `SafetyAbort`/`EmbodimentFault` that escapes `eval()` (raised outside a
trial) propagates out of `eval_set`. A halt inside a trial is absorbed by
`_run_eval` into that task's error log and the set continues to the next
task, exactly as before this plan. Making `eval_set` stop after such a task
is tracked in #423; the shipped prose (CHANGELOG, `docs/guide/cli.md`, the
`eval_set` docstring) states the narrower truth.
