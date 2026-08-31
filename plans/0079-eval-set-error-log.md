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

- Wrap the `eval(...)` call in `try/except Exception as exc`. Never catch
  `BaseException`: `KeyboardInterrupt` must keep propagating, and
  `_cmd_eval_set` already handles it. Check `_CancelledTrial`'s base class;
  if it derives from `Exception`, re-raise it explicitly before the generic
  handler.
- On failure append one `EvalLog` built by a private helper
  `_error_log_for(task, policy, embodiment, *, seed, exc) -> EvalLog`:
  `status="error"`, `error=f"{type(exc).__name__}: {exc}"`,
  `results=EvalResults(total_scenes=0, total_trials=0)`,
  `stats=EvalStats(started_at=now, completed_at=now, duration_s=0.0,
  total_steps=0)`, `samples=()`, and an `EvalSpec` whose `task`, `policy`,
  and `embodiment` names are the registry strings when strings were passed
  and `task.name` / `policy.info.name` / `embodiment.info.name` when
  objects were passed, `seed=seed`, `max_steps=None`, `max_seconds=None`.
  Reuse whatever `_run_eval` already uses for `created`,
  `inspect_robots_version`, and `git_commit` so the two spec paths cannot
  drift; factor a tiny helper if that code is currently inline.
- The synthetic log is **not** pushed through the sinks and **not** written
  to `log_dir`: it never had an `on_eval_start`, and a zero-trial JSON file
  for a task that never ran would show up in `inspect-robots view` as a
  run. In-memory only; the CLI summary renders it.
- The loop continues with the remaining tasks. This is consistent with the
  existing behaviour that a `SafetyAbort`/`EmbodimentFault` inside a task
  degrades to an error log and `eval_set` proceeds to the next task.
- Docstring: replace the sentence about `success` with the full contract,
  including "a task that raises before or without producing a log
  contributes one `status="error"` log carrying the exception text, and the
  remaining tasks still run".

CLI: verify, do not change. `_print_eval_set_summary` in `cli.py` already
prints `log.error` as the row detail when a log has no metrics, and the
command already exits 1 when `success` is false. If either turns out not to
hold for a zero-sample log, the fix belongs in that function and stays
minimal.

## Tests

- `tests/test_eval_log.py`, next to `test_eval_set_runs_multiple_tasks`
  (line 321): `test_eval_set_reports_failed_task_and_keeps_completed_logs`.
  Two `Task`s with `ScriptedPolicy()` / `CubePickEmbodiment()`, the second
  built with `Epochs(count=1, reducer="bogus")`. Assert `success is False`,
  `len(logs) == 2`, `logs[0].status == "success"` with its task name,
  `logs[1].status == "error"`, `"bogus"` in `logs[1].error`,
  `logs[1].eval.task` is the second task's name, `logs[1].samples == ()`,
  and exactly one JSON file for the first task exists in `tmp_path`.
- Same file: one case with string policy/embodiment names to cover the
  string branch of the spec names.
- `tests/test_registry_cli.py`, mirroring
  `test_cli_eval_set_runs_multiple_exact_tasks` (line 931): register a task
  with an invalid reducer, run `eval-set` with a good task first, assert
  `rc == 1`, the good task's `[ok]`/metrics row and the bad task's
  `[error]` row with the `ConfigError` text both appear in stdout, and the
  good task's log is on disk.
- A `KeyboardInterrupt` raised by a stub `eval` propagates out of
  `eval_set` (monkeypatch `inspect_robots.eval.eval`).

Coverage stays at 100%.

## Docs

- `CHANGELOG.md` under `## [Unreleased]` / `### Fixed`, `**Core:**` prefix,
  referencing #298.
- Grep `docs/guide` for `eval_set`; where its return contract is described,
  add the one-sentence rule above.

## Out of scope

Honoring `retry_attempts` / resumption (#136). Containing scorer or
`before_scoring` exceptions inside `_run_eval` so that the failing task's
own partial log survives; that is a separate change to `eval()`'s contract
and is noted on the issue as a follow-up.
