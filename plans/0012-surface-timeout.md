# Surface rollout timeouts in the eval log and CLI

Date: 2026-07-14
Status: approved (log design and CLI surface options confirmed by Jay)

## Problem

A real-robot run that hits the rollout step limit currently looks identical to a
clean finish. Observed on a yam ad-hoc run (`logs/adhoc_7425493a.json`,
`total_steps: 1200` at 10 Hz): the episode was cut off by `max_steps`, yet the
CLI printed `status: success` and the eval log recorded nothing about the
truncation.

The information exists and then gets dropped:

1. `rollout()` sets `TrialRecord.truncated = True` and
   `termination_reason = "max_steps"` when the loop exhausts the horizon
   (`rollout.py`, the `while` loop's `else` arm).
2. `eval()` builds `SceneResult` from the trial records but discards
   `truncated` / `termination_reason` (`eval.py`, `SceneResult(...)`).
3. `_print_run_summary()` in `cli.py` only reads `log.status`, which means
   "no errors", so the run reports success.

## Requirements

1. The eval log records, per trial, why the trial ended (so a timeout is
   visible in the persisted artifact).
2. The post-run CLI summary tells the operator when trials hit the step limit,
   including the limit in steps and (when the control rate is known) seconds.
3. The summary reminds the operator how to raise the limit: `--max-steps N` on
   ad-hoc runs, or `inspect-robots config set max_steps N`.
4. `inspect-robots inspect <log>` surfaces the same information for logs read
   back later.
5. Old logs (schema v1, written before these fields) keep reading back
   without error; schema version stays 1 (additive change, same pattern as
   `SceneResult.operator_judgements`).

## Design

### `log.py`

- `SceneResult.termination_reasons: tuple[str | None, ...] = ()` — strictly
  parallel to `epochs`: one entry per scored trial, holding
  `TrialRecord.termination_reason` (`"max_steps"`, an embodiment reason such
  as `"success"`/`"failure"`, a policy stop reason, or `None`). Errored trials
  are not scored and get no entry, matching `epochs`/`operator_judgements`.
- `EvalSpec.max_steps: int | None = None` — the task horizon, so a log is
  self-describing about the limit that produced a `"max_steps"` reason.
- `EvalLog.from_dict` coerces `termination_reasons` back to a tuple with a
  `()` default, and `EvalSpec` gains its field with a `None` default, so
  logs written before this change still read (newer reads older).

### `eval.py`

- Collect `record.termination_reason` alongside `record.operator_judgement`
  in the scored-trial branch; pass the tuple into `SceneResult`.
- Pass `task.max_steps` into `EvalSpec`.

### `cli.py`

- `_print_run_summary()`: count scored trials whose reason is `"max_steps"`
  across all samples. When nonzero, print (yellow, before the metrics):

  ```
  note: 1/1 trials hit the step limit before terminating (max_steps=1200, ~120s at 10 Hz)
  hint: raise it with --max-steps N or: inspect-robots config set max_steps N
  ```

  - The seconds figure uses `log.eval.max_steps` and
    `log.eval.embodiment_info["control_hz"]`; omit the parenthetical seconds
    part when either is missing or the rate is not a positive number.
  - `log.status` semantics are unchanged: a truncated run is still
    `success` (no errors); the note is advisory.
  - The `config set` hint applies to ad-hoc runs; registered tasks own their
    horizon, so when `log.eval.task != "adhoc"` the hint line instead says the
    task defines its own `max_steps`.
- `_cmd_inspect()`: per scene, when any termination reason is `"max_steps"`,
  append a short `(N/M trials hit max_steps)` marker to the scene line, and
  reuse the same note line at the top when the count is nonzero.

### Out of scope

- The yam plugin's live countdown showing the horizon (`t = 42s / 120s`)
  requires setting the `max_steps_hint` embodiment arg in the operator's
  config; that is a config/docs follow-up in the `inspect-robots-yam` repo,
  not a core change.
- Changing `EvalLog.status` or scoring semantics for truncated trials.

## Testing

TDD; the core gate is 100% coverage, mypy strict, ruff. New/updated tests:

- `log.py`: round-trip with `termination_reasons` and `max_steps`; read-back
  of a v1 dict missing both fields defaults them (golden/back-compat test).
- `eval.py`: an eval whose trial truncates at `max_steps` produces a
  `SceneResult` carrying `("max_steps",)` and an `EvalSpec.max_steps` equal to
  the task horizon; errored trials contribute no entry.
- `cli.py`: run summary prints the note + hint when a reason is
  `"max_steps"` (with and without a usable `control_hz`; adhoc vs registered
  task hint); prints nothing extra otherwise; `inspect` output shows the
  marker.
- `tests/test_api_snapshot.py`: only if the public surface changes (fields on
  existing dataclasses should not alter `__all__`).
