# 0026: Seconds-based task horizons

Issue: [#160](https://github.com/robocurve/inspect-robots/issues/160)
Status: proposed (implementation follows the issue sketch and resolves its open questions)

## 1. Problem

`Task.max_steps` fixes the rollout budget before a task is paired with an
embodiment, but a step has a physical duration only after the embodiment's
`control_hz` is known. The same 600-step task therefore receives 60 seconds on
a 10 Hz embodiment and 40 seconds on a 15 Hz embodiment. Benchmarks that define
physical-time protocols cannot currently express their horizon without
hard-coding one embodiment's rate.

## 2. Design

`Task` accepts exactly one horizon:

- `max_steps: int | None`, preserving the existing step-based contract; or
- keyword-only `max_seconds: float | None`, a finite positive physical-time
  budget resolved by `eval()`.

After policy binding, `eval()` runs compatibility checking. A seconds-based
task is incompatible with an embodiment whose `control_hz` is missing,
non-finite, or non-positive. Once compatibility passes, the task resolves a
`TaskEnvelope` with:

```python
max_steps = ceil(max_seconds * embodiment.info.control_hz)
```

The resolved envelope is passed to `bind_task()` and the rollout. This changes
the hook order from bind-then-compatibility to compatibility-then-bind. The
ordering is required because an adapter must never receive a seconds-derived
step budget based on an invalid rate.

`EvalSpec` records both values:

- `max_seconds`: the declared physical-time budget, or `None` for a
  step-based task;
- `max_steps`: the resolved integer budget used by the rollout.

The schema version remains 1 because `max_seconds` is additive and defaults to
`None` when newer code reads an older log.

## 3. Open-question resolutions

- **Missing or zero rate:** reject at compatibility-check time. Event-driven
  embodiments must use `max_steps` because physical seconds cannot be resolved
  from their declared contract.
- **Scorers:** stay steps-only in this change. They already receive the recorded
  trajectory, and adding task configuration to the scoring interface is a
  separate API decision.
- **`eval-set`:** show the declared seconds and resolved steps on each task row.
  Saved-log inspection and the HTML report show the same pair.

## 4. Non-goals

- No wall-clock pacing in `rollout()`; self-paced embodiments still own their
  cadence.
- No CLI `--max-seconds` override for ad-hoc runs. This change lets benchmark
  authors declare physical-time protocols; the existing ad-hoc `--max-steps`
  control is unchanged.
- No scorer API or metric changes.
- No dependency changes.

## 5. Validation

- Task construction rejects missing, duplicate, non-positive, and non-finite
  horizon declarations.
- Compatibility reports invalid embodiment rates before `bind_task()` or
  rollout.
- Resolution uses `ceil`, including non-integral products, and the exact
  resolved value reaches `TaskEnvelope`, `rollout()`, and `EvalLog`.
- Older schema-v1 logs without `max_seconds` still read with `None`.
- CLI run/inspect/eval-set output and HTML reports surface the declared and
  resolved horizons without changing old step-only output.
- Ruff, formatting, strict mypy, 100% line and branch coverage, and the strict
  Docusaurus build remain green.
