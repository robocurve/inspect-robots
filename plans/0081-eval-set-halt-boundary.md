# 0081: Stop an eval set at an in-trial robot halt

Closes #423.

## Problem

`_run_eval()` catches an in-trial `SafetyAbort` or `EmbodimentFault`, preserves
the partial trial in an error `EvalLog`, and returns normally. The local
`halted` state is not represented in the log. `eval_set()` therefore sees only
an ordinary error log and starts the next task, whose rollout resets the same
embodiment that just reported an unsafe or faulted state.

An equivalent halt raised outside a trial escapes `eval()` and already stops
the set. The boundary is inconsistent with the invariant in `errors.py`: a
robot halt must never auto-advance unattended.

## Design

Add one backward-compatible field to `EvalLog`:

```python
# True when the run stopped at a condition that forbids unattended
# continuation, including a robot halt or operator cancellation.
halted: bool = False
```

- `_run_eval()` passes its existing `halted` state into the final log. This is
  true for `SafetyAbort`, `EmbodimentFault`, and `_CancelledTrial`. Cancellation
  still raises after the log is persisted, so `eval_set()` never consumes that
  log; the field nevertheless records the run accurately for readers.
- `EvalLog.from_dict()` reads `data.get("halted", False)`. No schema-version
  bump is needed because older schema-v1 logs remain readable and the new field
  is additive.
- `eval_set()` stores the logs returned by each task, then immediately returns
  `(False, logs)` when any returned log has `halted is True`. The halt marker,
  rather than the human-facing status text, is authoritative. The halted task's
  persisted log remains included and the next task is never called.
- An outside-trial `SafetyAbort` or `EmbodimentFault` still propagates exactly
  as it does today. Ordinary contained errors still append an in-memory error
  row and continue. `fail_on_error` remains a policy-error tolerance and does
  not control the safety stop.
- A failure from `close()` cannot replace an established in-trial or escaping
  robot halt with an ordinary exception that `eval_set()` would contain. The
  cleanup failure emits a `RuntimeWarning`; the halted log or original halt
  retains control-flow precedence. A close failure after a non-halted run
  still raises normally.
- No opt-out is added. Resuming the remaining tasks requires an explicit
  operator re-run, as recommended in #423.

An explicit field is preferable to parsing `SceneResult.error`: error text is
human-facing, may accumulate extra details, and is not a stable control-flow
interface.

## Tests

- Parametrize a two-task `eval_set()` over `SafetyAbort` and
  `EmbodimentFault` raised by the first task's embodiment `step()`.
- Assert `success is False`, exactly one halted error log is returned, and the
  shared embodiment's `reset()` count is one. A second reset would prove the
  next task started.
- Inject a synthetic successful log with `halted=True` to prove the structural
  marker forces `(False, logs)` and prevents a second task independently of
  human-facing status text.
- Keep the existing outside-trial propagation test unchanged.
- Round-trip `halted=True` through `EvalLog.to_dict()` / `from_dict()` and
  assert a schema-v1 dictionary without the field reads as `False`.
- A three-task success/error/success regression proves an ordinary contained
  error still advances to the following task and never sets the halt marker.
- Registry-owned embodiment tests make `close()` fail after both a returned
  and escaping robot halt, and assert the set cannot advance. A non-halted
  control proves cleanup errors are not broadly swallowed.
- Existing multi-task tests continue to prove the ordinary success path.

Coverage remains at 100 percent line and branch.

## Documentation

- Update the `eval_set()` docstring and CLI guide to state that an in-trial
  robot halt returns its log and stops the set.
- Update the package module map and the existing #298 changelog entry so the
  previous continuation language cannot survive beside the new contract.
- Add a focused `Fixed` entry linking this plan and #423.

## Out of scope

Resumption and `retry_attempts` (#136). A public continue-after-halt option.
Changes to policy-error tolerance. The optional frame-documentation nits from
#405.
