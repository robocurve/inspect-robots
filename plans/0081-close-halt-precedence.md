# 0081: Preserve stop signals across owned cleanup failure

## Problem

`eval()` closes an embodiment that it resolves from a registry name in a
`finally` block. If `_run_eval()` raises an escaping `SafetyAbort`,
`EmbodimentFault`, or `KeyboardInterrupt` and `embodiment.close()` then raises
an ordinary exception, Python replaces the original stop signal with the
cleanup failure.

That replacement breaks the existing `eval_set()` contract. The set re-raises
escaping safety and cancellation signals, but it contains an ordinary cleanup
exception as a task error and advances to the next task. A disconnected robot
can therefore lose the signal that should have stopped the set.

Reproduced on `main` at `84a4322`: a registry-owned embodiment raises a stop
signal from `bind_task()` and `RuntimeError("disconnect failed")` from
`close()`. `eval_set()` returns two synthesized error logs instead of
re-raising the stop before the second task.

## Design

Keep the existing lifecycle and exception policy, with one narrow precedence
rule:

| `_run_eval()` outcome | `close()` outcome | Result |
| --- | --- | --- |
| return | return | return the evaluation log |
| return | ordinary exception | propagate the cleanup exception, unchanged |
| ordinary exception | return | propagate the original exception, unchanged |
| ordinary exception | ordinary exception | propagate the cleanup exception, unchanged |
| escaping safety or cancellation signal | return | propagate the original signal |
| escaping safety or cancellation signal | ordinary exception | emit a `RuntimeWarning` and propagate the original signal |

The protected signals are `SafetyAbort`, `EmbodimentFault`, and
`KeyboardInterrupt`, including `_CancelledTrial`. The cleanup diagnostic uses
a local `warnings.simplefilter("always", RuntimeWarning)` so a caller's warning
filter cannot convert the diagnostic into another exception and mask the stop
signal. Cleanup is still attempted exactly once.

The helper catches `Exception`, not `BaseException`. A new
`KeyboardInterrupt` raised by cleanup is therefore not swallowed.

## Tests

`tests/test_eval_orchestration.py` covers:

- each protected signal escaping from a registry-owned embodiment whose
  `close()` raises;
- propagation of the original type and message;
- exactly one `bind_task()` and one `close()` call, proving that `eval_set()`
  does not advance to the second task;
- the cleanup warning; and
- a negative control proving that ordinary exception precedence is unchanged.

The protected tests fail on the base commit because no stop signal escapes.

## Documentation

Update the `eval()` and `eval_set()` docstrings, the package agent guide, and
the changelog. No public type or schema changes.

## Out of scope

- Changing the retained behavior for a halt caught inside a trial. It still
  ends that task with an error log, and `eval_set()` continues as documented.
- Adding `EvalLog.halted` or any other log field.
- Adding a stop-after-halt rule to `eval_set()`.
- Changing cleanup precedence for ordinary exceptions or successful runs.
