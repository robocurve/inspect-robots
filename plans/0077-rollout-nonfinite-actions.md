# 0077: Reject non-finite actions in the rollout loop

Closes #356.

## Problem

`rollout()` is the framework's last boundary before `embodiment.step()`, and
it validates an action's dimension but not its finiteness. With the default
`AutoApprover()` (the `eval()` default, and the CLI fallback when a space
declares no bounds and no `--max-action-delta` is given) a policy that emits
`nan`/`inf` has that value delivered to the embodiment on every step.
Reproduced on `main` (`f4b6f03f`): with the `CubePick` mock the trial is
recorded as `status="success"` with poisoned state, and the action-log
side-car disables itself for the whole trial after the first non-finite
value. Real embodiments are out-of-tree and inherit the gap.

`ClampApprover` already aborts on NaN and `DeltaLimitApprover` is gaining the
same on ±inf (#376), but that protection is configuration-dependent. The
`GuardrailContribution` docstring only asks contributors to validate.

## Design

Two checks in `rollout()` (`src/inspect_robots/rollout.py`), no new public
API:

1. **Pre-review, policy-attributed.** Immediately after the existing
   dimension check (the `emitted_dim != expected_dim` block), if
   `not np.isfinite(action.data).all()`, raise
   `_record_failure(record, PolicyError(...), t)` with a message that names
   the embodiment and the offending value class, mirroring the dimension
   message. Rationale for `PolicyError` rather than `SafetyAbort`: the policy
   emitted garbage, nothing reached hardware, and the trial is attributed
   exactly like a wrong-dimension action (recorded as an errored trial,
   `fail_on_error` semantics apply, the eval continues to the next trial).
   Because this runs before `approver.review()`, no approver ever sees a
   non-finite value, so the `ClampApprover` NaN path becomes defence in depth
   for direct `approver.review()` callers only.
2. **Post-review, safety-attributed.** Immediately before `embodiment.step()`,
   if the reviewed action is non-finite, raise
   `_record_failure(record, SafetyAbort(...), t)`. The only way to get here
   is an approver that introduced the value (a contributed guardrail with a
   bug), and the repo invariant is that `SafetyAbort` halts the eval. The
   message names the approver class.

Both checks use `np.isfinite(...).all()` on `action.data` (already an
`np.ndarray` per the `Action` contract); no dtype conversion, no new deps.

`_policy_error()` is not used for check 1 because there is no underlying
exception to wrap; construct `PolicyError` directly as the dimension check
does.

## Tests

`tests/test_rollout_hardening.py`, after `test_wrong_dim_action_attributed_to_policy`:

- `test_non_finite_action_attributed_to_policy`, parametrized over
  `nan`, `inf`, `-inf`: a policy modelled on `_WrongDimPolicy` that emits the
  value in one dimension; assert `pytest.raises(PolicyError)`,
  `record.status == "error"`, the error message mentions "non-finite", and a
  spy on `embodiment.step` was never called.
- `test_approver_introduced_non_finite_action_is_a_safety_abort`: an
  approver whose `review()` returns `replace(action, data=<nan array>)`;
  assert `pytest.raises(SafetyAbort)`, `step` never called.
- One `eval()`-level test (in `tests/test_eval_action_log.py`, replacing
  `test_non_finite_action_degrades_without_affecting_eval` at line 306, which
  currently asserts `log.status == "success"` for a NaN action and therefore
  codifies the bug): the trial is recorded as errored with the `PolicyError`
  text in the sample, no `RuntimeWarning` about the action log is emitted
  (the NaN never reaches `sink.log_step`), and the eval still returns a log.
  The sibling test at line 340 that exercises the JSONL writer directly
  stays as is.

Coverage stays at 100% (both new branches are exercised).

## Docs

- `rollout()` docstring: the paragraph listing when `PolicyError` is raised
  gains "or a non-finite action"; the `SafetyAbort` sentence gains the
  approver-introduced case.
- `src/inspect_robots/CLAUDE.md`: the invariants list gets one line: the
  rollout rejects non-finite actions before review and before `step()`.
- `src/inspect_robots/approver.py`, `GuardrailContribution` docstring: note
  that the rollout now enforces finiteness, so contributors need not.
- `CHANGELOG.md` under `## [Unreleased]` / `### Fixed`, `**Core:**` prefix,
  referencing #356. Mention that an approver-introduced non-finite action is
  now a `SafetyAbort`.

## Out of scope

Validating observations or embodiment state for finiteness; changing
`AutoApprover`; any change to the mock's `step()`.
