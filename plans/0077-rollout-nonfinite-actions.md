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
same on ±inf (#376), but that protection is configuration-dependent: the CLI
falls back to `AutoApprover` when the space declares no bounds,
`DeltaLimitApprover` refuses, and no `contribute_guardrails` approver was
added. The `GuardrailContribution` docstring only asks contributors to
validate.

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

Both checks call one private helper, `_non_finite_detail(data) -> str |
None`, which owns the coercion `np.asarray(data, dtype=np.float64)` (the
same expression `approver.py` and `controller.py` use bare) inside a `try`
catching `(TypeError, ValueError)` (arbitrary objects raise `TypeError`;
string or ragged arrays raise `ValueError`; note `None` coerces to `nan`
rather than raising) and then `np.isfinite(...).all()`. It returns a short
reason string ("contains nan", "contains inf", "is not numeric: <exc>") or
`None`. One code path, two attributions, and the single policy-side test of
the `except` covers it for both checks. Messages: check 1
`"policy emitted a non-finite action ({detail}) for embodiment {name!r}"`,
check 2 `"approver {cls} returned a non-finite action ({detail})"`; tests
pin the substring `"non-finite"`. No new deps.

**User-visible semantics change, to be stated in the CHANGELOG and PR
body.** On the default CLI chain a NaN action today halts the whole eval
(`ClampApprover.review` raises `SafetyAbort`). After check 1 the same run
records one errored trial and, under the default `fail_on_error=False`,
advances to the next scene, exactly as a wrong-dimension action does today.
Rationale: the value never reached hardware, so it is a malformed action
(policy bug), not a safety event; `SafetyAbort` stays reserved for the
approver-introduced case and for approvers rejecting values that *are*
finite but unsafe. Operators who want stop-on-first-error keep it via
`fail_on_error=True` (or a threshold). `ClampApprover`'s NaN branch and
PR #376 remain useful for direct `approver.review()` callers and are not
superseded.

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
- `test_non_numeric_action_attributed_to_policy`, parametrized over
  `np.array([object(), 1.0], dtype=object)` (`TypeError`) and
  `np.array(["a", "b"])` (`ValueError`): a `PolicyError`, never a bare
  `TypeError`/`ValueError`. `Action(data=...)` needs
  `# type: ignore[arg-type]` because `StateArray` is
  `npt.NDArray[np.floating[Any]]` and mypy strict covers tests.
- `test_approver_introduced_non_finite_action_is_a_safety_abort`: an
  approver whose `review()` returns `replace(action, data=<nan array>)`;
  assert `pytest.raises(SafetyAbort)`, `step` never called.

`tests/test_eval_action_log.py`: replace
`test_non_finite_action_degrades_without_affecting_eval` (line 306, which
asserts `log.status == "success"` for a NaN action and therefore codifies
the bug) with a test that the trial is recorded as errored: with one trial
`log.status == "error"` and `log.error` is the all-trials-errored text, the
sample's `error` starts with `"PolicyError:"` and contains `"non-finite"`;
`_NaNTolerantEmbodiment` becomes dead scaffolding, so use plain
`CubePickEmbodiment` plus a `step` spy; that the action side-car is the
header-only JSONL with `record.metadata["actions"]` set (the error is at
`t=0`, so `record.steps` is empty; see
`test_start_failed_trial_writes_header_only_action_log`); and that no
`RuntimeWarning` is emitted, asserted explicitly with
`warnings.catch_warnings(record=True)` + `simplefilter("always")` since
pyproject has no `filterwarnings = error`. The sibling test at line 340 that
exercises the JSONL writer directly stays as is.

`tests/test_strict_json.py`: `test_nan_action_halts_as_safety_abort_and_log_reaches_disk`
(lines 105-115) drives `_NaNPolicy` through `ClampApprover` and asserts
`"NaN" in log.error`; after check 1 that path is a `PolicyError` and the
run's `log.error` becomes the all-trials-errored text. Rewrite it to inject
the NaN from an approver (the check-2 `SafetyAbort` path) so its real
intent, a halted run still writes strict-parseable JSON, survives (assert
`"non-finite"` and the approver class name in `log.error` instead of
`"NaN"`), and fix the module docstring (lines 1-7) that describes the old
path. This is also
the eval-level test for check 2.

Coverage stays at 100% (every new branch, including the coercion `except`,
is exercised).

## Docs

- `rollout()` docstring: the paragraph listing when `PolicyError` is raised
  gains "or a non-finite action"; the `SafetyAbort` sentence gains the
  approver-introduced case.
- `src/inspect_robots/CLAUDE.md`: the invariants list gets one line: the
  rollout rejects non-finite actions before review and before `step()`.
- `src/inspect_robots/approver.py`, `GuardrailContribution` docstring: note
  that the rollout now enforces finiteness, so contributors need not.
- `docs/guide/logging-and-rerun.md:160-163` says "A non-finite action or
  filesystem failure emits a warning, leaves no final file or metadata
  pointer, and does not change the eval status." Reword: filesystem
  failures degrade with a warning; a non-finite policy action is rejected by
  the rollout before any step is recorded and errors the trial.
- `docs/guide/concepts.md:66-77` (error taxonomy, already says a crashing
  approver becomes a `SafetyAbort`): one sentence that a non-finite action
  from the policy is a `PolicyError` and one introduced by an approver is a
  `SafetyAbort`.
- `CHANGELOG.md` under `## [Unreleased]` / `### Fixed`, `**Core:**` prefix,
  ending `([plan 0077](plans/0077-rollout-nonfinite-actions.md), [#356](https://github.com/robocurve/inspect-robots/issues/356))`
  like the neighbouring entries. State both the fix and the semantics
  change (a NaN action on the default CLI chain now errors the trial and
  continues under `fail_on_error=False` instead of halting the eval), and
  that an approver-introduced non-finite action is a `SafetyAbort`.

## Out of scope

Validating observations or embodiment state for finiteness; changing
`AutoApprover`; any change to the mock's `step()`.
