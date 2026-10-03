# Grader preflight and loud ungraded trials

PR: [#472](https://github.com/robocurve/inspect-robots/pull/472) (reworked)

## Outcome

A misconfigured VLM grader fails **before the robot moves**, and a grading
failure during a run can no longer hide inside an ordinary-looking result:

1. **Preflight.** Before any rollout, `eval()` sends one tiny grading request
   with the run's exact grader configuration. An HTTP 4xx rejection (bad model,
   bad key, unsupported `effort`, no image support) raises `ConfigError`
   immediately.
2. **Loud ungraded trials.** A grading failure after a rollout no longer stops
   the run and no longer scores as a failure. The trial is recorded as
   ungraded with its reason, the `operator` scorer abstains on it
   (`Score(value=None)`, from #436/#479), and the run ends with
   `status == "error"` and a message such as
   `3 of 20 trial(s) ungraded: grader failed (HTTP 400: ...)`.

## Why

Two failure modes today:

- **Silent skew.** On `main`, a VLM grading failure prints one stderr note and
  leaves `operator_judgement` unset. The `operator` scorer then returns
  `Score(False, "no operator judgement recorded")`
  (`src/inspect_robots/scorer.py:279-280`), so a grading outage silently counts
  every affected trial as a robot **failure**, and the run reports `success`.
- **Late discovery.** A rejected explicit `effort` (or any request-shape error)
  is only discovered after the first rollout, and then repeats on every trial.

The original #472 stopped the whole evaluation on the first 400/422 from an
explicit-effort request. The maintainer decided (2026-10-03) on a better split:
catch configuration errors before any robot time is spent, and never abort a
session mid-run over grading, because robot trials are the expensive resource
and a mid-run 400 is often content-specific (one oversized image, a provider
refusal) rather than configuration.

## Design

### 1. Distinguishing failure kinds in `_chatwire.py`

`chat_completion` currently raises plain `ConfigError` for every failure,
including network errors (`_urllib_post` wraps `URLError`). Add two private
subclasses so callers can tell them apart without parsing prose:

- `_ChatHTTPError(ConfigError)` with a `status: int` attribute, raised for any
  non-2xx response (same message as today).
- `_ChatTransportError(ConfigError)`, raised by `_urllib_post` for `URLError`
  (same message as today).

Both remain `ConfigError`, so every existing caller and `except ConfigError`
behaves exactly as before. Malformed 2xx replies keep raising plain
`ConfigError`.

### 2. Preflight (optional duck-typed grader hook)

- `Grader` protocol docstring documents an **optional** `preflight()` method,
  duck-typed like the existing optional `config()` hook
  (`eval.py:_grader_identity`), so out-of-tree graders without it keep
  satisfying the protocol.
- `eval()` calls `preflight()` once, right after `_grading_hook` resolves the
  grader and **before** resolving the task, policy and embodiment
  (`eval.py:392`), so no hardware is touched when it fails. `eval_set()`
  resolves its grader once and calls `preflight()` once, before its first task.
  `_VLMGrader` memoizes success (`self._preflight_done`), so the per-task
  `eval()` calls inside `eval_set` do not repeat the request.
- `_VLMGrader.preflight()` sends one `chat_completion` with the same
  `base_url`, `api_key`, `model`, `effort` and `http_post`, `what="grading
  preflight"`, and a minimal user message: one short text part asking for
  `GRADE: success`, plus one 1x1 PNG `image_url` part (via `png_data_url`), so
  a model without image input is rejected here rather than on every trial. The
  reply's content is not validated; a 2xx with a parseable reply passes.
- Outcomes:
  - `_ChatHTTPError` with `400 <= status < 500`: raise `ConfigError` with the
    original message plus the requested effort (when explicit) and
    `fix: check -G model, -G effort, the API key and that the model accepts
    images; this request was rejected before any rollout`.
  - `_ChatHTTPError` with 5xx, or `_ChatTransportError`: do not block the
    session. Print one stderr warning (`vlm grader preflight: ...; continuing,
    trials may end up ungraded`) and continue; mid-run failures are then
    handled by section 3. Not memoized, so a later `eval()` retries it.
  - Malformed 2xx reply (plain `ConfigError`): raise; the endpoint is not
    OpenAI-compatible.
- The builtin `operator` grader has no `preflight()`.

### 3. Recording an ungraded trial

`_VLMGrader.grade` keeps its "never raise after a rollout" contract, but no
longer leaves the record unchanged on failure. In its existing `except
Exception` branch it sets `record.metadata["grading_error"]` to a bounded
string (`f"{type(exc).__name__}: {exc}"`, first 500 characters) and keeps the
stderr note. Successful grading never sets the key. Trials adopted from a
console verdict or a definitive termination are unchanged.

`TrialRecord.metadata` already persists into `SceneResult` per-trial metadata
in the saved log (verify the exact field during implementation and test the
round trip).

### 4. Scoring an ungraded trial

`_OperatorScorer.__call__`: when `operator_judgement is None` **and**
`record.metadata.get("grading_error")` is set, return
`Score(value=None, explanation=f"ungraded: grader failed: {reason}")`, an
abstention. With no grading error, behavior is unchanged
(`Score(False, "no operator judgement recorded")`), so a human operator who
skips a trial or an unattended run with no grader still scores as today.

Abstentions are already excluded from reducers and metrics and counted in
`EvalResults.abstentions` (#479), so success rates are computed over graded
trials only and the count is shown beside the metric in `inspect` and `view`.

### 5. Failing the run

In `_run_eval`, after `before_scoring(record, scene)` returns, count trials
whose record carries `grading_error`, and keep the first reason. After all
scenes, when the run would otherwise end with `status == "success"` and at
least one trial is ungraded, set `status = "error"` and
`error = f"{n} of {graded_attempts} trial(s) ungraded: grader failed ({first_reason})"`,
where `graded_attempts` is the number of trials `before_scoring` ran for.
This does not count toward `fail_on_error` (grading is not a policy error) and
never interrupts the loop. An existing more specific error (halt, all trials
errored, fail_on_error) is not overwritten.

The CLI already prints a failed run's `error` and exits nonzero, and `eval_set`
already reports a task with `status == "error"` as failed.

### 6. What is dropped from the original #472

The mid-run stop, `_ExplicitEffortRejected`, and deferring a raise past
`on_eval_end` are removed. The branch is rebuilt from `main`; the original
diff stays visible in the PR history. `effort` handling on `main` is already
exact (`"none"` is sent verbatim; `grader.py` `vlm_grader`), so nothing from
that part of #472 is still needed beyond fixing the stale "requests the
minimum" wording in `vlm_grader`'s docstring and `docs/guide/cli.md`.

## Docs

- `docs/guide/cli.md`, "Automated grading: `--grader vlm`": describe the
  preflight (one request before the robot moves; 4xx stops the run, 5xx or
  network warns and continues) and the new ungraded behavior (abstains, run
  ends in error with a count). Replace the two "leaves trials ungraded with a
  stderr note" sentences and the "An unjudged trial honestly scores as failure"
  sentence so it distinguishes a skipped judgement (failure, unchanged) from a
  grader failure (abstention).
- `docs/guide/scoring.md`: one sentence linking ungraded VLM trials to
  abstention.
- Docstrings: `Grader` protocol (optional `preflight()`, grading_error marker),
  `_VLMGrader.grade`, `vlm_grader`, `_OperatorScorer`/`operator_scorer`.
- Writing-style rule from `AGENTS.md`: no em dashes in new prose.
- Changelog fragments: `+grader-preflight.added.md` and
  `+ungraded-trials-abstain.changed.md`.

## Files

```
src/inspect_robots/_chatwire.py      # _ChatHTTPError, _ChatTransportError
src/inspect_robots/grader.py         # preflight(), grading_error marker, docstrings
src/inspect_robots/scorer.py         # operator scorer abstains on grading_error
src/inspect_robots/eval.py           # call preflight; count ungraded; fail run
tests/test_vlm_grader.py             # preflight + grading_error tests
tests/test_eval_orchestration.py     # run status, abstention, eval_set preflight once
tests/test_scorers.py                # operator scorer abstention
docs/guide/cli.md, docs/guide/scoring.md
changelog.d/+grader-preflight.added.md
changelog.d/+ungraded-trials-abstain.changed.md
plans/0085-grader-preflight-and-loud-ungraded.md
```

## Tests (mocked `http_post`, no network)

- Preflight: 400 and 422 with explicit effort raise `ConfigError` before any
  rollout (assert the embodiment was never constructed or reset and no trial
  ran); 401/403/404 raise too; 500 and a transport error warn and the run
  proceeds; malformed 2xx raises; success memoizes (one request across an
  `eval_set` of two tasks); the request carries the exact effort and an image
  part; graders without `preflight` work unchanged.
- Grading failure mid-run: the record gets `grading_error`; the operator
  scorer abstains with the reason; metrics exclude the trial and
  `abstentions` counts it; the run ends `status == "error"` with the
  `n of m trial(s) ungraded` message; later trials still run and grade.
- Unchanged paths: a skipped human verdict and a no-grader run still score
  `False`; a definitive termination is adopted without a request; a more
  specific run error is not overwritten.
- Full gates: ruff, ruff format, strict mypy over src and tests, pytest at
  100% coverage.
