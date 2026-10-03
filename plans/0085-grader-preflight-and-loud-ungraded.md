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
  (same message as today) and also for `OSError` (including `TimeoutError` on
  `response.read()`) and `http.client.HTTPException` (e.g.
  `RemoteDisconnected`), which today escape as raw exceptions. Handler order:
  `HTTPError` first, then `URLError`, then `OSError`/`HTTPException` (`URLError`
  subclasses `OSError`). `exc.read()` inside the `HTTPError` branch can itself
  raise `OSError` or `http.client.IncompleteRead`; then the branch returns
  `(exc.code, b"")`, so the status still becomes a `_ChatHTTPError` and a 4xx
  keeps its meaning. New
  messages keep the `chat request failed:` prefix so taskgen's rewording regex
  (`taskgen.py:221-232`) still applies.

Both remain `ConfigError`, so every existing caller and `except ConfigError`
behaves exactly as before. Malformed 2xx replies keep raising plain
`ConfigError`.

### 2. Preflight (optional duck-typed grader hook)

- `Grader` protocol docstring documents an **optional** `preflight()` method,
  duck-typed like the existing optional `config()` hook
  (`eval.py:_grader_identity`), so out-of-tree graders without it keep
  satisfying the protocol.
- **CLI (the main `--grader vlm` path).** `run` (`cli.py:~1686`) and
  `eval-set` (`cli.py:~1922`) resolve the embodiment and policy (robot
  connection, claim, weights), run auto-task generation (an LLM call), and
  start sessions before reaching `eval()`. So both commands build the grader
  **first**, with `operator_session=None`, call `preflight()` immediately,
  and only then resolve components, generate tasks, and start sessions,
  attaching the session afterwards via the grader's existing
  `connect_session` (already used by `_build_grader`). A preflight
  `ConfigError` is converted with `raise SystemExit(str(exc)) from exc`, the
  same as other grader config errors (`_resolve_or_exit`, `cli.py:738-740`):
  a clean message and nonzero exit, no traceback, no embodiment constructed.
- **Python API.** `eval()` calls `preflight()` right after `_grading_hook`
  resolves the grader and before resolving string components
  (`eval.py:392`). `eval_set()` calls it after the empty-task-list check
  (`eval.py:~1005`) and **before** the task loop, outside the per-task
  `try/except Exception` (`eval.py:1009-1047`), so a preflight `ConfigError`
  propagates out of `eval_set` instead of becoming the first task's error log
  while later tasks run.
- **Once per grader object, outcome cached.** `_VLMGrader` caches the
  preflight outcome: passed or warned (no further requests), or the raised
  `ConfigError`, which every later `preflight()` call re-raises. A cached
  "warned" outcome does not print the warning again. So the CLI,
  `eval_set` and each per-task `eval()` make at most one request per grader,
  and a caller that catches the error and calls `eval()` again with the same
  grader is still stopped.
- **Contract for other graders.** The `Grader` docstring states that an
  optional `preflight()` must be idempotent and cheap to call repeatedly
  (callers may invoke it from the CLI, `eval_set` and every `eval()`), and
  should cache its own outcome as the builtin does.
- `_VLMGrader.preflight()` sends one `chat_completion` with the same
  `base_url`, `api_key`, `model`, `effort` and `http_post`, `what="grading
  preflight"`, and a minimal user message: one short text part asking for
  `GRADE: success`, plus one 64x64 solid-color PNG `image_url` part (via
  `png_data_url`; large enough to avoid "image too small" rejections), so a
  model without image input is rejected here rather than on every trial. The
  reply's content is not validated; a 2xx with a parseable reply passes.
- Outcomes:
  - The preflight passes its own `fix_hint` naming `-G` flags, and the raised
    error flattens the original message (dropping its `fix:` line), so the user
    sees exactly one `fix:` line.
  - `_ChatHTTPError` with `400 <= status < 500`, except 408 and 429: raise
    `ConfigError` with the
    original message plus the requested effort (when explicit) and
    `fix: check -G model, -G effort, the API key and that the model accepts
    images; this request was rejected before any rollout`.
  - `_ChatHTTPError` with 5xx, 408 or 429, `_ChatTransportError`, or any
    exception that is not a `ConfigError` (an injected `http_post` can raise
    anything): do not block the session. Print one stderr warning (`vlm grader preflight: ...; continuing,
    trials may end up ungraded`) and continue; mid-run failures are then
    handled by section 3.
  - Malformed 2xx reply (plain `ConfigError`): raise; the endpoint is not
    OpenAI-compatible.
- The builtin `operator` grader has no `preflight()`.

### 3. Recording an ungraded trial

`_VLMGrader.grade` keeps its "never raise after a rollout" contract, but no
longer leaves the record unchanged on failure. In its existing `except
Exception` branch it sets `record.metadata["grading_error"]` to a one-line
reason, capped at 500 characters: `str(exc)` with its `fix:` lines dropped and
all whitespace collapsed (provider error bodies are often pretty-printed
JSON), e.g. `grading request failed with HTTP 400: { "error": ... }`; for an
exception that is not a `ConfigError`, prefixed with its class name. The same
flattening applies to the preflight warning and error. It keeps the stderr
note. Successful grading never sets the key. Trials adopted from a
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
whose record carries `grading_error` **and** has no `operator_judgement`
(matching when the scorer abstains), and keep the first reason. Scene-level
status is deliberately left `success`: the trials ran and were scored (as
abstentions); the run-level error carries the count. After all
scenes, when the run would otherwise end with `status == "success"` and at
least one trial is ungraded, set `status = "error"` and
`error = f"{n} of {graded_attempts} trial(s) ungraded: grader failed ({first_reason})"`,
where `graded_attempts` is the number of trials `before_scoring` ran for.
This does not count toward `fail_on_error` (grading is not a policy error) and
never interrupts the loop. When the run already ended with a more specific
error (halt, all trials errored, fail_on_error, a reducer failure such as
`pass_at_k` with fewer graded epochs than k after abstentions), that status
and message stay, and `; N of M trial(s) ungraded` is appended so the count is
not lost (just `N of M trial(s) ungraded`, no leading separator, when the
existing `error` is empty, e.g. some cancelled runs).

The CLI already prints a failed run's `error` and exits nonzero, and `eval_set`
already reports a task with `status == "error"` as failed. The `eval-set`
summary row (`_print_eval_set_summary`, `cli.py:~1886`) currently shows
metrics *or* the error; an ungraded task still has metrics, so print the error
next to the metrics whenever `log.status != "success"` and `log.error` is set.

### 6. What is dropped from the original #472

The mid-run stop, `_ExplicitEffortRejected`, and deferring a raise past
`on_eval_end` are removed. The branch is rebuilt from `main`; the original
diff stays visible in the PR history. `effort` handling on `main` is already
exact (`"none"` is sent verbatim; `grader.py` `vlm_grader`), so nothing from
that part of #472 is still needed beyond fixing the stale "requests the
minimum" wording: `vlm_grader`'s docstring, `_VLMGrader.config()`'s docstring
(`grader.py:~161`, "asks for the minimum"), and `docs/guide/cli.md` (around
lines 176, 191 and 436).

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
- Docstrings and comments: the `Grader` protocol docstring's "leaving the
  record unchanged" (`grader.py:36-37`) becomes "leaving the judgement unset;
  it may record `metadata["grading_error"]`", plus the optional idempotent
  `preflight()`; the inline comment at `grader.py:278-279` ("must leave the
  record unchanged, marker included") is rewritten; `_VLMGrader.grade`, `vlm_grader`,
  `_OperatorScorer`/`operator_scorer`, and the `eval()`/`eval_set()`
  docstrings (preflight before rollouts; ungraded trials fail the run).
- `src/inspect_robots/CLAUDE.md` module map (around line 19): grading failures
  no longer degrade with only a stderr note.
- Writing-style rule from `AGENTS.md`: no em dashes in new prose.
- Changelog fragments: `+grader-preflight.added.md` and
  `+ungraded-trials-abstain.changed.md`.

## Files

```
src/inspect_robots/_chatwire.py      # _ChatHTTPError, _ChatTransportError
src/inspect_robots/grader.py         # preflight(), grading_error marker, docstrings
src/inspect_robots/scorer.py         # operator scorer abstains on grading_error
src/inspect_robots/eval.py           # call preflight; count ungraded; fail run
src/inspect_robots/cli.py            # build grader first; preflight before components
tests/test_registry_cli.py           # CLI preflight: clean exit, no embodiment built
tests/test_vlm_grader.py             # preflight + grading_error tests
tests/test_eval_orchestration.py     # run status, abstention, eval_set preflight once
tests/test_scorers.py                # operator scorer abstention
tests/test_grader_config.py          # existing http_post doubles see the preflight request
tests/test_chatwire.py               # new transport and HTTPError-read branches
src/inspect_robots/CLAUDE.md         # module map wording
docs/guide/cli.md, docs/guide/scoring.md
changelog.d/+grader-preflight.added.md
changelog.d/+ungraded-trials-abstain.changed.md
plans/0085-grader-preflight-and-loud-ungraded.md
```

## Tests (mocked `http_post`, no network)

Test conventions:

- Preflight adds one request before the first grading request whenever a VLM
  grader goes through `eval()`. Existing doubles that count or index requests
  (`tests/test_vlm_grader.py:127` asserts two requests;
  `tests/test_grader_config.py:96,113,126,151-153,182` index `post.bodies[0]`)
  are updated to expect the preflight request first, identified by a fixed
  marker in its text part (a module constant such as `_PREFLIGHT_PROMPT`),
  or to filter it out with a small shared helper.
- CLI preflight tests cannot pass `http_post` through `-G`; they monkeypatch
  `inspect_robots._chatwire._urllib_post` (looked up at call time).
- The real `_urllib_post` branches are covered in `tests/test_chatwire.py` by
  monkeypatching `urllib.request.urlopen` (pattern at
  `tests/test_chatwire.py:58-78`): one test each for `TimeoutError` from
  `response.read()`, `http.client.RemoteDisconnected`, and `exc.read()`
  raising `IncompleteRead` inside the `HTTPError` branch, asserting the class
  (`_ChatTransportError` / `_ChatHTTPError`), `status`, and the
  `chat request failed:` prefix.


- CLI: `run` and `eval-set` with a preflight 400 exit nonzero with the guided
  message and no traceback, and the embodiment factory and auto-task generation
  are never called; a 503 warns and the run proceeds.
- `eval_set` with a preflight 400: raises before the first task; no task runs
  and no embodiment reset happens. A caller that catches it and calls `eval()`
  with the same grader gets the cached error with no new request.
- Preflight: 400 and 422 with explicit effort raise `ConfigError` before any
  rollout (assert the embodiment was never constructed or reset and no trial
  ran); 401/403/404 raise too; 500 and a transport error warn and the run
  proceeds; 408 and 429 warn and proceed; a raw `TimeoutError` from the
  transport warns and proceeds; malformed 2xx raises; one request per grader
  across an `eval_set` of two tasks, whether the first attempt succeeded or
  only warned; the request carries the exact effort and an image
  part; graders without `preflight` work unchanged.
- Grading failure mid-run: the record gets `grading_error`; the operator
  scorer abstains with the reason; metrics exclude the trial and
  `abstentions` counts it; the run ends `status == "error"` with the
  `n of m trial(s) ungraded` message; later trials still run and grade.
- A halted run with ungraded trials keeps the halt message with
  `; N of M trial(s) ungraded` appended; `pass_at_k` with partial abstention (k=2 over
  3 epochs, 2 ungraded) degrades to the existing reducer-failure error plus
  the count.
- The exact run error string for an HTTP 400 mid-run is
  `1 of N trial(s) ungraded: grader failed (grading request failed with HTTP 400: ...)`,
  single line, no class name, no `fix:` line.
- `eval-set` CLI summary prints the ungraded error next to the metrics.
- Unchanged paths: a skipped human verdict and a no-grader run still score
  `False`; a definitive termination is adopted without a request; a more
  specific run error is not overwritten.
- Full gates: ruff, ruff format, strict mypy over src and tests, pytest at
  100% coverage.
