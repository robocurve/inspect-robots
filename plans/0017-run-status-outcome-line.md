# 0017: `run status:` label and `outcome:` line in the CLI

Issue: [#122](https://github.com/robocurve/inspect-robots/issues/122)
Status: approved (critique rounds 1-3 applied; round 3 found no substantive design issues)

## 1. Problem

`_print_run_summary` and `_cmd_inspect` print `status: {log.status}`. The
value `success` means "the eval ran to completion without a halting error,
and at least one trial survived" (under the default `fail_on_error=False`,
per-trial `PolicyError`s are caught and the run stays `success` unless every
trial errored). It says nothing about the task verdict, which lives in the
metrics lines.

Operators misread it. A run whose only trial hit the step horizon, or whose
LLM agent exhausted its `max_llm_calls` budget and was forced into `give_up`,
prints a green `status: success` and exits 0 even though every scorer said
failure. `max_steps` truncations at least trigger the yellow step-limit note;
`give_up` (and every other policy-stop reason) surfaces nowhere in the
summary.

## 2. Non-goals

- No change to the on-disk `EvalLog` schema. `EvalLog.status` /
  `SceneResult.status` keep their `"success" | "error" | "cancelled"`
  (+ transient `"started"`) values; old logs stay readable and the Python API
  is untouched.
- No change to exit codes. For runs that reach the summary, `run` and
  `inspect` keep exiting 0 iff `log.status == "success"`; the cancellation
  path (`run` returns 130 before the summary prints, cli.py:768-774) is
  untouched.
- No invented task-success verdict. The `outcome:` line reports the
  termination reasons the log already records; it does not aggregate scorer
  values (the operator can score a truncated trial as a success, and the
  metrics lines already carry that).

## 3. Design

### 3a. `run status:` label, `completed` display value (options A+B)

In both `_print_run_summary` and `_cmd_inspect`, the label becomes
`run status:` and the displayed value maps through a tiny shared helper:

```python
_STATUS_DISPLAY = {"success": "completed"}

def _display_status(status: str) -> str:
    ...  # _STATUS_DISPLAY.get(status, status)
```

Unmapped values (`error`, `cancelled`, a transient or hand-edited `started`)
display as-is via the `dict.get(status, status)` fallback.

- Run summary: `run status: completed` (value green), `run status: error`
  (value red). Label styling stays exactly as today: cyan in the run summary
  only. Note `run status: cancelled` is unreachable here: `eval()` re-raises
  the interrupt and `_cmd_run` returns 130 before `_print_run_summary` runs;
  cancelled logs surface through `inspect`.
- Inspect view: labels in that block are plain unstyled prints and stay so
  (ANSI codes would also break the fixed-width alignment). The literal is
  `"run status:  "`: 11-char label + 2 spaces = the block's 13 columns
  (matching `task:        `, 13 chars).

### 3b. `outcome:` line (option E)

A new line aggregating per-trial termination reasons across
`log.samples[*].termination_reasons` (`read_eval_log` backfills `()` for logs
written before the field existed, so the line is simply omitted for them —
see 3c).

Reason-to-phrase mapping (`_OUTCOME_PHRASES`):

| recorded reason | phrase |
|---|---|
| `success` | `succeeded` |
| `failure` | `failed` |
| `max_steps` | `hit step limit` |
| `give_up` | `gave up` |
| `done` | `reported done` |
| `policy_stop` | `stopped by policy` (rollout's default when a stop request names no reason, rollout.py:259) |
| `truncated` | `truncated` |
| any other non-empty string | the raw reason string, unquoted (foreign text; see 3d) |
| `None` or `""` | `no reason recorded` |

(A policy that explicitly sets `stop_reason=None` records the literal string
`"None"` today — rollout.py:259 applies `str()` — and that renders raw via
the unmapped path. Pre-existing rollout behavior, accepted; not a bug in the
outcome line.)

`None` is deliberately neutral, not "did not finish": an adapter may
terminate a successful trial without naming a reason
(`StepResult.termination_reason` defaults to `None`), and errored/cancelled
trials also record `None`. Error visibility already comes from the
`(N errored)` trials suffix and the per-scene failure list, so the outcome
phrase must not guess. `""` (a policy setting `stop_reason=""`) folds into
the same phrase so no group renders blank.

Hand-edited logs: `EvalLog.from_dict` does no field validation, so
`termination_reasons` entries can be ints, bools, lists, or dicts. Every
non-`None` entry is coerced with `str()` before the phrase lookup (an int
`3` renders as the raw reason `3`), keeping the degrade-don't-crash norm the
inspect reader already follows for `max_steps` (cli.py:464-470). No
unhashable keys or mixed-type sorts can reach the aggregation.

Grouping key is the *phrase*, after mapping and coercion. A policy-authored
raw reason that collides with a mapped phrase (e.g. `stop_reason="gave up"`
alongside `give_up` trials) merges into one group; the line still routes
through the degraded printer because an unmapped raw reason was present
(see 3d).

Format: counts joined by commas, largest first, ties broken alphabetically by
phrase. The count prefix is dropped when the collected reasons list (not
`results.total_trials`, which can disagree in hand-edited logs) has exactly
one entry:

```text
outcome: 2 succeeded, 1 gave up, 1 hit step limit   # multiple reasons recorded
outcome: gave up                                    # exactly one reason recorded
```

`outcome: 2 no reason recorded` is accepted as slightly awkward English; it
is factual, rare, and must not be showcased in the docs examples (the docs
writing-style gate applies to `docs/guide/cli.md`).

Placement:

- Run summary: immediately after the `run status:` line (before the `error:`
  line and per-scene failure list), so the two lines read as a pair:
  machinery status, then what the trials did. Label cyan, value unstyled (a
  factual digest, not an alarm; the yellow step-limit note keeps the alarm
  role).
- Inspect view: `outcome:     ...` (8-char label + 5 spaces = 13 columns)
  directly under `run status:`, plain like its siblings.

### 3c. Omission rule

No recorded reasons (`termination_reasons` empty across all samples: an old
log, or a failure before any trial was recorded) → no `outcome:` line. An
all-`None` reasons list still prints (`outcome: 2 no reason recorded`): that
is signal, not absence of data. (A cancelled run normally does record its
partial trial's `None` reason, so it prints the line; only a cancel outside
rollout leaves reasons empty.)

### 3d. Foreign text

Unmapped reasons are foreign text from two sources: policy-authored
`action.meta["stop_reason"]` (a hostile model server writes this freely, so
it can contain lone UTF-16 surrogates that crash `print` on strict-UTF-8
stdout) and embodiment-authored `StepResult.termination_reason` (open
vocabulary; types.py names `"collision"` as an example). The raw reason is
kept verbatim (no `repr`-quoting, which would neutralize surrogates into
ASCII escapes and defeat the degradation path), and the whole outcome line is
emitted via `_print_degraded` whenever at least one unmapped reason is
present. `_print_degraded` is byte-identical to `print` for clean text, so
benign embodiment reasons like `collision` render normally. Lines built only
from mapped phrases keep plain `print`.

### 3e. Scene-level status vocabulary untouched

The per-scene markers (`[success]` / `[error]` / `[cancelled]` in both the
run summary's failure list and inspect's scenes block) keep the raw
`SceneSample.status` values. Only the top-level line gets the display
mapping; the mixed vocabulary (`run status: completed` above `[success] s0`)
is a deliberate scope cut, not a bug.

### 3f. Step-limit notice unchanged

`_print_step_limit_notice` stays as-is: it carries the `max_steps=N, ~Ss at
R Hz` parenthetical and the horizon-ownership hint, which the outcome digest
does not duplicate. The overlap is one count, and the note only fires when
the step limit was actually hit.

## 4. Touched files

Helper contract: `_outcome_line(log: EvalLog) -> tuple[str, bool] | None`
returns `(digest, has_unmapped)` or `None` when there is nothing to print
(3c). It does NOT print. Each call site formats its own label and padding
and picks `print` vs `_print_degraded` from `has_unmapped`, which is what
gives each call site its own omit-vs-print and degraded-vs-plain branches
(see tests 5-6).

```
src/inspect_robots/cli.py         # _display_status, _outcome_line, both call sites
tests/test_registry_cli.py        # updated assertions + new outcome cases
tests/test_string_resolution.py   # "status:      success" assertion at :44
docs/guide/cli.md                 # inspect example block; "status: error" wording
```

No new modules, no API surface change (`__all__` untouched), no schema bump.

## 5. Tests

Update existing assertions (`status: success` → `run status: completed`,
`status: error` → `run status: error`, the inspect padded variants in both
test files).

New cases, all through the public CLI entry points with the CubePick mock or
synthesized logs (existing `_transcript_log`-style fixtures):

1. Timeout run: summary shows `run status: completed` and
   `outcome: hit step limit` (single reason, no count).
2. Multi-trial mixed reasons: counts and ordering
   (`2 succeeded, 1 hit step limit`).
3. `give_up` truncation (synthesized record): `outcome: gave up`.
4. Errored trial (reason `None`): `outcome: no reason recorded`; the errored
   scene list and `(N errored)` suffix still print.
5. Unmapped foreign reason containing a lone surrogate: line prints via the
   degraded path (assert the U+FFFD replacement, mirroring the existing
   transcript-degradation test). Exercised through BOTH the run summary and
   inspect (branch coverage runs at 100%, and each call site carries its own
   degraded-vs-plain branch).
6. Old log with no `termination_reasons`: no `outcome:` line. Also exercised
   through BOTH call sites (omit-vs-print branch each).
7. Inspect view prints both `run status:` and `outcome:` with 13-column
   block padding; a cancelled log shows `run status:  cancelled` plus its
   partial trial's `outcome: no reason recorded`. NOTE: the existing
   `_transcript_log(status="cancelled")` fixture has `termination_reasons=()`
   (omission branch) and three epochs; this test needs a synthesized log with
   exactly one recorded trial whose reason is `None`, or the output is the
   counted plural form.
8. `started` (hand-edited/in-flight) status displays raw via the `.get`
   fallback.
9. Hand-edited log with non-string `termination_reasons` entries (int, bool,
   list): coerced via `str()`, no crash (mirrors the existing
   non-numeric-max_steps hand-edit test).
10. Empty-string reason folds into `no reason recorded`.
11. Phrase collision (`stop_reason="gave up"` next to `give_up`): one merged
    group, line still degraded-printed.

Gates: `ruff check`, `ruff format --check`, `mypy --strict` (src+tests),
`pytest --cov` at 100%.

## 6. Docs

- `docs/guide/cli.md`: update the inspect example block (add `run status:`
  and an `outcome:` line) and the `status: error` prose at line ~197. Add one
  sentence stating that `completed` is the display form of the log's
  `success` status value, so users grepping their JSON for `completed` are
  not stranded (the on-disk field and the Python API docs keep `"success"`).
- README has no CLI status output; its `log.status` Python example is
  unaffected.
