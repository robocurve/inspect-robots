# 0021 — Mandatory per-tool-call notes: see what the agent sees and decides

Issue: #130. Status: draft.

## Problem

The persisted agent transcript (plan 0015) records observations, tool calls,
and tool results, but nothing records what the model believed it was seeing or
why it chose an action. A user replaying a trial sees `move_by({"deltas":
{"x": 0.05}})` and has to guess the intent. Users running evals explicitly
want that visibility: what did the model see, what did it decide, why.

The model *could* volunteer assistant text before each tool call, but in
practice tool-trained models often emit bare tool calls, and nothing today
tells them anyone is reading. Visibility that depends on the model's mood is
not visibility.

## Design

Make the note a **mandatory argument of the move tool**, enforced the same way
every other argument mistake is enforced: a structured tool error the model
must correct. No core changes; the notes ride inside the tool-call arguments,
so they land in the persisted policy transcript, `inspect-robots inspect
--transcript`, the stderr echo (plan 0019), and the Rerun transcript stream
(plan 0020) with zero schema or plumbing changes.

Three coordinated edits, all in the agent plugin:

### 1. Tool schema (`_tools.py`, `Toolset.schemas()`)

The move tool (whichever of `move_joints`/`move_to`/`move_by` was built) gains
a `note` property and lists it in `required` alongside the values key:

```json
"note": {
  "type": "string",
  "description": "What you observe right now in the observation (images, if
    any, and state), and why you chose this motion. The user reads these
    notes live and in the saved transcript to follow what you see and what
    you decide. Write for them, in one or two plain sentences."
}
```

(The wrapping above is presentational; the wire string is a single line.
Camera-less embodiments exist, hence "if any".)

(The `required` array carrying both the values key and `note` is standard
JSON Schema; OpenAI-compatible endpoints accept multi-entry `required`
lists. "Mandatory" is expressed by `required`, not repeated in the
description.)

Naming note: `ToolResult.note` (the success confirmation string sent back to
the model) is an unrelated, pre-existing field. The implementation keeps both
but disambiguates in docstrings ("the call's note argument" vs "the result
note").

`done` and `give_up` already carry mandatory free-text fields (`summary` /
`reason`) that serve the same purpose at trial end; they are unchanged.

### 2. Enforcement (`_tools.py`, `Toolset._move()`)

A move call whose `note` is missing, not a string, or empty/whitespace-only
returns `ToolResult(error="note is required: describe what you observe and
why you chose this motion")`. This reuses the existing correction loop in
`policy.py` (`failures` counter, `_MAX_CONSECUTIVE_FAILURES`); no new control
flow.

Placement, precisely: the existing broken-sensor guard in `_move()`
(non-finite proprioceptive reference **raises** before any argument
validation, so a malformed call can never mask a dead sensor) stays first,
and it lives inside the `if self._absolute:` block. The note check goes
**after that block, unconditionally** — it must fire for displacement mode
(`move_by`) exactly as for absolute mode — and before the `values`
validation. Error strings stay single-issue like every existing `ToolResult`
error; the note error leads because the note is the new contract and the
most likely omission from a model whose training predates it. Worst case, a
call wrong in both ways costs two correction turns (2 LLM calls, 2 of the 3
consecutive-failure budget); acceptable, and simpler than a combined
multi-error message that no existing error path uses (every current
`ToolResult` error reports a single problem).

The note is a sibling of `targets`/`deltas` in the arguments object, so it
never enters the per-dimension `values` loop; no motion code changes.

### 3. System prompt (`policy.py`, `_SYSTEM_TEMPLATE`)

One added sentence, stating the *reason* (the user wants to see through the
model's eyes) and the *contract* (every move carries a note):

> Every move tool call must include a `note`: in one or two sentences, say
> what you observe in the current observation and why you chose this
> motion. The user is watching these notes to see what you see and what you
> decide, so write them for a human reader.

The `reset()` code path is untouched (the template already flows through
`str.format`; the new sentence has no placeholders).

## What is deliberately NOT changed

- **No new echo line.** The plan-0019 echo already prints every tool call
  verbatim as `[agent] << tool_call move_by({...})`; the note is inside the
  arguments and therefore already visible live. Adding a second, prettified
  line for the same data would duplicate plan 0019's one-line-per-message
  contract.
- **No `ToolResult` change.** The note travels in the request arguments, not
  the result; the tool-result message the LLM sees back stays the compact
  `executing … over N steps` confirmation.
- **No note on `done`/`give_up`.** Their existing mandatory `summary`/`reason`
  fields are the note; requiring a second free-text field would be noise.
  Their empty-string laxness (`str(arguments.get("summary") or ...)`) is
  pre-existing and out of scope.
- **`_forced_give_up()` untouched.** The synthetic budget-exhaustion call is
  plugin-generated, not model output, and never enters the transcript.
- **No core/log/CLI changes.** Downstream rendering (the HTML viewer, planned
  separately) can extract `note` from tool-call arguments generically.

## Compatibility

- Old transcripts (no notes) still render everywhere; nothing reads the field
  by name in this plan.
- The stricter schema is announced in both the tool description and the
  system prompt, and enforcement produces a self-explanatory correction
  error, but this **does change eval outcomes for models that persistently
  omit the note**: three consecutive note-less calls hit
  `_MAX_CONSECUTIVE_FAILURES` and the trial errors (unscored `PolicyError`),
  exactly like any other persistently malformed call. Each correction turn
  also burns one `max_llm_calls` unit, and a budget exhausted mid-correction
  ends the trial as a scored forced `give_up` instead. This is the intended
  trade: notes are a hard contract, and a model that cannot follow the tool
  schema after two corrections is already failing the existing contract.
- Agent plugin version: `0.9.0 → 0.10.0` (behavioral change to the tool
  surface).

## Tests (plugin suite, `plugins/inspect-robots-agent/tests/`)

`test_tools_motion.py`:
- `schemas()`: move tool declares `note` as a string property and its
  `required` list is exactly `[values_key, "note"]`; `done`/`give_up`
  schemas unchanged. (`test_schemas_match_control_mode_and_remove_duration`
  asserts the exact `required` lists today and is updated in place.)
- `execute()` on a move call with `note` missing → structured error, no
  chunk — asserted on **both** an absolute toolset and a displacement
  toolset, so the unconditional placement can't regress to
  absolute-only.
- Same for `note: ""`, `note: "   "`, and non-string `note` (e.g. `42`).
- A valid note leaves the produced chunk byte-identical to today's (motion
  math untouched); the note never counts as an unknown dimension.
- Broken-sensor ordering: non-finite reference still raises even when the
  same call also omits the note.
- Mechanical churn, named so it can be audited: the `_call` /
  `_execute_absolute` helpers gain a default note (single edit point for
  most fixtures); `test_tool_errors_are_messages_not_exceptions` asserts an
  unknown-dimension error mentioning `left_elbow` on a call that would now
  fail the note check first, so its fixture gains a note to keep testing
  what it names; `test_stray_duration_key_is_ignored` likewise.

`test_policy_e2e.py`:
- System prompt contains the note contract sentence.
- End-to-end with the mock transport: a first response missing the note gets
  the correction error as a tool message and a second, corrected response
  succeeds; the persisted transcript contains the note text inside the
  tool-call arguments.
- `_tool_response` **and** `_multi_tool_response` payloads gain notes.
  Two tests are re-checked against silent drift:
  `test_llm_call_budget_forces_give_up` must keep sending valid noted calls
  (or it silently tests note correction instead of the budget path), and
  `test_extra_tool_calls_are_answered_but_not_executed` builds its executed
  first call via `_multi_tool_response` — without a note there, its
  assertions all still pass while it stops exercising "the first call
  executes".

Gates (what actually blocks): ruff, `mypy` on the plugin src, plugin pytest
green, and the core suite untouched/green. CI runs plugin coverage at
`--cov-fail-under=0` (visibility only, no 100% gate for plugins); this
change keeps the plugin's report at 100% anyway since every new branch is
enumerated above.

## Docs

- `plugins/inspect-robots-agent/README.md`: the "How it works" tool-surface
  description gains the `note` argument (what it is, that it is required,
  and why: the user reads it), and the transcript-echo prose gains one
  sentence noting that notes appear inside the echoed tool-call arguments
  (the section is prose only today; this is an addition, not an edit of an
  example line). Follows the repo's public-text style rules (no em dashes,
  no mid-sentence bold).
- `CHANGELOG.md`: an entry under the agent plugin for the new mandatory
  note argument (behavioral change to the tool surface).

## Rollout

Single PR (`Closes #130`), agent plugin only. **Before** cutting the
release, audit downstream embodiment docs (`EmbodimentInfo.docs`
cheat-sheets, e.g. the yam plugin's) for example tool calls that lack a
note: such examples would show the model a counterexample inside the same
system prompt, and rigs install from PyPI the moment the release lands, so
a post-release audit ships the contradiction for a whole release window.
Then release `inspect-robots-agent 0.10.0` with the next core release cut
immediately after merge.
