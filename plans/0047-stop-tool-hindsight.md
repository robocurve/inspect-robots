# Stop-tool hindsight Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** LLM agents relearn the same rig- and task-specific facts on every
rollout. Collect them at the terminal tool call: `done` and `give_up` gain a
required `hindsight` argument asking what the agent wishes it had known at
episode start, the system prompt announces the question up front so the model
tracks learnings during the rollout, and the answer persists in two
harvestable places (stop-action meta, and `trial_metadata` in the JSON log).
Feeding hindsight forward into future rollouts is explicitly out of scope.
Closes #305.

**Architecture:** the `hindsight` parameter is schema-required on both stop
tools but execution-lenient: `_stop` records it into the stop action's meta
as `stop_hindsight` only when present and non-blank, and never errors on its
absence, because the budget-exhausted `_forced_give_up` path synthesizes the
call with no LLM available to answer. The policy stashes the value when it
detects the stop chunk (the existing `request_stop` site), resets it per
trial, and `on_trial_end` writes `record.metadata["hindsight"]` next to the
existing `llm_usage`/`transcript` keys, which lands it in the sample's
`trial_metadata` in the JSON log. Both system-prompt templates gain one
sentence so the model knows the question is coming.

**Tech stack:** stdlib only, all inside `plugins/inspect-robots-agent`.
pytest with the existing fake-client patterns in
`plugins/inspect-robots-agent/tests/`.

## Global Constraints

- Gates (all blocking), run from the worktree root: `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`, `uv run pytest --cov` at
  **100% coverage**.
- Every public module/class/function needs a docstring stating the contract
  (ruff D1).
- Repo root is the `wt-ir-hindsight` worktree at
  `~/robocurve/wt-ir-hindsight`; run everything via `uv run ...` there.
- Existing tests pass untouched EXCEPT: tests that enumerate the stop tools'
  required parameters or snapshot the system prompt text may be updated
  mechanically to the new schema/text; list every such edit in the commit
  message. Everything else is off limits.
- The plugin version bump touches THREE files or CI fails (established
  release rule): `plugins/inspect-robots-agent/pyproject.toml` `version`,
  `uv.lock` (workspace member version is locked), and the hardcoded pin in
  `plugins/inspect-robots-agent/tests/test_package.py:9`. Bump 0.21.0 →
  0.22.0 (new tool argument = minor).
- Docs follow the repo writing rules (no em dashes in prose).
- Commit messages: imperative, scoped; reference #305.

## Reference: current wiring (main @ 01fd2659)

- `plugins/.../src/inspect_robots_agent/_tools.py:191-216` — `done` schema
  (required `summary`) and `give_up` schema (required `reason`).
- `_tools.py:265-266` — dispatch: both names route to `_stop`.
- `_tools.py:307-318` — `_stop`: builds the hold action with meta
  `{request_stop: True, stop_reason: name, stop_detail: summary-or-reason}`,
  returns `note=f"{name}: {detail}"`.
- `_tools.py` take_pic `note` validation (~321-325) — the existing pattern
  for an execution-level argument check; `hindsight` deliberately does NOT
  get one (leniency contract above).
- `policy.py:100-134` — `_SYSTEM_TEMPLATE` and `_ON_DEMAND_SYSTEM_TEMPLATE`;
  both end with "When the goal is achieved call done; if it cannot be
  achieved call give_up. You have a budget of {budget} LLM calls...".
- `policy.py:955` — the one site that detects the stop chunk:
  `stopped = bool(chunk.actions[0].meta.get("request_stop"))`.
- `policy.py:1011-1018` — `_forced_give_up`: synthesizes
  `ToolCall(id="budget", name="give_up", arguments=json.dumps({"reason": why}))`
  and requires `result.error` to be None, so `_stop` must not reject a
  missing `hindsight`.
- `policy.py:704-737` — `on_trial_end(record, ...)`: writes
  `record.metadata["wire_capture"|"transcript"|"llm_usage"]`. The
  per-trial reset lives wherever the policy resets `_messages`/counters for
  a new trial (locate `on_trial_begin` or equivalent; wire the hindsight
  stash reset there).
- `src/inspect_robots/rollout.py:106-107` — `TrialRecord.metadata`
  ("Extensible metadata for the trial (e.g. populated by policies)");
  `src/inspect_robots/eval.py:469,521` copies it into the sample's
  `trial_metadata`; no core changes are needed.
- `plugins/inspect-robots-agent/tests/test_policy_e2e.py:52-54,684` —
  imports both templates and parametrizes over them;
  `test_goal_runs_to_done_and_config_lands_in_log` (:385) is the pattern
  for asserting log contents end to end;
  `test_transcript_echo_reports_forced_give_up` (:1228) covers the forced
  path.
- `tests/test_tools_motion.py:1144` — existing done/give_up toolset test.

## Task 1: schema and `_stop` meta

- [ ] **Step 1: failing tests.** In the toolset tests: (a) both stop tools'
  schemas declare `hindsight` with `required == [<existing>, "hindsight"]`
  and a description that asks what the agent wishes it had known at the
  start; (b) `_stop` on a call carrying
  `{"summary": "...", "hindsight": "the jar lid needs two approach angles"}`
  puts `stop_hindsight` with that exact string into the action meta,
  alongside the unchanged `request_stop`/`stop_reason`/`stop_detail`;
  (c) a call WITHOUT `hindsight` (and one with `hindsight: "  "`) still
  succeeds, meta has NO `stop_hindsight` key, and `note` is unchanged.
- [ ] **Step 2: implement.** Add the parameter to both schemas. Description
  (same string for both, single source constant): "What do you know now
  that you wish you had known at the start of this episode? Concrete,
  transferable facts about this rig, task, or embodiment (geometry, grip
  offsets, camera quirks, motion behavior), written as advice to a future
  agent attempting the same task. Say 'none' if nothing qualifies." In
  `_stop`, read `arguments.get("hindsight")`; when it is a non-blank
  string, add `stop_hindsight` (stripped) to the meta. No validation
  error path.
- [ ] **Step 3: gates green, commit.**

## Task 2: policy stash and `trial_metadata` persistence

- [ ] **Step 1: failing tests.** E2E, following the
  `test_goal_runs_to_done_and_config_lands_in_log` pattern: a scripted run
  whose `done` call carries `hindsight` produces a log where
  `samples[0]["trial_metadata"][0]["hindsight"]` equals the string.
  Second test: a forced give_up run (budget exhaustion, mirror
  `test_transcript_echo_reports_forced_give_up`) produces trial_metadata
  WITHOUT a `hindsight` key. Third test: two trials in one run where only
  the first supplies hindsight; the second trial's metadata has no
  `hindsight` key (per-trial reset, no bleed-through).
- [ ] **Step 2: implement.** At the `policy.py:955` stop-chunk site, stash
  `chunk.actions[0].meta.get("stop_hindsight")` on the policy when
  `request_stop` is set. Reset the stash wherever per-trial state resets.
  In `on_trial_end`, write `record.metadata["hindsight"]` when the stash
  is a non-blank string. Note `on_trial_end` currently returns early when
  there is no transcript (`policy.py:717-719`); the hindsight write must
  happen BEFORE that early return, or a transcript-less trial silently
  drops it.
- [ ] **Step 3: gates green, commit.**

## Task 3: system prompt announcement

- [ ] **Step 1: failing tests.** Parametrized over both templates (existing
  `:684` pattern): the rendered system prompt contains the announcement
  sentence.
- [ ] **Step 2: implement.** Insert one sentence into BOTH templates,
  immediately before the "You have a budget" sentence: "Note what you are
  learning about this rig and task as you go: done and give_up will ask
  what you wish you had known from the start." Update any prompt-snapshot
  assertions (permitted mechanical edits).
- [ ] **Step 3: gates green, commit.**

## Task 4: version bump, docs, changelog

- [ ] **Step 1:** Bump the plugin to 0.22.0 in all THREE places (pyproject
  `version`, `uv lock` regeneration, `tests/test_package.py:9` pin).
- [ ] **Step 2:** Docs sweep: the plugin/agent docs page that lists the
  tools (grep `docs/` for `give_up`) documents the new argument and both
  persistence surfaces. Root `CHANGELOG.md` entry under Unreleased
  referencing #305 and plan 0047, following the sibling-entry convention.
- [ ] **Step 3: gates green, commit.**
