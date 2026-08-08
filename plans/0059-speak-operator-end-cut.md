# Operator-ended trials cut `--speak` narration immediately — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rig operators report that ending a run from the console (Esc or `/stop`) does
not stop narration: the in-flight note keeps playing and the run only exits after
`on_eval_end`'s success-path drain waits for speech to finish (`_DRAIN_TIMEOUT = 15.0`).
The drain exists so a natural `done()`/`give_up()` summary plays out; an operator-ended
trial is the opposite case — the human explicitly asked the run to stop. Cut speech the
moment the sink learns the trial was operator-terminated. (Issue #343.)

**Shape:** voice plugin only (`inspect-robots-voice` 0.5.0 → 0.5.1). No core changes at
all: `SpeakerSink` already receives `on_trial_end(record)` through the `LogSink`
protocol (currently inherited as `NullSink`'s no-op), `TrialRecord.termination_reason`
carries the reason, and `OPERATOR_END == "operator_end"` is public core API
(`inspect_robots.types`, re-exported in `inspect_robots.__all__`).

## Mechanism and ordering facts (verified against main @ 6008edb1)

- `rollout()` sets `record.termination_reason = OPERATOR_END` and breaks on an
  end-poll (`src/inspect_robots/rollout.py:320-334`). It does not call any sink hook at
  the break.
- `eval()` delivers `bus.on_trial_end(record)` per trial (`src/inspect_robots/eval.py:529`)
  **after** the grading hook (`before_scoring`, line 478) — so with an attended
  `operator` grader, the verdict prompt happens before the sink hears about the trial
  end. Consequence (documented, accepted): the tail of the in-flight note can still
  play while the operator answers the verdict prompt; the cut lands before the eval's
  exit path (the drain), which is where the reported multi-second wait lives. Cutting
  earlier would need a new core seam or reordering `on_trial_end` before grading, which
  would change what every sink sees in the record (no `operator_judgement` yet) — out
  of scope.
- The worker's generation checks (`self._speech_gen != gen` after pop, after
  synthesis, between chunks) run in **every** mode — plan 0057 added the mechanism for
  interrupt mode but did not gate it by mode. A generation bump therefore cuts speech
  in blocking and queue modes too, with no worker changes.
- Residual latency after the cut: at most one 0.1 s playback chunk, or the remainder of
  an already-running `engine.synthesize()` call (whose result is then discarded). Both
  are far below one note's playback time, and `_drain` waits on `queue or _inflight`,
  so `on_eval_end` returns as soon as the abandonment lands.

## Design decisions (binding)

1. **The cut applies in all three modes.** Operator intent outranks delivery-mode
   semantics — queue mode's "old behavior" contract is about how *live* notes are
   delivered, not about overriding an explicit stop. Docs say so.
2. **Only `OPERATOR_END` cuts.** `policy_stop` (the policy's own `done()`/`give_up()`)
   and embodiment termination keep today's behavior: the terminal summary plays and a
   successful eval end drains it. `record.terminated`/`truncated` flags are not
   consulted; the reason string is the contract. Note the string has three producers
   (console end-poll, `action.meta["stop_reason"]`, embodiment
   `result.termination_reason`), so a policy or embodiment reporting the literal
   `"operator_end"` also cuts — desirable (an embodiment-side operator e-stop should
   silence narration too) and accepted.
3. **The cut is a generation bump + queue clear + `notify_all`, under the condition.**
   Identical mechanics to an interrupt-mode delta. All sink hooks run synchronously on
   the one control thread, so no blocking-mode waiter can actually be parked while
   `on_trial_end` runs; `notify_all` (matching `on_trial_start`) is costless,
   symmetric, and defensive against future threaded callers, and the waiter-release
   property is covered by a defensive test of a production-unreachable interleaving.
   A woken waiter re-checks `queue or _inflight` and proceeds only once the worker's
   abandonment *lands* (next chunk boundary or synthesis return) — that residual is
   the same one-chunk/synthesis-remainder bound stated above. No `_dropped` counting:
   this is a deliberate discard, same rationale as interrupt-mode clears (plan 0057
   decision 9). The hook must never assume duck-typed extras beyond the `TrialRecord`
   dataclass fields: `_EvalSinkBus.on_trial_end` has no exception isolation, so an
   `AttributeError` here would surface into the eval loop.
4. **No drain-skip logic in `on_eval_end`.** After the cut the queue is empty and
   `_inflight` clears within one chunk/synthesis-return, so the existing drain
   degenerates to a near-no-op on its own. One mechanism, not two.
5. **Multi-trial runs:** a cut at trial N's end does not affect trial N+1's narration
   (its deltas arrive with the new generation). No trial-start changes.

## Tasks

### Task 1: `_speaker.py`

- [ ] Import `OPERATOR_END` from `inspect_robots.types` at module scope (core is
      already a module-scope import source here via `inspect_robots.logging.sink`; the
      lazy-import invariant covers audio/model deps only).
- [ ] Add `on_trial_end(self, record: TrialRecord) -> None` (type via the existing
      `TYPE_CHECKING` block, importing `TrialRecord` from `inspect_robots.rollout`):
      if `record.termination_reason != OPERATOR_END`, return; else under
      `self._condition`: `self._speech_gen += 1`, `self._queue.clear()`,
      `notify_all()`. Docstring states the contract: an operator-ended trial silences
      narration in every mode; natural completions still play out.
- [ ] Module docstring: add one sentence for the operator-end cut.

### Task 2: tests (`tests/test_speaker.py`)

- [ ] Helper to build a minimal `TrialRecord` with a given `termination_reason`
      (`termination_reason` is a defaulted constructor field:
      `TrialRecord(scene_id=..., epoch=..., seed=..., termination_reason=...)`).
- [ ] **Fix the existing test the hook breaks:**
      `test_trial_end_does_not_clear_just_enqueued_summary` currently calls
      `sink.on_trial_end(cast(TrialRecord, object()))`; the new hook's
      `record.termination_reason` access raises `AttributeError` on it. Update it to
      pass a real `TrialRecord` with reason `None` (it then doubles as part of the
      non-operator-reason coverage).
- [ ] **Cut on operator end (default mode):** gated in-flight utterance plus a queued
      text; `on_trial_end(operator_end record)`; assert queue empties; then release
      the gate and assert the utterance was abandoned (no post-cut writes) and a
      subsequent successful `on_eval_end` returns without waiting (bounded event
      asserts, existing `_GatedPlayback` pattern).
- [ ] **Cut applies in queue mode:** same shape with `mode="queue"`.
- [ ] **Cut releases a blocking-mode waiter (defensive):** blocking sink, gated
      in-flight utterance, a hook call parked in the blocking wait on another thread;
      `on_trial_end(operator_end record)`, **then release the gate**; assert the
      waiter returns promptly after the worker's abandonment lands and that no
      post-cut chunks were written. (The waiter cannot re-proceed before the
      abandonment lands — `_inflight` holds until then — so the gate release is part
      of the test, not an afterthought. This interleaving is unreachable in
      production, where all hooks share the control thread; the test pins the
      defensive property.)
- [ ] **Non-operator reasons do not cut:** `on_trial_end` with `termination_reason`
      of `None` and of `"policy_stop"` leaves queue and generation untouched
      (assert `_speech_gen` unchanged), and drain-on-success still plays the summary
      (the existing drain test already covers the positive path; keep it green).

### Task 3: docs, CHANGELOG, version

- [ ] `docs/guide/voice-mode.md`: one short paragraph in the speech-modes section:
      ending a trial from the console cuts narration immediately in every mode; only
      natural completions drain the final summary. Be mode-accurate about the verdict
      prompt: in the default interrupt mode the tail of one note may still be audible
      through it, while in `mode=queue` the whole backlog (up to 4 notes) keeps
      playing until the verdict is entered, because the cut lands after grading. Also
      note the operator's own `/stop` note text is never spoken (the speaker reads
      policy narration only).
- [ ] `plugins/inspect-robots-voice/CLAUDE.md`: extend the `SpeakerSink` invariant
      bullet with the operator-end cut.
- [ ] `CHANGELOG.md` `## [Unreleased]` → Fixed: voice 0.5.1, operator-ended trials cut
      `--speak` narration instead of draining it at eval end (plan 0059, #343).
- [ ] Version 0.5.0 → 0.5.1 in plugin pyproject + `__init__.py` + the
      `test_package_exports_and_version` assertion; `uv lock` if dirtied. (0.5.0 was
      published to PyPI in the v0.47.1 release run on 2026-08-07, so a distinct 0.5.1
      is required; re-verify `gh release list` at merge time per the concurrent-release
      convention.)

### Task 4: verification

- [ ] Plugin gates: ruff check, ruff format --check, mypy (plugin config, src+tests),
      pytest `--cov=inspect_robots_voice`.
- [ ] Core gates unchanged but run anyway: `ruff check .`, `ruff format --check .`,
      `mypy`, `pytest --cov -q` at 100%.

## Out of scope

- Cutting speech before the verdict prompt (needs a core seam or `on_trial_end`
  reordering; revisit only if operators still complain after this lands).
- Making the blocking-mode in-loop wait responsive to pending operator input (the
  documented Esc-latency caveat of plan 0057 stands).
