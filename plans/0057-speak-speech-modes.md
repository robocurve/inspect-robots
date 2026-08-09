# `--speak` speech modes: blocking, interrupt (new default), queue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In `--speak` runs the narration lags: the bounded queue backlogs and the note
being spoken can describe a turn 5–10 inferences in the past. Add a `mode` option to the
speaker sink with three behaviors, selectable from the CLI via the existing repeatable
`-S k=v` channel (issue #336):

1. **`blocking`** — a note starts speaking at the moment its joint commands are issued
   (the agent policy emits both in the same turn) and the motors move while it plays.
   New observations and the next LLM inference may proceed while the note is still
   playing, but the *next* turn's joint commands wait until the current note finishes;
   then the joints move and the new note starts.
2. **`interrupt`** (**new default**) — a newly issued note cuts off whatever is currently
   being spoken and plays immediately. Narration always describes the current turn.
3. **`queue`** — today's behavior: notes queue (bounded, drop-oldest) and may lag.

**Shape:** one PR, two packages, almost all plugin.

1. **Voice plugin (`inspect-robots-voice` 0.4.0 → 0.5.0):** `SpeakerSink(mode=...)` plus
   an utterance-generation abort mechanism in the worker and a bounded pre-enqueue wait
   for blocking mode. Factory validation for `-S mode=`.
2. **Core (help text only):** extend the `-S` argparse help on `run` to name the mode
   option. No behavior, signature, or public-API change; no new core tests needed
   (argparse help strings are not covered assertions).

**Key architectural fact this plan relies on:** `rollout()` calls the sink's
`log_policy_messages(t, entries)` immediately after an inference completes and *before*
`approver.review()` / `embodiment.step()` in the same loop iteration
(`src/inspect_robots/rollout.py:358-414`). So "block the joints until the previous note
finishes" is implementable entirely inside the sink: `log_policy_messages` waits
(bounded) for the in-flight utterance to finish, then enqueues the new note and returns;
the rollout then executes the joints while the new note plays. Steps that execute the
rest of an open-loop chunk perform no inference, so the hook does not fire on them and
they are never delayed. No rollout/eval/core-sink-protocol change.

## Global Constraints

- Core gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
  The core change is one help string, so this is a formality, but the gates still run.
- Plugin gates mirror the `plugin-voice` CI job: ruff + ruff format + mypy (plugin
  config, strict) + pytest with `--cov=inspect_robots_voice`.
- Plugin tests must keep running without PortAudio/audio hardware/model downloads:
  injected fakes and threading events, **no sleeps as synchronization** (bounded
  `Event.wait(timeout=...)` is the tool; a bare `time.sleep` to "give the thread time"
  is not).
- D1 docstrings on public defs; contract, not name. Line length 100.
- No dependency changes, so no `uv lock` churn expected. If a pyproject changes anyway
  (version bump line only), the lockfile records plugin versions — run `uv lock` and
  commit it if `git status` shows it dirty.
- Public core API: no new symbols; `inspect_robots.__all__` and
  `tests/test_api_snapshot.py` untouched.
- Writing-style rules for public-facing text (README, docs, CHANGELOG): no em dashes in
  prose, no decorative emoji, no mid-sentence bold, headers use colons.

## Reference: current speaker behavior (main @ cf2a4dd3)

`plugins/inspect-robots-voice/src/inspect_robots_voice/_speaker.py`:

- `log_policy_messages` extracts note/summary/reason texts and appends to a
  `deque` bounded at `_QUEUE_SIZE = 4` with drop-oldest (`_dropped` counter). Never
  blocks, never touches audio.
- One daemon worker pops a text, sets `_inflight`, synthesizes, plays in
  `_CHUNK_SECONDS = 0.1` chunks, checking `_stop` between chunks; on any exception it
  sets `_disabled`, sets `_stop`, clears the queue, notifies, warns once, and exits.
  A `finally` clears `_inflight` and `notify_all()`s.
- `on_trial_start` clears the queue and reports the accumulated drop count to stderr.
- `on_eval_end` drains (queue empty and not inflight, `_DRAIN_TIMEOUT = 15`) only on
  `status == "success"`, then `close()`.
- `close()` is idempotent: sets `_stop`, clears, notifies, joins, closes playback.

All coordination already goes through one `threading.Condition` (`self._condition`).
The plan builds on that; it adds no second lock.

## Design decisions (binding)

1. **Mode selection surface: `-S mode=...`, not a new flag.** The `-S` channel exists,
   is validated in the plugin factory (typed scalars, unknown-key `TypeError`), and
   keeps core free of plugin vocabulary. A dedicated `--speak-mode` flag would force
   core to know and validate plugin enum values, or to pass strings blind with worse
   errors. Rejected.
2. **Default flips to `interrupt`.** This changes behavior for existing `--speak` users:
   backlogged narration is replaced by always-current narration. That is the point of
   the issue, the maintainer's explicit choice, and a minor-version bump of a 0.x
   plugin. `-S mode=queue` restores the old behavior; the CHANGELOG says so explicitly.
3. **Interrupt granularity: between playback chunks and around synthesis, not
   mid-`engine.synthesize()`.** Kokoro synthesis is a single blocking call (typically
   well under the length of the audio it produces). The worker aborts a stale utterance
   at the next 0.1 s chunk boundary and discards a stale synthesis result before playing
   it. Worst-case residual speech after an interrupt is one chunk (~0.1 s) plus the tail
   of an already-running synthesize call. Acceptable; anything sharper needs engine
   cooperation that kokoro-onnx does not offer.
4. **Interrupt mechanism: a generation counter, not a second Event.** `self._speech_gen:
   int` incremented under `self._condition`. The worker snapshots the generation when it
   pops a text and abandons the utterance when `self._speech_gen != gen` (checked
   wherever `_stop` is checked today: after the pop, after synthesis, between chunks).
   Reading an `int` attribute is atomic under the GIL; all writes stay under the
   condition. A per-utterance `Event` object would need careful swap semantics for the
   same result.
5. **Interrupt scope: a new *delta* interrupts, texts within one delta queue.** One
   `log_policy_messages` call can carry several spoken texts (a `note` plus a terminal
   `summary` in the same turn). Interrupting mid-batch would silence the note the moment
   the summary arrives, losing this turn's narration to this turn's own text. So in
   interrupt mode the hook: bumps the generation once, clears the queue once, then
   enqueues every text of the delta. They play in order; the next delta interrupts them.
6. **Blocking mode: bounded wait, fail-open, and a permanent degrade path.** The hook
   waits for `not self._queue and not self._inflight` under the condition, bounded by
   `_BLOCK_TIMEOUT = 15.0` seconds (matching `_DRAIN_TIMEOUT`; a two-sentence note is
   5–8 s of audio, so 15 s is real margin, and the constant must be read as a module
   global at call time — mirroring how `on_eval_end` reads `_DRAIN_TIMEOUT` — so tests
   can monkeypatch it). Exit behavior is per-predicate:
   - **Timeout:** stop waiting, enqueue anyway (drop-oldest applies, counted in
     `_dropped`), print one stderr warning, and set a `self._block_degraded` flag that
     permanently skips the wait for the rest of the run. The failure this guards
     against (playback stalled on a dead audio server — `OutputStream.write` blocks
     rather than raises, so the worker's exception→`_disabled` path never fires) is
     persistent: without the flag every later turn would eat the full timeout,
     turning the run into a slideshow. One slow turn, then narration degrades to
     queue-mode enqueueing while control continues at full rate.
   - **`_stop` set, `_disabled` set, or worker thread dead:** return *without*
     enqueueing, matching the existing entry guard — never append to a queue that
     `close()` just cleared on a sink with no worker.
   The worker's exception path and `close()` both `notify_all()`, so these wakeups are
   prompt. Speech must never be able to halt a robot indefinitely: this is a narration
   feature, not a safety interlock, and the embodiment's own guardrails
   (Clamp/DeltaLimit, `self_paced` step cadence) remain the real safety layer.
7. **Blocking-mode drop reporting stays.** If the timeout path ever drops a note it is
   real signal (TTS wedged, audio device stalled) and the existing
   `on_trial_start` stderr line reports it.
8. **`on_trial_start` keeps today's behavior in every mode:** clear the backlog, let
   the in-flight utterance finish. In interrupt mode the new trial's *first delta*
   bumps the generation the moment it has something to say, which already guarantees no
   old note plays over new narration; a bump at `on_trial_start` itself would cut
   terminal `done()`/`give_up()` summaries during the otherwise-silent reset window,
   quietly reversing plan 0054's deliberate choice to let them play out. Blocking
   mode's first hook call of the new trial waits for a still-playing summary, which is
   exactly blocking semantics.
9. **Interrupt-mode queue clears are not "drops".** The `_dropped` counter keeps meaning
   "notes lost to backpressure or timeout", so its stderr report stays actionable. By
   design interrupt mode discards superseded notes; counting those would make the
   report fire on every healthy interrupt-mode run. Known edge, accepted: a single
   interrupt-mode delta carrying more than `_QUEUE_SIZE` texts still hits the
   drop-oldest bound and counts those drops; real deltas carry one or two texts.
10. **`_QUEUE_SIZE` bound stays in all modes.** It is load-bearing for queue mode,
    harmless in interrupt mode (queue is cleared on every delta), and a backstop in
    blocking mode's timeout path.
11. **`on_eval_end` drain is unchanged** in all modes: on success, wait for the last
    note to finish (bounded); otherwise abort. In interrupt mode the queue holds at most
    the final delta, so the drain is short by construction.
12. **Mode is constructor state, not mutable at runtime.** No per-trial or per-delta
    mode switching; one run, one mode.
13. **Known and accepted properties, to be stated in the docs rather than left
    implied:**
    - The blocking gate only fires on deltas that carry speakable text. An inference
      turn with no note (note-less tool call, malformed arguments, or live streaming
      disabled after a hook exception) issues its joints without waiting for the
      previous note.
    - Speech begins once Kokoro synthesis of the note completes (~1–2 s), not at the
      literal instant the joints are commanded; the blocking wait covers the previous
      note's synthesis plus playback.
    - Blocking mode delays the top-of-loop `operator_input.poll()` for attended runs:
      Esc//stop response time grows by up to one note duration (and by one
      `_BLOCK_TIMEOUT` in the degraded case, once). It also deliberately inserts
      multi-second gaps before `embodiment.step()`; rigs with command-cadence
      watchdogs should prefer interrupt mode.

## Tasks

### Task 1: `SpeakerSink` mode plumbing and worker generation checks

`plugins/inspect-robots-voice/src/inspect_robots_voice/_speaker.py`

- [ ] Add module constants: `_BLOCK_TIMEOUT = 15.0`, `_MODES = ("blocking", "interrupt",
      "queue")`, `_DEFAULT_MODE = "interrupt"`. These are the single source of truth:
      the factory imports them rather than restating literals, and `_BLOCK_TIMEOUT` is
      read as a module global at wait time (not captured at construction) so the
      timeout test can monkeypatch it.
- [ ] `SpeakerSink.__init__`: accept `mode: str = _DEFAULT_MODE`, store `self.mode`;
      initialize `self._speech_gen = 0` and `self._block_degraded = False`. Guard
      direct constructors with `if mode not in _MODES: raise ValueError(...)` (the
      factory owns `-S` validation and raises `TypeError` per its convention; the
      constructor guard is insurance for programmatic use).
- [ ] `_worker`: snapshot `gen = self._speech_gen` inside the condition block where the
      text is popped. Replace every bare `if self._stop.is_set(): return` mid-utterance
      check with a stale-check that *continues to the next queue item* instead of
      returning when only the generation moved: stop still returns; generation mismatch
      abandons the current utterance but keeps the worker alive (`break` out of the
      chunk loop / skip playing the synthesis result, then fall through to the
      `finally` and loop again).
- [ ] `log_policy_messages`: keep the existing extract + liveness guard. Then branch on
      mode inside the same `with self._condition` block:
      - `queue`: exactly today's code.
      - `interrupt`: `self._speech_gen += 1`; `self._queue.clear()` (the clear itself
        does no `_dropped` counting); then append all texts **via the existing bounded
        drop-oldest counting append loop** (this is what makes design decision 9's
        intra-delta-overflow edge true); `notify()`.
      - `blocking`: unless `self._block_degraded`, run a bounded wait loop (deadline
        via `time.monotonic()` against the module-global `_BLOCK_TIMEOUT`) on
        `(self._queue or self._inflight)`. On every wakeup re-check the same compound
        guard as the hook's entry, in the same order (`thread is None or not
        thread.is_alive() or self._disabled or self._stop.is_set()` — checking
        `is_alive` on a thread `close()` nulled must not raise): any of those →
        **return without enqueueing** (never append to a queue `close()` just cleared
        on a sink with no worker). On deadline expiry → set
        `self._block_degraded = True` and remember to warn; the one stderr warning
        (`speaker: blocking wait timed out; narration may lag`) prints **after
        releasing the condition**, matching how `on_trial_start` and the worker print
        outside the lock. Then (normal completion, degraded skip, or timeout
        fall-through) append all texts with the existing drop-oldest counting bound
        and `notify()`.
- [ ] `on_trial_start`: unchanged in all modes (design decision 8).
- [ ] Update the module docstring (it currently reads "Non-blocking policy-note speech
      with bounded stale-work loss"), the `SpeakerSink` class docstring ("Speak policy
      narration asynchronously without delaying the control loop"), and the
      `log_policy_messages` docstring ("without blocking on audio") — all three state
      the pre-plan contract and all three are now mode-dependent. The invariant that
      survives in every mode: the hook never synthesizes or plays audio on the control
      thread; only blocking mode may *wait* on it, boundedly and fail-open.

### Task 2: factory validation

`plugins/inspect-robots-voice/src/inspect_robots_voice/__init__.py`

- [ ] Add `"mode"` to the allowed `-S` keys. Read `mode = kwargs.get("mode",
      _DEFAULT_MODE)` (import `_DEFAULT_MODE`/`_MODES` from `_speaker` — one source of
      truth); require `isinstance(mode, str)` and membership in `_MODES`; `TypeError`
      otherwise with a message that lists the valid values (match the existing error
      style: lowercase, states the constraint).
- [ ] Pass `mode=mode` through to `SpeakerSink`.
- [ ] Extend the `speaker_sink` docstring's supported-keys sentence with `mode`
      (string, one of `"blocking"`, `"interrupt"`, `"queue"`, default `"interrupt"`).

### Task 3: plugin tests

`plugins/inspect-robots-voice/tests/test_factory.py`:

- [ ] `mode` accepted for each of the three values; result's `.mode` matches.
- [ ] Default is `interrupt`.
- [ ] Non-string mode → `TypeError`; unknown string mode → `TypeError`.

`plugins/inspect-robots-voice/tests/test_speaker.py` (use the existing fake engine /
fake playback / event patterns):

- [ ] **Existing queue-behavior tests:** audit every test that exercises queueing,
      drop-oldest, or multi-note ordering and pin `mode="queue"` where the old
      semantics are the subject under test. Two are known to *fail* (not just drift in
      meaning) under the new default and must be pinned:
      `test_trial_start_clears_queued_prior_trial_notes` (the second delta would bump
      the generation and stale-abandon the held-open utterance right after its
      synthesis, so `_wait_until(len(playback.writes) == 3)` times out at write 1) and
      `test_overflow_drops_oldest_and_reports_once_at_next_trial`
      (five separate deltas would collapse to two utterances with `_dropped == 0`).
      Tests asserting mode-independent behavior (lifecycle, error disable, close
      idempotence, drain-on-success) stay on the default and must still pass.
- [ ] **Interrupt:** with a fake playback that blocks on an event mid-utterance, deliver
      a second delta; assert the first utterance stops within one chunk (worker moves
      on), the queue holds only the new delta's texts, the new note plays, and
      `_dropped` stays 0. Also: a stale synthesis result is discarded (interrupt lands
      between pop and play), and the worker survives interrupts (plays the next note
      rather than exiting).
- [ ] **Interrupt, multi-text delta:** a delta carrying two texts plays both, in order.
- [ ] **Interrupt trial boundary:** an in-flight utterance from trial N keeps playing
      across `on_trial_start` (summaries play out, decision 8) and is cut by trial
      N+1's first delta.
- [ ] **Blocking:** with an in-flight utterance held open by an event,
      `log_policy_messages` from a second thread does not return until the utterance
      completes; after release it enqueues and the note plays. Non-return is inherently
      a timed negative assertion, which is sanctioned here as the one exception to the
      no-sleep rule: `assert not returned_event.wait(small_timeout)` (false-pass
      possible, never false-fail — the same pattern
      `test_successful_eval_end_drains_inflight_utterance_before_close` already uses).
      Positive assertions (it *did* return after release) use generous event timeouts.
- [ ] **Blocking fail-open on worker death:** worker's engine raises while a blocked
      `log_policy_messages` waits; the call returns promptly (disabled path) without
      enqueueing, long before the full timeout.
- [ ] **Blocking fail-open on close:** `close()` from another thread releases a blocked
      hook promptly, without enqueueing.
- [ ] **Blocking timeout degrade:** with `_BLOCK_TIMEOUT` monkeypatched tiny and
      playback held open, the hook returns after the deadline, the note is enqueued
      (drop-oldest accounting intact), one stderr warning is printed, and the *next*
      hook call skips the wait entirely (`_block_degraded` behavior).
- [ ] **Queue mode regression:** `mode="queue"` still drop-oldest-bounds at
      `_QUEUE_SIZE` and reports drops at trial start.

### Task 4: core help text

`src/inspect_robots/cli.py`:

- [ ] Extend the `-S` help string to mention the mode option, e.g. `"pass an argument
      to the inspect-robots-voice speaker, e.g. -S mode=blocking|interrupt|queue
      (requires --speak)"`. Nothing else in core changes.

### Task 5: docs, CHANGELOG, versions, agent guides

- [ ] `docs/guide/voice-mode.md`: add a "Speech modes" subsection under the `--speak`
      section: the three modes, interrupt as default, one example per non-default mode,
      blocking mode's bounded fail-open wait and one-time degrade, and the design
      decision 13 properties (gate skipped on note-less turns, speech starts after
      synthesis, operator-input latency and watchdog-cadence caveats for blocking).
      Also **rewrite** the existing sentence "Speech synthesis and playback run off the
      control loop." (now false for blocking mode) and add a `mode` row to the
      existing `-S` key table in that file.
- [ ] `plugins/inspect-robots-voice/README.md`: the README is currently input-only (it
      never mentions `--speak`). Add a short `--speak` paragraph naming the three
      modes and the interrupt default, matching the README's existing prose style and
      the repo writing-style rules; do not add a table.
- [ ] `docs/guide/plugins.md`: the voice-plugin bullet says "bounded drop-oldest
      buffering keeps speech output off the control loop", which the new default
      falsifies twice (mechanism is generation-abort; blocking mode deliberately
      delays the loop). Rewrite it to mirror the corrected voice-mode.md invariant
      wording.
- [ ] `docs/guide/cli.md`: extend the `-S` key enumerations ("select the output voice,
      speed, volume, device, language, or offline model paths", and the run-section
      equivalent) to include the speech mode.
- [ ] `plugins/inspect-robots-voice/CLAUDE.md`: update the `_speaker.py` module-map row
      and the `SpeakerSink` invariant bullet: the hook never synthesizes or plays audio
      on the control thread in any mode; blocking mode may wait boundedly; interrupt is
      the default and aborts via a generation counter.
- [ ] `plugins/inspect-robots-voice/pyproject.toml` + `__init__.py`: version 0.4.0 →
      0.5.0 (feature plus default-behavior change on a 0.x plugin), **and** update the
      hard assertion in `test_factory.py::test_package_exports_and_version` to match.
      Run `uv lock` if it dirties; commit whatever changes.
- [ ] `CHANGELOG.md` under `## [Unreleased]`: an Added entry (voice: speech modes) and a
      Changed entry (voice: default `--speak` behavior is now interrupt; `-S mode=queue`
      restores the old queueing), linking plan 0057 and issue #336.

### Task 6: verification

- [ ] `uv sync --locked --all-packages --extra dev` in the worktree.
- [ ] Plugin: ruff check, ruff format --check, mypy (plugin config, src + tests),
      pytest with coverage over `inspect_robots_voice`.
- [ ] Core: full gates (`ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov -q`
      at 100%).
- [ ] Re-read the final `_speaker.py` top-to-bottom for lock-ordering and
      wait-predicate correctness (every `wait()` in a loop with a predicate; every
      state change that a waiter watches followed by a notify under the same lock).

## Out of scope

- `eval-set --speak` (unchanged from plan 0054's scoping).
- Mid-synthesis interruption (needs engine cooperation kokoro-onnx lacks).
- Pausing or rate-adapting TTS speed to catch up (a possible future mode).
- Any rollout/eval/core sink-protocol change.
