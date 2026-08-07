# Escape ends the episode (bare Enter becomes a no-op) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **Critique status:** R1 (2026-08-07, vs main @ c8ac6f09) found 7 substantive issues
> (feed-level ESC resolution, time-floored grace, mode-aware deduped usage reminder,
> corrected test inventory). R2 found 3 more (/stop note rides the message path instead
> of EndRequest.note; arming restricted to byte-feeding pumps so a quiet pump cannot
> re-stamp the grace into never elapsing; ESC+newline leaves the editor buffer
> untouched). R3 found 2 stale-text contradictions from earlier revisions plus 1 design
> gap (`/stop <text>` would confirm as `[sent]` though never delivered; ending polls now
> confirm as `[noted]`). All folded in. R4 pending.

**Goal:** Close #333. On attended runs with the operator console (`policy=agent` being the
motivating case), pressing Enter on an empty input line currently ends the episode. Enter
is also the key the operator presses reflexively after typing feedback, so one accidental
bare Enter kills a run. Change the end-of-episode gesture so plain Enter never ends the
run:

- **Esc ends the episode.** In footer mode (attended TTY, cbreak) the session reads raw
  bytes, so a bare `0x1b` keypress is detectable and distinguishable from arrow-key
  escape sequences.
- **`/stop [note]` ends the episode from any mode.** In plain line mode stdin is
  canonical, so an Esc keypress reaches the process only after a newline flushes the
  buffer; an explicit command is the discoverable fallback. Trailing text is NOT an
  `EndRequest.note`: rollout persists `end.note` only alongside a verdict
  (`rollout.py:320-334`), and the post-trial `prompt_verdict` would clobber a
  verdict-less note. Instead `_parse` returns the trailing text as a normal operator
  *message* alongside the end request — messages are recorded to the log before the end
  check (`rollout.py:302-308`), so persistence and provenance come free and rollout is
  untouched.
- **A line consisting of exactly ESC (`"\x1b"`) also ends the episode**, so Esc-then-Enter
  works in plain line mode too, and footer mode can reuse the same parser via a sentinel
  line.
- **Bare Enter (empty/whitespace-only line) becomes a usage reminder**, not an end
  request: the console prints the (updated) usage line so an operator who expects the old
  behavior immediately learns the new gesture. The reminder is printed **at most once per
  poll** (Enter autorepeat must not flood the footer with repaints) and is **mode-aware**:
  the console owns a `usage` string chosen at construction, so end-only sessions remind
  with `USAGE_END_ONLY`, not the type-a-message text (today's `poll` hardcodes `USAGE`,
  a pre-existing wart this change would otherwise promote to the mainline path).

**Why not Cmd+Enter:** terminal emulators do not forward the Cmd modifier to stdin; macOS
Terminal.app and iTerm2 both bind Cmd+Enter at the app layer (fullscreen), and what
reaches the pty for any Enter variant is the same `\r`. Cmd+Enter is therefore
indistinguishable from plain Enter without opting into the kitty keyboard protocol, which
is out of scope. The issue and CHANGELOG note this so the decision is discoverable.

**Architecture:** Two seams change, both already covered by injectable tests.

1. `console._parse` (pure function) — owns the new line grammar: empty line -> usage
   reminder; `"\x1b"` line -> `EndRequest()`; `/stop` -> `EndRequest()` and
   `/stop <text>` -> the text as an ordinary operator *message* plus `EndRequest()`
   (never `EndRequest.note` — see the Goal bullet); `/y /n /p` unchanged; unknown
   `/cmd` unchanged.

   **Confirmation labeling:** `_confirm_console_messages` currently echoes every
   console message as `[<footer label>]`, i.e. `[sent]` on message-accepting sessions.
   A message arriving in the same poll as an end request is recorded to the log but
   never delivered to the policy (rollout records messages, then breaks on the end
   before the next inference), so `[sent]` would be a false delivery claim for
   `/stop <text>`. Rule: when `poll.end is not None`, `_confirm_console_messages` uses
   the label `noted` instead of the session's footer label. End-only sessions already
   use `noted`, so nothing changes there; message-accepting sessions truthfully report
   `[noted] <text>` for text that ends the episode. Covered by a session-level footer
   test for `/stop <text>`.
2. `session._LineEditor` + `session._pump_input` — owns bare-Esc detection at two
   levels.

   **Feed-level resolution (new ESC-while-pending rules).** Today ESC+any-non-`[`/`O`
   byte is an alt-combo discard, which would make the new gesture self-cancelling: Esc
   pressed twice would eat itself (odd/even mashing), and Esc-then-Enter would silently
   discard both keys. `feed()` therefore resolves a pending ESC explicitly. With
   `_escape_pending` set, the next byte `b` is handled as:
   - `[` / `O` -> CSI/SS3 discard (unchanged);
   - `0x1b` (a second ESC) -> the pending ESC was a bare keypress: emit the `"\x1b"`
     sentinel as a completed line, stay pending for the new ESC (each mashed Esc except
     the last fires at feed time; the last resolves via the grace or a later byte);
   - `\r` / `\n` -> the pending ESC was a bare keypress: emit the sentinel, clear
     pending, and consume the newline (so Esc-then-Enter works in footer mode too;
     accepted cost: Alt+Enter, which sends `ESC \r`, also ends the episode). The editor
     `_buffer` is left untouched in both sentinel cases: a partial line typed before the
     Esc stays a partial, is never completed as a line, never merged into the sentinel,
     and is simply abandoned when the trial ends;
   - anything else -> alt-combo discard (unchanged).

   **Pump-level grace with a time floor.** Arming is restricted to pumps that fed
   bytes: when a pump that saw at least one byte ends with the editor `escape_pending`,
   the session arms itself and stamps `now_fn()` (a new injectable seam defaulting to
   `time.monotonic`). A quiet pump (zero bytes) NEVER arms or re-stamps — it only
   tests: if armed and `now_fn() - stamp >= _ESC_GRACE_S` (0.15 s), the pending escape
   is consumed as a bare Esc keypress and the sentinel line is enqueued into
   `_line_queue`. (Without this restriction a quiet pump would refresh the stamp every
   poll and the grace would never elapse on a fast control loop.) If bytes arrive while
   armed they are fed normally, so a split escape sequence still resolves as a sequence
   (the existing cross-poll split test keeps passing) and any feed-level rule above may
   fire instead; the pump then disarms unless the editor ended pending again (a fresh
   press re-stamps). EOF while armed never fires an end request (EOF already latches
   the console).

   The time floor (vim-style escape timeout) is what makes the grace robust on fast
   control loops: with 10-50 ms polls, a single-poll grace would turn an SSH-delayed
   arrow-key tail into a spurious end request. 150 ms is one-to-a-few polls on fast
   loops and rounds up to the next poll on slow ones. The accepted residual trade-offs,
   stated in the `_LineEditor`/pump docstrings: a sequence tail delayed beyond the grace
   still misreads as bare Esc; on a very slow self-paced loop a lone Esc waits for the
   next poll; and a printable key pressed while an Esc is pending resolves it as an
   alt-combo (inherited behavior, but under the new gesture it silently cancels the
   intended end and eats the keystroke). An operator who wants zero latency presses Esc
   twice.

**Mode-aware usage plumbing.** `OperatorConsole` gains a `usage: str = USAGE`
constructor arg stored and printed by `poll` (replacing the hardcoded module constant),
and `OperatorSession` gains a `console_usage: str | None = None` kwarg forwarded when it
builds its own console (caller-injected consoles keep their own). `cli._build_operator_session`
constructs the session after computing `accepts_messages` and passes
`console_usage=USAGE_END_ONLY` for end-only sessions.

No protocol, `__all__`, or `EvalLog` schema changes (`USAGE` is not exported from
`inspect_robots/__init__`; cli and tests import it directly). `EndRequest` and
`ConsolePoll` are untouched. Attached feedback-only inputs (voice) still cannot end an
episode (`OperatorSession.poll` already ignores their `end`), and voice-transcribed text
bypasses `_parse` entirely, so it can never trigger `/stop`.

**Tech Stack:** Python 3.10+, stdlib only; pytest; no new deps.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
- D1 docstrings state contracts. Line length 100. Core stays numpy+stdlib.
- Zero behavior change outside end-of-episode input: footer rendering, feedback
  delivery, verdict prompts, and the plain-mode ticker are untouched. One intended
  exception: the per-poll usage dedup also collapses multiple unknown commands in a
  single poll to one usage print.
- Update `src/inspect_robots/CLAUDE.md` rows (`console.py`, `session.py`), CHANGELOG
  under Unreleased, and `docs/guide/cli.md`. Writing-style rules for docs prose (no em
  dashes, no mid-sentence bold).
- Worktree: `../ir-wt-escape-end` on branch `escape-ends-episode`. Reference #333 in
  commits; end with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  trailer. **No git operations from Codex.**
- Before merge: re-check `git ls-tree origin/main plans/` for a 0056 collision and
  renumber if a concurrent session took it.

## Reference: current wiring (main @ c8ac6f09)

- `src/inspect_robots/console.py` — `USAGE` (`:16-19`) and `USAGE_END_ONLY` (`:20-23`)
  say "Enter alone ends the episode" / "Enter ends the episode". `_parse` (`:68-81`):
  empty/whitespace line returns `(None, EndRequest(), False)` (`:71-72`); verdict map
  `/y /n /p` (`:75-78`); unknown slash command returns `show_usage=True` (`:79-80`).
  `OperatorConsole.poll` prints `USAGE` via `output_fn` when `show_usage` (`:120-121`).
- `src/inspect_robots/session.py` — `_LineEditor` (`:77-133`): `_escape_pending` /
  `_csi_discard` discard state persists across `feed()` calls; docstring (`:80-84`)
  currently promises the cross-poll persistence outright. `_pump_input` (`:225-233`)
  drains `_fd_readable`/`_fd_read` into the editor and extends `_line_queue`.
  `_enter_footer` (`:245-262`) and `_reset_footer_state` (`:285-291`) reset editor
  state. `_dispatch_read` (`:214-223`) hands queued lines (newline-joined) to the
  console.
- Tests that break and must be updated:
  - `tests/test_console.py:52` (`test_whitespace_only_line_requests_an_unscored_end`)
    asserts empty line -> `EndRequest()`.
  - `tests/test_console.py:111` (`test_first_end_request_wins_while_later_lines_still_parse`)
    scripts an empty first line as the winning end and asserts `output == [USAGE]`;
    post-change the winning end must come from `/stop` or `"\x1b"` and the usage output
    changes.
  - `tests/test_session.py:1292` (`test_footer_poll_does_not_confirm_end_requests_or_verdicts`;
    the `(b"\n", EndRequest())` tuple is at `:1288`) — the `b"\n"` case must move to
    `b"/stop\n"`.
- Tests that pin behavior this plan preserves, all three staying unmodified:
  `tests/test_session.py:1394` (ESC+byte alt-combo swallow — still true for non-ESC,
  non-newline bytes), `:1404` (in-pump split CSI), `:1414` (cross-poll split CSI whose
  tail arrives on the next poll).
- Docs — `docs/guide/cli.md:152-153,165,195` document Enter-ends-episode (prose, the
  footer example's status line, and the quoted usage string). Do NOT touch
  `docs/guide/cli.md:135` ("Bare Enter or whitespace-only input records no note") — that
  is the post-trial grader-notes prompt, unrelated — nor historical CHANGELOG entries.
- Out of repo: the hardware embodiment (inspect-robots-yam) composes the footer status
  text ("... | Enter ends the episode"); file a follow-up issue there after merge. The
  docs example here should switch to "Esc ends the episode" to match the new gesture.

## Tasks

### 1. Console grammar (`console.py`, `cli.py` + `tests/test_console.py`)

- [ ] `_parse`: empty/whitespace-only line returns `(None, None, True)` (usage
      reminder, no end request).
- [ ] `_parse`: a stripped line equal to `"\x1b"` returns `(None, EndRequest(), False)`.
      Add a comment (or dedicated test) for the non-obvious fact the sentinel hinges
      on: `str.strip()` removes `\x1c`-`\x1f` but not `\x1b`.
- [ ] `_parse`: `/stop` (case-insensitive) returns `(None, EndRequest(), False)`;
      `/stop <text>` returns `(stripped_text, EndRequest(), False)` — the text rides
      the existing operator-message path (recorded to the log before the end check),
      never `EndRequest.note`.
- [ ] `OperatorConsole(usage: str = USAGE)` — keyword-positioned after `output_fn` —
      store and print `self._usage` in `poll`, at most once per `poll()` call
      regardless of how many lines requested it (this also collapses two unknown
      commands in one poll from two prints to one; intended).
- [ ] `OperatorSession(console_usage: str | None = None)` forwards to its owned
      console; `cli._build_operator_session` passes `console_usage=USAGE_END_ONLY` when
      the policy does not accept messages (construct the session after computing
      `accepts_messages`).
- [ ] Rewrite `USAGE` and `USAGE_END_ONLY`: "Esc (or /stop) ends the episode" (the
      "or /stop" hedge matters: in the plain-mode fallback a lone Esc does nothing
      until Enter); keep the feedback and verdict phrasing and the
      `"operator console: "` prefix — `cli.py` `removeprefix`s it for styling.
- [ ] Docstrings: `console.py` needs no docstring edits beyond the `USAGE` constants
      (neither the module docstring nor `OperatorConsole`'s promises
      Enter-ends-episode; `:85` describes line completion, which is unchanged).
- [ ] Tests: update `test_whitespace_only_line_requests_an_unscored_end` (usage
      reminder, no end) and `test_first_end_request_wins_while_later_lines_still_parse`
      (first end via `/stop` or `"\x1b"`); add `/stop`, `/STOP note`, `"\x1b\n"`,
      usage-dedup-per-poll, and custom-`usage` cases; keep multiline-paste tests
      passing.

### 2. Bare-Esc detection (`session.py` + `tests/test_session.py`)

- [ ] `_LineEditor.feed`: implement the ESC-while-pending resolution rules from the
      Architecture section (second ESC -> emit sentinel + stay pending; `\r`/`\n` ->
      emit sentinel + consume newline; `[`/`O` and other bytes unchanged). Emit the
      sentinel by appending `"\x1b"` to the returned completed-lines list.
- [ ] `_LineEditor`: expose `escape_pending` (read) and `take_bare_escape()` (consume);
      update the class docstring with the resolution rules and the accepted Alt+Enter /
      delayed-tail trade-offs.
- [ ] `_pump_input`: track whether this pump saw bytes; ONLY a pump that fed bytes and
      ended `escape_pending` arms and stamps `self._now_fn()`; a quiet pump never
      stamps — if armed and `now_fn() - stamp >= _ESC_GRACE_S` (0.15 s module
      constant), consume via `take_bare_escape()` and enqueue one `"\x1b"` sentinel
      line; never fire on EOF; disarm whenever the editor is no longer pending.
- [ ] `OperatorSession.__init__` gains the injectable `now_fn: Callable[[], float]`
      seam (default `time.monotonic`), consistent with the existing seam style.
- [ ] Reset the armed flag and stamp alongside editor resets (`_enter_footer`,
      `_reset_footer_state`).
- [ ] `_confirm_console_messages`: when `poll.end is not None`, confirm console
      messages with the label `noted` instead of the session footer label (a message in
      an ending poll is recorded, never delivered; `[sent]` would be a false delivery
      claim for `/stop <text>`).
- [ ] Tests (fake clock via `now_fn`): bare ESC fires on a quiet poll past the grace;
      a quiet poll before the grace does not fire AND a second consecutive quiet poll
      past the grace does (guards against a quiet-pump re-stamp bug that would make the
      grace never elapse); ESC-tail split across polls within the grace still discards
      (existing `:1414` unmodified); double-ESC in one chunk emits one sentinel at feed
      time with the second ESC resolving via the grace; ESC-then-`\r` (same chunk and
      across polls) fires at feed time, completes no line, and leaves a previously
      typed partial in the buffer untouched (the input row repaints the stale partial
      for one frame until `end_trial` clears it; cosmetic, pinned by the test);
      session-level footer `/stop <text>` confirms as `[noted]` even on a
      message-accepting (`sent`-labeled) session; ESC as the last byte after typed text
      preserves the partial and still fires via the grace; EOF while armed fires
      nothing; ESC+letter same-chunk alt-combo still swallows (existing `:1394`
      unmodified); update `test_footer_poll_does_not_confirm_end_requests_or_verdicts`
      (the `(b"\n", EndRequest())` case at `:1288`) to use `b"/stop\n"`.

### 3. Docs, changelog, module map

- [ ] `docs/guide/cli.md`: prose (`:152-153`), footer example status line (`:165`), and
      quoted usage line (`:195`) switch to the Esc / `/stop` gesture; one sentence on
      why Cmd+Enter is not offered (terminals do not forward Cmd). Follow the
      writing-style rules.
- [ ] CHANGELOG Unreleased: behavior change entry (bare Enter no longer ends attended
      episodes; Esc or `/stop` does).
- [ ] `src/inspect_robots/CLAUDE.md`: touch the `console.py` and `session.py` rows to
      mention the Esc-to-end gesture and the time-floored (150 ms) bare-Esc grace.

### 4. Verification

- [ ] All four gates green in the worktree (`ruff check`, `ruff format --check`,
      `mypy`, `pytest --cov -q` at 100%).
- [ ] `uv run pytest tests/test_console.py tests/test_session.py -q` as the focused
      loop during development.
- [ ] Manual smoke (optional, TTY): `uv run inspect-robots "wave" --policy noop
      --embodiment cubepick` style attended run; Enter prints usage, Esc ends.

## Follow-ups (not this PR)

- inspect-robots-yam: footer status suffix still says "Enter ends the episode"; update
  to "Esc ends the episode" once this lands (file an issue referencing #333).
