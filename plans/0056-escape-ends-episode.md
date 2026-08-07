# Escape ends the episode (bare Enter becomes a no-op) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **Critique status:** R1 (2026-08-07, vs main @ c8ac6f09) found 7 substantive issues;
> all folded in below (feed-level ESC resolution, time-floored grace, mode-aware deduped
> usage reminder, corrected test inventory). R2 pending.

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
  buffer; an explicit command is the discoverable fallback. An optional note is captured
  like the `/y /n /p` notes.
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
   reminder; `"\x1b"` line -> `EndRequest()`; `/stop [note]` -> `EndRequest(note=...)`;
   `/y /n /p` unchanged; unknown `/cmd` unchanged.
2. `session._LineEditor` + `session._pump_input` — owns bare-Esc detection at two
   levels.

   **Feed-level resolution (new ESC-while-pending rules).** Today ESC+any-non-`[`/`O`
   byte is an alt-combo discard, which would make the new gesture self-cancelling: Esc
   pressed twice would eat itself (odd/even mashing), and Esc-then-Enter would silently
   discard both keys. `feed()` therefore resolves a pending ESC explicitly. With
   `_escape_pending` set, the next byte `b` is handled as:
   - `[` / `O` -> CSI/SS3 discard (unchanged);
   - `0x1b` (a second ESC) -> the pending ESC was a bare keypress: emit the `"\x1b"`
     sentinel as a completed line, stay pending for the new ESC (mashing Esc fires
     immediately, once per press);
   - `\r` / `\n` -> the pending ESC was a bare keypress: emit the sentinel, clear
     pending, and consume the newline (so Esc-then-Enter works in footer mode too;
     accepted cost: Alt+Enter, which sends `ESC \r`, also ends the episode);
   - anything else -> alt-combo discard (unchanged).

   **Pump-level grace with a time floor.** When a pump drains and the editor is still
   `escape_pending`, the session arms itself and records `now_fn()` (a new injectable
   seam defaulting to `time.monotonic`). On a later pump that saw no bytes, if still
   armed and at least `_ESC_GRACE_S = 0.15` seconds have elapsed, the pending escape is
   consumed as a bare Esc keypress and the sentinel line is enqueued into `_line_queue`.
   If bytes arrive while armed they are fed normally, so a split escape sequence still
   resolves as a sequence (the existing cross-poll split test keeps passing) and any
   feed-level rule above may fire instead. EOF while armed never fires an end request
   (EOF already latches the console).

   The time floor (vim-style escape timeout) is what makes the grace robust on fast
   control loops: with 10-50 ms polls, a single-poll grace would turn an SSH-delayed
   arrow-key tail into a spurious end request. 150 ms is one-to-a-few polls on fast
   loops and rounds up to the next poll on slow ones. The accepted residual trade-offs,
   stated in the `_LineEditor`/pump docstrings: a sequence tail delayed beyond the grace
   still misreads as bare Esc, and on a very slow self-paced loop a lone Esc waits for
   the next poll. An operator who wants zero latency presses Esc twice.

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
  delivery, verdict prompts, and the plain-mode ticker are untouched.
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
  - `tests/test_session.py:1285` (`test_footer_poll_does_not_confirm_end_requests_or_verdicts`)
    parametrizes `(b"\n", EndRequest())` and asserts no extra output; the `b"\n"` case
    must move to `b"/stop\n"` (or `b"\x1b\x1b"`).
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
- [ ] `_parse`: `/stop` (case-insensitive) returns
      `(None, EndRequest(note=note or None), False)` with the same note handling as the
      verdict commands (stripped; `None` when absent).
- [ ] `OperatorConsole(usage: str = USAGE)`: store and print `self._usage` in `poll`,
      at most once per `poll()` call regardless of how many lines requested it.
- [ ] `OperatorSession(console_usage: str | None = None)` forwards to its owned
      console; `cli._build_operator_session` passes `console_usage=USAGE_END_ONLY` when
      the policy does not accept messages (construct the session after computing
      `accepts_messages`).
- [ ] Rewrite `USAGE` and `USAGE_END_ONLY`: "Esc (or /stop) ends the episode" (the
      "or /stop" hedge matters: in the plain-mode fallback a lone Esc does nothing
      until Enter); keep the feedback and verdict phrasing and the
      `"operator console: "` prefix — `cli.py` `removeprefix`s it for styling.
- [ ] Update module/class docstrings that promise Enter-ends-episode (`console.py:85`
      stays: it describes line completion, which is unchanged).
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
- [ ] `_pump_input`: track whether this pump saw bytes; on a drain that leaves
      `escape_pending`, arm and stamp `self._now_fn()`; on a later quiet pump, if armed
      and `now_fn() - stamp >= _ESC_GRACE_S` (0.15 s module constant), consume via
      `take_bare_escape()` and enqueue one `"\x1b"` sentinel line; never fire on EOF;
      disarm whenever the editor is no longer pending.
- [ ] `OperatorSession.__init__` gains the injectable `now_fn: Callable[[], float]`
      seam (default `time.monotonic`), consistent with the existing seam style.
- [ ] Reset the armed flag and stamp alongside editor resets (`_enter_footer`,
      `_reset_footer_state`).
- [ ] Tests (fake clock via `now_fn`): bare ESC fires after a quiet poll past the
      grace; quiet poll before the grace does not fire; ESC-tail split across polls
      within the grace still discards (existing `:1414` unmodified); double-ESC in one
      chunk fires immediately, once per press; ESC-then-`\r` (same chunk and across
      polls) fires immediately and completes no line; ESC as the last byte after typed
      text preserves the partial line and still fires; EOF while armed fires nothing;
      ESC+letter same-chunk alt-combo still swallows (existing `:1394` unmodified);
      update `test_footer_poll_does_not_confirm_end_requests_or_verdicts` to use
      `b"/stop\n"` for its end-request case.

### 3. Docs, changelog, module map

- [ ] `docs/guide/cli.md`: prose (`:152-153`), footer example status line (`:165`), and
      quoted usage line (`:195`) switch to the Esc / `/stop` gesture; one sentence on
      why Cmd+Enter is not offered (terminals do not forward Cmd). Follow the
      writing-style rules.
- [ ] CHANGELOG Unreleased: behavior change entry (bare Enter no longer ends attended
      episodes; Esc or `/stop` does).
- [ ] `src/inspect_robots/CLAUDE.md`: touch the `console.py` and `session.py` rows to
      mention the Esc-to-end gesture and the one-poll bare-Esc grace.

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
