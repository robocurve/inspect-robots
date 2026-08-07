# Escape ends the episode (bare Enter becomes a no-op) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **Critique status:** pending first round.

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
  behavior immediately learns the new gesture.

**Why not Cmd+Enter:** terminal emulators do not forward the Cmd modifier to stdin; macOS
Terminal.app and iTerm2 both bind Cmd+Enter at the app layer (fullscreen), and what
reaches the pty for any Enter variant is the same `\r`. Cmd+Enter is therefore
indistinguishable from plain Enter without opting into the kitty keyboard protocol, which
is out of scope. The issue and CHANGELOG note this so the decision is discoverable.

**Architecture:** Two seams change, both already covered by injectable tests.

1. `console._parse` (pure function) — owns the new line grammar: empty line -> usage
   reminder; `"\x1b"` line -> `EndRequest()`; `/stop [note]` -> `EndRequest(note=...)`;
   `/y /n /p` unchanged; unknown `/cmd` unchanged.
2. `session._LineEditor` + `session._pump_input` — owns bare-Esc detection with a
   **one-poll grace period**. `feed()` keeps today's in-chunk semantics exactly
   (ESC+`[`/`O` starts a CSI/SS3 discard; ESC+other-byte is an alt-combo discard). What
   changes is the *cross-poll* resolution of a trailing ESC: when a pump drains and the
   editor is still in `escape_pending` state, the session arms a flag instead of
   swallowing the next byte later. On the next pump:
   - if new bytes arrived, they are fed normally (a split escape sequence still resolves
     as a sequence — the existing cross-poll split test must keep passing);
   - if no bytes arrived and the armed flag is still set, the pending escape is consumed
     as a bare Esc keypress: the session enqueues the `"\x1b"` sentinel line into
     `_line_queue`, and the composed console's `_parse` turns it into `EndRequest()`.

   The grace period costs one poll interval (~one control period) of latency on Esc,
   imperceptible to an operator, and is what disambiguates "bare Esc keypress" from "an
   escape sequence whose tail has not arrived yet" without blocking the control loop.
   EOF while armed does not fire an end request (EOF already latches the console).

No protocol, `__all__`, or `EvalLog` schema changes. `EndRequest` and `ConsolePoll` are
untouched. Attached feedback-only inputs (voice) still cannot end an episode
(`OperatorSession.poll` already ignores their `end`).

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
- Tests — `tests/test_console.py:52` (`test_whitespace_only_line_requests_an_unscored_end`)
  asserts the old empty-line semantics. `tests/test_session.py:1394,1404,1414` cover
  ESC+byte swallow, in-pump split sequences, and the cross-poll split sequence; the
  first and third pin behavior this plan must preserve (alt-combo discard; split
  sequence still discarded when its tail arrives on the next poll).
- Docs — `docs/guide/cli.md:152-153,165,195` document Enter-ends-episode (prose, the
  footer example's status line, and the quoted usage string).
- Out of repo: the hardware embodiment (inspect-robots-yam) composes the footer status
  text ("... | Enter ends the episode"); file a follow-up issue there after merge. The
  docs example here should switch to "Esc ends the episode" to match the new gesture.

## Tasks

### 1. Console grammar (`console.py` + `tests/test_console.py`)

- [ ] `_parse`: empty/whitespace-only line returns `(None, None, True)` (usage
      reminder, no end request).
- [ ] `_parse`: a stripped line equal to `"\x1b"` returns `(None, EndRequest(), False)`.
- [ ] `_parse`: `/stop` (case-insensitive) returns
      `(None, EndRequest(note=note or None), False)` with the same note handling as the
      verdict commands (stripped; `None` when absent).
- [ ] Rewrite `USAGE` and `USAGE_END_ONLY`: Esc (or `/stop`) ends the episode; keep the
      feedback and verdict phrasing. Keep the `"operator console: "` prefix — `cli.py`
      `removeprefix`s it for styling.
- [ ] Update module/class docstrings that promise Enter-ends-episode (`console.py:85`
      stays: it describes line completion, which is unchanged).
- [ ] Tests: update `test_whitespace_only_line_requests_an_unscored_end` to assert
      usage-reminder-no-end; add `/stop`, `/STOP note`, and `"\x1b\n"` cases; keep
      first-end-wins and multiline-paste tests passing.

### 2. Bare-Esc detection (`session.py` + `tests/test_session.py`)

- [ ] `_LineEditor`: expose the pending-escape state (`escape_pending` property) and a
      `take_bare_escape()` consumer that clears and reports it; keep `feed()`'s
      in-chunk semantics byte-identical.
- [ ] Update the `_LineEditor` docstring: cross-poll discard still holds when the tail
      arrives on the next poll; a trailing ESC that stays unanswered for a full poll is
      a bare Esc keypress (decision: one-poll grace).
- [ ] `_pump_input`: track whether this pump saw bytes; maintain an armed flag
      (`_esc_armed`); fire at most one `"\x1b"` sentinel line into `_line_queue` when a
      quiet pump finds the armed pending escape; never fire on EOF; re-arm/disarm from
      the editor state after every pump.
- [ ] Reset `_esc_armed` alongside editor resets (`_enter_footer`,
      `_reset_footer_state`).
- [ ] Tests: bare ESC fires an end request on the next quiet poll; ESC-tail split
      sequence across polls still discards (existing test must pass unmodified); ESC as
      the last byte after typed text preserves the partial line and still fires; EOF
      while armed fires nothing; ESC+byte same-chunk alt-combo still swallows (existing
      test unmodified).

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
