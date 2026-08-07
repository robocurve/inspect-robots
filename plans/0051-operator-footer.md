# Operator footer: fixed status line and an owned input line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Critique status:** 10 fresh-context rounds; R10 returned no substantive issues
> against main @ 4e5cc772 (2026-08-06). R4-R9 findings and fixes are in this
> branch's `plan:` commit history.

**Goal:** PR 3 of the plan-0048 arc (#308). On attended runs with the console enabled and
a real TTY, `OperatorSession` renders a fixed two-line footer: a status line that rewrites
in place and never moves, and a `> ` input line the session itself echoes into. Typing is
never torn by ticker repaints, sent feedback is confirmed in scrollback, and the timer
stays put:

```
  [sent] you might wanna move the right arm out of the way
  t = 61s / 120s | Enter ends the episode
  > is there anything I can hand you█
```

Off-TTY, on Windows, when the console channel is off, or when termios is unavailable,
rendering stays exactly as today (plain in-place status line, tty-driver echo).

**Architecture:** Footer mode lives entirely inside `session.py` plus one small,
optional rollout hook. The session takes terminal ownership per trial: a new duck-typed
`end_trial()` on operator input (called best-effort by `rollout()` in the same `finally`
that already exists for trial teardown) pairs with `begin_trial()` so the session can
enter cbreak-with-echo-off on trial start and always restore termios on trial end,
Ctrl-C included (`ISIG` stays on; an `atexit` fallback catches process death). While in
footer mode the session wraps the composed console's `read` seam with a line editor:
raw chunks are edited (printable append, backspace, Ctrl-U) and echoed into the input
line by the session, and the console receives only completed lines, so its parser and
`begin_trial()` drain discipline are untouched. `status(line)` repaints the status line
above the input line; `write_line(text)` inserts scrollback above both and repaints;
`poll()` delegates then prints one `[sent] <text>` confirmation per returned feedback
message. Everything is injectable; the only `# pragma: no cover` lines are the genuine
termios/tty syscalls, isolated in module-private raw-mode helpers.

**Tech Stack:** Python 3.10+, stdlib only (`termios` is POSIX-stdlib; imported lazily
inside the helpers so core imports stay clean on Windows); pytest; no new deps.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
- D1 docstrings state contracts. Line length 100. Core stays numpy+stdlib.
- Zero behavior change off the footer path: plain mode must be byte-identical to
  today's `session.py` rendering, and unattended/no-console runs never touch termios.
- Public API: no new exports expected (`OperatorSession` is already public; `end_trial`
  is a duck-typed method on it). If anything public is added, update
  `inspect_robots.__all__` and `tests/test_api_snapshot.py` together.
- Update `src/inspect_robots/CLAUDE.md` rows (`session.py`, `rollout.py`), CHANGELOG
  under Unreleased, and the docs operator pages (`docs/guide/cli.md` live-feedback
  section gains a short footer description). Writing-style rules for prose.
- Worktree: `.claude/worktrees/operator-footer`. Reference #308 in commits; end with
  the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. **No git
  operations from Codex.**
- Before merge: re-check `git ls-tree origin/main plans/` for a 0051 collision and
  renumber if a concurrent session took it.

## Reference: current wiring (main @ 4e5cc772, post-voice #316)

- `src/inspect_robots/session.py` — the whole module (209 lines): `_flush_stdin_fd`,
  `OperatorSession.__init__` seams (`console`, `input_fn`, `write`, `flush_fn`),
  `_write`/`_input` late-binding, `status`/`write_line` state machine
  (`_status_open`, `_status_line`; the `status(None)` close branch is `:116-119`),
  `gate`, verdict prompts. Since voice (#316), `poll()` (`:70-91`) is NOT a pure
  delegate: it polls the console first, then merges every healthy attached input
  (`attach_input(source, label=...)`, `:106-112`), echoing each attached message as
  `f"{label}: {text}"` via `write_line` and returning
  `ConsolePoll(messages, end, sources)` with `sources` parallel to `messages`
  (`"console"` vs the attach label; empty tuple = all-console). `begin_trial()`
  (`:93-104`) fans out to the console and every attached input with
  detach-on-raise.
- `src/inspect_robots/console.py` — `OperatorConsole(readable, read, output_fn)`
  (`read: Callable[[], str] | None`, `:90`); `poll()` assembles lines from chunks
  (`:99-122`; reads only `while self._readable()`, `:106`; EOF latch `:108-111`);
  `begin_trial()` drains (`:124-134`). `_stdin_readable` is at `:60-61`; the
  console's default `read` is the str-typed fd-level `_stdin_read` (`:64-65`);
  `ConsolePoll.sources` defaults to `()` (`:38-44`). The session builds its
  default console with only `output_fn=self.write_line`.
- `src/inspect_robots/rollout.py:277-285` — `operator_input.begin_trial()` inside a
  try/except that disables the channel on failure (`console_ok`, init `:256`);
  `:291-301` — the per-iteration poll with the same degrade discipline; `:302-308`
  — the per-message loop now records `poll.sources[i]` provenance; `:461-465` —
  KeyboardInterrupt conversion re-raised through the per-trial `finally` at
  `:466`, which is where `end_trial()` goes. The degrade tests to mirror live in
  `tests/test_rollout_observation_step.py` (NOT test_rollout_hardening.py, which
  has no console tests).
- `src/inspect_robots/cli.py` — `_build_operator_session` (`:729-779`, decision-9
  ladder; `USAGE_END_ONLY` chosen at `:747`): the footer activates only on rows
  where the console turns on; the helper tells the session (see decision 3).
  Voice wiring (`_build_voice_input`/`_start_voice_input`/`_close_voice_input`,
  `:782-823`) attaches a `label="voice"` input around the session — the footer
  must compose with it (decisions 6/7).
- Tests: `tests/test_session.py` (exact-byte rendering tests to extend),
  `tests/test_rollout_observation_step.py` (the poll/begin_trial channel-degrade
  tests to mirror for `end_trial`; the new rollout `end_trial` tests can live
  beside them), `tests/test_registry_cli.py` (matrix rows unchanged).

## Design decisions (and why)

1. **Per-trial raw-mode window via `begin_trial()`/`end_trial()`.** Terminal mode must
   be canonical between trials: the verdict prompts (`input()`), the yam gates, and
   any embodiment homing output run there. `begin_trial()` already exists on the
   `OperatorInput` protocol; `end_trial()` joins as a **duck-typed optional** method
   (same pattern as every other optional hook) so third-party `OperatorInput`
   implementations stay conformant. `rollout()` calls it best-effort in the trial's
   `finally`: wrapped in the same catch-warn-disable discipline as `poll()`, and it
   must run even when the trial raised (`EmbodimentFault` mid-trial must not leave the
   user's terminal in raw mode). `end_trial()` owns the FULL footer teardown, not just
   termios: idempotently close the footer (clear the input row and the status row,
   reset `_status_open`) and then restore termios — no core code calls
   `session.status(None)` (only plugins do), so on `EmbodimentFault`, Ctrl-C, or a
   plugin that never closes its status line, `end_trial` is the only thing standing
   between the verdict prompt and a stale `> ` row. The restore callable is an
   injectable, idempotent method (tests call it directly — a closure registered only
   with `atexit` would run at interpreter exit and fail the 100% gate), registered
   with `atexit` once per session as a crash net; restore clears the saved termios
   attrs so the atexit fallback can never replay a stale saved state over a later
   trial's raw window. `KeyboardInterrupt` is converted by `rollout()` to its
   cancelled-trial path and re-raised **through** the same `finally`
   (rollout.py:461-466), so Ctrl-C restores too. Guard scope: `end_trial()` runs
   whenever `operator_input is not None` and the hook exists — it does NOT check
   `console_ok`. Only the catch-warn part mirrors poll's discipline, not its guard: a
   footer-mode `poll()` that raises (the pump is the likeliest raiser) flips
   `console_ok` False, and skipping restore on that flag would leave the verdict
   prompt running in cbreak with echo off — the exact scenario the teardown exists
   for.
2. **Footer mode gates on: console channel enabled AND `sys.stdin.isatty()` AND
   termios importable AND raw-mode entry succeeded.** Any failure (non-tty, Windows,
   exotic terminal, `termios.error`) falls back to plain mode silently at
   `begin_trial()` time — plain mode is today's behavior, which is always safe. The
   mode is re-decided each trial (a TTY cannot appear mid-run, but a failed entry
   must not poison later trials).
3. **The CLI opts the session into footer mode explicitly** (`session.enable_footer()`
   or a constructor flag set inside `_build_operator_session` exactly where the
   console turns on — implementer's choice, but it must be the CLI's decision, not
   ambient sniffing inside `poll()`): direct API users of `OperatorSession` (yam
   routes `status()` through it on connected runs) get footer behavior only when the
   run actually has the interjection console, which only the CLI knows.
   Two contract points: (a) on a session constructed with a caller-injected
   `console=...`, `enable_footer()` is a documented no-op — the footer requires the
   session-built console whose seams are the decision-7 dispatching closures;
   checked at `begin_trial()` alongside the tty/termios gates. (b) the confirmation
   label is CLI-chosen at enable time: `[sent]` on feedback rows (messages reach the
   policy), `[noted]` on end-only rows (`USAGE_END_ONLY` says lines are saved to the
   log; "sent" would imply delivery to the policy that is not happening).
4. **Line editing is deliberately minimal:** printable characters append, backspace
   (`\x7f`/`\x08`) deletes one **character** — strip trailing UTF-8 continuation
   bytes (`0x80`–`0xBF`) plus one lead byte, never a lone byte, or backspace after
   "é" leaves a dangling lead byte that mangles the delivered message — Ctrl-U kills
   the line, `\n`/`\r` completes it (echoing `\r` as line completion matches
   raw-mode Enter), everything else non-printable is ignored. Ignoring `\x1b` alone
   is not enough: in cbreak an arrow key arrives as `\x1b [ A` whose tail bytes are
   printable and would append as literal `[A` junk — so on `\x1b` followed by `[`
   or `O` (CSI **and SS3**: xterm-family terminals send F1–F4 as `\x1bOP`…`\x1bOS`,
   and arrows arrive as `\x1bOA`…`\x1bOD` whenever application-cursor mode is left
   set) discard bytes until the final byte (`0x40`–`0x7E`); on any other bare
   `\x1b` discard the next byte. The discard state is editor state persisting
   across `fd_read` chunks AND across polls: the pump can land a read between
   `\x1b[` and its final byte (select goes quiet mid-sequence), the tail arriving a
   whole control period later — a chunk-local scan would append the stray final
   byte, the same desync split UTF-8 gets below. Multi-byte UTF-8 arrives whole
   from `os.read` chunks in
   practice, but the editor must tolerate split sequences: buffer bytes, decode
   with `errors="replace"` at display time for the echo, and decode each completed
   line **once** with `errors="replace"` at queue handoff — the console's `read`
   seam is str-typed (`read: Callable[[], str]`, console.py:90), so bytes cannot
   pass through it, and a strict decode there would turn malformed paste bytes into
   a raising `poll()` that kills the channel; decoding only complete lines
   preserves content exactly where today's per-64KiB-chunk decode could not. Paste
   works because chunk
   processing is loop-per-byte over arbitrarily large reads. No history, no arrow
   keys, no cursor movement (YAGNI; the plan-0048 arc can add them later if operators
   ask).
5. **Rendering model: the input row is the footer-window constant; the status row is
   optional.** The input row is drawn (empty `> `) at footer `begin_trial()` and
   exists for the whole footer window. Before drawing it, footer `begin_trial()`
   closes any plain-open status line using the existing plain semantics (write
   `"\n"`, reset `_status_open` — session.py:116-119): a prior plain trial's ticker
   line survives the zero-byte plain `end_trial()` (and a plugin can call
   `status()` during `embodiment.reset()`, which runs before `begin_trial()`), and
   drawing the footer onto it would append `> ` to the stale line AND start the
   renderer in the two-row branch with no footer status row above — desyncing
   every relative repaint into scrollback. This closing write is on the footer
   path only, so plain byte-identity is untouched. The status row exists only while
   `_status_open` — and on most attended runs it NEVER opens (no core code calls
   `session.status()`; only plugins like the yam ticker do), so the one-row state
   is the default, not an edge case. The cursor lives at the end of the input line.
   All updates are relative repaints using `\r` and ANSI `\x1b[A` (cursor up) /
   `\x1b[K` (clear to end), and every sequence branches on `_status_open`:
   - **Two-row state** (status open): status update = save input, move up, clear,
     write status, move down, rewrite input; scrollback insert = `\r ESC[K`-clear
     the input row FIRST, move up, clear, write text + newline, write status +
     newline, rewrite input; `status(None)` = clear input row, move up, clear
     status row, write input row there (the footer collapses upward one line into
     the one-row state; typing continues on the retained input row).
   - **One-row state** (status never opened, or closed): scrollback insert = clear
     input row, write text + newline, redraw input row — NO cursor-up, there is no
     status row above and moving up would `ESC[K` real scrollback; the first
     `status(line)` = clear input row, write status + newline, redraw input row
     (creates the status row above, entering the two-row state).
   Repaints always clear before writing, which also self-heals after any stray
   third-party print (one smudged frame, then clean). BOTH rows clip to the
   terminal width (injectable `width_fn` defaulting to
   `shutil.get_terminal_size().columns`, re-read at each repaint so resizes
   re-clip): the input row clips to `width - 4` showing the tail; the status row
   clips to `width - 1`. An unclipped row that wraps to two physical lines desyncs
   every relative `\r`/`ESC[A` movement afterward — clipping is correctness, not
   cosmetics. `end_trial()` tears down whatever exists (clear input row; clear
   status row too if open) idempotently as the backstop (decision 1), leaving a
   clean terminal for the verdict prompt; plain mode keeps its existing
   close-with-newline semantics.
6. **Confirmations come from `poll()`, not from the editor — and only for
   console-sourced messages.** The editor cannot know whether a completed line is
   feedback, an end request, a verdict, or a usage typo — the console's parser
   decides. Footer-mode `poll()` delegates, then for each message in the returned
   `ConsolePoll` **whose source is the console** writes
   `write_line(f"[{label}] {text}")` where `label` is the CLI-chosen decision-3(b)
   value (`sent` on feedback rows, `noted` on end-only rows — do not hardcode
   `sent`). Source filtering follows the `ConsolePoll.sources` contract: empty
   `sources` means every message is console-sourced; otherwise confirm exactly the
   indices where `sources[i] == "console"`. Voice-sourced messages (#316) are
   already echoed by the merge loop as `voice: <text>` — a blanket iteration over
   `messages` would print a second, `[sent]`-stamped line for input the footer
   never delivered-confirmed. End requests and verdict lines print nothing (the
   episode visibly ends); usage hints already arrive via the console's `output_fn`,
   which is `write_line`. The confirmation therefore appears at the next rollout
   poll, at most one control period after Enter — imperceptible.
7. **The console keeps line assembly; footer-mode `poll()` pumps the editor first.**
   The pump cannot live in a passive `readable` wrapper: `OperatorConsole.poll()`
   only calls `read()` while `readable()` is true, so a queue-checking `readable`
   plus an fd-draining `read` would never drain the fd at all — no echo, no
   completed line, deadlock. Instead, footer-mode `session.poll()` runs an explicit
   pump **before** the console-delegate portion of the merged poll
   (session.py:70-91) — pump → `console.poll()` → attached-input merge, with the
   merge loop (voice echo, detach-on-raise, `sources` stamping) untouched after
   it: while the fd is readable, `os.read` chunks, feed the
   editor (echoing per decision 5), and append each completed raw line to a queue;
   then call `console.poll()`, whose session-provided seams are dispatching
   closures — `readable` returns "queue non-empty or real EOF seen", `read` pops
   the queued lines concatenated **each with its terminating `"\n"`** (the console
   completes lines only on `"\n" in buffer`, so a line handed over without its
   newline would sit unparsed until the next Enter; `""` returns only on real EOF,
   preserving the console's EOF contract); in plain mode the same closures delegate
   to the fd seams. Seam types: `fd_readable: () -> bool` defaults to `console.py`'s
   existing pragma'd module-private `_stdin_readable` (imported — same package);
   `fd_read: () -> bytes` is **bytes-typed** so the editor can buffer raw bytes
   across split UTF-8 sequences (decision 4) — its default is a new one-line
   module-private helper `_stdin_read_bytes()` returning
   `os.read(sys.stdin.fileno(), 65536)` (pragma'd body; the one new fd pragma line,
   see decision 8). Console.py's str-typed `_stdin_read` cannot be the default: it
   decodes per chunk, destroying split sequences before the editor sees them. The
   plain-mode dispatching `read` closure decodes each chunk itself with
   `decode("utf-8", errors="replace")` — the same str per chunk that the console's
   own default produces today, and pure covered logic over the injected `fd_read`.
   This keeps `OperatorConsole` byte-for-byte untouched:
   the fd-desync rationale of plan 0042 decision 9 targets the buffered
   TextIOWrapper, which this pump is not. Footer-mode `session.begin_trial()`:
   drain the fd raw with **no echo** (stale pre-trial bytes must not paint into a
   not-yet-drawn footer), clear the editor buffer and the queue, then run the
   existing `begin_trial()` fan-out (session.py:93-104) — `console.begin_trial()`'s
   own drain loop sees an empty queue and is a no-op, it clears its internal
   buffer as today, and the attached-input fan-out with detach-on-raise runs
   unchanged after.
   Echo cadence note (document in docs, it is a property, not a bug): the pump runs
   at the rollout poll cadence, once per control step, so on a self-paced robot
   executing a chunk, keystroke echo can lag up to one `embodiment.step()`. Threads
   are excluded by the 0048 arc; state it.
8. **Raw-mode syscalls are the only NEW uncovered lines.** `_enter_cbreak()` /
   `_restore()` module-private helpers own `termios.tcgetattr`/`tcsetattr`
   (`# pragma: no cover` bodies, lazy `import termios`), injectable as
   `raw_mode_fn`-style seams on the session for tests. The session's fd access:
   `fd_readable` reuses `console.py`'s already-pragma'd module-private
   `_stdin_readable`; `fd_read` defaults to the new bytes-typed
   `_stdin_read_bytes()` (decision 7), whose one-line body is the only new fd
   pragma line — decoding lives in covered closure code. The editor, renderer,
   dispatching closures, `end_trial` paths, and mode fallback are all pure logic
   over injected callables — fully covered.

## Behavior notes (document in CHANGELOG/docs, not code comments)

- Operators see their typing on a stable `> ` line; the timer above never tears.
- Feedback confirms as `[sent] …` scrollback (`[noted] …` in end-only mode); Enter
  still ends; `/y /n /p` still verdict; typos still print the usage hint — all above
  the footer.
- Keystroke echo is pumped at the rollout poll cadence: on a self-paced robot it can
  lag up to one control step. Property of the threadless design, documented.
- Off-TTY/Windows/piped stdin: unchanged.
- Third-party prints (driver chatter) can smudge one frame; the next repaint heals it.

---

### Task 1: line editor + footer renderer (pure logic)

**Files:** `src/inspect_robots/session.py`, `tests/test_session.py`

- [ ] **Step 1: failing tests.** Editor + pump (decision 7): scripted `fd_readable`/
  `fd_read` sequences drive `poll()` — append/backspace/Ctrl-U/Enter incl. split
  UTF-8 and 64KiB paste; backspace after a multi-byte character removes the whole
  character (buffer and echo both clean); arrow-key/Delete/F-key escape sequences
  (`\x1b[A`, `\x1b[3~`, SS3 `\x1bOP`, bare `\x1b` + byte) are swallowed whole — no
  junk in the buffer or the echo — including an escape sequence split across two
  `fd_read` chunks and across two `poll()` calls (discard state persists); bytes
  without a newline echo but deliver nothing; the
  completed line reaches the console parser verbatim on the same `poll()`; the
  dispatching `readable`/`read` closures satisfy the console's EOF contract (`""`
  only on real EOF); footer `begin_trial()` discards stale fd bytes with no echo and
  clears editor + queue. Renderer, exact byte sequences in BOTH states of the
  decision-5 machine: input row drawn at footer `begin_trial()`; footer
  `begin_trial()` on a session with a plain-open status line emits the closing
  `"\n"` then a clean `> ` row, and the next `status(line)` takes the one-row
  branch (no `ESC[A` into scrollback); one-row
  `write_line` (no cursor-up — prior scrollback untouched); first `status(line)`
  creates the status row from the one-row state; two-row status update; two-row
  `write_line` insert (input row cleared first); input echo in both states,
  including typing after `status(None)`; `status(None)` collapses two-row to
  one-row (input row retained); `end_trial()` teardown from the one-row state,
  the two-row state, and (idempotently) after a prior teardown; tail-clipping of
  the input row at `width - 4`, clipping of the status row at `width - 1`,
  re-clip after a `width_fn` change. Plain-mode tests untouched.
- [ ] **Step 2-4: fail → implement → green.**

### Task 2: raw-mode window — `begin_trial()`/`end_trial()` + rollout hook

**Files:** `src/inspect_robots/session.py`, `src/inspect_robots/rollout.py`,
`tests/test_session.py`, `tests/test_rollout_observation_step.py`

- [ ] **Step 1: failing tests.** Session: footer entered only when enabled+tty+raw-ok
  AND the console is session-built (caller-injected `console=...` → documented no-op);
  each gate falsified via injected seams; failed entry → plain mode this trial and
  the next trial retries; `end_trial()` closes the footer rows and restores exactly
  once (idempotent; saved termios attrs cleared on restore so atexit cannot replay
  stale state); restore-on-exception. Rollout: `end_trial` called in the per-trial
  `finally` (rollout.py:466) on success, on trial-ending exceptions, and on the
  KeyboardInterrupt cancelled-trial path; skipped without error when the hook is
  absent (a bare `OperatorInput`); a raising `end_trial` warns and never masks the
  trial's own outcome (mirror the `poll()` degrade tests in
  `tests/test_rollout_observation_step.py`); `end_trial` still called after a
  raising `poll()` flipped `console_ok` False mid-trial (decision 1 guard scope);
  `end_trial()` on a session that never entered footer mode writes zero bytes and
  does not touch `_status_open` (plain-mode byte-identity backstop).
- [ ] **Step 2-4: fail → implement → green.**

### Task 3: CLI opt-in + `[sent]` confirmations

**Files:** `src/inspect_robots/cli.py`, `src/inspect_robots/session.py`,
`tests/test_registry_cli.py`, `tests/test_session.py`

- [ ] **Step 1: failing tests.** Footer enabled on exactly the plan-0048 decision-9
  rows where the console turns on (both new-hook and legacy ladders), never
  otherwise; the CLI passes the confirmation label per row — `[sent]` on feedback
  rows, `[noted]` on end-only rows (decision 3, contract point b); footer `poll()` emits one labeled
  confirmation per CONSOLE-sourced feedback message, nothing for ends/verdicts;
  a poll carrying a voice-sourced message (attached input, #316) emits no
  confirmation beyond the merge loop's existing `voice: <text>` echo — both with
  non-empty `sources` and in the empty-`sources` all-console case (decision 6);
  matrix outputs otherwise unchanged.
- [ ] **Step 2-4: fail → implement → green.**

### Task 4: docs, module map, changelog, gates

**Files:** `src/inspect_robots/CLAUDE.md`, `CHANGELOG.md`, `docs/guide/cli.md`

- [ ] Rows, entry (Added: footer; the arc's user-visible payoff — reference plans
  0048/0051 and #308), docs paragraph with the footer example block.
- [ ] **Full gates** at 100%.
