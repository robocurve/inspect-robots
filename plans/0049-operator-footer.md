# Operator footer: fixed status line and an owned input line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

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
- Before merge: re-check `git ls-tree origin/main plans/` for a 0049 collision and
  renumber if a concurrent session took it.

## Reference: current wiring (main @ 6c022ead)

- `src/inspect_robots/session.py` — the whole module (162 lines): `_flush_stdin_fd`,
  `OperatorSession.__init__` seams (`console`, `input_fn`, `write`, `flush_fn`),
  `_write`/`_input` late-binding, `poll`/`begin_trial` delegation, `status`/`write_line`
  state machine (`_status_open`, `_status_line`), `gate`, verdict prompts.
- `src/inspect_robots/console.py` — `OperatorConsole(readable, read, output_fn)`;
  `poll()` assembles lines from chunks (`:90-113`); `begin_trial()` drains (`:115-124`).
  The console's default `read` is fd-level `os.read` (`:55-56`); the session currently
  builds its default console with only `output_fn=self.write_line`.
- `src/inspect_robots/rollout.py:276-283` — `operator_input.begin_trial()` inside a
  try/except that disables the channel on failure; `:288-306` — the per-iteration poll
  with the same degrade discipline; the trial's `finally` teardown block (find with
  `grep -n "finally" src/inspect_robots/rollout.py`) is where `end_trial()` goes.
- `src/inspect_robots/cli.py` — `_build_operator_session` (decision-9 ladder): the
  footer activates only on rows where the console turns on; the helper tells the
  session (see decision 3).
- Tests: `tests/test_session.py` (exact-byte rendering tests to extend),
  `tests/test_rollout_hardening.py` (channel-degrade discipline to mirror for
  `end_trial`), `tests/test_registry_cli.py` (matrix rows unchanged).

## Design decisions (and why)

1. **Per-trial raw-mode window via `begin_trial()`/`end_trial()`.** Terminal mode must
   be canonical between trials: the verdict prompts (`input()`), the yam gates, and
   any embodiment homing output run there. `begin_trial()` already exists on the
   `OperatorInput` protocol; `end_trial()` joins as a **duck-typed optional** method
   (same pattern as every other optional hook) so third-party `OperatorInput`
   implementations stay conformant. `rollout()` calls it best-effort in the trial's
   `finally`: wrapped in the same catch-warn-disable discipline as `poll()`, and it
   must run even when the trial raised (`EmbodimentFault` mid-trial must not leave the
   user's terminal in raw mode). Restore is idempotent and additionally registered
   with `atexit` once per session as a crash net; `KeyboardInterrupt` unwinds through
   the `finally` like any exception.
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
4. **Line editing is deliberately minimal:** printable characters append, backspace
   (`\x7f`/`\x08`) deletes one, Ctrl-U kills the line, `\n`/`\r` completes it
   (echoing `\r` as line completion matches raw-mode Enter), everything else
   non-printable is ignored. Multi-byte UTF-8 arrives whole from `os.read` chunks in
   practice, but the editor must tolerate split sequences: buffer bytes, decode with
   `errors="replace"` only at display time, and hand the console the raw completed
   line so message content is preserved exactly as today. Paste works because chunk
   processing is loop-per-byte over arbitrarily large reads. No history, no arrow
   keys, no cursor movement (YAGNI; the plan-0048 arc can add them later if operators
   ask).
5. **Rendering model: the cursor lives at the end of the input line.** All updates
   are relative two-line repaints using `\r` and ANSI `\x1b[A` (cursor up) /
   `\x1b[K` (clear to end): status update = save input, move up, clear, write status,
   move down, rewrite input; scrollback insert = move up, clear, write text + newline,
   write status + newline, rewrite input. Repaints always clear before writing, which
   also self-heals after any stray third-party print (one smudged frame, then clean).
   The input display clips to `terminal width - 4` showing the tail (injectable
   `width_fn` defaulting to `shutil.get_terminal_size().columns`). In footer mode
   `status(None)` tears down both lines (clear input line, clear status line) so the
   trial ends with a clean terminal for the verdict prompt; plain mode keeps its
   existing close-with-newline semantics.
6. **`[sent]` confirmations come from `poll()`, not from the editor.** The editor
   cannot know whether a completed line is feedback, an end request, a verdict, or a
   usage typo — the console's parser decides. Footer-mode `poll()` delegates, then
   for each message in the returned `ConsolePoll.messages` writes
   `write_line(f"[sent] {text}")`. End requests and verdict lines print nothing (the
   episode visibly ends); usage hints already arrive via the console's `output_fn`,
   which is `write_line`. The confirmation therefore appears at the next rollout
   poll, at most one control period after Enter — imperceptible.
7. **The console keeps line assembly; the editor feeds it completed lines.** The
   session wraps the `readable`/`read` seams it passes to its default
   `OperatorConsole`: footer-mode `read` drains the real fd (`os.read`), runs bytes
   through the editor (echoing per decision 5), and returns only the completed lines
   (`"line\n"` joined) — an empty string return means EOF exactly as today, and "no
   completed lines yet" returns `""`? No: `""` signals EOF to the console. The
   wrapper must instead report `readable() == False` until a completed line exists —
   i.e. the session buffers keystrokes itself and its wrapped `readable` returns
   True only when completed lines are queued (or real EOF was seen). This keeps
   `OperatorConsole` byte-for-byte untouched: its fd-desync rationale (plan 0042
   decision 9) doesn't apply because the wrapper is not the buffered TextIOWrapper —
   it drains the fd fully on every rollout poll. `begin_trial()` clears the editor
   buffer, queued lines, and the composed console's buffer (existing behavior).
8. **Raw-mode syscalls are the only uncovered lines.** `_enter_cbreak()` /
   `_restore()` module-private helpers own `termios.tcgetattr`/`tcsetattr`
   (`# pragma: no cover` bodies, lazy `import termios`), injectable as
   `raw_mode_fn`-style seams on the session for tests. The editor, renderer, wrapped
   seams, `end_trial` paths, and mode fallback are all pure logic over injected
   callables — fully covered.

## Behavior notes (document in CHANGELOG/docs, not code comments)

- Operators see their typing on a stable `> ` line; the timer above never tears.
- Feedback confirms as `[sent] …` scrollback; Enter still ends; `/y /n /p` still
  verdict; typos still print the usage hint — all above the footer.
- Off-TTY/Windows/piped stdin: unchanged.
- Third-party prints (driver chatter) can smudge one frame; the next repaint heals it.

---

### Task 1: line editor + footer renderer (pure logic)

**Files:** `src/inspect_robots/session.py`, `tests/test_session.py`

- [ ] **Step 1: failing tests.** Editor: append/backspace/Ctrl-U/Enter over scripted
  chunk sequences incl. split UTF-8 and 64KiB paste; queued-lines `readable` protocol
  (False until a line completes; True after; EOF passthrough). Renderer: exact byte
  sequences for status update, write_line insert, input echo, teardown on
  `status(None)`, tail-clipping at injected width. Plain-mode tests untouched.
- [ ] **Step 2-4: fail → implement → green.**

### Task 2: raw-mode window — `begin_trial()`/`end_trial()` + rollout hook

**Files:** `src/inspect_robots/session.py`, `src/inspect_robots/rollout.py`,
`tests/test_session.py`, `tests/test_rollout_hardening.py`

- [ ] **Step 1: failing tests.** Session: footer entered only when enabled+tty+raw-ok
  (each gate falsified via injected seams); failed entry → plain mode this trial and
  the next trial retries; `end_trial()` restores exactly once (idempotent);
  restore-on-exception. Rollout: `end_trial` called in `finally` on success, on
  trial-ending exceptions, and skipped without error when the hook is absent
  (plain `OperatorConsole`); a raising `end_trial` warns and never masks the trial's
  own outcome (mirror the `poll()` degrade tests).
- [ ] **Step 2-4: fail → implement → green.**

### Task 3: CLI opt-in + `[sent]` confirmations

**Files:** `src/inspect_robots/cli.py`, `src/inspect_robots/session.py`,
`tests/test_registry_cli.py`, `tests/test_session.py`

- [ ] **Step 1: failing tests.** Footer enabled on exactly the decision-9 rows where
  the console turns on (both new-hook and legacy ladders), never otherwise; footer
  `poll()` emits one `[sent]` per feedback message, nothing for ends/verdicts; matrix
  outputs otherwise unchanged.
- [ ] **Step 2-4: fail → implement → green.**

### Task 4: docs, module map, changelog, gates

**Files:** `src/inspect_robots/CLAUDE.md`, `CHANGELOG.md`, `docs/guide/cli.md`

- [ ] Rows, entry (Added: footer; the arc's user-visible payoff — reference plans
  0048/0049 and #308), docs paragraph with the footer example block.
- [ ] **Full gates** at 100%.
