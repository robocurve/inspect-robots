# OperatorSession: one owner for attended-run operator I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the terminal exactly one owner during attended runs. Today stdin has four
readers (core's `OperatorConsole`, yam's readiness gate, yam's legacy end-keypress poll,
the CLI's verdict prompt) and stdout has three writers (yam's `\r` status ticker, CLI
prints, driver chatter), coordinated by hand-tuned drains and the `defer_operator_end()`
truce hook. The visible symptom is operator interjections garbled by ticker redraws;
the structural problem is that no component owns the terminal. This plan introduces a
core `OperatorSession` that owns all operator I/O for a run, **behavior-identically** —
rendering and prompts keep today's exact output so every existing test passes untouched.

**The arc (this is PR 1 of 3):**

1. **This plan (core):** `OperatorSession` absorbs the interjection console, the
   post-trial verdict/notes prompts, a status-line renderer, and a readiness-gate API.
   New duck-typed embodiment hook `connect_operator_session(session)`. No visible change.
2. **yam plan (separate repo):** the yam embodiment implements the hook — routes its
   ticker through `session.status(...)`, its readiness gates through `session.gate(...)`,
   deletes its `defer_operator_end` stdin choreography, and bumps its `inspect-robots`
   floor. The legacy keypress poll (`default_poll_end`) is **kept as the fallback** for
   runs where the hook is never called (`--no-prompt`, non-TTY, win32, direct
   `rollout()` — see decision 5a) so those runs can still end an episode before
   `max_steps`. Session-connected attended VLA runs switch from "any key ends the
   episode" to "Enter ends the episode" (unified; bumped keys no longer kill episodes)
   and gain `/y /n /p` verdicts and logged operator notes.
3. **core follow-up plan:** footer rendering inside `OperatorSession` — cbreak mode,
   echo off, a fixed two-line footer (status line + owned input line), sent-message
   confirmations in scrollback. Purely a rendering change inside one class by then.

**Architecture:** A new core module `session.py` (console.py keeps its single purpose:
the polled fd-level input primitive). `OperatorSession` composes an `OperatorConsole`
and implements the `OperatorInput` Protocol by delegation, so `rollout()` and `eval()`
are untouched. It adds four capabilities behind injectable seams: `status(line)` (an
exact port of yam's `\r`-rewriting one-line renderer — the future single render path),
`write_line(text)` (scrollback print that closes an open status line first and repaints
it after), `gate(prompt, hint=None)` (blocking readiness confirmation that fd-flushes
stale console input first and raises `EmbodimentFault` on dead stdin), and the verdict
prompts `prompt_verdict` / `prompt_verdict_on_operator_end` (moved verbatim from
`cli.py`). The CLI builds one session per attended run, passes it to `eval()` as
`operator_input` and its bound prompt method as `before_scoring`, and offers it to the
embodiment via the new duck-typed `connect_operator_session(session)` hook. When an
embodiment accepts the hook it thereby promises never to touch stdin/stdout status
itself, so the session's console turns on for **every** policy (someone must own the
end-of-episode keypress); until an embodiment ships the hook, the existing
`defer_operator_end` / `is_simulated` gating applies unchanged.

**Tech Stack:** Python 3.10+, stdlib + numpy only in core; pytest; no new deps.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
- D1 docstrings on public defs; state the contract, not the name. Line length 100.
- Zero behavior change: no embodiment implements the new hook yet, so every run takes
  today's paths. Existing tests pass untouched except where a module-private CLI helper
  moved (those tests move with it).
- Public API is fenced by `inspect_robots.__all__` **and** `tests/test_api_snapshot.py`
  — update both together (`OperatorSession` becomes public).
- Update `src/inspect_robots/CLAUDE.md`'s module table (new `session.py` row; adjust
  `cli.py` and `console.py` rows) and `CHANGELOG.md` under `## [Unreleased]` →
  `### Added`/`### Changed` (Keep a Changelog format, plan + issue reference).
- TTY-bound default callables are isolated, injectable, and `# pragma: no cover` on
  the TTY-bound lines only — the same discipline as `console.py` and plan 0042.
- Worktree: `.claude/worktrees/operator-session`; run everything via `uv run ...`
  there. Reference the tracking issue in commit messages; end each with the
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. **No git operations
  from Codex** — the orchestrating session commits.

## Reference: current wiring (main @ 01fd2659)

- `src/inspect_robots/console.py` (125 lines, read it whole) — `OperatorConsole`
  (injectable `readable`/`read`/`output_fn`), `ConsolePoll`, `EndRequest`,
  `OperatorInput` Protocol, `USAGE`. The usage hint on a bad slash command prints via
  `output_fn` (`:112`) — in the session this routes through `write_line`.
- `src/inspect_robots/cli.py:697-702` — `_PROMPT`, `_NOTES_PROMPT`, `_PROMPT_ANSWERS`,
  `_DEFINITIVE_REASONS`; `:705-750` — `_prompt_operator` (verdict adoption, prompt
  loop, notes, `operator_event` emission); `:753-763` — `_prompt_operator_on_operator_end`;
  `:766-772` — `_attended`; `:775-805` — `_build_operator_console` (Windows notice,
  `accepts_operator_messages` gate, `defer_operator_end` call site, `is_simulated`
  fallback, legacy notice, USAGE print); `:1375-1387` — `run`'s `before_scoring` +
  console wiring (ad-hoc operator-scorer branch vs `OPERATOR_END`-only branch);
  `:1426` — `eval(..., before_scoring=...)`; `:1522-1544` — `eval-set` equivalents.
- `src/inspect_robots/rollout.py:288-336` — the poll/drain/inject loop: messages →
  store + `operator_message_event`, end → `OPERATOR_END` + judgement/note stamping,
  tail-since-last-inference injection under `policy_extra["operator_messages"]`.
  Rollout is **not modified** by this plan.
- `src/inspect_robots/embodiment.py:72-135` — Protocol docstring documenting the
  optional duck-typed hooks (`bind_task` at `:72`, `defer_operator_end` at `:85`);
  document `connect_operator_session` here.
- `src/inspect_robots/errors.py` — `EmbodimentFault` (always-halt) for `gate()`.
- `src/inspect_robots/__init__.py:23,74-87` — console exports; add `OperatorSession`.
- Sibling repo (context only, do not touch):
  `inspect-robots-yam/src/inspect_robots_yam/embodiment.py:316-321` — `_default_status`
  (the renderer `status()` ports: `print(f"\r  {line}   ", end="", flush=True)`; `None`
  closes with a newline); `operator.py:41-76` — `wait_ready(drain, flush_first)` whose
  flush-first safety rationale `gate()` inherits; yam plan 0019 records the truce this
  arc dismantles.
- Tests to mirror or move: `tests/test_console.py`, `tests/test_registry_cli.py`
  (CLI harness + prompt tests — `grep -n "prompt_operator\|_PROMPT" tests/*.py`),
  `tests/test_api_snapshot.py`.

## Design decisions (and why)

1. **New module `session.py`, composing — not replacing — `console.py`.** The console
   stays the threadless input primitive with its hard-won fd-level discipline (0042
   decision 9). The session is the orchestration layer: it owns *who* reads and *what*
   renders. One class per concern keeps the 100%-coverage tests small and the PR-3
   footer change local to the session.
2. **The session implements `OperatorInput` by delegation** (`poll`/`begin_trial`
   forward to the composed console). `rollout()` and `eval()` signatures and behavior
   are untouched; the CLI simply passes the session where it passed the console. When
   the console channel is disabled (Windows, VLA policy on a hook-less embodiment) the
   session still exists — it owns the verdict prompts — and `poll()` returns empty
   results; the CLI passes `operator_input=None` in exactly the cases it does today, so
   rollout's channel bookkeeping is unchanged.
3. **`status(line: str | None)` is an exact port of yam's `_default_status`**, plus
   one piece of state: the session remembers whether a status line is open. `None`
   closes it with a newline (idempotent — closing a closed line prints nothing, unlike
   the yam original, so embodiment teardown paths cannot stack blank lines).
   `write_line(text)` closes an open status line, prints the text as a normal line, and
   repaints the remembered status line after. This is dormant until PR 2 routes the yam
   ticker here, but it ships tested now so PR 2 is wiring only. The console's usage
   hint (`output_fn`) is constructed to point at `write_line` so a typo'd slash command
   never tears the ticker once both route through the session.
4. **`gate(prompt, *, hint=None)` raises `EmbodimentFault` on dead stdin** — it is
   called from inside `embodiment.reset()` where a missed safety confirmation must
   halt the eval (yam's `wait_ready` already established this). It fd-flushes pending
   input *before* printing the prompt (stale interjection lines must never stand in
   for a "stand clear" confirmation — yam plan 0019 decision 3), and never drains
   *after* (pending lines belong to the console; `begin_trial()` handles them). `hint`
   lets the embodiment append rig-specific advice to the fault message.
5. **`connect_operator_session(session)` supersedes `defer_operator_end()`; both stay
   supported.** The new hook is a strict superset: accepting the session *is* the
   stand-down promise (an embodiment holding a session must never read stdin or print
   status itself). The CLI tries the new hook first, then falls back to today's exact
   `defer_operator_end` / `is_simulated` / notice-and-off ladder. `defer_operator_end`
   is documented as superseded in the Protocol docstring but keeps working — old-yam +
   new-core must run through the transition window (rigs install from PyPI).

   **5a. The hook is an optional input, not a guarantee.** `connect_operator_session`
   is only called on attended, console-capable runs; an embodiment that ships it still
   runs with it never called under `--no-prompt`, non-TTY stdin, win32, direct
   `rollout()`/`eval()` API use, and older cores. The Protocol docstring must mandate
   the same graceful-fallback language `bind_task` uses (`embodiment.py:79-83`):
   an embodiment may not depend on the hook firing, and must keep a working
   end-of-episode path when it never does (yam keeps its legacy keypress poll for
   exactly this case; see arc item 2).
6. **On a session-aware embodiment the console turns on for every policy, not just
   message-accepting ones.** Once the embodiment stands down, someone must own the
   end-of-episode keypress; that is the session. `accepts_operator_messages` now
   selects the usage wording, not the channel: the full feedback line for accepting
   policies, an end-only line (`operator console: Enter ends the episode; /y /n /p
   [note] records a verdict; typed notes are saved to the log`) otherwise. Typed
   messages are recorded as transcript events and injected under
   `extra["operator_messages"]` unconditionally — non-accepting policies never read
   that reserved key (it is additive `extra`, invisible to VLA adapters), and the
   grader gets the operator's notes in the log either way. This makes `/y`-style
   verdicts and logged notes work for VLA runs the moment yam ships PR 2 — the
   "any key" → "Enter" unification is called out in the yam changelog, not here.
   Today this branch is dead code behind the hook check (no embodiment has the hook),
   which is what keeps this PR behavior-identical; it ships tested via a stub
   embodiment so PR 2 activates it with zero core changes.
7. **The verdict prompts move to the session verbatim** — same constants, same loop,
   same `operator_event` emissions, same EOFError tolerance — as
   `prompt_verdict(record, scene)` and `prompt_verdict_on_operator_end(record, scene)`,
   reading through the session's injectable `input_fn`, printing via `write_line`. The CLI keeps its
   branch choosing which one becomes `before_scoring` (ad-hoc operator-scored runs
   prompt every non-definitive trial; registered tasks prompt only operator-ended
   trials — R6). The module-private `cli._prompt_operator*` helpers are deleted, their
   tests move to the session's test file, and `_PROMPT`/`_NOTES_PROMPT` constants move
   with them. Prompting through `input()` while the console also polls stdin stays
   safe for the reason 0042 established: the console is threadless, so nothing is
   blocked on stdin between polls, and prompts only run between trials.
8. **One session per CLI run, built whenever `_attended(args)`** — even when the
   console channel is off (session still owns the prompts). `_build_operator_console`
   becomes `_build_operator_session(policy, embodiment) -> tuple[OperatorSession,
   OperatorSession | None]`: called only from inside the `_attended(args)` branches,
   it always returns the session (first element — the prompts owner) and returns the
   same object or `None` as the second element depending on console enablement (what
   gets passed to `eval()` as `operator_input`). Keep it a single function so the
   enablement rules stay in one place.
9. **Enablement matrix and check order.** The helper's checks run in this exact
   order, chosen to preserve today's observable behavior on every existing path
   (notably: a non-accepting policy on a hook-less embodiment is **silently** skipped
   before the platform check, even on Windows — `cli.py:777-787` checks the policy
   first, and `test_policy_without_opt_in_skips_console_silently_before_platform_gate`
   pins it):

   1. embodiment has `connect_operator_session` → new-hook branch (see below);
   2. else policy does not accept messages → console off, **silent** (today);
   3. else win32 → console off + today's Windows notice;
   4. else `defer_operator_end` present → call it, console on, feedback usage (today);
   5. else `is_simulated` → console on, feedback usage (today);
   6. else → console off + today's legacy-hardware notice.

   New-hook branch (dead code until PR 2 — no embodiment ships the hook yet, which is
   what keeps this PR behavior-identical; ships tested via a stub embodiment):
   on win32 the console stays off with the Windows notice (the hook is **not**
   called — no console exists for the session to own); otherwise call the hook
   exactly once, before `eval()`, regardless of policy, then console on with the
   feedback usage line for accepting policies or the end-only line for the rest:

   | Platform | Policy accepts msgs | console (`operator_input`) | usage line |
   |---|---|---|---|
   | win32 | any | off + Windows notice | none |
   | posix | yes | on | feedback |
   | posix | no | on | end-only |
10. **Public API and docs.** `OperatorSession` joins `inspect_robots.__all__` and the
    API snapshot. The `Embodiment` Protocol docstring documents the new hook's full
    contract (stand-down promise, what the session provides, `gate` fault semantics).
    `docs/` gets a sweep: `grep -rn "operator console\|any key" docs/` and update any
    page describing the console gating (end-only mode is new prose).

---

### Task 1: `session.py` — status renderer, `write_line`, console delegation

**Files:**
- Create: `src/inspect_robots/session.py`
- Test: `tests/test_session.py`

**Interfaces (public, D1 docstrings):**
- `class OperatorSession:` constructor
  `(*, console: OperatorConsole | None = None, input_fn: Callable[[str], str] | None =
  None, write: Callable[[str], None] | None = None, flush_fn: Callable[[], None] |
  None = None)` — `console=None` constructs a default `OperatorConsole` whose
  `output_fn` is this session's `write_line`; `flush_fn` defaults to a module-private
  fd-level flush (zero-timeout `select` + `os.read` discard loop, no-op off-TTY,
  TTY-bound lines `# pragma: no cover`, mirroring
  `inspect_robots_yam.operator._flush_stdin_fd`).
- **Late-bound defaults, not def-time bindings.** `input_fn=None` resolves to the
  `input` builtin at **call time** (a def-time `= input` default binds the original
  function object at import, breaking the existing end-to-end CLI tests that
  monkeypatch `builtins.input` and run `main()` —
  `test_operator_prompt_records_verdict_and_reprompts_on_typos` et al. must pass
  untouched). Likewise `write=None` resolves `sys.stdout` per call (write + flush),
  never binding `sys.stdout.write` at construction, so capsys/monkeypatched stdout
  keeps working.
- There is exactly **one** output seam: `write` receives exact strings (including
  `\r` prefixes, no implicit newline). `write_line` and `status` are built on it;
  everything the session prints (usage hints, prompt announcements, verdict
  reprompts) goes through `write_line`. No `print`-shaped second path — two
  independent stdout paths would reintroduce the tearing this arc removes.
- `poll() -> ConsolePoll`, `begin_trial() -> None` — delegate to the composed console;
  the session satisfies `isinstance(..., OperatorInput)` (runtime-checkable).
- `status(line: str | None) -> None` — `\r  {line}   ` in place via the `write` seam;
  tracks open/closed; `None` closes with a newline only if open.
- `write_line(text: str) -> None` — close open status, print text, repaint status.

- [x] **Step 1: failing tests.** Delegation (`poll`/`begin_trial` forward; default
  console wired with `write_line` as its `output_fn`; `OperatorInput` isinstance).
  Status lifecycle: open → repaint → close → double-close is a no-op; exact byte
  sequences asserted on a recording `write` seam (`\r  t = 3s   `, closing `\n`).
  `write_line` while open: close, line, repaint (exact order); while closed: plain line.
- [x] **Step 2: run, confirm FAIL.**
- [x] **Step 3: implement.**
- [x] **Step 4: green + `ruff` + `mypy` clean.**

### Task 2: `gate()`

**Files:** `src/inspect_robots/session.py`, `tests/test_session.py`

**Interface:** `gate(prompt: str, *, hint: str | None = None) -> None` — call
`flush_fn()`, then `input_fn(prompt)`; on `EOFError`/`OSError` raise
`EmbodimentFault` whose message states the problem, remedies (real TTY / injectable
input / unattended mode), and appends `hint` when given. Never drains after.

- [x] **Step 1: failing tests.** Flush ordering (flush recorded strictly before
  input); success path returns None; each of EOFError/OSError → `EmbodimentFault`,
  message contains the remedy text and the hint iff provided; no trailing drain
  (input_fn called exactly once, flush_fn exactly once).
- [x] **Step 2-4: fail → implement → green.**

### Task 3: verdict prompts move into the session

**Files:** `src/inspect_robots/session.py`, `src/inspect_robots/cli.py`,
`tests/test_session.py`, prompt tests moved from their current home
(`grep -rn "prompt_operator\|grader notes" tests/`)

**Interfaces:** `prompt_verdict(record, scene)` and
`prompt_verdict_on_operator_end(record, scene)` — verbatim ports of
`cli._prompt_operator` / `_prompt_operator_on_operator_end` (constants move too),
reading via `input_fn`, printing via `write_line` (equivalent to `print` while no
status line is open — which is always true between trials today).

- [x] **Step 1: move the existing prompt tests** to `tests/test_session.py`, adapted
  to injected `input_fn`/recorded output; add one test that announcements go through
  `write_line`. CLI tests keep passing with `before_scoring` now a bound session method.
- [x] **Step 2-4: fail → implement (delete the cli helpers, wire bound methods at
  `cli.py:1375-1387` and `:1522-1528`) → green.**

### Task 4: CLI enablement — `_build_operator_session` and the connect hook

**Files:** `src/inspect_robots/cli.py`, `tests/test_registry_cli.py` (or the CLI
console-gating test file found via `grep -rn "defer_operator_end" tests/`)

**Interface:** one helper replacing `_build_operator_console`, returning
`(session, operator_input)` per the decision-9 matrix and check order; it prints the
applicable usage/notice lines and calls `connect_operator_session` (preferred) or
`defer_operator_end` (fallback) exactly once. The end-only usage text lives in
`console.py` next to `USAGE` as `USAGE_END_ONLY` (module-level, not exported —
`USAGE` itself is not in `__all__` either).

- [x] **Step 1: failing tests.** Matrix rows as parametrized cases with stub policies
  (with/without `accepts_operator_messages`) and stub embodiments (with the new hook /
  with only `defer_operator_end` / simulated / bare hardware): assert hook call counts,
  which usage text printed, and whether `operator_input` is the session or `None`.
  Windows row via `sys.platform` monkeypatch. New-hook rows assert the console is on
  even for a non-accepting policy (end-only wording).
- [x] **Step 2-4: fail → implement → green.**

### Task 5: API, protocol docs, module map, changelog, docs sweep

**Files:** `src/inspect_robots/__init__.py`, `tests/test_api_snapshot.py`,
`src/inspect_robots/embodiment.py` (docstring only), `src/inspect_robots/CLAUDE.md`,
`CHANGELOG.md`, `docs/**` (sweep)

- [x] `OperatorSession` exported + snapshot updated together.
- [x] Protocol docstring: `connect_operator_session` contract; `defer_operator_end`
  marked superseded-but-supported.
- [x] CLAUDE.md module table: `session.py` row; `console.py`/`cli.py` rows adjusted.
- [x] CHANGELOG under Unreleased (Added: OperatorSession + hook; Changed: verdict
  prompts relocated — behavior identical). Plan + issue references.
- [x] Docs sweep for operator-console prose; add end-only mode where gating is
  described. Writing-style rules apply (no em dashes, no mid-sentence bold).
- [x] **Full gates:** `uv run ruff check . && uv run ruff format --check . &&
  uv run mypy && uv run pytest --cov -q` — 100%.
