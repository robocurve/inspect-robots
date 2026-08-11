# Footer Echo Pump Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operators see what they type the moment they type it, even while the policy is blocked in a long LLM inference (#367).

**Architecture:** The footer window runs cbreak with `ECHO` off, so the only echo path is `OperatorSession._pump_input()`, which today runs solely inside `poll()` — once per rollout iteration, and each iteration blocks in `controller.next_action()`. For the `agent` policy that is a whole LLM round-trip, so typing is invisible until the returned chunk plays back ("text only shows when the robot moves"). Fix: a session-owned daemon **echo pump thread**, alive only between footer entry and `end_trial()`, that wakes every `echo_interval_s` seconds (`Event.wait` cadence — no new fd seams; the existing zero-timeout `fd_readable`/`fd_read` seams stay the only stdin readers) and runs the same `_pump_input()` under a shared `threading.RLock`. Completed lines still queue for `poll()`; the console primitive stays threadless; thread exceptions are stored and re-raised from the next `poll()` so rollout's existing degrade-and-warn path handles them.

This amends plan 0042 (`plans/0042-operator-console.md` — not the identically numbered config-file plan) design decision 1 ("polled, not threaded"). That decision's objection was a reader blocked in `readline()` racing the post-trial `input()` verdict prompt for stdin. The echo pump has neither property: it never blocks in a read (it drains only `select`-readable bytes), and it is joined in `end_trial()` before any verdict prompt can run, so no second stdin reader ever coexists with `input()`. Decision 1's latency claim ("one control period, imperceptible") was true for VLAs and false for LLM agent policies, where one loop iteration contains a full inference.

**Tech Stack:** Python 3.10+, stdlib only (`threading` joins `termios`/`select` in session.py); pytest; no dependency changes.

**Critique record:** round 1 — 12 findings (1 blocker: self-contradictory docs; 6 should-fix; 5 nits), all applied; lock coverage, deadlock freedom, degrade path, opt-in coverage, and 100%-coverage reachability traced and confirmed sound. Round 2 — 9 findings (1 blocker: wrong test-file name, the CLI tests live in `tests/test_registry_cli.py`; 3 should-fix, all test determinism/hygiene; 5 nits), all applied; locking, teardown ordering, crash-net interaction, degrade path, and per-line coverage re-verified sound. Round 3 — pending.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage** for core.
- Core stays NumPy-only; `threading` is stdlib.
- Ruff D1: docstring on every public module/class/function; state the contract, not the name.
- Public API is fenced by `inspect_robots.__all__` + `tests/test_api_snapshot.py`; this plan adds no package-level names (a keyword-only parameter with a default on `enable_footer` and a module constant are not snapshot entries).
- No `git` operations from the implementing agent (Codex writes files only); the main session commits.

## Design decisions (and why)

1. **Per-footer-window thread lifetime.** The thread starts as the last statement of `_enter_footer()` and is stopped + joined at the top of `end_trial()` (before the `_footer_active` guard, so a stale handle from an aborted entry can never survive; before the row-clearing writes and before `_restore_terminal()`). In-repo, verdict prompts and readiness gates only run outside the footer window, so the plan-0042 verdict-prompt guarantee holds — but that is convention, not construction, so the invariant is documented at the seams third parties see: `gate()` and `enable_footer()` docstrings state that the footer window owns stdin and blocking reads are only legal outside it. As a crash net, `_restore_terminal()` (the atexit-registered callable) also sets `_pump_stop` when one exists, so a daemon pump alive at interpreter shutdown stops echoing into a terminal whose termios was just restored.
2. **`Event.wait(interval)` cadence, default 0.05 s.** The loop is `while not stop.wait(interval): pump()`. No blocking reads, no new seams: the pump body is the existing `_pump_input()`, which drains only bytes the zero-timeout `fd_readable` seam reports. Echo latency is bounded by the interval; the 150 ms Esc grace (`_ESC_GRACE_S`) now settles within grace + interval instead of waiting out an entire inference, so a bare Esc is queued promptly (the episode still ends at the rollout loop's next iteration). **Amended accepted cost:** the grace floor is now enforced continuously rather than only at poll boundaries — an escape-sequence tail split more than 150 ms from its ESC byte (very laggy SSH) is consumed as a bare Esc even mid-inference, where today's starved poll cadence would have drained ESC and tail together. This is the same misread plan 0051 already accepts on any self-paced fast control loop (whose poll cadence matches the pump's 50 ms); the grace value stays 150 ms rather than growing a pumped/unpumped conditional, and the `_ESC_GRACE_S` comment is updated to name the pump as the governing cadence.
3. **Opt-in via `enable_footer(..., echo_interval_s=...)`, wired by the CLI.** Default `None` starts no thread, keeping every existing injected-seam test synchronous and deterministic. `cli._build_operator_session` — the only `enable_footer` caller — passes the session module's `ECHO_INTERVAL_S` constant at both call sites, so every real attended run gets the fix.
4. **One `threading.RLock` guards all shared footer state**: the line editor, `_line_queue`, `_esc_stamp`, `_pump_eof`, `_pump_error`, and every footer-row write (`_pump_input`'s echo, `_footer_status`, `_footer_write_line`). The lock MUST be re-entrant and its scope in `poll()` MUST end before the attached-input merge; both are load-bearing. Re-entrant because `console.poll()` runs inside the locked region and its usage-reminder path prints through `output_fn=self.write_line` (console.py:142-145), which takes the lock again — a plain `Lock` deadlocks the first time an operator hits bare Enter mid-inference. Scope-limited because the merge calls attached `source.poll()` (e.g. voice) and `_confirm_console_messages`, which must not run under the lock: attached sources can be slow and take only rollout-thread state, and `write_line` re-acquires the lock itself for exactly as long as it needs. `begin_trial()` wraps the console's drain (`self._console.begin_trial()`) in the lock too: with the thread already running, the drain reads `_line_queue` through the dispatch seams.
5. **Thread errors surface through the next `poll()`.** The pump loop catches `Exception`, stores it in `_pump_error`, and exits; `poll()` re-raises a stored error first thing. Rollout already treats a raising `poll()` as "disable the console for this trial, warn once, close the footer" (plan 0051 decision 1) — the thread reuses that path instead of inventing its own error channel. `_reset_footer_state()` clears `_pump_error` (safe: `poll()` consumes the error before raising, so the degrade path's own `end_trial()` cannot swallow the error it just surfaced) — clearing only in `_enter_footer()` would let a stale error from trial N kill trial N+1's plain-mode console when N+1's footer entry fails the gate ladder. **Accepted cost:** an error stored in the trial's final inter-poll window, with no `poll()` before `end_trial()`, is never surfaced — the window's typed input is lost and no warning fires. Fd-read errors are rare enough that a dedicated side channel is not worth its complexity.
6. **Threads never print or raise on their own.** The pump thread's only outputs are the same bytes `_pump_input()` always wrote, under the lock. No logging, no stderr, no daemon-thread tracebacks.

---

### Task 1: echo pump thread in `session.py`

**Files:**
- Modify: `src/inspect_robots/session.py`
- Modify: `src/inspect_robots/console.py` (module docstring only: the stdin-availability claim now also rests on the session joining its echo pump in `end_trial()` — add a pointer clause)
- Modify: `src/inspect_robots/CLAUDE.md` (session.py row: footer echo pump thread, per-window lifetime, poll-surfaced errors)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: existing `OperatorSession` seams (`fd_readable`, `fd_read`, `write`, `isatty_fn`, `raw_mode_fn`, `restore_fn`, `now_fn`, `atexit_register`, `width_fn`) — unchanged signatures.
- Produces: `ECHO_INTERVAL_S: float = 0.05` module constant in `session.py`; `enable_footer(*, label: str, echo_interval_s: float | None = None)`; internals `_echo_pump_loop(stop: threading.Event, interval: float) -> None`, `_stop_echo_pump() -> None`, attributes `_lock` (RLock), `_pump_thread: threading.Thread | None`, `_pump_stop: threading.Event | None`, `_pump_error: Exception | None`, `_echo_interval_s: float | None`. Task 2 relies on `ECHO_INTERVAL_S` and the `echo_interval_s` parameter existing with exactly these names.

- [ ] **Step 1: Write the failing tests** in `tests/test_session.py` (new test group; follow the file's existing seam-injection style for building footer sessions — TTY true, raw mode returning a token, recording `write` fn, scripted `fd_readable`/`fd_read` closures). Add a module-level helper:

```python
def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    """Poll ``predicate`` until true or fail the test after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    pytest.fail("timed out waiting for background echo pump")
```

Tests (each builds a footer-eligible session; `enable_footer(label="sent", echo_interval_s=0.001)` unless stated; `begin_trial()` enters the window):

```python
def test_echo_pump_thread_echoes_between_polls() -> None:
    """Typed bytes are echoed by the background pump with no poll() call at all."""
    # scripted seams: fd_readable True while chunks remain, fd_read pops b"hi"
    # after begin_trial (which drains stale bytes first, so arm the chunk list
    # only after begin_trial returns), never call poll();
    # _wait_until(lambda: any(w.endswith("> hi") for w in writes))

def test_echo_pump_lines_are_delivered_by_later_poll() -> None:
    """A line completed by the pump thread arrives as a normal message on the next poll."""
    # arm b"hello" and b"\n" as two chunks after begin_trial;
    # _wait_until(lambda: bool(session._line_queue))  # GIL-safe read; do NOT wait on
    # the echo text: the newline clears the editor, so the final draw is "> " alone
    # poll() returns ConsolePoll(messages=("hello",), ...)

def test_end_trial_joins_pump_thread_before_clearing() -> None:
    """end_trial stops the thread; no footer bytes are written after the clearing write."""
    # after begin_trial, capture session._pump_thread; end_trial();
    # assert thread is not None and not thread.is_alive()
    # assert session._pump_thread is None
    # record len(writes); sleep(0.01); assert len(writes) unchanged

def test_each_footer_window_gets_a_fresh_pump_thread() -> None:
    """A second begin_trial starts a new thread after the first window closed."""
    # begin_trial; t1 = session._pump_thread; end_trial();
    # begin_trial; t2 = session._pump_thread
    # assert t2 is not None and t2 is not t1 and t2.is_alive(); end_trial()

def test_pump_thread_settles_esc_grace_without_poll() -> None:
    """A bare Esc typed mid-inference is queued as the end sentinel by the pump alone."""
    # inject now_fn over a mutable clock; arm b"\x1b" after begin_trial;
    # _wait_until(lambda: session._editor.escape_pending is False or session._line_queue)
    # is racy — instead: _wait_until(lambda: session._esc_stamp is not None);
    # advance clock past _ESC_GRACE_S; _wait_until(lambda: END_SENTINEL in session._line_queue);
    # poll() returns an end request

def test_pump_thread_error_is_raised_from_next_poll() -> None:
    """A pump-thread exception disables nothing silently: the next poll raises it."""
    # fd_read raises RuntimeError("boom") once fd_readable flips True after begin_trial;
    # _wait_until(lambda: session._pump_error is not None)
    # thread = session._pump_thread; assert thread is not None
    # _wait_until(lambda: not thread.is_alive())  # error is stored under the lock,
    # BEFORE thread teardown finishes; waiting on _pump_error alone races is_alive()
    # with pytest.raises(RuntimeError, match="boom"): session.poll()
    # assert session._pump_error is None  (consumed)

def test_enable_footer_without_interval_starts_no_thread() -> None:
    """Default echo_interval_s=None keeps the footer fully synchronous."""
    # enable_footer(label="sent"); begin_trial();
    # assert session._pump_thread is None; poll/echo behavior unchanged; end_trial()

def test_stale_pump_error_is_cleared_on_next_window() -> None:
    """An unsurfaced pump error from a closed window cannot fail a later plain-mode poll."""
    # seams driven by a mutable flag: fd_read raises RuntimeError while the flag is set;
    # _wait_until(lambda: session._pump_error is not None)  # REQUIRED before end_trial:
    # without it, stop.set() can beat the pump's first wake and the test passes vacuously
    # then end_trial() WITHOUT poll(); clear the flag;
    # assert session._pump_error is None immediately after end_trial (cleared by
    # _reset_footer_state, not deferred to the next footer entry);
    # poll() returns an empty ConsolePoll

def test_atexit_restore_stops_a_live_pump() -> None:
    """The atexit crash net stops the pump so it cannot echo after termios is restored."""
    # begin_trial with echo_interval_s=0.001; thread = session._pump_thread;
    # call session._restore_terminal() directly (the atexit-registered callable);
    # _wait_until(lambda: not thread.is_alive()); restore_fn recorded exactly one call
```

Seam scripting must be arm-after-begin_trial: `begin_trial` → `_enter_footer` drains readable bytes with no echo before the thread starts, so chunks armed too early are silently consumed by the drain. Use a list the test appends to after `begin_trial()` returns, with `fd_readable = lambda: bool(chunks)` and `fd_read = lambda: chunks.pop(0)`.

Every test that starts a pump MUST call `end_trial()` in a `finally` (or use a small fixture that does): a leaked 1 ms daemon thread keeps pumping test-local closures for the rest of the pytest run and poisons other tests on any assertion failure. `tests/test_session.py` will also need `import time` for `_wait_until`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_session.py -q -k "echo_pump or stale_pump or without_interval or atexit_restore"`
Expected: FAIL — `enable_footer() got an unexpected keyword argument 'echo_interval_s'` and missing attributes.

- [ ] **Step 3: Implement in `session.py`**

Add `import threading` to the module imports. Add the constant next to `_ESC_GRACE_S`:

```python
# Background echo pump cadence. Echo latency while the policy blocks in inference is
# bounded by this interval; the Esc grace settles within _ESC_GRACE_S + this interval.
ECHO_INTERVAL_S = 0.05
```

Constructor additions (with the other state fields):

```python
self._echo_interval_s: float | None = None
self._lock = threading.RLock()
self._pump_thread: threading.Thread | None = None
self._pump_stop: threading.Event | None = None
self._pump_error: Exception | None = None
```

`enable_footer` gains the parameter and stores it (docstring: the interval opts the footer into a background echo pump thread so typing echoes while the rollout thread is blocked in policy inference; `None` keeps the footer synchronous — every embedded/test use — and the CLI passes `ECHO_INTERVAL_S`):

```python
def enable_footer(self, *, label: str, echo_interval_s: float | None = None) -> None:
    if self._owns_console:
        self._footer_requested = True
        self._footer_label = label
        self._echo_interval_s = echo_interval_s
        ...  # unchanged atexit registration
```

`_enter_footer()` — start the thread as the **last** statements, after the initial input-row draw (`_pump_error` is cleared in `_reset_footer_state()` only — per closed window, per decision 5 — not here):

```python
        if self._echo_interval_s is not None:
            stop = threading.Event()
            thread = threading.Thread(
                target=self._echo_pump_loop,
                args=(stop, self._echo_interval_s),
                name="operator-echo-pump",
                daemon=True,
            )
            # Assign only after a successful start: join() on a never-started thread
            # raises RuntimeError, and _try_enter_footer's abort path must leave no
            # handle a later end_trial's _stop_echo_pump would trip over.
            thread.start()
            self._pump_stop = stop
            self._pump_thread = thread
```

The loop and the stopper:

```python
    def _echo_pump_loop(self, stop: threading.Event, interval: float) -> None:
        """Echo typed bytes between rollout polls while the policy blocks (plan 0066).

        Runs the same pump ``poll()`` runs, under the shared lock, every ``interval``
        seconds until ``stop`` is set. A raising pump is stored for the next ``poll()``
        to re-raise (rollout's existing degrade path) and ends this thread; the thread
        itself never prints or raises.
        """
        while not stop.wait(interval):
            with self._lock:
                try:
                    self._pump_input()
                except Exception as exc:
                    self._pump_error = exc
                    return

    def _stop_echo_pump(self) -> None:
        """Idempotently stop and join the echo pump thread for this footer window."""
        if self._pump_thread is None:
            return
        assert self._pump_stop is not None
        self._pump_stop.set()
        self._pump_thread.join()
        self._pump_thread = None
        self._pump_stop = None
```

`end_trial()` — first statement, BEFORE the `_footer_active` guard (defense-in-depth: with start-then-assign no aborted entry can leave a handle today, but the stop must not depend on that invariant surviving future edits):

```python
        self._stop_echo_pump()
        if not self._footer_active:
            return
```

`_reset_footer_state()` — add `self._pump_error = None` to the existing resets (decision 5: cleared per closed window, not per opened one).

`_restore_terminal()` — before restoring, arm the crash net (decision 1):

```python
        if self._pump_stop is not None:
            self._pump_stop.set()
```

(placed after the `_NO_TERMIOS_STATE` early return, so a session that never entered footer mode is untouched; no join — atexit must not block on a wedged thread, and a set stop event is enough to end the loop before its next echo).

Update the `_ESC_GRACE_S` comment: the grace is now tested at the echo pump's cadence (`ECHO_INTERVAL_S`) whenever a pump is running, not only at quiet rollout polls; the misread-tail accepted cost is unchanged from plan 0051, it just applies mid-inference too.

Docstring edits carrying decision 1's invariant to the seams: `enable_footer()` states the footer window owns stdin between `begin_trial()` and `end_trial()` and blocking stdin reads are only legal outside it; `gate()` states it must not be called while a footer window is open; `console.py`'s module docstring gains a pointer clause (its no-background-reader guarantee now also relies on `OperatorSession` joining its echo pump in `end_trial()`).

`poll()` — surface a stored thread error, and take the lock around the pump + console poll (the console's dispatch seams read `_line_queue`):

```python
    def poll(self) -> ConsolePoll:
        """Poll the console first, then merge every healthy attached input in order."""
        with self._lock:
            if self._pump_error is not None:
                error = self._pump_error
                self._pump_error = None
                raise error
            if self._footer_active:
                self._pump_input()
            console_poll = self._console.poll()
        ...  # attached-input merge unchanged (write_line re-acquires the RLock)
```

`status()` and `write_line()` — wrap only the footer branches:

```python
        if self._footer_active:
            with self._lock:
                self._footer_status(line)   # / self._footer_write_line(text)
            return
```

`begin_trial()` — wrap the console drain (the thread may already be running):

```python
        self._try_enter_footer()
        with self._lock:
            self._console.begin_trial()
```

Update the `session.py` module docstring's ownership sentence and the `src/inspect_robots/CLAUDE.md` session.py row: footer windows may own a per-trial echo pump thread (interval from `enable_footer`, CLI passes `ECHO_INTERVAL_S`), joined in `end_trial()` before teardown, errors re-raised from the next `poll()`. The row's Esc-grace phrase ("on quiet polls") must also change to name the pump cadence, matching the updated `_ESC_GRACE_S` comment.

- [ ] **Step 4: Run the new tests, then the full gates**

Run: `uv run pytest tests/test_session.py -q` → PASS, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: all green, coverage 100%.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/session.py src/inspect_robots/console.py src/inspect_robots/CLAUDE.md tests/test_session.py
git commit -m "feat(session): footer echo pump thread so typing echoes during inference (#367)"
```

(Committed by the main session, not the implementing agent.)

---

### Task 2: CLI wiring + docs

**Files:**
- Modify: `src/inspect_robots/cli.py` (`_build_operator_session`, both `enable_footer` call sites)
- Modify: `docs/guide/cli.md` (footer paragraph — the stale cadence caveat, see Step 3)
- Modify: `CHANGELOG.md` (Unreleased → Fixed)
- Test: `tests/test_registry_cli.py`

**Interfaces:**
- Consumes: `ECHO_INTERVAL_S` and `enable_footer(*, label, echo_interval_s)` from Task 1.
- Produces: nothing new — behavior wiring only.

- [ ] **Step 1: Write the failing test** in `tests/test_registry_cli.py`, next to the existing `_build_operator_session` tests and reusing their policy/embodiment fakes:

```python
def test_build_operator_session_enables_echo_pump() -> None:
    """Attended CLI sessions get the background echo pump at the module's default cadence."""
    # build via _build_operator_session with a console-safe embodiment and an
    # accepts_operator_messages policy (same fixtures the neighboring tests use)
    # assert session._echo_interval_s == inspect_robots.session.ECHO_INTERVAL_S
    # repeat for the connect_operator_session-hook path (cli.py's other call site)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_registry_cli.py -q -k echo_pump`
Expected: FAIL — `_echo_interval_s` is `None`.

- [ ] **Step 3: Implement**

In `cli.py`, extend the existing `from inspect_robots.session import ...` import with `ECHO_INTERVAL_S`, and change both call sites:

```python
session.enable_footer(label="sent" if accepts_messages else "noted", echo_interval_s=ECHO_INTERVAL_S)
```

```python
session.enable_footer(label="sent", echo_interval_s=ECHO_INTERVAL_S)
```

In `docs/guide/cli.md`, the footer paragraph (around line 183) currently claims the opposite of this change: "Keystroke echo is pumped at the rollout poll cadence, so on a self-paced robot it can lag up to one control step." Replace that sentence with: "Keystroke echo runs on its own background cadence, so typing shows up immediately even while the policy is still thinking." Do not also add an echo sentence to the earlier feedback paragraph: the footer paragraph is the echo's home, and two claims in one page will drift. (Writing-style rules: no em dashes, no bold, no "not just X, but Y".)

In `CHANGELOG.md`, add under `## [Unreleased]` → `### Fixed`:

```markdown
- **Core:** the operator footer now echoes typing on a background cadence, so
  feedback is visible while an agent policy is blocked in inference instead of
  appearing only when the robot moves
  ([plan 0066](plans/0066-footer-echo-pump.md),
  [#367](https://github.com/robocurve/inspect-robots/issues/367)).
```

- [ ] **Step 4: Run the test and the full gates**

Run: `uv run pytest tests/test_registry_cli.py -q -k echo_pump` → PASS, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: all green, coverage 100%.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/cli.py docs/guide/cli.md CHANGELOG.md tests/test_registry_cli.py
git commit -m "feat(cli): wire the footer echo pump for attended runs (#367)"
```

(Committed by the main session, not the implementing agent.)
