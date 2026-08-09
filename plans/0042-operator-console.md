# Operator console — live feedback to agent policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During an attended run with an opted-in policy (the `agent` plugin), the operator can type a line of feedback + Enter and have it delivered to the policy at its next inference ("the gripper is farther than it looks"), can end the episode with a bare Enter (unchanged muscle memory), and can end-and-score in one stroke with `/y`, `/n`, `/p` plus an optional grader note (#273). VLA runs and unattended runs are byte-for-byte unchanged.

**Architecture:** A new core module `console.py` owns a **polled, threadless** stdin reader: `OperatorConsole.poll()` does a zero-timeout `select()` on stdin, reads pending bytes **at the fd level** (`os.read`, with in-module line assembly — see Design decision 9), and parses complete lines into feedback messages or an end request. No background thread means no second stdin reader: the post-trial verdict prompt (`cli._prompt_operator`, plain `input()`) keeps working because nothing else is ever blocked on stdin. `rollout()` accepts an optional `operator_input` and, mirroring the `extra["approvals"]` channel (#217), injects messages under the reserved `extra["operator_messages"]` key with the same delivered-once-per-inference tail discipline; an end request terminates the trial with the existing `OPERATOR_END` reason and, when the line carried a verdict, stamps `operator_judgement`/`operator_note` directly so the post-run prompt is skipped. A raising `poll()` degrades (warn once, channel off for the trial) — it never breaks the "always persist an EvalLog" invariant. The CLI enables the console only when `_attended(args)`, the policy declares the duck-typed class attribute `accepts_operator_messages = True` (set by `LLMAgentPolicy`), the platform supports `select` on stdin (not Windows), **and the embodiment is console-safe**: it either offers the duck-typed `defer_operator_end()` hook (stdin-polling embodiments stand down when called) or is `is_simulated` (sims never read stdin). Real-hardware embodiments without the hook — today's yam — keep exactly the current keypress behavior, with a printed notice, until they ship the hook; this makes day one race-free instead of racy. Feedback messages are recorded as typed transcript events **and persisted per-trial in the saved log** via a new `SceneResult.operator_messages` parallel field (`TrialRecord.events` themselves are never persisted — `JsonLogSink` drops records), so the HTML view and summarize can show what the operator said and when.

**Tech Stack:** Python 3.10+, stdlib + numpy only in core (`select` + `sys` are stdlib); pytest; agent-plugin rendering change in `plugins/inspect-robots-agent`.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage** for core. Agent plugin tests run from its own package scope (`uv run pytest plugins/inspect-robots-agent/tests -q`).
- D1 docstrings on public defs; state the contract, not the name. Line length 100.
- Core stays NumPy-only; no new deps. `select` does not support stdin on Windows: the TTY-bound default callables are isolated and injectable exactly like `inspect_robots_yam.operator.default_poll_end`, and are excluded from coverage with `# pragma: no cover` on the TTY-bound lines only.
- Zero behavior change when `operator_input is None` (the default everywhere): every existing test passes untouched.
- Public API is fenced by `inspect_robots.__all__` **and** `tests/test_api_snapshot.py` — update both together.
- Update `src/inspect_robots/CLAUDE.md`'s module table (new `console.py` row; extend the `rollout.py` and `types.py` rows).
- Worktree: `.claude/worktrees/operator-console`; run everything via `uv run ...` there. Reference #273 in commit messages; end each with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. **No git operations from Codex** — the orchestrating session commits.

## Reference: current wiring (main @ ba873203)

- `src/inspect_robots/rollout.py:211-223` — `rollout()` signature (keyword-only after `scene`). `:272-306` — loop top: approvals tail (`_APPROVALS_KEY`, `_rollout_last_approvals_idx`) is sliced since the last inference and injected via `replace(obs, extra={**obs.extra, "env_step": t, "approvals": tail_approvals})`; the index advances inside `if len(inferences) > prev_inferences:` (`:288-290`). `:375-389` — termination bookkeeping (`terminated`/`truncated`/`termination_reason` + `break`). Mirror both patterns.
- `src/inspect_robots/types.py:25-44` — `Observation` docstring reserves `extra["env_step"]` and `extra["approvals"]`; `:85-89` — `OPERATOR_END = "operator_end"` and its rationale comment.
- `src/inspect_robots/transcript.py:52-64` — `operator_event(t, verdict, source="prompt", note=None)`; new sibling factory goes here.
- `src/inspect_robots/eval.py:135-148` (`eval` signature), `:207-221` (`_run_eval` call), `:229-243` (`_run_eval` signature), `:369` (the `rollout(...)` call site), `:424-428` (`before_scoring` invocation), `:570-608` (`eval_set` pass-through). Thread `operator_input` through all of these the same way `before_scoring` flows.
- `src/inspect_robots/cli.py:659-664` — prompt constants; `:667-708` — `_prompt_operator` (embodiment-verdict adoption precedent at `:677-684`); `:711-719` — `_prompt_operator_on_operator_end`; `:722-728` — `_attended`; `:1280-1290` — `_cmd_run` before_scoring wiring (console construction goes here); `:1307-1321` — the `eval(...)` call; `:1413-1433` — `_cmd_eval_set` equivalents.
- `src/inspect_robots/embodiment.py:33-41` — capability flags; `:44-66` — `EmbodimentInfo`; the `Embodiment` Protocol docstring lists optional duck-typed hooks (`bind_task`) — document `defer_operator_end()` there.
- `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py:244` — `class LLMAgentPolicy(PolicyBase)`; `:686-712` — `act()` builds `_observation_content(...)`; `:1017-1033` — `_approvals_line` (defensive-parsing template); `:1036-1056` — `_observation_content` (insertion point: after the approvals line, before `narration`). Find the system-prompt builder with `grep -n "system" plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py`.
- `plugins/inspect-robots-yam/...` (SEPARATE REPO, not touched here): `operator.py:default_poll_end` reads stdin per step and returns `terminated=True, reason="operator_end"`. The `defer_operator_end()` contract exists so that repo can stand down in a follow-up.
- Tests to mirror: `tests/test_rollout_observation_step.py` (approvals/env_step injection tests), `tests/test_rollout_hardening.py`, `tests/test_registry_cli.py` (CLI harness), `tests/test_api_snapshot.py`.

## Design decisions (and why)

1. **Polled, not threaded.** A background thread blocked in `readline()` would race the post-trial `input()` verdict prompt for stdin — the thread's pending read would swallow the operator's answer. A zero-timeout `select()` poll from the rollout thread (once per control step, the same cadence yam polls today) has no second reader, no locks, and no lifecycle. Latency is one control period, imperceptible to a human.
2. **Line grammar** (parsed from `line.rstrip("\n")`, then `.strip()` for classification):
   - stripped empty → end request, no verdict (post-run prompt collects it, exactly as today). Bare Enter ending the run is today's behavior, so no retraining.
   - `/y [note]`, `/n [note]`, `/p [note]` (case-insensitive command token, note is everything after the first space, stripped, `None` if empty) → end request with verdict `"y"`/`"n"`/`"partial"` and the note. These map onto the existing `_PROMPT_ANSWERS` vocabulary, and `"partial"` keeps its scores-as-failure semantics in the operator scorer.
   - any other line starting with `/` → print a one-line usage hint via the injectable `output_fn`, queue nothing (a typo must not enter the model context or end the run).
   - anything else → one feedback message, verbatim (original spacing preserved apart from the trailing newline).
3. **`extra["operator_messages"]`** is a reserved key like `approvals`: a list of `{"t": int, "text": str}` dicts, the tail since the last inference, stamped with the step at which the rollout drained them. Consumers see each message exactly once across inferences (chunked controllers included) because the tail index advances only when an inference happened — the identical mechanism approvals use.
4. **End beats feedback typed in the same poll**, but messages drained in the ending iteration are still recorded as transcript events (the grader sees them; the policy never will — there is no further inference).
5. **`terminated=True, reason=OPERATOR_END`** for a console end, matching what stdin-polling embodiments emit today, so `_prompt_operator_on_operator_end`, the summarizer, and the HTML view need no new vocabulary.
6. **Verdict-carrying ends skip the prompt** via a guard in the CLI: prompting is skipped when `record.operator_judgement` is already set (console verdicts, like embodiment-definitive adoptions, are announced instead). The console records the `operator_event` itself with `source="console"`.
7. **Gating is a duck-typed policy class attribute** (`accepts_operator_messages`), not a `PolicyInfo` field: it follows the `bind`/`transcript` precedent, requires no dataclass change, and VLAs never declare it.
8. **`defer_operator_end()`** is a duck-typed embodiment hook and also the **console-safety gate**. The CLI enables the console only when the embodiment offers the hook (which the CLI then calls) **or** `embodiment.info.is_simulated` is true. A real-hardware embodiment without the hook may be polling stdin itself (unpatched yam does, inside `step()`), and two consumers of one stdin race: a feedback line typed during a slow LLM inference would be eaten by the embodiment's poll and end the episode. Rather than ship that race, the console stays off there — one styled notice explains why — and lights up the moment the embodiment ships the hook. Mock/sim runs work on day one; the yam implementation is a follow-up issue in its repo.
9. **The default reader is fd-level, not `sys.stdin.readline`.** Pairing fd-level `select()` with the buffered `TextIOWrapper` desynchronizes on multi-line input: one `readline()` call slurps every pending byte into the user-space buffer and returns only the first line, after which `select()` reports the fd not-readable while complete lines sit invisible in the buffer — delivering messages one keypress late and defeating `begin_trial()`'s drain. The console therefore injects `readable: Callable[[], bool]` and `read: Callable[[], str]` where the default `read` is `os.read(sys.stdin.fileno(), 65536)` decoded UTF-8 (`errors="replace"`), and the console assembles lines itself, keeping any partial trailing line in an internal buffer until its newline arrives. `begin_trial()` clears both the fd (drain loop) and the internal buffer.
10. **A raising `poll()` must not lose the log.** Every other in-loop call is wrapped so `eval()` can persist the partial record; `poll()` gets the same treatment as transcript streaming (`rollout.py:293-306`): catch `Exception` (never `KeyboardInterrupt` — Ctrl-C during a poll must keep today's cancelled-trial path), warn once, and disable the channel for the rest of the trial. On Windows `select.select` on stdin raises `OSError`, so the CLI additionally never constructs a console there (`sys.platform == "win32"` check with a notice) — degrade at the gate, not at step 0.

---

### Task 1: `console.py` — parser and polled reader

**Files:**
- Create: `src/inspect_robots/console.py`
- Test: `tests/test_console.py`

**Interfaces (all public, D1 docstrings):**
- `@dataclass(frozen=True) class EndRequest: verdict: str | None = None; note: str | None = None`
- `@dataclass(frozen=True) class ConsolePoll: messages: tuple[str, ...] = (); end: EndRequest | None = None`
- `@runtime_checkable class OperatorInput(Protocol):` with `def poll(self) -> ConsolePoll: ...` and `def begin_trial(self) -> None: ...` — the contract `rollout()` consumes.
- `class OperatorConsole:` constructor takes `readable: Callable[[], bool] | None = None`, `read: Callable[[], str] | None = None`, `output_fn: Callable[[str], None] = print`. `None` defaults bind the TTY-bound implementations — `select.select([sys.stdin], [], [], 0)` for `readable` and `os.read(sys.stdin.fileno(), 65536)` decoded UTF-8 (`errors="replace"`) for `read` (Design decision 9) — and those default callables are the **only** `# pragma: no cover` lines in the module. `poll()` loops while `readable()`, appends each `read()` chunk to an internal text buffer, splits off **complete** lines only (a trailing partial line stays buffered for a later poll), applies the grammar in Design decision 2, accumulates messages, and keeps the **first** end request of the poll (later lines in the same burst still parse: feedback after an end line is still returned as a message so it reaches the transcript). An empty string from `read()` (EOF) stops the loop for good (an internal `_eof` flag makes every later `poll()` **and** `begin_trial()` a cheap no-op); a buffered partial line at EOF is discarded (never submitted without its Enter). `begin_trial()` drains the fd (loop `readable()`/`read()` discard, **also stopping and latching `_eof` on an empty `read()`** — on a closed/redirected stdin `select` reports readable forever while `read` returns nothing, and a literal drain loop would spin) and clears the internal buffer so text typed during scene setup or the previous trial's scoring never leaks into a trial.
- Module-level `USAGE = "operator console: type a message + Enter to send it to the policy; Enter alone ends the episode; /y /n /p [note] ends it with a verdict"` — printed by the CLI at run start and by the console on an unknown `/` command.

- [ ] **Step 1: failing tests** in `tests/test_console.py`, driving `OperatorConsole` with scripted `readable`/`read` (closures over a list of chunks): feedback line verbatim; leading/trailing spaces preserved on messages but a whitespace-only line is an end; `/y`/`/N`/`/p note text here` verdict + note mapping; `/oops` prints `USAGE` via a recording `output_fn` and queues nothing; **one chunk carrying multiple lines (paste) yields every line in the same poll**; a chunk ending mid-line keeps the partial buffered and a later chunk completes it; multiple lines in one poll; end-then-feedback in one poll returns both the end and the message; EOF stops parsing, discards a buffered partial line, and later polls return empty; `begin_trial()` discards both pending chunks and the internal partial buffer; `begin_trial()` with an always-readable EOF source (`readable` always true, `read` returns `""`) returns instead of spinning, and later calls are no-ops; `OperatorConsole` satisfies `isinstance(..., OperatorInput)`.
- [ ] **Step 2: run, confirm FAIL** (`uv run pytest tests/test_console.py -v`).
- [ ] **Step 3: implement** `console.py` per the interfaces above. Keep the parser a module-private pure function `_parse(line: str) -> tuple[str | None, EndRequest | None, bool]`-style helper or similar so grammar tests stay direct — implementer's choice, but no thread, no global state.
- [ ] **Step 4: run to green.** Gates on the new files (ruff, mypy).

---

### Task 2: rollout delivery + transcript events

**Files:**
- Modify: `src/inspect_robots/rollout.py`, `src/inspect_robots/transcript.py`, `src/inspect_robots/types.py` (Observation docstring only)
- Test: `tests/test_rollout_observation_step.py` (same file as the approvals tests), `tests/test_console.py` if helpers fit better there

**Interfaces:**
- `transcript.operator_message_event(t: int, text: str) -> Event` — `kind="operator_message"`, `data={"text": text}`. Docstring: live feedback typed at the console during the trial, distinct from the post-hoc `operator` verdict event.
- `rollout(..., operator_input: OperatorInput | None = None)` (keyword-only). Import `OperatorInput` under `TYPE_CHECKING` from `inspect_robots.console` to keep import-time surface flat (runtime access is duck-typed through the parameter).
- Loop changes, all inside `while t < max_steps:` **before** `controller.next_action`:
  1. `poll = operator_input.poll()` when `operator_input is not None` **and the channel is still live** — wrapped like transcript streaming (`rollout.py:293-306`): `except Exception` sets a `console_ok = False` flag and warns once (`RuntimeWarning`, "Operator console disabled for this trial after ..."); `KeyboardInterrupt` is not caught, preserving the cancelled-trial path (`:396-400`). A dead channel means later iterations skip polling entirely; the trial and its log are unaffected.
  2. For each message: append `{"t": t, "text": text}` to `store[_OPERATOR_MSGS_KEY]` and `record.events.append(operator_message_event(t, text))`.
  3. Inject the tail (same slicing as approvals, its own `_rollout_last_operator_msgs_idx` advanced in the same `len(inferences) > prev_inferences` block) as `extra["operator_messages"]` — only when `operator_input is not None`, so the reserved key never appears otherwise.
  4. If `poll.end` is not None: set `record.terminated = True`, `record.termination_reason = OPERATOR_END`; when `poll.end.verdict` is set, also `record.operator_judgement = poll.end.verdict`, `record.operator_note = poll.end.note`, and append `operator_event(t=t, verdict=poll.end.verdict, source="console", note=poll.end.note)`; `break` before any inference or motion this step.
  5. Call `operator_input.begin_trial()` once, right after `embodiment.reset(...)` returns — scene-setup typing is discarded. A raising `begin_trial()` gets the same degrade treatment as a raising `poll()`.
- `types.py` `Observation` docstring: add `extra["operator_messages"]` to the reserved-keys sentence.

- [ ] **Step 1: failing tests.** A `FakeOperatorInput` (scripted list of `ConsolePoll`s per call) drives `rollout()` with the mock embodiment/policy: (a) message typed at t=0 with a chunked controller (chunk length ≥ 2) appears in the observation of the **next inference only**, and never again; (b) two messages across different steps both arrive, stamped with their drain step; (c) `poll.end` with no verdict → `terminated`, `termination_reason == OPERATOR_END`, no judgement set, **no** step executed that iteration (embodiment step count unchanged); (d) `/y`-style end → judgement `"y"`, note recorded, one `operator` event with `source="console"`; (e) messages in the ending poll land in `record.events` as `operator_message` events; (f) `begin_trial` called exactly once per rollout, after reset (assert ordering via the fake); (g) `operator_input=None` (default) → no `operator_messages` key in any policy-facing observation; (h) a `poll()` that raises → `RuntimeWarning`, no further polls, trial completes normally with an intact record; a raising `begin_trial()` degrades the same way; (i) a `poll()` raising `KeyboardInterrupt` → the existing cancelled-trial path (`_CancelledTrial`, `status == "cancelled"`).
- [ ] **Step 2: run, confirm FAIL.**
- [ ] **Step 3: implement** per the interface notes. Keep the approvals and messages tail bookkeeping textually adjacent so the parallel is obvious to the next reader.
- [ ] **Step 4: run to green**, including the full existing rollout suite.

---

### Task 3: eval/eval_set threading + log persistence

**Files:**
- Modify: `src/inspect_robots/eval.py`, `src/inspect_robots/log.py`
- Test: `tests/test_eval_orchestration.py`, `tests/test_eval_log.py`

**Interfaces:**
- `eval(..., operator_input: OperatorInput | None = None)` → `_run_eval` → the `rollout(...)` call at `eval.py:369`; `eval_set(...)` passes it to each per-task `eval` exactly like `before_scoring`. Docstring one-liner on `eval`: attended-console input source; `None` (default) disables the channel.
- **Persistence** (`TrialRecord.events` are dropped by `JsonLogSink` — `logging/json_log.py:79-81` — so the log needs its own field): `SceneResult.operator_messages: tuple[tuple[dict[str, Any], ...], ...] = ()` — one inner tuple per epoch, each entry the `{"t", "text"}` dict, strictly parallel to `epochs`/`termination_reasons` (`log.py:63-87`). Assemble in `_run_eval` beside `termination_reasons` (`eval.py:341, :462, :497-507`) by filtering `record.events` for `kind == "operator_message"`. Back-compat default in `read_eval_log` following `:135-140`, but note this is the module's first **two-layer** sequence field: the single-level `tuple(sample.get(key, ()))` coercion would leave inner JSON lists as lists, violating both the declared type and log.py's shallow-immutability contract (`log.py:8-14`). Coerce both layers in `EvalLog.from_dict`: `tuple(tuple(msgs) for msgs in sample.get("operator_messages", ()))`; the round-trip test must assert tuple-of-tuples **by type**, not just by content. Older logs load as empty tuples; **no** `SCHEMA_VERSION` bump — additive with a default, same as `operator_notes` was. Empty for every run without a console.
- Surface: `_summarize.py:179-184` gains an `operator feedback:` line per message next to the existing judgement/note fields; `_html.py:800-812` renders messages in the same block as `operator_notes` (plain text, escaped like everything else there).

- [ ] **Step 1: failing tests** — (a) an eval over ≥ 2 trials with a fake `OperatorInput` sees `begin_trial` once per trial and a message "typed" during trial 1's scoring window must not surface in trial 2 (assert the discard); (b) a run whose fake console emitted messages round-trips them through `write → read_eval_log` in `SceneResult.operator_messages`, parallel to epochs; (c) a pre-field log JSON (fixture without the key) still loads, empty tuples; (d) summarize and HTML outputs contain the message text.
- [ ] **Step 2: confirm FAIL. Step 3: implement. Step 4: green.**

---

### Task 4: CLI — gating, hint, verdict-skip, defer hook

**Files:**
- Modify: `src/inspect_robots/cli.py`, `src/inspect_robots/embodiment.py` (Protocol docstring only)
- Test: `tests/test_registry_cli.py`

**Interfaces:**
- One shared helper `_build_operator_console(policy, embodiment) -> OperatorConsole | None` used by `_cmd_run` (`:1280` region) and `_cmd_eval_set` (`:1413` region), consulted only when `_attended(args)`. Gate, in order (Design decisions 8 and 10):
  1. `getattr(policy, "accepts_operator_messages", False)` falsy → `None`, silently (VLAs: status quo).
  2. `sys.platform == "win32"` → `None` + one degraded notice (select cannot watch stdin there).
  3. `hook = getattr(embodiment, "defer_operator_end", None)`; if callable → call it, build the console.
  4. No hook but `embodiment.info.is_simulated` → build the console (sims do not read stdin).
  5. Otherwise → `None` + one styled notice: this embodiment predates the operator console and still owns the end-of-episode keypress, so feedback typing stays off (prevents the two-readers race with e.g. unpatched yam).
  When a console is built: print `USAGE` (styled like the existing `rerun:` lines) and pass it as `operator_input=...` to `eval`/`eval_set`.
- `_prompt_operator_on_operator_end`: add `record.operator_judgement is None` to the gate — a console verdict already answered, so print an adoption line (mirror `:683`) instead of prompting. `_prompt_operator` (ad-hoc path) gets the same early-return guard.
- `embodiment.py` Protocol docstring: document `defer_operator_end()` next to `bind_task` — "an embodiment that polls stdin for the end-of-episode keypress must stop doing so when called; the framework console owns stdin for this run. Offering the hook is also how a real-hardware embodiment declares itself console-safe."

- [ ] **Step 1: failing tests** (reuse the CLI harness in `tests/test_registry_cli.py`, monkeypatching `sys.stdin.isatty`; positive-path tests must monkeypatch `sys.platform` to a POSIX value so they also pass on the non-blocking Windows `test-extra` tier, and the win32 test monkeypatches it the other way): console constructed + hint printed only when attended, policy opts in, and the embodiment is console-safe (hook called when present; `is_simulated` fallback works); real-hardware embodiment without the hook → `operator_input is None` + the notice; `win32` → `None` + notice; `--no-prompt` or a policy without the attribute → `operator_input is None` (spy on `eval`); `_prompt_operator_on_operator_end` skips when judgement pre-set and prompts when not; `_prompt_operator` early-returns on a pre-set judgement.
- [ ] **Step 2: confirm FAIL. Step 3: implement. Step 4: green.**

---

### Task 5: public API + docs surface

**Files:**
- Modify: `src/inspect_robots/__init__.py`, `tests/test_api_snapshot.py`, `src/inspect_robots/CLAUDE.md`, `docs/` (only if an existing page covers the attended-run/operator flow — `grep -rn "operator" docs/ --include="*.md" -l`; otherwise skip, no new page)

- [ ] Export `OperatorConsole`, `OperatorInput`, `ConsolePoll`, `EndRequest` from `inspect_robots` (lazy-import friendly, matching how other symbols are re-exported); update `__all__` and the snapshot test **together**.
- [ ] CLAUDE.md module table: add `console.py` row; extend the `rollout.py` (operator input channel), `transcript.py` (`operator_message` kind), and `log.py` (`operator_messages` parallel field) rows.
- [ ] Full gates: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest --cov -q` (100%).

---

### Task 6: agent plugin — rendering + opt-in

**Files:**
- Modify: `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py`
- Test: `plugins/inspect-robots-agent/tests/` (alongside the existing approvals-line tests — `grep -rn "_approvals_line\|approvals" plugins/inspect-robots-agent/tests -l`)
- Modify: `plugins/inspect-robots-agent/README.md` (short "Operator feedback" subsection; follow the repo writing-style rules — no em dashes in prose, no mid-sentence bold)

**Interfaces:**
- `LLMAgentPolicy.accepts_operator_messages: bool = True` as a class attribute with a docstring-comment stating the framework contract.
- `_operator_lines(observation) -> list[str]`: defensive exactly like `_approvals_line` (non-list, non-dict entries, non-int `t`, non-str `text` all skipped); each valid entry renders as `operator feedback (step {t}): {text}`. Inserted in `_observation_content` after the approvals line, before `narration`.
- System prompt: one added sentence — the model may receive operator feedback lines mid-run and should treat them as trusted guidance from the human supervising the robot. The plugin has **two** system templates (`_SYSTEM_TEMPLATE` and `_ON_DEMAND_SYSTEM_TEMPLATE`, policy.py:604 region); add the sentence after template selection (or to both templates) so both camera modes carry it.

- [ ] **Step 1: failing tests**: rendering with one and multiple messages; malformed entries skipped; absent key renders nothing; class attribute is `True`.
- [ ] **Step 2: confirm FAIL. Step 3: implement. Step 4:** plugin tests green (`uv run pytest plugins/inspect-robots-agent/tests -q`) **and** core gates still green.

---

### Interim behavior until the yam follow-up lands

On real yam hardware the console stays **off** (gate step 5): the run behaves exactly as today (any Enter ends the episode via yam's own poll), plus one notice explaining that feedback typing needs a yam update. There is no release-ordering hazard: core can ship first, and the feature lights up on hardware when yam adds `defer_operator_end()`.

### Out of scope (follow-ups filed, not implemented here)

- `inspect-robots-yam`: implement `defer_operator_end()` (skip its `poll_end` when deferred) — issue to file in that repo after merge; the console then activates on hardware automatically.
- Richer HTML rendering of feedback (e.g. timeline placement) beyond the operator-notes block added in Task 3.
