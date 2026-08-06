# Grader: judgement capture as a first-class component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the *capture* side of R6 a first-class, registered component. R6
(plan 0001 §9) splits judging in two: a judgement is captured once around the
rollout, and scorers are pure functions that read it from the record (that is
what keeps scoring reproducible from a saved log). The scoring half is a
registered abstraction; the capture half is CLI plumbing — two `before_scoring`
callbacks on `OperatorSession`, selected by string-sniffing the scorer list
(`any(s.name == "operator")` in `cli.py`), with divergent behavior across
`run --instruction`, `run --task`, and `eval-set`.

**The bug this fixes (issue #320, the acceptance bar for this plan):** when the
`agent` policy ends a trial via its done()/give_up() tools, the trial ends as a
policy-requested stop (`rollout.py`: `truncated=True`,
`termination_reason="done"|"give_up"` — deliberately not `OPERATOR_END` and not
an embodiment `terminated`). The attended verdict prompt then never fires on
most paths, so the operator cannot give a judgement or grader notes, and the
`operator` scorer silently records failure ("no operator judgement recorded"):

- Registered tasks (`run --task`, attended) and `eval-set` use
  `prompt_verdict_on_operator_end`, which prompts only for `OPERATOR_END`.
- Ad-hoc `--instruction` runs prompt only when the resolved scorer list
  contains a scorer literally named `operator`; `--scorer reached_goal_state`
  or a config-default scorer silently drops grading.

After this plan, an attended trial ended by done()/give_up() is prompted for a
verdict and grader notes on **every** path (ad-hoc with any scorer, registered
task, eval-set), and unattended runs still never prompt.

## Architecture

New core module `src/inspect_robots/grader.py`:

```python
@runtime_checkable
class Grader(Protocol):
    """Capture a judgement onto a trial record, once, before scorers run."""

    name: str

    def grade(self, record: TrialRecord, scene: Scene) -> None: ...
```

- A grader runs **only for trials that will be scored** (never for errored or
  cancelled trials — any non-success status is skipped, `eval.py`), after the
  rollout returns and before scorers run — exactly the existing
  `before_scoring` seam in `eval.py`. It may be interactive (human prompt) or
  expensive (a future VLM call over final frames), and it **mutates the
  record**: `operator_judgement`, `operator_note`, and an `operator_event`
  with a `source` tag. Scorers stay pure readers, so re-scoring a saved log
  stays deterministic. (This is also why the reserved `VLMScorer` (R10) can
  only ever be implemented correctly as a grader; it stays untouched here and
  a VLM grader ships later as a plugin.)
- Builtin: `operator_grader(session: OperatorSession | None = None)`, whose
  `grade()` delegates to `OperatorSession.prompt_verdict`. When constructed
  without a session (e.g. `eval(grader="operator")` from the Python API, where
  nothing calls `connect_session`), it lazily constructs a default
  `OperatorSession` on first `grade()` — keeping every
  behavior that method already has: adopt a console-captured verdict, adopt a
  definitive embodiment verdict (`success`/`failure`) without re-asking, note
  the `max_steps` case, EOFError-safe prompting.
- Registry: new kind `"grader"`, entry-point group `inspect_robots.graders`,
  decorator `grader()` in `registry.py`. The decorator export shadows the
  `inspect_robots.grader` submodule in the package namespace exactly as the
  `scorer` decorator already shadows `scorer.py` — established precedent, not
  a new hazard.
- Duck-typed seam for terminal access: after the CLI resolves a grader by
  name, it calls `connect_session(session)` on it if that attribute is
  callable, passing the run's `OperatorSession`. The builtin operator grader
  implements it; a VLM grader simply won't have it. This keeps the registry
  factory signature clean (`resolve("grader", name)` needs no special-cased
  kwargs).

## Binding decisions

1. **`eval()` / `eval_set()` API:** new keyword `grader: Grader | str | None =
   None`, resolved via the registry when a string. Internally it is adapted to
   the existing `before_scoring` seam (`before_scoring=grader.grade`).
   `before_scoring` stays public and documented (it is the lower-level hook and
   the internal plumbing); passing **both** `grader` and `before_scoring`
   raises `ConfigError` — two writers for one seam is always a caller bug.
   `eval()` validates a resolved or passed grader with
   `isinstance(g, Grader)` (the protocol is `runtime_checkable`) and raises
   `ConfigError` on mismatch, so a broken entry point fails at configuration
   time, not deep inside scoring. The literal `"none"` is CLI/config
   vocabulary only (see decision 2); the API spelling is `grader=None`.
2. **CLI selection:** shared `--grader NAME` argument on `run` and `eval-set`
   (same anti-drift helper as the other shared eval args), plus a
   `[defaults] grader` config key. Resolution order:
   `--grader` > `[defaults] grader` > built-in default. The literal name
   `none` disables grading (`--grader none` is the escape hatch; also valid in
   config). `none` is resolved by the CLI **before** any registry lookup, so it
   is effectively a reserved name — a plugin registering a grader called
   `none` is shadowed; document it as reserved.
3. **Default:** `operator` when the run is attended (`_attended(args)`: a real
   TTY and no `--no-prompt`), `none` when unattended. This preserves R6's hard
   invariant — CI/unattended runs never block on a prompt — while making every
   attended run graded by default.
4. **Explicit flag wins; config-sourced `operator` needs an attended
   terminal:** an explicit `--grader` is honored regardless of TTY-ness — a
   VLM grader must be able to run unattended, and a user who types
   `--grader operator` into a TTY-less invocation gets the safe degradation
   (`prompt_verdict` swallows `EOFError` on dead stdin and returns). A
   **config-sourced** `[defaults] grader = operator` is weaker intent and is
   downgraded to `none` with a stderr note whenever the run cannot actually
   be attended: under `--no-prompt` **or** when `sys.stdin.isatty()` is
   false. Without the TTY half, a one-time config edit would make every later
   cron/CI run block forever at `input()` when stdin is an open pipe —
   violating decision 3's hard invariant. The one rejected contradiction is
   the explicit flag pair `--no-prompt --grader operator`
   (`SystemExit`, checked in `_check_shared_run_conflicts` so `run` and
   `eval-set` share it). Non-operator graders are unaffected by
   `--no-prompt`, and its help text must say it suppresses "the operator
   grader", not all prompting.
5. **The scorer-name sniffing gate is deleted**, as is
   `prompt_verdict_on_operator_end` (both call sites replaced by the grader
   wiring; the method's only job was the narrow OPERATOR_END-only policy this
   plan removes). `prompt_verdict` itself — and therefore the operator grader —
   is the single grading behavior everywhere.
6. **Behavior change, accepted:** attended registered-task and eval-set runs
   start prompting after each non-definitive trial (previously prompt-free
   unless the operator pressed the end key). The operator is standing at the
   rig in an attended run; `--grader none` (or `--no-prompt`) restores the old
   behavior. With `epochs > 1` this means one prompt per epoch trial — the
   operator-facing cost of the change, consistent with existing ad-hoc
   behavior. Unattended behavior is unchanged everywhere.
7. **Early-end context line:** `prompt_verdict` gains one note (mirroring the
   existing `max_steps` note): when the trial is truncated with a reason other
   than `max_steps`, it prints `note: this trial ended early ('<reason>')`
   before asking. The wording is deliberately neutral: `TrialRecord` cannot
   distinguish a policy-requested stop from an embodiment truncation (both
   set the same two fields in `rollout.py`), so claiming "the policy ended
   this" could be false for an embodiment truncation. The reason string
   (`done`, `give_up`, `time_limit`, ...) carries the specifics.
8. **No log-schema change:** the judgement/note/event already land in the
   `EvalLog`; recording the grader *name* in the log header is a possible
   follow-up, not this PR.
9. **`inspect-robots list`** does **not** pick the kind up automatically: it
   iterates `_KIND_BY_PLURAL` (`cli.py`), and the `list` positional's
   `choices` come from the same map. Add a `"graders"` entry there
   (`_PLURAL_BY_KIND` is derived from it by comprehension and follows
   automatically). CLI grader resolution goes through `_resolve_or_exit`,
   **not** `_pick_component` — the latter's no-default error path indexes
   `_ENV_BY_KIND[kind]`, which has no `"grader"` entry.

## Tasks

- [ ] **grader.py + registry kind.** New module with the `Grader` protocol and
  `operator_grader()`; `registry.py` gains kind `"grader"`, group
  `inspect_robots.graders`, and the `grader()` decorator; `_builtins.py`
  registers `operator`. Unit tests: protocol conformance, registry round-trip,
  `operator_grader().grade` delegates to `prompt_verdict` (injected
  `input_fn`), `connect_session` rebinds the session, and the lazy
  default-session branch (no session, first `grade()`) is exercised
  explicitly.
- [ ] **eval.py: `grader=` kwarg.** Accept object or registry name on `eval()`
  and `eval_set()`; adapt to `before_scoring`; `ConfigError` when both are
  passed or when the resolved object fails `isinstance(g, Grader)`.
  Docstrings updated (state the once-per-scored-trial contract and the
  mutual exclusion). Tests: grader called exactly once per scored trial, never
  for errored trials, string resolution works, both-passed raises,
  non-conforming resolved object raises.
- [ ] **session.py: early-end note + delete `prompt_verdict_on_operator_end`.**
  Add the decision-7 neutral note line to `prompt_verdict`; remove the narrow
  variant and its tests (`tests/test_session.py` has ~14 references); port
  any still-relevant assertions onto `prompt_verdict`. The deletion leaves
  `OPERATOR_END` imported but unused in `session.py` — drop the import.
- [ ] **cli.py + defaults.py wiring.** Shared `--grader` arg; `[defaults]
  grader` config key (`defaults.py`: `Defaults` field, parser, `_CONFIG_KEYS`);
  resolution per decisions 2–4; both `run` paths and `eval-set` build the
  grader (connecting the `OperatorSession` via `connect_session`) and pass
  `grader=` to `eval()`/`eval_set()`; delete the scorer-sniffing gate and both
  `before_scoring =` wirings; `--no-prompt --grader operator` exits with an
  error in `_check_shared_run_conflicts`, and `--no-prompt` downgrades a
  config-sourced `operator` to `none` with a stderr note. Add `"graders"` to
  `_KIND_BY_PLURAL` so `inspect-robots list` shows them. Help strings and any
  user-facing wording follow the repo writing-style rule (no em dashes, no
  slogans).
- [ ] **Regression tests for issue #320 (the acceptance bar).** Attended
  (injected `input_fn`/TTY seams), a policy that emits
  `action.meta["request_stop"]` (stop reason `done`): the operator is prompted
  and judgement + note land in the record/log on (a) ad-hoc run with a
  **non-operator** scorer, (b) registered task via `run --task`, (c)
  `eval-set`. Plus: unattended run never prompts and records `None` (R6),
  `--grader none` suppresses the prompt on an attended run, `--no-prompt`
  downgrades a config-sourced `operator` grader (no prompt, stderr note), and
  a config-sourced `operator` with non-TTY stdin also downgrades (the
  decision-4 cron/CI case).
- [ ] **Public API + docs.** Export `Grader`, `operator_grader`, and the
  `grader` decorator via `__init__.py` `__all__`. Import ordering matters for
  the shadowing: `from inspect_robots.grader import ...` must run before
  `from inspect_robots.registry import ... grader ...` so the submodule
  import completes first and the decorator name wins in the package
  namespace (alphabetical placement gives this for free). Update
  `tests/test_api_snapshot.py` in the same commit. Update `src/inspect_robots/CLAUDE.md`
  module map (new `grader.py` row; `session.py`/`cli.py` rows), root
  `CLAUDE.md` if it names the affected flows, and the docs pages that describe
  operator grading / CLI flags (`docs/`). Regenerate nothing by hand that
  `scripts/gen_api_docs.py` owns.
- [ ] **Gates.** `ruff check .`, `ruff format --check .`, `mypy` (strict, src +
  tests), `pytest --cov` at 100% (`--cov-fail-under=100`), core-only import
  unaffected (no new deps).

## Out of scope

- A VLM grader (ships later as a plugin via `inspect_robots.graders`).
- Removing/deprecating `VLMScorer` (reserved stub stays as-is).
- Recording the grader name in the `EvalLog` header (possible follow-up).
- Multiple graders per run (one per run in v0; revisit if a real need appears).
- Any change to unattended/CI behavior.
