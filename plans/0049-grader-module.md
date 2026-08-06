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

- A grader runs **once per scored trial** (never for errored trials), after the
  rollout returns and before scorers run — exactly the existing
  `before_scoring` seam in `eval.py`. It may be interactive (human prompt) or
  expensive (a future VLM call over final frames), and it **mutates the
  record**: `operator_judgement`, `operator_note`, and an `operator_event`
  with a `source` tag. Scorers stay pure readers, so re-scoring a saved log
  stays deterministic. (This is also why the reserved `VLMScorer` (R10) can
  only ever be implemented correctly as a grader; it stays untouched here and
  a VLM grader ships later as a plugin.)
- Builtin: `operator_grader(session: OperatorSession | None = None)`, whose
  `grade()` delegates to `OperatorSession.prompt_verdict` — keeping every
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
2. **CLI selection:** shared `--grader NAME` argument on `run` and `eval-set`
   (same anti-drift helper as the other shared eval args), plus a
   `[defaults] grader` config key. Resolution order:
   `--grader` > `[defaults] grader` > built-in default. The literal name
   `none` disables grading (`--grader none` is the escape hatch; also valid in
   config).
3. **Default:** `operator` when the run is attended (`_attended(args)`: a real
   TTY and no `--no-prompt`), `none` when unattended. This preserves R6's hard
   invariant — CI/unattended runs never block on a prompt — while making every
   attended run graded by default.
4. **Explicit choice always wins:** an explicit `--grader` (or config value) is
   honored verbatim regardless of attendedness — a VLM grader must be able to
   run unattended, and the operator grader already degrades safely on dead
   stdin (`prompt_verdict` swallows `EOFError` and returns). One contradiction
   is rejected loudly: `--no-prompt --grader operator` is a `SystemExit` error
   (the flag promises no prompts; the grader exists to prompt).
5. **The scorer-name sniffing gate is deleted**, as is
   `prompt_verdict_on_operator_end` (both call sites replaced by the grader
   wiring; the method's only job was the narrow OPERATOR_END-only policy this
   plan removes). `prompt_verdict` itself — and therefore the operator grader —
   is the single grading behavior everywhere.
6. **Behavior change, accepted:** attended registered-task and eval-set runs
   start prompting after each non-definitive trial (previously prompt-free
   unless the operator pressed the end key). The operator is standing at the
   rig in an attended run; `--grader none` (or `--no-prompt`) restores the old
   behavior. Unattended behavior is unchanged everywhere.
7. **Policy-stop context line:** `prompt_verdict` gains one note (mirroring the
   existing `max_steps` note): when the trial ended as a policy-requested stop
   (truncated, reason neither `max_steps` nor an adopted definitive reason),
   it prints `note: the policy ended this trial ('<reason>')` before asking —
   the operator should know the robot stopped itself before judging it.
8. **No log-schema change:** the judgement/note/event already land in the
   `EvalLog`; recording the grader *name* in the log header is a possible
   follow-up, not this PR.
9. **`inspect-robots list`** picks up the new kind automatically if it
   iterates `KINDS`; verify and include graders in the listing either way.

## Tasks

- [ ] **grader.py + registry kind.** New module with the `Grader` protocol and
  `operator_grader()`; `registry.py` gains kind `"grader"`, group
  `inspect_robots.graders`, and the `grader()` decorator; `_builtins.py`
  registers `operator`. Unit tests: protocol conformance, registry round-trip,
  `operator_grader().grade` delegates to `prompt_verdict` (injected
  `input_fn`), `connect_session` rebinds the session.
- [ ] **eval.py: `grader=` kwarg.** Accept object or registry name on `eval()`
  and `eval_set()`; adapt to `before_scoring`; `ConfigError` when both are
  passed. Docstrings updated (state the once-per-scored-trial contract and the
  mutual exclusion). Tests: grader called exactly once per scored trial, never
  for errored trials, string resolution works, both-passed raises.
- [ ] **session.py: policy-stop note + delete `prompt_verdict_on_operator_end`.**
  Add the decision-7 note line to `prompt_verdict`; remove the narrow variant
  and its tests; port any still-relevant assertions onto `prompt_verdict`.
- [ ] **cli.py + defaults.py wiring.** Shared `--grader` arg; `[defaults]
  grader` config key (`defaults.py`: `Defaults` field, parser, `_CONFIG_KEYS`);
  resolution per decisions 2–4; both `run` paths and `eval-set` build the
  grader (connecting the `OperatorSession` via `connect_session`) and pass
  `grader=` to `eval()`/`eval_set()`; delete the scorer-sniffing gate and both
  `before_scoring =` wirings; `--no-prompt --grader operator` exits with an
  error. Verify `inspect-robots list` shows graders.
- [ ] **Regression tests for issue #320 (the acceptance bar).** Attended
  (injected `input_fn`/TTY seams), a policy that emits
  `action.meta["request_stop"]` (stop reason `done`): the operator is prompted
  and judgement + note land in the record/log on (a) ad-hoc run with a
  **non-operator** scorer, (b) registered task via `run --task`, (c)
  `eval-set`. Plus: unattended run never prompts and records `None` (R6), and
  `--grader none` suppresses the prompt on an attended run.
- [ ] **Public API + docs.** Export `Grader`, `operator_grader`, and the
  `grader` decorator via `__init__.py` `__all__`; update
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
