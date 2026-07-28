# Operator-End Grading Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trial that terminates with `termination_reason="operator_end"` — the standard reason for "a human ended this episode by keypress" — gets the operator verdict prompt (y/n/partial/skip + grader note) on **any** attended run, registered tasks included. Fixes [#194](https://github.com/robocurve/inspect-robots/issues/194); enabler for [inspect-robots-yam#79](https://github.com/robocurve/inspect-robots-yam/issues/79).

**Architecture:** Today `before_scoring = _prompt_operator` is installed only for ad-hoc runs whose task carries an `operator`-named scorer (`cli.py:972-981`), so a registered task never prompts — deliberate, to keep R6's non-blocking guarantee for unattended runs. This plan keeps that guarantee but adds an evidence-gated path: when stdin is a TTY and `--no-prompt` is absent, non-(adhoc+operator-scorer) runs get a conditional hook that prompts **only** for trials whose `termination_reason == OPERATOR_END` — the operator was demonstrably present (they pressed the key), so prompting cannot block an unattended run. The reason string becomes a public constant on the `StepResult` vocabulary.

**Tech Stack:** Python 3.12, uv, pytest. Run everything from the worktree root: `/home/robocurve/robocurve/worktrees/feat-operator-end-gate`.

## Global Constraints

- Test command: `uv run pytest -q`. Coverage is enforced for `src/` (`branch = true`, `fail_under = 100`, `pyproject.toml:124-129`); every new branch needs a test.
- Lint/format/type gates: `uv run ruff check . && uv run ruff format --check . && uv run mypy` before every commit. mypy is `strict = true` and covers `tests/` too (`pyproject.toml:104-109`) — every test snippet below must be fully typed.
- Public-API discipline: `tests/test_api_snapshot.py::test_public_api_snapshot` asserts `set(inspect_robots.__all__) == EXPECTED`; any export change must update `EXPECTED` **and** add a `CHANGELOG.md` `[Unreleased]` entry (the snapshot file's docstring requires both).
- The constant's value is exactly `"operator_end"` — inspect-robots-yam will emit this literal; changing the spelling breaks the cross-repo contract.
- Behavior that must NOT change (existing tests enforce it): ad-hoc + operator-scorer runs prompt for every non-definitive trial exactly as today; non-TTY and `--no-prompt` never prompt (`tests/test_registry_cli.py:2154`); registered tasks never prompt for trials that end any other way — `test_registered_task_never_prompts_even_with_operator_scorer_on_tty` (`tests/test_registry_cli.py:2179`) must still pass unmodified, since the cubepick embodiment never emits `operator_end`.
- `_prompt_operator` itself is untouched: `OPERATOR_END` is not in `_DEFINITIVE_REASONS`, so it falls through to the full prompt loop naturally.

---

### Task 1: `OPERATOR_END` constant on the termination vocabulary

**Files:**
- Modify: `src/inspect_robots/types.py` (constant next to `StepResult`, lines 83-98; docstring mention), `src/inspect_robots/__init__.py` (export), `tests/test_api_snapshot.py` (add `"OPERATOR_END"` to `EXPECTED`), `CHANGELOG.md` (`[Unreleased]` → Added)
- Test: `tests/test_types_spaces.py` (the module that covers `types.py`)

**Interfaces:**
- Produces: `inspect_robots.OPERATOR_END == "operator_end"` (also importable from `inspect_robots.types`). Task 2 imports it in `cli.py`.

- [x] **Step 1: Write the failing test**

```python
def test_operator_end_constant_is_public_vocabulary() -> None:
    import inspect_robots
    from inspect_robots.types import OPERATOR_END

    assert OPERATOR_END == "operator_end"
    assert inspect_robots.OPERATOR_END is OPERATOR_END
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/ -q -k operator_end_constant`
Expected: FAIL with `ImportError: cannot import name 'OPERATOR_END'`

- [x] **Step 3: Implement**

In `src/inspect_robots/types.py`, directly above `class StepResult`:

```python
# Standard termination reason for "a human operator ended this episode by
# keypress, without giving a verdict". Non-definitive on purpose: the verdict
# (y/n/partial/skip + optional grader note) is collected by the CLI's operator
# prompt, which fires for attended trials that end with this reason (#194).
OPERATOR_END = "operator_end"
```

Extend the `StepResult` docstring's reason examples (line 88-89) to include `"operator_end"` alongside `"success"`, `"collision"`, `"fault"`, `"out_of_bounds"`, noting it is the standard reason for operator-ended episodes awaiting a verdict.

In `src/inspect_robots/__init__.py`, import `OPERATOR_END` from `.types` alongside the existing `StepResult` import and add `"OPERATOR_END"` to `__all__` (ASCII-sorted: it slots before `"Observation"`).

- [x] **Step 4: Update the API snapshot and changelog**

`tests/test_api_snapshot.py::test_public_api_snapshot` asserts the exact `__all__` set — add `"OPERATOR_END"` to `EXPECTED` (it's a set; position is free). In `CHANGELOG.md`, the `[Unreleased]` section currently has only `### Removed` and `### Changed` (lines 8-22) — create an `### Added` subsection first (Keep-a-Changelog order: Added before Removed) containing:

```markdown
- `OPERATOR_END` termination-reason constant (`"operator_end"`): the standard
  reason for "a human ended this episode by keypress, verdict pending". Attended
  runs now prompt for exactly those trials — registered tasks and `eval-set`
  included (#194).
```

- [x] **Step 5: Run the test to verify it passes, then the full suite**

Run: `uv run pytest tests/ -q -k "operator_end_constant or api_snapshot" && uv run pytest -q`
Expected: PASS; full suite green.

- [x] **Step 6: Commit**

```bash
git add src/inspect_robots/types.py src/inspect_robots/__init__.py tests/test_types_spaces.py tests/test_api_snapshot.py CHANGELOG.md
git commit -m "feat(types): OPERATOR_END termination-reason constant (#194)"
```

---

### Task 2: Conditional prompt hook for operator-ended trials

**Files:**
- Modify: `src/inspect_robots/cli.py` — new `_prompt_operator_on_operator_end` beside `_prompt_operator` (after line ~575), gate rewrite at lines 972-981, `--no-prompt` added to the `eval-set` subparser (mirror `p_run`'s at line 233), `before_scoring` threaded into the `eval_set()` call (lines 1107-1119), `"operator_end"` entry in `_OUTCOME_PHRASES` (lines 94-102)
- Test: `tests/test_registry_cli.py` — unit tests beside the existing `_prompt_operator` unit tests (which start at line 2223; place the new ones after the last of them — find the end with `grep -n "def test_prompt_operator" tests/test_registry_cli.py`), CLI-level tests following the in-test embodiment-registration pattern (`tests/test_registry_cli.py:631-652`)

**Interfaces:**
- Consumes: `OPERATOR_END` from Task 1 — imported in `cli.py` as a **top-level runtime import** (`from inspect_robots.types import OPERATOR_END`). Do NOT add it to the `if TYPE_CHECKING:` block at `cli.py:67-73` (that import is erased at runtime and the hook would `NameError`).
- Produces: `_prompt_operator_on_operator_end(record, scene)` — module-level in `cli.py` so tests can import it, same signature as `_prompt_operator`.

- [x] **Step 1: Write the failing unit tests**

Add after the last `_prompt_operator` unit test, mirroring their record-construction style. There is **no** shared `_terminated_record` helper — `test_prompt_operator_unit_semantics` uses a local `_record()` closure (`tests/test_registry_cli.py:2387`); read `test_prompt_operator_still_prompts_without_definitive_verdict` (line 2270) and inline the same `TrialRecord`/steps construction with the reason swapped. Where the snippets below say `_terminated_record(...)`, substitute that inlined construction:

```python
def test_prompt_on_operator_end_prompts_and_records_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_robots.cli import _prompt_operator_on_operator_end

    answers = iter(["partial", "left gripper slipped"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    record = _terminated_record(reason="operator_end")  # same construction as the test at line 2270, reason swapped
    _prompt_operator_on_operator_end(record, Scene(id="s0", instruction="reach"))
    assert record.operator_judgement == "partial"
    assert record.operator_note == "left gripper slipped"
    capsys.readouterr()


def test_prompt_on_operator_end_ignores_other_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots.cli import _prompt_operator_on_operator_end

    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("must not prompt: not operator_end")
    )
    for reason, truncated in [("max_steps", True), ("success", False), (None, False)]:
        record = _terminated_record(reason=reason, truncated=truncated)
        _prompt_operator_on_operator_end(record, Scene(id="s0", instruction="reach"))
        assert record.operator_judgement is None
```

(If no `_terminated_record` helper exists, inline the same record construction the neighboring tests use — do not invent a new fixture style.)

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_registry_cli.py -q -k operator_end`
Expected: FAIL with `ImportError: cannot import name '_prompt_operator_on_operator_end'`

- [x] **Step 3: Implement the hook and the gate**

In `src/inspect_robots/cli.py`, after `_prompt_operator` (around line 575):

```python
def _prompt_operator_on_operator_end(record: TrialRecord, scene: Scene) -> None:
    """Prompt only for trials the operator demonstrably ended (R6-safe).

    ``OPERATOR_END`` means a human pressed the end-episode key, so prompting
    here can never block an unattended run — every other reason (``max_steps``
    included) keeps R6's non-blocking behavior for registered tasks.
    """
    if record.termination_reason == OPERATOR_END:
        _prompt_operator(record, scene)
```

Add a **top-level runtime import** near `cli.py`'s other runtime imports: `from inspect_robots.types import OPERATOR_END`. (Not the `if TYPE_CHECKING:` block at lines 67-73 — that import vanishes at runtime and the hook would `NameError`.)

Add the outcome phrase so operator-ended runs don't render as degraded/unmapped in the run summary (`_outcome_line` marks unknown reasons `has_unmapped` and prints via `_print_degraded`, `cli.py:635-636, 758-759`). In `_OUTCOME_PHRASES` (lines 94-102), add:

```python
    "operator_end": "ended by operator",
```

Replace the gate at lines 972-981:

```python
        before_scoring = None
        if not args.no_prompt and sys.stdin.isatty():
            if is_adhoc and any(s.name == "operator" for s in task.scorers):
                # Ad-hoc operator-scored runs: every non-definitive trial is
                # prompted, exactly as before.
                before_scoring = _prompt_operator
            else:
                # Any task, registered included: a trial the operator ended by
                # keypress owes a verdict; everything else stays non-blocking
                # and unattended-safe (R6).
                before_scoring = _prompt_operator_on_operator_end
```

Also add a phrase-map unit test (there are no existing direct `_OUTCOME_PHRASES` tests — outcome behavior is tested via rendering around `tests/test_registry_cli.py:1308-1463`; place this beside the new hook unit tests):

```python
def test_outcome_phrase_maps_operator_end() -> None:
    from inspect_robots.cli import _OUTCOME_PHRASES

    assert _OUTCOME_PHRASES["operator_end"] == "ended by operator"
```

Both Step 1 snippets use `Scene(...)`: import it function-locally (`from inspect_robots.scene import Scene`), matching the neighboring tests (`tests/test_registry_cli.py:2234, 2277`).

- [x] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_registry_cli.py -q -k operator_end`
Expected: PASS

- [x] **Step 5: Write the CLI-level registered-task test**

Following the in-test registration pattern (`tests/test_registry_cli.py:628-640` for the embodiment, `:2180-2219` for the registered task + cleanup):

```python
def test_registered_task_prompts_when_operator_ends_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace as dc_replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.types import OPERATOR_END, Action, StepResult

    class _OperatorEndsEmbodiment(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:  # every step: human ends the episode
            result = super().step(action)
            return dc_replace(
                result, terminated=True, truncated=False, termination_reason=OPERATOR_END
            )

    name = "operator-ends-cubepick-for-test"
    embodiment_decorator(name)(_OperatorEndsEmbodiment)
    _tty_stdin(monkeypatch)
    answers = iter(["y", "smooth grasp"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    log_dir = tmp_path / "logs"
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "-T",
                "num_scenes=1",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(log_dir),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    log = _read_only_log(log_dir)
    assert log.samples[0].operator_judgements == ("y",)
    assert log.samples[0].operator_notes == ("smooth grasp",)
    capsys.readouterr()
```

The `-T num_scenes=1` is load-bearing: `cubepick-reach` defaults to `num_scenes=4` (`src/inspect_robots/_builtins.py:42-50`), which would mean 4 prompted trials and a `StopIteration` from the 2-answer iterator (`before_scoring` exceptions propagate out of `main()`, `eval.py:389-393`). `-T` is defined at `cli.py:217` and reaches the task factory via `_parse_kvs` (`cli.py:132-139, 956`); `parse_value` coerces `"1"` to `int` (`_defaults.py:34-54`).

- [x] **Step 6: Wire eval-set the same way**

`eval-set` currently never prompts: `_cmd_eval_set` doesn't pass `before_scoring` to `eval_set()` (`cli.py:1107-1119`) even though `eval_set()` accepts it (`eval.py:538`), and its subparser has no `--no-prompt`. Three edits:

1. Move the `--no-prompt` definition (currently on `p_run`, `cli.py:232-236`) into `_add_shared_eval_args` (`cli.py:142-148`) so both commands get it from one place — that helper's docstring exists precisely so shared run/eval-set flags don't drift between two copies. Do not leave a second copy on `p_run`.
2. In `_cmd_eval_set`, before the `eval_set(...)` call:

```python
        before_scoring = None
        if not args.no_prompt and sys.stdin.isatty():
            # Registered tasks only here: prompt for exactly the trials a human
            # ended by keypress (OPERATOR_END); everything else stays
            # non-blocking and unattended-safe (R6).
            before_scoring = _prompt_operator_on_operator_end
```

and pass `before_scoring=before_scoring` to `eval_set(...)`.

3. Add a CLI-level test mirroring Step 5's, using the same `_OperatorEndsEmbodiment` registration pattern but `main(["eval-set", "cubepick-reach", "--policy", "scripted", "--embodiment", name, "--log-dir", str(log_dir)])`. `eval-set` has no `-T`, so the default 4 scenes stand; epochs default to 1, so that's 4 prompted trials: feed `answers = iter(["y", ""] * 4)` (empty note → recorded as `None`). The log shape is **one `SceneResult` per scene with one tuple entry per epoch** (`eval.py:456-470`), so assert:

```python
    assert [s.operator_judgements for s in log.samples] == [("y",)] * 4
    assert [s.operator_notes for s in log.samples] == [(None,)] * 4
```

4. Two more cases in the same test (or siblings): (a) non-TTY stdin with `input` set to `pytest.fail` — `eval-set` completes and `[s.operator_judgements for s in log.samples] == [(None,)] * 4`; (b) TTY stdin **with `--no-prompt`** and `input` set to `pytest.fail` — same silent result. Case (b) is the only behavioral coverage of the new flag (branch coverage alone won't catch its absence: `not args.no_prompt` short-circuits into the same false-exit as a non-TTY).

- [x] **Step 7: Run the CLI-level tests, then the full suite and gates**

Run: `uv run pytest tests/test_registry_cli.py -q -k operator_end && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all green — including the untouched `test_registered_task_never_prompts_even_with_operator_scorer_on_tty`.

- [x] **Step 8: Commit**

```bash
git add src/inspect_robots/cli.py tests/test_registry_cli.py
git commit -m "feat(cli): grade attended trials that end with operator_end, registered tasks included (#194)"
```

---

### Task 3: Docs — the operator_end contract

**Files:**
- Modify: `docs/guide/scoring.md` (operator-scorer section), `docs/guide/policies-and-embodiments.md` (embodiment-author contract), `docs/guide/cli.md:107-109` (stale prompting claim)

**Interfaces:**
- Consumes: the constant and gate from Tasks 1-2. Nothing downstream.

- [x] **Step 1: Document the contract for embodiment authors**

In `docs/guide/policies-and-embodiments.md`, in the `StepResult` section (lines 61-101; anchor on `def step(self, action: Action) -> StepResult:` at line 82), add:

```markdown
When a human operator ends an episode without giving a verdict (an end-episode
keypress), terminate with `termination_reason=inspect_robots.OPERATOR_END`
(`"operator_end"`). Attended runs (interactive terminal, no `--no-prompt`) then
get the operator prompt — `did the robot succeed? [y/n/partial/skip]` plus an
optional grader note — for exactly those trials, on registered tasks and ad-hoc
runs alike. Do **not** ask your own verdict prompt in the embodiment and
terminate with a definitive `"success"`/`"failure"`: that suppresses the
framework prompt and locks the run out of partial/skip verdicts and notes.
```

- [x] **Step 2: Document the scoring pairing**

In `docs/guide/scoring.md`, in the operator-scorer section (anchor: line 17 `operator_scorer,  # reads a human verdict recorded during the rollout` and line 62), add after the existing prose:

```markdown
Trials that end with `termination_reason="operator_end"` are prompted on any
attended run — registered tasks included — so judgement-reading scorers (the
`operator` scorer, or task scorers that fall back to `operator_judgement`)
work with operator-in-the-loop embodiments. `success_at_end` reads only
embodiment-detected `"success"` terminations and scores operator-graded trials
as failures; pair attended operator-graded runs with a judgement-reading
scorer instead.
```

- [x] **Step 3: Fix the stale CLI-guide claim**

The stale sentence spans `docs/guide/cli.md:107-110`: "Piped/CI stdin, `--no-prompt`, or a registered `--task` run never prompt or adopt an embodiment verdict, and an unjudged trial honestly scores as failure with 'no operator judgement recorded'." The registered-task clause becomes false; keep the scoring clause. Replace with:

```markdown
Piped/CI stdin or `--no-prompt` never prompt. A registered `--task` or
`eval-set` run prompts only for trials that end with
`termination_reason="operator_end"` — a human pressed the end-episode key — and
never adopts an embodiment verdict or prompts for any other ending, so
unattended runs stay non-blocking. An unjudged trial honestly scores as
failure with "no operator judgement recorded".
```

- [x] **Step 4: Run gates and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: PASS

```bash
git add docs/guide/scoring.md docs/guide/policies-and-embodiments.md docs/guide/cli.md plans/0029-operator-end-grading-gate.md
git commit -m "docs: operator_end grading contract for embodiment authors (#194)"
```
