# Wizard option slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let embodiment plugins declare boolean behavior toggles (first consumer: inspect-robots-yam's `auto_start`, yam#87) that the `inspect-robots setup` wizard interviews as yes/no questions and writes into `[embodiment.args]`.

**Architecture:** Mirror the existing `DeviceSlot` / `DEVICE_SLOTS` protocol one-for-one: a frozen `OptionSlot` dataclass and a defensive `option_slots()` reader in `conformance.py`, plus a small `_options_section()` interview in `_setup.py` that runs after the device/camera section for the configured embodiment. Answered options become managed args (re-runs re-prompt with the carried value as the suggestion; `_render_config` writes them from the interview, not the carry-through). Plugins on older cores are unaffected; cores with no declaring plugin ask nothing.

**Tech Stack:** Python 3.10+, stdlib only, pytest with injected `input_fn`/`out` seams.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict), `uv run pytest --cov` at **100% coverage**.
- Every public module/class/function needs a docstring; state the contract.
- Repo root is the `wizard-auto-start` worktree at `.claude/worktrees/wizard-auto-start`; run everything via `uv run ...` there.
- Wizard behavior for embodiments that declare no `OPTION_SLOTS` must be byte-for-byte unchanged (all existing golden-config tests must pass untouched).
- Commit messages: imperative, scoped; reference yam#87 as motivation where apt.

## Reference: current wiring

- `conformance.py:32-70`: `@dataclass(frozen=True) class DeviceSlot` and `device_slots(factory)` — the defensive-reader pattern to copy (`getattr` in try/except, `Iterable` check, per-entry `isinstance` filter).
- `_setup.py:130-147`: `_ask_yes_no(prompt, default, *, input_fn, out) -> bool` — the interview primitive; renders `[Y/n]`/`[y/N]`, Enter accepts the default.
- `_setup.py:15`: `_parse_value` is already imported; it coerces `"true"`/`"false"` (any case) to bool and other strings to other scalars.
- `_setup.py:1013-1043` (`run_setup`): computes `slots = device_slots(...)` for the configured embodiment, then either `_device_section` (sets `managed_args = tuple(slot.arg ...)`) or `_camera_section` (sets `managed_args = CAMERA_KEYS`). Everything interview-shaped sits inside the `try` whose `except (EOFError, KeyboardInterrupt)` aborts with nothing written.
- `_setup.py:893-945` (`_render_config(defaults, embodiment_args, carried, managed_args)`): writes managed keys from `embodiment_args`, then carries any `[embodiment.args]` key NOT in `managed_args` verbatim — so once an option key is managed, the interview's answer wins over the carried line and there is no duplicate.
- `tests/test_setup.py:122-138`: `_register_device_slots(monkeypatch, slots, name)` fakes `inspect_robots.registry.registered` with a `_Factory` class carrying `DEVICE_SLOTS`; the autouse `_empty_registry` fixture empties the registry otherwise. Scripted runs drive `run_setup` with `input_fn=iter(answers).__next__`-style callables and a `StringIO` out (see `test_run_setup_defaults_and_numbered_cameras_write_golden_config`, line 553, for the env/tmp_path harness to copy).

---

### Task 1: `OptionSlot` + `option_slots` in conformance.py

**Files:**
- Modify: `src/inspect_robots/conformance.py` (below `device_slots`, ~line 70)
- Test: `tests/test_conformance.py`

**Interfaces:**
- Produces: `OptionSlot(arg: str, label: str, default: bool = False)` (frozen dataclass) and `option_slots(factory: object) -> tuple[OptionSlot, ...]`. Task 2 consumes both. Plugins import `OptionSlot` from `inspect_robots.conformance` (same path as `DeviceSlot`; not added to the package `__all__`, matching `DeviceSlot`, unless `DeviceSlot` IS in `__all__` — mirror whatever `DeviceSlot` does).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_conformance.py` (extend its conformance import line with `OptionSlot, option_slots`; `ClassVar` may need adding to the typing imports):

```python
def test_option_slots_reads_declared_tuple_in_order() -> None:
    declared = (
        OptionSlot(arg="auto_start", label="Skip the operator start prompts (auto_start)"),
        OptionSlot(arg="verbose", label="Verbose", default=True),
    )

    class _Factory:
        OPTION_SLOTS: ClassVar[tuple[OptionSlot, ...]] = declared

    assert option_slots(_Factory) == declared


def test_option_slots_defaults_to_false() -> None:
    assert OptionSlot(arg="a", label="A").default is False


def test_option_slots_none_and_absent_return_empty() -> None:
    class _Bare:
        pass

    assert option_slots(None) == ()
    assert option_slots(_Bare()) == ()


def test_option_slots_malformed_never_crash() -> None:
    class _NotIterable:
        OPTION_SLOTS: ClassVar[int] = 7

    class _MixedEntries:
        OPTION_SLOTS: ClassVar[tuple[object, ...]] = (
            OptionSlot(arg="ok", label="OK"),
            "junk",
            None,
        )

    class _RaisingIteration:
        def __iter__(self) -> "Iterator[OptionSlot]":
            raise RuntimeError("boom")

    class _RaisingIterationFactory:
        OPTION_SLOTS: ClassVar[object] = _RaisingIteration()

    class _RaisingGetattr:
        @property
        def OPTION_SLOTS(self) -> tuple[OptionSlot, ...]:
            raise RuntimeError("boom")

    assert option_slots(_NotIterable()) == ()
    assert option_slots(_MixedEntries()) == (OptionSlot(arg="ok", label="OK"),)
    assert option_slots(_RaisingIterationFactory()) == ()
    assert option_slots(_RaisingGetattr()) == ()
```

Before finalizing, compare with how `tests/test_conformance.py` already exercises `device_slots`' defensive paths (lines ~63-111) and match its idioms exactly: it annotates fixture class attributes with `ClassVar[...]`, breaks iteration by raising in `__iter__`, asserts `device_slots(None) == ()`, and parametrizes whole-value garbage — mirror whichever of those apply, reusing its parametrize lists where they fit, in preference to the sketch above where they differ.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conformance.py -k option_slot -v`
Expected: FAIL at import (`ImportError: cannot import name 'OptionSlot'`).

- [ ] **Step 3: Implement**

In `src/inspect_robots/conformance.py`, directly after `device_slots`:

```python
@dataclass(frozen=True)
class OptionSlot:
    """One boolean behavior toggle the setup wizard interviews.

    ``arg`` is the ``[embodiment.args]`` key to write (``true``/``false``);
    ``label`` is the yes/no question shown to the operator ("Skip the
    operator start prompts (auto_start)"); ``default`` is the suggested
    answer when the key is absent from an existing config.
    """

    arg: str
    label: str
    default: bool = False


def option_slots(factory: object) -> tuple[OptionSlot, ...]:
    """The declared option slots, defensively read.

    Reads ``OPTION_SLOTS`` off ``factory``; anything that is not an iterable
    of ``OptionSlot`` instances has the offending entries ignored, never
    crashes the wizard. Returns a tuple in declaration order.
    """
    try:
        slots = getattr(factory, "OPTION_SLOTS", None)
    except Exception:
        return ()
    if not isinstance(slots, Iterable):
        return ()
    try:
        return tuple(slot for slot in slots if isinstance(slot, OptionSlot))
    except Exception:
        return ()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conformance.py -v`
Expected: PASS (new and existing).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/conformance.py tests/test_conformance.py
git commit -m "conformance: OptionSlot protocol for wizard-interviewed behavior toggles"
```

---

### Task 2: wizard interviews declared options

**Files:**
- Modify: `src/inspect_robots/_setup.py` (import at line 14; new `_options_section` near `_device_section`; wiring in `run_setup` ~line 1013-1043)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `OptionSlot`, `option_slots` from Task 1; existing `_ask_yes_no`, `_parse_value`, `_render_config` managed-args semantics.
- Produces: `_options_section(options, carried, *, input_fn, out) -> dict[str, str]` (private) and the run_setup behavior below.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_setup.py` a registrar mirroring `_register_device_slots` (import `OptionSlot` alongside `DeviceSlot`):

```python
def _register_option_slots(
    monkeypatch: pytest.MonkeyPatch,
    options: tuple[OptionSlot, ...],
    name: str = "option-body",
) -> None:
    class _Factory:
        OPTION_SLOTS: ClassVar[tuple[OptionSlot, ...]] = options

    monkeypatch.setattr(
        "inspect_robots.registry.registered",
        lambda kind: {name: _Factory} if kind == "embodiment" else {},
    )
```

Then tests, following the harness style of `test_run_setup_defaults_and_numbered_cameras_write_golden_config` (env with `XDG_CONFIG_HOME=tmp_path` and `DISPLAY: ":0"` — without DISPLAY the headless rerun note fires and flips the rerun suggestion, adding transcript noise; scripted answers via the harness's `_scripted_input`, which RECORDS every prompt string into a `prompts` list; `io.StringIO()` out; read back the written config with the same mechanism the golden test uses). The scripted answer sequences must be derived from the ACTUAL prompt order in the current wizard: `_prompt_defaults` asks exactly 6 questions (the `SUGGESTED` keys), then the camera/device section, then options. The options-only factory declares no `DEVICE_SLOTS`, so `_camera_section` runs and its "Configure cameras?" question must be answered "n" (check the real prompt text before writing the list).

IMPORTANT: prompt strings (including the `[y/N]`/`[Y/n]` suffix) go to `input_fn`, NOT to `out` — all prompt assertions must target the recorded `prompts` list, the way existing tests assert on `prompts[6]` (see `test_setup.py` ~lines 1090-1128). Asserting prompt text "not in out" is vacuously true and guards nothing.

```python
AUTO_START = OptionSlot(arg="auto_start", label="Skip the operator start prompts (auto_start)")


def test_run_setup_writes_declared_option_yes(tmp_path, monkeypatch) -> None:
    # embodiment answer must name the registered factory ("option-body");
    # answer "y" at the option prompt -> auto_start = true in [embodiment.args];
    # the option prompt (prompts[7], after 6 defaults + "Configure cameras?")
    # contains AUTO_START.label.
    ...


def test_run_setup_writes_declared_option_default_no(tmp_path, monkeypatch) -> None:
    # Enter at the option prompt with no carried value -> auto_start = false,
    # and the recorded option prompt carries the "[y/N]" suffix (default False).
    ...


def test_run_setup_option_suggestion_comes_from_carried_config(tmp_path, monkeypatch) -> None:
    # Pre-write a config with [embodiment.args] auto_start = true (and the
    # matching [defaults] embodiment = option-body so the same factory is
    # suggested); Enter keeps it -> auto_start = true survives, and the
    # recorded option prompt carries "[Y/n]".


def test_run_setup_option_carried_garbage_falls_back_to_default(tmp_path, monkeypatch) -> None:
    # Pre-write auto_start = banana -> recorded option prompt carries "[y/N]"
    # (the slot default), and Enter writes false.


def test_run_setup_option_answer_overrides_carried_value_without_duplicate(tmp_path, monkeypatch) -> None:
    # Pre-write auto_start = true; answer "n" -> the written config text
    # contains exactly one "auto_start" occurrence and it is false.


def test_run_setup_abort_during_options_writes_nothing(tmp_path, monkeypatch) -> None:
    # input_fn raises EOFError at the option prompt -> exit code 1, no config file.


def test_run_setup_interviews_options_alongside_device_slots(tmp_path, monkeypatch) -> None:
    # The real first-consumer shape (yam declares both): register ONE factory
    # class carrying both DEVICE_SLOTS (a single "can" slot, following
    # _register_device_slots) and OPTION_SLOTS = (AUTO_START,). Decline the
    # device section ("n" at "Configure devices?", so _preserve_managed_args
    # runs; pre-write the device arg in [embodiment.args] to assert it is
    # preserved), then answer "y" at the option prompt. Assert both keys are
    # written exactly once: the carried device value and auto_start = true.
    ...


def test_run_setup_option_colliding_with_managed_key_is_skipped(tmp_path, monkeypatch) -> None:
    # A factory with no DEVICE_SLOTS but OPTION_SLOTS naming a camera key
    # (OptionSlot(arg="top_cam_device", ...)) plus AUTO_START: the colliding
    # option is never prompted (no prompt contains its label), only
    # auto_start is interviewed, and the written config has no duplicate
    # top_cam_device line. Also cover a duplicate arg WITHIN options
    # ((AUTO_START, OptionSlot(arg="auto_start", label="dupe"))): one prompt,
    # first declaration wins.


def test_run_setup_no_declared_options_asks_nothing(tmp_path, monkeypatch) -> None:
    # A factory without OPTION_SLOTS: the recorded prompts list contains no
    # option-label text and has exactly the same length as before this
    # feature (6 defaults + camera/device prompts) — the regression guard
    # for unchanged wizards.
```

Flesh each `...`/comment into real code against the actual harness (the plan cannot transcribe the golden test's env scaffold; copy it from the file). Assert written-file contents via the same mechanism the golden test uses.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_setup.py -k option -v`
Expected: FAIL — no option prompt is consumed, so scripted-answer counts and file contents mismatch (the abort test fails because setup completes instead of aborting).

- [ ] **Step 3: Implement**

In `src/inspect_robots/_setup.py`:

1. Extend the conformance import: `from inspect_robots.conformance import DeviceSlot, OptionSlot, device_slots, option_slots`.

2. Add after `_device_section` (module-private, docstringed):

```python
def _options_section(
    options: tuple[OptionSlot, ...],
    carried: dict[str, dict[str, str]],
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> dict[str, str]:
    """Interview plugin-declared behavior toggles as yes/no questions.

    The carried config value (parsed as a bool) is the suggested answer when
    present and boolean; otherwise the slot's declared default. Answers are
    written explicitly (``true``/``false``) so declining a previously enabled
    toggle turns it off rather than silently carrying it forward.
    """
    existing_args = carried.get("embodiment.args", {})
    answers: dict[str, str] = {}
    for option in options:
        suggested = option.default
        if option.arg in existing_args:
            parsed = _parse_value(existing_args[option.arg])
            if isinstance(parsed, bool):
                suggested = parsed
        enabled = _ask_yes_no(option.label, suggested, input_fn=input_fn, out=out)
        answers[option.arg] = "true" if enabled else "false"
    return answers
```

3. In `run_setup`, alongside the `slots = device_slots(...)` computation, add the parallel `options = option_slots(embodiment_factories[configured_embodiment]) if configured_embodiment in embodiment_factories else ()` (keep the existing membership-test style). After the `if slots: ... else: ...` device/camera block, still inside the `try`:

```python
        # Options whose arg collides with an already-managed key (a device
        # slot or camera key) or repeats an earlier option are skipped:
        # interviewing them would write the key twice, and a duplicate key
        # makes configparser reject the whole file on the next setup run.
        interviewed: list[OptionSlot] = []
        taken = set(managed_args)
        for option in options:
            if option.arg in taken:
                continue
            taken.add(option.arg)
            interviewed.append(option)
        if interviewed:
            managed_args = managed_args + tuple(option.arg for option in interviewed)
            embodiment_args.update(
                _options_section(tuple(interviewed), carried, input_fn=input_fn, out=out)
            )
```

(`embodiment_args.update` is safe: `_device_section`, `_camera_section`, and `_preserve_managed_args` all return freshly built dicts, never references into `carried` — verify this still holds before relying on it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_setup.py -v`
Expected: all PASS — new option tests and every pre-existing golden/scripted test unchanged.

- [ ] **Step 5: Full gate set**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: clean, 100% coverage (both suggestion sources, both answers, garbage fallback, abort path, the collision-skip `continue`, the combined device+option shape, and the no-options fall-through are all exercised above).

- [ ] **Step 6: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: interview plugin-declared option slots (yam#87 auto_start)"
```

---

### Task 3: docs

**Files:**
- Modify: `README.md` (wizard description ~line 67)
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Added`; match the file's existing heading structure)
- Modify: `docs/guide/adapters.md` (the `DEVICE_SLOTS` section, ~lines 42-70)
- Modify: `src/inspect_robots/CLAUDE.md` (module-map rows for `conformance.py` and `_setup.py`)

- [ ] **Step 1: README**

Extend the wizard sentence ("The wizard picks your defaults and finds your cameras, then writes...") with one clause noting it also asks about behavior toggles the embodiment plugin declares (e.g. yam's `auto_start`). Match the surrounding prose style; no em dashes if the README avoids them (check nearby prose and mirror it).

- [ ] **Step 2: adapters guide (`docs/guide/adapters.md`)**

This is the plugin-author-facing protocol, so it needs the authoring docs. After the `DEVICE_SLOTS` section, add a parallel `OPTION_SLOTS` subsection in the same voice and format, with a minimal declaration example:

```python
from inspect_robots.conformance import OptionSlot


class MyEmbodiment:
    OPTION_SLOTS: ClassVar[tuple[OptionSlot, ...]] = (
        OptionSlot(
            arg="auto_start",
            label="Skip the operator start prompts (auto_start)",
        ),
    )
```

State the contract in prose: each slot is one yes/no question; `arg` is the `[embodiment.args]` key written as `true`/`false`; the carried config value is the suggested answer on re-runs; declarations whose `arg` collides with a device slot, camera key, or earlier option are skipped. Adapt the example to the surrounding doc's actual class/ClassVar conventions.

- [ ] **Step 3: CHANGELOG**

```markdown
- `OptionSlot` / `OPTION_SLOTS` (plan 0032): embodiment plugins can declare
  boolean behavior toggles that `inspect-robots setup` interviews as yes/no
  questions and writes into `[embodiment.args]`. First consumer:
  inspect-robots-yam's `auto_start` (yam#87).
```

Match the existing entry indentation/format exactly.

- [ ] **Step 4: Module map (`src/inspect_robots/CLAUDE.md`)**

Extend the `conformance.py` row to mention `OptionSlot`/`option_slots` next to `DeviceSlot`/`device_slots`, and the `_setup.py` row to mention the option interview, matching each row's existing phrasing density.

- [ ] **Step 5: Gates + commit**

Run: `uv run pytest -q` (green tree), then:

```bash
git add README.md CHANGELOG.md docs/guide/adapters.md src/inspect_robots/CLAUDE.md
git commit -m "docs: describe wizard option-slot interviews"
```

---

## Out of scope

- Non-boolean option kinds (choice/text): YAGNI until a plugin needs one; the dataclass can grow a `kind` later without breaking declarers.
- The yam-side `OPTION_SLOTS` declaration: separate PR in inspect-robots-yam after this ships in a core release (it needs the new dependency floor).
- Prompting for options of unregistered embodiments: without the plugin installed there is no declaration to read; the wizard already warns about missing plugins.
