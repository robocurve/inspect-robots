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


def test_option_slots_absent_or_malformed_never_crash() -> None:
    class _Bare:
        pass

    class _NotIterable:
        OPTION_SLOTS = 7

    class _MixedEntries:
        OPTION_SLOTS = (OptionSlot(arg="ok", label="OK"), "junk", None)

    class _RaisingGetattr:
        @property
        def OPTION_SLOTS(self) -> tuple[OptionSlot, ...]:
            raise RuntimeError("boom")

    class _RaisingIteration:
        class _Evil:
            def __iter__(self) -> "_RaisingIteration._Evil":
                return self

            def __next__(self) -> OptionSlot:
                raise RuntimeError("boom")

        OPTION_SLOTS = _Evil()

    assert option_slots(_Bare()) == ()
    assert option_slots(_NotIterable()) == ()
    assert option_slots(_MixedEntries()) == (OptionSlot(arg="ok", label="OK"),)
    assert option_slots(_RaisingGetattr()) == ()
    assert option_slots(_RaisingIteration()) == ()
```

Before finalizing, compare with how `tests/test_conformance.py` already exercises `device_slots`' defensive paths and match its idioms (it may already have equivalents of `_RaisingGetattr` etc. to crib; property-on-class raise only fires on instances, hence the instance calls above — keep whichever instantiation style the device_slots tests use).

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

Then tests, following the harness style of `test_run_setup_defaults_and_numbered_cameras_write_golden_config` (env with `XDG_CONFIG_HOME=tmp_path`, scripted answers list, `io.StringIO()` out, read back the written config with the existing raw-read helper or `configparser`). The scripted answer sequences must be derived from the ACTUAL prompt order in the current wizard (defaults prompts first, then the camera/device section, then options) — count the prompts in the existing golden test and extend its answer list; the factory here declares no `DEVICE_SLOTS`, so the camera path (`_camera_section`) runs, and its "Configure cameras?"-style question must be answered "n" (check the real prompt text in `_camera_section` before writing the list).

```python
AUTO_START = OptionSlot(arg="auto_start", label="Skip the operator start prompts (auto_start)")


def test_run_setup_writes_declared_option_yes(tmp_path, monkeypatch) -> None:
    # embodiment answer must name the registered factory ("option-body");
    # answer "y" at the option prompt -> auto_start = true in [embodiment.args].
    ...


def test_run_setup_writes_declared_option_default_no(tmp_path, monkeypatch) -> None:
    # Enter at the option prompt with no carried value -> auto_start = false,
    # and the prompt suffix shown in `out` is [y/N] (default False).
    ...


def test_run_setup_option_suggestion_comes_from_carried_config(tmp_path, monkeypatch) -> None:
    # Pre-write a config with [embodiment.args] auto_start = true (and the
    # matching [defaults] embodiment = option-body so the same factory is
    # suggested); Enter keeps it -> auto_start = true survives, prompt shows [Y/n].


def test_run_setup_option_carried_garbage_falls_back_to_default(tmp_path, monkeypatch) -> None:
    # Pre-write auto_start = banana -> suggestion is the slot default (False).


def test_run_setup_option_answer_overrides_carried_value_without_duplicate(tmp_path, monkeypatch) -> None:
    # Pre-write auto_start = true; answer "n" -> written config contains
    # exactly one auto_start line and it is false.


def test_run_setup_abort_during_options_writes_nothing(tmp_path, monkeypatch) -> None:
    # input_fn raises EOFError at the option prompt -> exit code 1, no config file.


def test_options_section_absent_when_no_declared_options(tmp_path, monkeypatch) -> None:
    # _empty_registry / a factory without OPTION_SLOTS: no option prompt text
    # in `out`; this is the regression guard for unchanged wizards.
```

Flesh each `...` into real code against the actual harness (the plan cannot transcribe the golden test's 30-line env scaffold; copy it from the file). Assert written-file contents via the same mechanism the golden test uses.

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
        if options:
            managed_args = managed_args + tuple(option.arg for option in options)
            embodiment_args.update(
                _options_section(options, carried, input_fn=input_fn, out=out)
            )
```

(`embodiment_args` is a plain dict in both branches; if either branch returns a mapping that must not be mutated, use `embodiment_args = {**embodiment_args, **_options_section(...)}` instead — check `_camera_section`/`_device_section`/`_preserve_managed_args` return values; `_preserve_managed_args` builds a fresh dict, so `update` is safe, but verify.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_setup.py -v`
Expected: all PASS — new option tests and every pre-existing golden/scripted test unchanged.

- [ ] **Step 5: Full gate set**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: clean, 100% coverage (both suggestion sources, both answers, garbage fallback, abort path, and the no-options fall-through are all exercised above).

- [ ] **Step 6: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: interview plugin-declared option slots (yam#87 auto_start)"
```

---

### Task 3: docs

**Files:**
- Modify: `README.md` (wizard description ~line 67)
- Modify: `CHANGELOG.md` (`## Unreleased` → `### Added`; match the file's existing heading structure)

- [ ] **Step 1: README**

Extend the wizard sentence ("The wizard picks your defaults and finds your cameras, then writes...") with one clause noting it also asks about behavior toggles the embodiment plugin declares (e.g. yam's `auto_start`). Match the surrounding prose style; no em dashes if the README avoids them (check nearby prose and mirror it).

- [ ] **Step 2: CHANGELOG**

```markdown
- `OptionSlot` / `OPTION_SLOTS` (plan 0032): embodiment plugins can declare
  boolean behavior toggles that `inspect-robots setup` interviews as yes/no
  questions and writes into `[embodiment.args]`. First consumer:
  inspect-robots-yam's `auto_start` (yam#87).
```

Match the existing entry indentation/format exactly.

- [ ] **Step 3: Gates + commit**

Run: `uv run pytest -q` (green tree), then:

```bash
git add README.md CHANGELOG.md
git commit -m "docs: describe wizard option-slot interviews"
```

---

## Out of scope

- Non-boolean option kinds (choice/text): YAGNI until a plugin needs one; the dataclass can grow a `kind` later without breaking declarers.
- The yam-side `OPTION_SLOTS` declaration: separate PR in inspect-robots-yam after this ships in a core release (it needs the new dependency floor).
- Prompting for options of unregistered embodiments: without the plugin installed there is no declaration to read; the wizard already warns about missing plugins.
