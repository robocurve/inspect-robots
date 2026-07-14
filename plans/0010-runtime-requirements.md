# 0010 — Runtime-requirement preflight (`RUNTIME_REQUIREMENTS`)

Fixes #59. Adapters have runtime deps that install metadata cannot express
(the `i2rt` motor driver is git-only, so PyPI forbids depending on it); today
they surface as `EmbodimentFault: No module named 'i2rt'` at reset, with the
arms already in the loop. Components declare their runtime imports as data;
`setup` and `doctor` preflight them with `importlib.util.find_spec`, which
never executes the module.

## 1. Protocol (data only)

A component factory (class or function, whatever is registered) may carry:

```python
RUNTIME_REQUIREMENTS = {
    "i2rt": 'uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"',
    "cv2": 'uv pip install "inspect-robots-yam[cameras]"',
}
```

Keys are importable module names, values are remediation commands shown
verbatim. Absent attribute means no requirements. The declaration lives on
the factory so checking never constructs the component (constructors are
hardware-free by convention, but the whole point is to run before anything
plugin-shaped executes).

## 2. Core checker: `conformance.py`

```python
def missing_runtime_requirements(factory: object) -> dict[str, str]:
    """The declared runtime modules that are not importable here.

    Reads ``RUNTIME_REQUIREMENTS`` (module name -> remediation command) off
    ``factory`` and probes each with ``importlib.util.find_spec``. Top-level
    names (the intended use) are probed without executing anything; a dotted
    name imports its parent package when present, so declare top-level names.
    ANY probe failure counts as missing (broad ``except Exception``: a
    present-but-broken parent package propagates arbitrary errors from its
    ``__init__``, and this checker must never crash setup or doctor).
    Entries whose key or value is not ``str``, or a ``RUNTIME_REQUIREMENTS``
    that is not a ``Mapping``, are ignored (a plugin typo must not crash the
    preflight). Returns the missing subset, insertion-ordered.
    """
```

Lives in `inspect_robots.conformance` and is imported from there, matching
its siblings (`check_embodiment` is deliberately NOT in `__all__`; the
adapters guide teaches `from inspect_robots.conformance import ...`). No
`__all__` or API-snapshot change. `importlib.util` import stays
module-level (stdlib, numpy-only rule holds).

## 3. Consumers

### 3.1 `setup` wizard (final step, after `Wrote`)

For each of the two configured kinds (policy, embodiment) whose accepted
name IS registered: fetch the factory via `registered(kind)` (already loaded
by the per-prompt warning, so this is a dict lookup) and sweep. If anything
is missing, print a yellow checklist after the existing unregistered-name
reminder position:

```
setup complete, but 2 runtime dependencies are missing:
  ✗ i2rt (yam_arms) → uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"
  ✗ cv2 (yam_arms) → uv pip install "inspect-robots-yam[cameras]"
```

Singular "1 runtime dependency is" handled; the header counts lines. Exit
code stays 0: the config file is correct; the checklist is the actionable
part. No dedup: one line per (kind, name, module) — two components may need
the same module with different remedies, and collapsing across kinds could
silently drop a remedy. The `✗` is output only (functional mark, allowed);
`→` likewise.

### 3.2 `doctor`

`_cmd_doctor` is reordered so the sweep runs and PRINTS before construction
(construction can itself die on the missing module; `_resolve_or_exit`
catches only `KeyError`/`ConfigError`, so a constructor-time
`ModuleNotFoundError` would otherwise traceback with no guidance):

1. Print the `embodiment: {name} ({source})` header first (moved up from
   after construction).
2. Sweep `registered("embodiment").get(name)` (`.get`: an unknown name must
   fall through to `_resolve_or_exit`'s guided SystemExit, not KeyError;
   `missing_runtime_requirements(None)` returns `{}`). Print one line per
   missing module, before constructing:

```
  [error] runtime-requirement: i2rt missing → uv pip install "i2rt @ git+..."
```

3. Construct and run the conformance checks as today; the summary prints
   after (its first line repeats the embodiment name, which is acceptable).
4. Exit 1 when the report failed OR requirements were missing.

Doctor stays embodiment-only (its existing scope). A constructor that still
crashes on the missing module now crashes after the guided lines printed.

## 4. Docs and plugin side

- `docs/guide/adapters.md`: short section "Declare runtime requirements"
  with the dict example and the two consumers.
- `docs/guide/cli.md`: one sentence each under setup and doctor.
- CHANGELOG "Added" entry referencing #59.
- Module map: extend the `conformance.py` row.
- Plugin declaration for `yam_arms` is tracked separately
  (robocurve/inspect-robots-yam#30); this PR ships the protocol + consumers.
  The CubePick mock declares nothing (numpy-only, nothing to declare).

## 5. Tests (100% branch, mypy strict incl. tests)

`tests/test_conformance.py` (or wherever check_embodiment's tests live):
- no attribute → `{}`; all present (e.g. `{"os": "..."}`).
- missing top-level module → returned with its remedy, order preserved.
- submodule key with missing parent (`"definitely_missing_xyz.sub"`) →
  treated missing, no exception.
- probe raising an arbitrary exception (monkeypatched `find_spec` raising
  `OSError`, the broken-parent-package case) → treated missing, no crash.
- non-mapping attribute, and mapping entries with non-str key or value →
  ignored without crashing; `missing_runtime_requirements(None)` → `{}`.

`tests/test_setup.py`:
- registered fake factory (monkeypatched `registered`) carrying a missing
  requirement → checklist printed after `Wrote`, singular/plural forms,
  `(component)` attribution, remedy verbatim.
- requirements all present → no checklist.
- unregistered names → no sweep (existing reminder only).

`tests/test_registry_cli.py` (doctor):
- conformant embodiment whose factory declares a missing module → the
  `[error] runtime-requirement:` line and exit 1; the line appears BEFORE
  the conformance summary in captured output.
- declares only present modules → unchanged pass (exit 0).
- unknown embodiment name → the existing guided SystemExit, unchanged.

## 6. Execution

Single PR on `feat/runtime-requirements`: (1) checker + tests, (2) setup
consumer + tests, (3) doctor consumer + tests, (4) docs/changelog. Gates
per commit; Fable review-edit loop before merge.
