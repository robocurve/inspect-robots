# CAN pinning port fallback Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** The setup wizard's udev pinning suggestion must not give up when
CAN adapter serials are missing or duplicated (Innomaker gs_usb adapters all
report serial `SN0001`): fall back to port-pinned `KERNELS` rules so
multi-adapter rigs keep stable interface names across replugs and reboots.
Closes #275.

**Architecture:** `_suggest_can_pinning` becomes a three-rung ladder over one
shared derived-names computation: (1) `ATTRS{serial}` rules exactly as today
when every serial is present and distinct; (2) otherwise `KERNELS` rules
keyed on each interface's USB kernel device name (read through the same
sysfs `device` link `_can_serial` already uses) when those names are all
available and distinct, with copy telling the operator the name follows the
physical port; (3) otherwise today's bare warning. A dual-channel adapter
exposing two netdevs from one USB interface makes the kernel names collide,
which is exactly rung 3.

**Tech stack:** stdlib only. pytest with tmp_path fake sysfs trees, mirroring
the existing `_can_serial` fixtures.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, covers `src` and `tests`), `uv run pytest --cov` at
  **100% coverage**.
- Every public module/class/function needs a docstring stating the contract
  (ruff D1).
- Repo root is the `ir-wt-can-pinning` worktree at
  `~/robocurve/ir-wt-can-pinning`; run everything via `uv run ...` there.
- The serial rung's output stays byte-for-byte identical: the exact-copy
  tests at `tests/test_setup.py:3682`, `:3712`, `:3731` (and any sibling
  pinning tests) pass untouched. If one fails, treat it as a bug in the new
  code, not a test to update.
- Docs follow the repo writing rules (no em dashes in prose).
- Commit messages: imperative, scoped; reference #275.

## Reference: current wiring (main @ ba873203)

- `_setup.py:1206-1212` — `_can_serial(sysfs_net, ifname)`: resolves
  `<sysfs_net>/<ifname>/device`, reads `parent / "serial"`; broad
  `except Exception` returns None. The `device` link points at the USB
  *interface* directory (e.g. `.../3-2/3-2:1.0`), whose parent is the USB
  device directory holding `serial`. The interface directory's *name*
  (`3-2:1.0`) is what a udev `KERNELS==` clause matches.
- `_setup.py:1215-1280` — `_suggest_can_pinning`: relevance gates at
  1223-1237 (two-plus order-dependent `canN` names, at least one assigned, at
  least one on USB); the `warning` string and serial bail at 1239-1243
  (`if not all(serials) or len(set(serials)) != len(serials): print bare
  warning; return`); derived-name computation at 1245-1265 (slot-stem names,
  collision/length fallback to `can_a`, `can_b`, ...); rule rendering and the
  block print at 1267-1280.
- `tests/test_setup.py:747-765` — the `_can_serial` fixture idiom: build
  `sysfs_net/<ifname>` with a `device` symlink to a fake interface dir whose
  parent holds a `serial` file. Reuse this shape; for the fallback tests give
  interface dirs realistic kernel names (`3-2:1.0`, `3-4:1.0`).
- `tests/test_setup.py:3682-3760` — the three existing pinning tests,
  including one asserting the full block copy verbatim.
- `docs/guide/cli.md:132-135` — the CAN wizard paragraph ending in the
  pinning-suggestion sentence.

---

### Task 1: `_can_kernels` helper

**Files:**
- Modify: `src/inspect_robots/_setup.py` (directly after `_can_serial`)
- Test: `tests/test_setup.py` (next to the `_can_serial` tests, ~747)

- [ ] **Step 1: Write the failing tests**

- `test_can_kernels_reads_interface_kernel_name`: fixture as at `:747` with
  the device symlink pointing at `.../3-2/3-2:1.0`; `_can_kernels(sysfs_net,
  "can0") == "3-2:1.0"`.
- `test_can_kernels_missing_device_link_returns_none`: no `device` link (or
  a dangling one) returns None.

Extend the test file's `from inspect_robots._setup import ...` block.

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_setup.py -k can_kernels -v` — ImportError.

- [ ] **Step 3: Implement**

```python
def _can_kernels(sysfs_net: Path, ifname: str) -> str | None:
    """Kernel device name of a CAN interface's USB interface, if resolvable.

    The name (for example ``3-2:1.0``) is what a udev ``KERNELS==`` clause
    matches, pinning the interface to its physical USB port.
    """
    try:
        return (sysfs_net / ifname / "device").resolve(strict=True).name
    except OSError:
        return None
```

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_setup.py -k can_kernels -v`, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: read a CAN interface's kernel device name (#275)"
```

---

### Task 2: the port-pinned fallback rung

**Files:**
- Modify: `src/inspect_robots/_setup.py` (`_suggest_can_pinning`)
- Test: `tests/test_setup.py` (next to the existing pinning tests, ~3682)

- [ ] **Step 1: Write the failing tests**

Mirror the existing pinning fixtures (two adapters on USB, assigned
`left_channel=can0`, `right_channel=can1` through the same `DeviceSlot`
tuples the neighboring tests build), varying only serials and interface dirs:

- `test_suggest_can_pinning_shared_serials_fall_back_to_port_rules`: both
  adapters report serial `SN0001`, interface dirs `3-2:1.0` and `3-4:1.0`.
  Assert the exact block (same all-at-once style as the test at `:3682`):

  ```
  these CAN interfaces have order-dependent names; a replug can swap them.
  adapter serials are missing or shared, so pin them by USB port instead
  (paste into /etc/udev/rules.d/70-can-names.rules, then replug or reboot),
  and re-run setup to record the pinned names; a port-pinned name follows the
  physical USB port, so keep each adapter plugged into the same port:
    SUBSYSTEM=="net", ACTION=="add", KERNELS=="3-2:1.0", NAME="can_left"
    SUBSYSTEM=="net", ACTION=="add", KERNELS=="3-4:1.0", NAME="can_right"
  ```

- `test_suggest_can_pinning_missing_serials_fall_back_to_port_rules`: no
  `serial` files at all; the same `KERNELS` rules appear (spot-check the two
  rule lines rather than the full block).
- `test_suggest_can_pinning_dual_channel_adapter_stays_bare_warning`: two
  netdevs whose `device` links resolve to the SAME interface dir (one gs_usb
  dual-channel adapter), serials shared; output is exactly the single
  warning line, no rules.
- `test_suggest_can_pinning_unresolvable_kernels_stays_bare_warning`: serials
  shared and one interface's `device` link missing; single warning line only.

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_setup.py -k pinning -v` — the new tests fail on
missing rules/copy; the three existing ones still pass.

- [ ] **Step 3: Implement**

Restructure `_suggest_can_pinning` minimally:

1. Hoist the derived-names block (`_setup.py:1245-1265`) above the serial
   check; it never depended on serials (pure reorder, no behavior change).
2. Replace the bail at 1239-1243 with the ladder. Sketch:

```python
    serials = [_can_serial(sysfs_net, ifname) for ifname in order_dependent]
    if all(serials) and len(set(serials)) == len(serials):
        attribute_clauses = [f'ATTRS{{serial}}=="{serial}"' for serial in serials]
        intro = [
            warning,
            "pin them by adapter serial (paste into /etc/udev/rules.d/70-can-names.rules,",
            "then replug or reboot), and re-run setup to record the pinned names:",
        ]
    else:
        kernels = [_can_kernels(sysfs_net, ifname) for ifname in order_dependent]
        if not all(kernels) or len(set(kernels)) != len(kernels):
            print(_paint(warning, _YELLOW, out), file=out)
            return
        attribute_clauses = [f'KERNELS=="{kernel}"' for kernel in kernels]
        intro = [
            warning,
            "adapter serials are missing or shared, so pin them by USB port instead",
            "(paste into /etc/udev/rules.d/70-can-names.rules, then replug or reboot),",
            "and re-run setup to record the pinned names; a port-pinned name follows the",
            "physical USB port, so keep each adapter plugged into the same port:",
        ]
    rules = [
        f'  SUBSYSTEM=="net", ACTION=="add", {clause}, NAME="{name}"'
        for clause, name in zip(attribute_clauses, derived_names, strict=True)
    ]
    print(_paint("\n".join([*intro, *rules]), _YELLOW, out), file=out)
```

The serial rung's rendered block must remain byte-for-byte what the test at
`:3682` asserts today (same intro lines, same rule format); only the internal
shape changes. Drop the now-unused `serial_values` filter (in the serial rung
`all(serials)` already guarantees no None; mypy needs the narrowed list — a
comprehension over `serials` inside the `if` arm after the `all()` check may
still type as `str | None`, so build the clauses from `serials` with an
explicit narrowing (`[s for s in serials if s is not None]`) or `cast`,
whichever reads cleaner under strict mypy. Update the function docstring:
serial-pinned rules preferred, port-pinned fallback, bare warning last.

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_setup.py -k pinning -v`, then all four gates.
100% coverage means every ladder arm is hit; the four new tests plus the
three existing ones cover serial-rung, both fallback triggers (missing and
duplicated serials), and both rung-3 arms (duplicate kernels, unresolvable
kernels).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: fall back to port-pinned udev rules for shared-serial CAN adapters (#275)"
```

---

### Task 3: docs

**Files:**
- Modify: `docs/guide/cli.md` (the CAN sentence at lines 132-135)
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Fixed`; match the existing
  entry format)
- Modify: `src/inspect_robots/CLAUDE.md` (the `_setup.py` module map row,
  line 36)

- [ ] **Step 1: cli guide**

Extend the pinning sentence: the suggestion pins by adapter serial when
serials are present and distinct, and otherwise by USB port (`KERNELS`
rules), which is what rigs with several identical gs_usb adapters such as
Innomaker's (every unit reports `SN0001`) need; a port-pinned name stays
valid only while the adapter keeps its port. No em dashes in prose.

- [ ] **Step 2: CHANGELOG**

One `### Fixed` entry under `## [Unreleased]` (#275, plan 0043): the CAN
pinning suggestion no longer degrades to a bare warning for shared or
missing adapter serials; it emits port-pinned `KERNELS` rules instead.

- [ ] **Step 3: Module map**

Extend the `_setup.py` row's plan list with 0043 and rephrase "CAN udev
guidance" to mention serial-pinned or port-pinned rules, matching the row's
density.

- [ ] **Step 4: Gates + commit**

`uv run pytest -q` green, then:

```bash
git add docs/guide/cli.md CHANGELOG.md src/inspect_robots/CLAUDE.md
git commit -m "docs: describe the port-pinned CAN fallback (#275)"
```

---

## Out of scope

- Writing udev rules automatically: the suggestion stays paste-by-hand; a
  wizard writing to `/etc` would need root and a rollback story.
- Listing CAN interfaces by port in the wizard's interview (the unplug-diff
  flow already disambiguates identical adapters for assignment; this plan
  only fixes the persistence suggestion).
- Camera udev rule generation (tracked in plan 0040's out-of-scope list).
- The companion multi-rig config selection (#274, plan 0042).
