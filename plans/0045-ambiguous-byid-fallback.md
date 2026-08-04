# Ambiguous by-id camera fallback Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** The setup wizard must classify a by-id camera name as ambiguous
whenever udev can build the identical name for a second physical camera, not
only when two cameras share a non-empty serial. Two same-model cameras whose
USB serial is empty (rig-2's D405 pair reports `serial=''` in sysfs) produce
byte-identical by-id names; the symlink ping-pongs to whichever camera
enumerated last, unplug-identify becomes self-defeating (replugging the
camera under test steals the other camera's name), and the only exit the
wizard offers writes the same device into two slots. Closes #299.

**Architecture:** ambiguity moves from "serial shared by two cameras" to
"by-id identity claimable by two cameras". `_CameraNode` grows a `model`
field (sysfs `idVendor:idProduct`, the stable proxy for the vendor+model
part of the udev by-id name). A new `_ambiguous_identities(inventory)`
returns the set of `(model, serial)` identity keys claimed by more than one
physical camera, where a duplicated non-None serial keeps its current
meaning (ambiguous regardless of model, preserving plan-0040 behavior) and
a `None` serial is ambiguous exactly when a second camera has the same
model and also no serial. Every current `_duplicated_serials` consumer
switches to the new helper through one record-level predicate, which also
deletes the `record.serial is None` trust-by-fiat rung in
`_preferred_name`. The duplicate-assignment prompt refuses outright when
the colliding name is an ambiguous by-id name, because "use it for both
slots" can never be correct there.

**Tech stack:** stdlib only. pytest with the existing fake-sysfs camera
fixtures from plan 0040 in `tests/test_setup.py`.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, covers `src` and `tests`), `uv run pytest --cov` at
  **100% coverage**.
- Every public module/class/function needs a docstring stating the contract
  (ruff D1).
- Repo root is the `wt-ir-ambiguous-byid` worktree at
  `~/robocurve/wt-ir-ambiguous-byid`; run everything via `uv run ...` there.
- Existing behavior tests pass untouched. The only permitted edits to
  existing tests are mechanical call-signature updates in unit tests that
  call `_duplicated_serials` or `_preferred_name` directly; grep
  `tests/test_setup.py` for both names before starting and list the hits in
  the first commit message. Duplicate-non-empty-serial behavior (the
  Innomaker/plan-0040 cases) must stay byte-for-byte identical: same rows,
  same hints, same names.
- A single serial-less camera with no same-model twin keeps its by-id name
  (no behavior change for healthy single-camera rigs).
- Docs follow the repo writing rules (no em dashes in prose).
- Commit messages: imperative, scoped; reference #299.

## Reference: current wiring (main @ 13ca1e41)

- `_setup.py:1009-1024` — `_CameraNode`: `node`, `camera` (sysfs USB device
  dir, one per physical camera), `serial`, `by_id`, `by_path`.
- `_setup.py:1057-1097` — `_camera_inventory`: groups symlinks by resolved
  target, probes color capability, locates the USB device dir via
  `_usb_device_dir` (the ancestor holding `idVendor`), reads
  `serial` with `.strip() or None` (empty serial becomes None, which is how
  it escapes duplicate detection today).
- `_setup.py:1100-1109` — `_duplicated_serials`: counts non-None serials per
  physical camera; its docstring already describes the shared-name replug
  race this plan closes for the empty-serial case.
- `_setup.py:1112-1132` — `_preferred_name`: the trust ladder. The bug line:
  `trusted = by_id is not None and (record.serial is None or record.serial
  not in duplicated)` — a missing serial confers trust by fiat.
- `_setup.py:1135-1155` — `_camera_rows`: per-view listing rows through
  `_preferred_name`.
- `_setup.py:1158-1177` — `_reconcile_missing_current`: maps a dead by-id
  name to the camera carrying its embedded serial; calls `_preferred_name`
  with `_duplicated_serials` at 1177.
- `_setup.py:282-303` — `_print_camera_name_hint`: fallback predicate
  `record.by_id is None or (record.serial is not None and record.serial in
  duplicated)`.
- `_setup.py:306-325` — `_camera_view_state`: same predicate decides whether
  to advertise the 'p' toggle.
- `_setup.py:395-408` — identify flow's `name_of`: canonicalizes the
  physically identified camera through `_preferred_name`; this is the call
  that hands the stolen by-id name to the wizard after a replug.
- `_setup.py:514-668` — `_prompt_device_slot`: receives `inventory`; the
  duplicate-assignment branch at 638-668 prints "already assigned" and asks
  "Use X for both ...?".

## Task 1: model identity on the inventory record

- [ ] **Step 1: failing tests.** Extend the fake-sysfs camera fixture so a
  camera's USB device dir carries `idVendor` and `idProduct` files (it
  already must hold `idVendor` for `_usb_device_dir` to resolve; add real
  contents). Tests: `_camera_inventory` fills `model="8086:0b5b"`-style
  values; a camera without a resolvable sysfs dir gets `model=None`;
  unreadable `idProduct` gets `model=None`.
- [ ] **Step 2: implement.** Add `model: str | None` to `_CameraNode`
  (docstring: stable udev-name proxy, `idVendor:idProduct`). In
  `_camera_inventory`, read both files next to the existing serial read,
  same exception envelope, normalizing with `.strip()`; either file missing
  or empty yields `model=None`.
- [ ] **Step 3: gates green, commit.**

## Task 2: name-collision ambiguity replaces serial-collision ambiguity

- [ ] **Step 1: failing tests.** Fixture: two same-model cameras, both
  `serial` files empty, one owning the shared by-id symlink, the other
  by-path only (mirror the rig-2 listing). Assert: (a) `_camera_rows`
  by-id view lists BOTH cameras by their by-path names; (b)
  `_print_camera_name_hint` counts 2 fallback nodes; (c) the identify
  flow's canonical name for the symlink owner is its by-path name; (d) a
  lone serial-less camera (no twin) keeps by-id; (e) two serial-less
  cameras of different models both keep by-id; (f) the existing
  duplicate-serial tests pass unchanged.
- [ ] **Step 2: implement.** Add
  `_ambiguous_identities(inventory) -> set[tuple[str | None, str | None]]`
  per the Architecture paragraph, and a record predicate (module-level
  helper, e.g. `_is_ambiguous(record, ambiguous)`) so call sites stay
  one-liners. Convert `_preferred_name` to take the ambiguous-key set
  (drop the serial-None fiat rung), and update `_camera_rows`, `name_of`,
  `_reconcile_missing_current`, `_print_camera_name_hint`, and
  `_camera_view_state`. Delete `_duplicated_serials` or reduce it to a
  private detail of the new helper; do not leave two competing notions of
  ambiguity in the module.
- [ ] **Step 3: gates green, commit.**

## Task 3: refuse "use for both" on ambiguous by-id names

- [ ] **Step 1: failing test.** Drive `_prompt_device_slot` so the selected
  device equals an already-assigned ambiguous by-id name (type the name
  manually; after Task 2 the identify flow can no longer produce one).
  Assert the wizard prints a refusal explaining the name is shared between
  two cameras and re-prompts, and that `_ask_yes_no` is never reached
  (no "Use X for both" question in the transcript).
- [ ] **Step 2: implement.** In the duplicate-assignment branch
  (`_setup.py:638-668`), when the colliding `selected` is the `by_id` of
  any inventory record whose identity key is ambiguous, print the refusal
  (name the two cameras' by-path names so the operator knows what to pick)
  and `continue` instead of offering the yes/no. The yes/no stays for
  every other duplicate (typed raw paths, by-path duplicates from a
  genuinely shared device).
- [ ] **Step 3: gates green, commit.**

## Task 4: docs sweep

- [ ] **Step 1:** Grep `docs/` and the `_setup.py` module docstring for
  by-id trust or duplicate-serial language; update to "identity shared by
  two cameras (duplicate or missing serial)" wording where present. Update
  the `_duplicated_serials` docstring's successor to name the empty-serial
  case explicitly.
- [ ] **Step 2: gates green, commit.**
