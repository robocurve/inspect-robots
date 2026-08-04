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
part of the udev by-id name), defaulted to `None` so existing keyword
constructions of the dataclass stay valid. A new
`_ambiguous_identities(inventory)` returns the set of `(model, serial)`
identity keys claimed by more than one physical camera, with two carve-outs:
a duplicated non-None serial keeps its current meaning (ambiguous
regardless of model, preserving plan-0040 behavior — this needs an
explicit rule beyond a naive Counter over `(model, serial)`), and a record
with BOTH `model is None` and `serial is None` is never ambiguous (unknown
identity is not shared identity; two sysfs-less cameras must not demote
each other, matching the old fiat rung's effect for exactly that case).
Deduplication is per physical camera via `record.camera or record.node`,
exactly as `_duplicated_serials` does today. Every current
`_duplicated_serials` consumer switches to the new helper through one
record-level predicate, which also deletes the `record.serial is None`
trust-by-fiat rung in `_preferred_name`. Slot prompts refuse ambiguous
by-id names on EVERY acceptance path (Enter-accept of a carried current
value included, not just the duplicate-assignment collision), because a
name that can silently swap cameras is never a valid assignment.

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
- Existing behavior tests pass untouched. Permitted edits to existing
  tests, exhaustively: (a) mechanical call-signature updates in unit tests
  calling `_preferred_name` directly (`tests/test_setup.py:525-546`);
  (b) rewriting the `tests/test_setup.py:498` assertion on
  `_duplicated_serials(inventory) == {"SN0001"}` to the new helper's
  equivalent claim (same fixture, same meaning: the shared serial is
  ambiguous), including the matching import swap in the
  `tests/test_setup.py:22-30` import block; (c) nothing else — in
  particular
  `test_reconcile_missing_current_matches_serial_from_by_id_name`
  (`tests/test_setup.py:549-568`) constructs `_CameraNode` directly and
  must keep passing, which the defaulted `model=None` field guarantees.
  Duplicate-non-empty-serial behavior (the Innomaker/plan-0040 cases) must
  stay byte-for-byte identical: same rows, same hints, same names.
- A single serial-less camera with no same-model twin keeps its by-id name
  (no behavior change for healthy single-camera rigs).
- Docs follow the repo writing rules (no em dashes in prose).
- Commit messages: imperative, scoped; reference #299.

## Reference: current wiring (main @ 13ca1e41)

- `_setup.py:1009-1024` — `_CameraNode`: `node`, `camera` (sysfs USB device
  dir, one per physical camera), `serial`, `by_id`, `by_path`.
- `_setup.py:1058-1097` — `_camera_inventory`: groups symlinks by resolved
  target, probes color capability, locates the USB device dir via
  `_usb_device_dir` (`:1027-1041`, only returns ancestors where
  `idVendor.is_file()`, so `idVendor` exists whenever `camera` is not
  None), reads `serial` with `.strip() or None` (empty serial becomes
  None, which is how it escapes duplicate detection today).
- `_setup.py:1100-1109` — `_duplicated_serials`: counts non-None serials per
  physical camera (dedup key `record.camera or record.node` at `:1107`);
  its docstring already describes the shared-name replug race this plan
  closes for the empty-serial case.
- `_setup.py:1112-1131` — `_preferred_name`: the trust ladder. The bug line:
  `trusted = by_id is not None and (record.serial is None or record.serial
  not in duplicated)` — a missing serial confers trust by fiat.
- `_setup.py:1134-1152` — `_camera_rows`: per-view listing rows through
  `_preferred_name`; the empty-inventory raw-directory fallback at
  `:1147-1152` stays as is (no records means ambiguity is undetectable in
  principle).
- `_setup.py:1158-1177` — `_reconcile_missing_current`: maps a dead by-id
  name to the camera carrying its embedded serial; calls `_preferred_name`
  with `_duplicated_serials` at 1177. Its `_BY_ID_SERIAL` regex captures a
  model tail (e.g. "405") from serial-less names, but the
  `record.serial == serial` filter can never match a None-serial record,
  so the function is a safe no-op there — no change needed.
- `_setup.py:568-595` — the Enter-accept path for a carried `current`
  value: when the saved name still exists on the rig, no warning fires and
  `selected = current` — this path must also refuse ambiguous by-id names
  (Task 3), or slot 1 silently keeps the ping-pong name and the refusal
  only ever fires at slot 2.
- `_setup.py:282-303` — `_print_camera_name_hint`: fallback predicate
  `record.by_id is None or (record.serial is not None and record.serial in
  duplicated)`.
- `_setup.py:306-325` — `_camera_view_state`: same predicate decides whether
  to advertise the 'p' toggle.
- `_setup.py:406-408` — identify flow's `name_of`: canonicalizes the
  physically identified camera through `_preferred_name`; this is the call
  that hands the stolen by-id name to the wizard after a replug.
- `_setup.py:514-669` — `_prompt_device_slot`: receives `inventory`
  (`:525`); the duplicate-assignment branch at 638-668 prints "already
  assigned" and asks "Use X for both ...?".

## Task 1: model identity on the inventory record

- [x] **Step 1: failing tests.** The `_usb_device` fixture
  (`tests/test_setup.py:211`) already writes `idVendor` with content
  ("8086") but no `idProduct` at all: parameterize the fixture to take a
  product id (and expose the vendor for Task 2's different-model tests),
  writing the `idProduct` file. Tests: `_camera_inventory` fills
  `model="8086:0b5b"`-style values; a camera without a resolvable sysfs
  dir gets `model=None`; unreadable `idProduct` gets `model=None` —
  simulate unreadable with a DIRECTORY named `idProduct`
  (IsADirectoryError is an OSError; `chmod 000` is a no-op under root CI).
- [x] **Step 2: implement.** Add `model: str | None = None` to
  `_CameraNode` (defaulted, so direct constructions in existing tests
  stay valid; docstring: stable udev-name proxy, `idVendor:idProduct`).
  In `_camera_inventory`, read both files next to the existing serial
  read, same exception envelope, using the `.strip() or None` EXPRESSION
  form (an `if`-based empty check would add a branch no listed test
  covers); either file missing or empty yields `model=None`.
- [x] **Step 3: gates green, commit.**

## Task 2: name-collision ambiguity replaces serial-collision ambiguity

- [ ] **Step 1: failing tests.** Fixture: two same-model cameras, both
  `serial` files empty, one owning the shared by-id symlink, the other
  by-path only (mirror the rig-2 listing). Assert: (a) `_camera_rows`
  by-id view lists BOTH cameras by their by-path names; (b)
  `_print_camera_name_hint` counts 2 fallback nodes; (c) the identify
  flow's canonical name for the symlink owner is its by-path name; (d) a
  lone serial-less camera (no twin) keeps by-id; (e) two serial-less
  cameras of DIFFERENT models (parameterized fixture vendor/product) both
  keep by-id; (f) two cameras sharing a non-None serial across DIFFERENT
  models are both ambiguous (the plan-0040 rule needs its own rule beyond
  a naive `(model, serial)` Counter, and this test is what keeps that
  branch covered under the 100% gate — the existing `_shared_serial_rig`
  cameras share a hardcoded vendor and cannot exercise it); (g) two
  cameras with `camera=None` (no sysfs: `model=None, serial=None`) both
  KEEP their by-id names (unknown identity is not shared identity);
  (h) the existing duplicate-serial tests pass unchanged.
- [ ] **Step 2: implement.** Add
  `_ambiguous_identities(inventory) -> set[tuple[str | None, str | None]]`
  per the Architecture paragraph — dedup per physical camera via
  `record.camera or record.node` exactly as `_duplicated_serials` does
  today, duplicated non-None serials ambiguous regardless of model,
  `(None, None)` never ambiguous — and a record predicate (module-level
  helper, e.g. `_is_ambiguous(record, ambiguous)`) so call sites stay
  one-liners. Convert `_preferred_name` to take the ambiguous-key set
  (drop the serial-None fiat rung), and update `_camera_rows`, `name_of`,
  `_reconcile_missing_current`, `_print_camera_name_hint`, and
  `_camera_view_state`. Delete `_duplicated_serials` or reduce it to a
  private detail of the new helper; do not leave two competing notions of
  ambiguity in the module. Rewrite the `tests/test_setup.py:498`
  assertion to the new helper's equivalent claim (permitted-edit (b) in
  Global Constraints).
- [ ] **Step 3: gates green, commit.**

## Task 3: refuse ambiguous by-id names on every acceptance path

- [ ] **Step 1: failing tests.** Two paths, both driven by calling
  `_prompt_device_slot` directly (the existing helper pattern at
  `tests/test_setup.py:60-86`, but with a real camera inventory): (i) a
  freshly TYPED ambiguous by-id name with a pre-populated `assigned` dict
  holding the same name — once the feature exists this state is
  unreachable through the full wizard (every acceptance path refuses),
  so a run_setup-level transcript cannot test it; assert the refusal
  prints and no "Use X for both" question appears; (ii) the SLOT-1
  Enter-accept hole: a carried `current` equal to an ambiguous by-id
  name whose symlink exists — today `selected = current` succeeds at
  `_setup.py:595`; assert the refusal fires and the prompt re-asks.
  Test-fake wrinkle for (ii): the `_color_by_node` fake
  (`tests/test_setup.py:228`) keys on basename, and after Task 2 the
  ambiguous by-id name is in neither rows list, so the first Enter hits
  the not-in-listing warning block and burns a `continue`; either feed
  two Enters or make the color fake resolve symlinks (production is
  unaffected — the real probe resolves).
- [ ] **Step 2: implement.** Refuse at the single point where any
  `selected` value is about to be returned (covering typed names,
  Enter-accepted current values, and duplicate collisions alike): when
  `selected` is the `by_id` of any inventory record whose identity key is
  ambiguous, print the refusal (name the claimant cameras via
  `by_path or node`, mirroring `_print_camera_name_hint` at
  `_setup.py:294`, so the operator knows what to pick) and `continue`.
  The "use for both" yes/no stays for every other duplicate (typed raw
  paths, by-path duplicates from a genuinely shared device). The
  empty-inventory raw fallback (`_camera_rows:1147-1152`) keeps today's
  behavior: with no records, ambiguity is undetectable in principle.
  Non-camera slots are unaffected by construction (`_device_section`
  passes an empty inventory for non-v4l2 kinds, `_setup.py:896`).
  Accepted limitation, note in the refusal-block comment: a typed
  speed-qualified alias (`usbv2-`/`usbv3-`) of the same ambiguous camera
  bypasses the exact-string match; records store only the plain alias.
- [ ] **Step 3: gates green, commit.**

## Task 4: docs and changelog sweep

- [ ] **Step 1:** Update the user-facing strings and docs that state the
  old rule, at minimum: `_print_camera_name_hint`'s message
  (`_setup.py:296-299`, "a serial shared by two cameras" → wording that
  covers duplicate OR missing serials; tests assert only the "no usable
  by-id entry" substring, so rewording is safe) and `docs/guide/cli.md:161`
  ("when two cameras share one serial"). Grep `docs/` and the `_setup.py`
  module docstring for other by-id trust or duplicate-serial language.
  Update the `_duplicated_serials` docstring's successor to name the
  empty-serial case explicitly.
- [ ] **Step 2:** `CHANGELOG.md` entry under "Unreleased" per
  `CONTRIBUTING.md:131`, referencing #299.
- [ ] **Step 3: gates green, commit.**
