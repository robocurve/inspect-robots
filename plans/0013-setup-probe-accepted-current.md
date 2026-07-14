# Probe Enter-accepted stale camera devices in the setup wizard

Date: 2026-07-14
Status: draft; subagent critique round 1 in progress

## Problem

`_prompt_device_slot` (`src/inspect_robots/_setup.py`) Enter-accepts a
`(current, not detected)` config value with no validation: the accepted path
is never run through the color-capture probe added in #70. A stale pre-#70
config carrying a RealSense depth node sails through setup and fails minutes
later at runtime as `EmbodimentFault: cannot open top_cam` (observed on a yam
rig on 2026-07-14; see #82).

Separately, when udev's by-id table is missing capture-capable nodes that
by-path has (same rig: the D435 color node `/dev/video16` had no by-id
symlink at all), the only clue is the count-mismatch hint, whose stated cause
("identical cameras without serials collide") is wrong for this failure
mode. The correct device is not selectable from the default listing while
the wrong one is Enter-acceptable. This is the sharp end of #71.

## Requirements

1. Enter-accepting a camera device that the wizard can tell is wrong warns
   the operator at setup time, not minutes later at runtime.
2. Inconclusive probes never block acceptance (Windows workstations, unusual
   devices, configs authored for another machine). #71's lesson: only speak
   when the evidence is definitive.
3. A second Enter always accepts, so scripted or headless Enter-through runs
   terminate.
4. When the by-id listing cannot show a capture-capable node that by-path
   can, the wizard says so per device section instead of relying on the
   operator discovering the `p` toggle.
5. No public API changes; `tests/test_api_snapshot.py` unchanged.

## Design

### A. Probe on Enter-accept of a not-detected current value (v4l2 slots only)

In the `elif not entered and current is not None` branch of
`_prompt_device_slot`: when `kind == "v4l2"` and `current not in devices`:

- Path missing: warn once in the existing foreign-machine tone ("does not
  exist here (ok if this config is for another machine)") and re-prompt; a
  second Enter keeps the value.
- Path exists: call `_v4l2_color_capture(Path(current))`:
  - `False`: yellow warning that the device offers no color capture format
    (likely a depth or metadata node); re-prompt once; a second Enter keeps
    it.
  - `None` (inconclusive): accept silently, exactly as today.
- One re-prompt per slot at most (local flag scoped to the prompt loop), so
  repeated Enters terminate and deliberate unusual devices stay usable.
- CAN and serial slots keep today's behavior (no probe exists for them).

### B. Name the by-id gap instead of hand-waving

Where the count-mismatch hint prints: compute the set of by-path
color-capable nodes whose resolved target has no by-id entry (compare
resolved-path sets of the two scans). When non-empty, state the actual
situation: N capture-capable nodes have no by-id entry; press `p` to pick
them from the by-path listing (stable per physical USB port). Keep the
serial-collision wording only for the case where the resolved sets match
(pure name collision). No behavior change to listings or toggle logic.

## Tests

TDD against the existing `tests/test_setup.py` patterns (fake `input_fn`,
injected scans, monkeypatched `_v4l2_color_capture`):

1. Enter on a not-detected current whose probe returns `False`: warning
   printed, re-prompt; second Enter accepts and writes the value.
2. Same, but the operator picks a number after the warning: the picked
   device wins.
3. Probe returns `None`: accepted immediately, no warning.
4. Current path does not exist: foreign-machine warning, re-prompt once.
5. Detected current (in the active listing): no probe call at all (assert
   via monkeypatch counter); zero new friction on the happy path.
6. by-path-only capture-capable nodes present: new hint text names the count
   and the `p` toggle; resolved sets equal: old collision wording.
7. CAN slot with a not-detected current: unchanged single-Enter accept.

## Constraints

- 100% coverage, mypy strict (src and tests), ruff D1 docstrings on new
  helpers. Core stays NumPy-only; the probe already lazily imports `fcntl`.
- Respect #70's review resolutions (Bayer fourccs, bounded enumeration,
  Windows skips, native-endian structs).
