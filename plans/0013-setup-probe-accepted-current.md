# Probe Enter-accepted stale camera devices in the setup wizard

Date: 2026-07-14
Status: revised after subagent critique round 1 (probe-True acceptance, both
listings checked before probing, discriminator dropped, dual hint sites,
explicit test updates); round 2 pending

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
("identical cameras without serials collide") is one possible cause of
several, and whose remedy (the `p` toggle) is easy to miss. This is the
sharp end of #71.

## Requirements

1. Enter-accepting a camera device that the wizard can tell is wrong warns
   the operator at setup time, not minutes later at runtime.
2. Inconclusive probes never block acceptance (Windows workstations, unusual
   devices, configs authored for another machine). #71's lesson: only speak
   when the evidence is definitive.
3. A second Enter always accepts, so scripted or headless Enter-through runs
   terminate.
4. A healthy device never triggers new friction, including one configured
   via by-path or a raw `/dev/videoN` path while the by-id listing is
   active ("not detected" is listing-relative).
5. The count-mismatch hint names its possible causes and the `p` remedy
   without overclaiming what the fallback scan cannot know.
6. No public API changes; `tests/test_api_snapshot.py` unchanged.

## Design

### A. Probe on Enter-accept of a not-detected current value (v4l2 slots only)

In the `elif not entered and current is not None` branch of
`_prompt_device_slot`, when `kind == "v4l2"`:

- Skip all new logic when `current` is present in either scan
  (`by_id_devices` or `by_path_devices`), not just the active listing: a
  device the other listing knows is healthy, and probing it again is
  pointless (requirement 4).
- Otherwise, if the path does not exist: warn once in the existing
  foreign-machine tone ("does not exist here (ok if this config is for
  another machine)") and re-prompt; a second Enter keeps the value.
- Otherwise call `_v4l2_color_capture(Path(current))`:
  - `True` or `None`: accept silently, exactly as today. `None` covers
    Windows (`fcntl` ImportError), open/QUERYCAP `OSError`, and other
    inconclusive outcomes; never nag on those.
  - `False`: yellow warning that the device offers no color capture format
    (likely a depth or metadata node); re-prompt; a second Enter keeps it.
    Yellow, not the issue's "red": the wizard palette (`_setup.py:18-22`)
    defines no red and yellow is its warning convention; note the deviation
    in the PR body.
- Re-prompt bookkeeping: one local `warned_current` flag scoped to the
  prompt loop of the slot. Semantics: (i) the second Enter accepts
  unconditionally, with no re-probe; (ii) the flag survives `p` toggles and
  `u` rescans within the slot (the verdict is a property of the node, not
  the listing); (iii) the warning path must `continue` before the generic
  instruction block at `_setup.py:413-423` so the operator never gets two
  stacked messages; (iv) the group-retry loop (`_setup.py:691`) re-invokes
  `prompt_slot`, resetting the flag and warning again on a fresh pass;
  acceptable because it is operator-bounded.
- CAN and serial slots keep today's behavior (no probe exists for them).

### B. One honest hint, both sites

The count-mismatch hint prints in two places with identical text:
`_camera_section` (`_setup.py:488-497`) and `_device_section`
(`_setup.py:631-641`). Extract a shared helper (D1 docstring) and change the
message once:

- Compute how many by-path scan entries resolve (via `Path.resolve()` or
  `os.path.realpath`) to targets that no by-id scan entry resolves to.
- When that count is positive and the by-id listing is active, print: N of
  the detected camera nodes have no by-id entry (udev serial collision or
  missing symlink); by-path names are stable per physical USB port; press
  `p` to switch listing.
- When the active listing is already by-path (fresh systems where by-id is
  empty: `_setup.py:470/585` set `active_is_by_id = False`), drop the `p`
  clause; the operator is already looking at the right list.
- Do not claim the extra nodes are capture-capable: `_scan_cameras`
  (`_setup.py:736-743`) returns all entries when no node probes `True`
  (fallback mode) and discards per-entry verdicts, so capability is unknown
  here. Say "camera nodes". Threading verdicts out of the scan is #71's
  work, not this PR's.
- No realpath-set "collision vs missing symlink" discriminator: a serial
  collision also leaves the collided-away node's realpath absent from the
  by-id set, so the two causes are indistinguishable by construction and the
  old wording would be dead code. One message, both causes named.

## Tests

TDD against the existing `tests/test_setup.py` patterns (fake `input_fn`,
`_make_devices` fixtures, monkeypatched `inspect_robots._setup._v4l2_color_capture`):

1. Enter on a not-detected current whose probe returns `False`: warning
   printed, re-prompt; second Enter accepts and writes the value.
2. Same, but the operator picks a number after the warning: the picked
   device wins.
3. Probe returns `None`: accepted immediately, no warning. Same for `True`.
4. Current path exists in the by-path scan while by-id is active: accepted
   immediately, probe not called for that path (count probe invocations
   whose argument equals the current path, or drive `_prompt_device_slot`
   directly per the established private-helper import pattern,
   `tests/test_setup.py:16-30`; a blanket call counter cannot work because
   `_scan_cameras` probes every entry).
5. Current path does not exist: foreign-machine warning, re-prompt once,
   second Enter accepts.
6. Hint: by-path entries resolving to targets absent from by-id → new text
   with the count and the `p` clause; active listing already by-path → no
   `p` clause. Symlink fixtures pointing into a shared target dir, wrapped
   in the existing `pytest.skip` on `OSError` pattern
   (`tests/test_setup.py:89`) for the Windows advisory tier.
7. CAN slot with a not-detected current: unchanged single-Enter accept
   (guards `tests/test_setup.py:2251` behavior).

Existing tests that must be updated (they break by design):

- `tests/test_setup.py:1536` `test_run_setup_marks_undetected_current_camera_defaults`:
  currents are nonexistent `/remote/*` paths with a `[""] * 10` script; the
  new missing-path re-prompt consumes three extra Enters. Extend the script
  and assert the new warning.
- `tests/test_setup.py:1442-1449` and `:2305`: assert the exact old
  collision wording; update to the new message (their `_make_devices`
  fixtures live in disjoint tmp dirs, so the by-path-extras branch fires).

## Constraints

- 100% coverage, mypy strict (src and tests), ruff D1 docstrings on new
  helpers. Core stays NumPy-only; the probe already lazily imports `fcntl`.
- Respect #70's review resolutions (Bayer fourccs, 64-entry bounded
  enumeration, native-endian structs, Windows ImportError → None), all
  reused unmodified.
- Known probe caveat, acceptable: an `OSError` mid `ENUM_FMT` returns
  `False` (`_setup.py:725-727`), so an I/O-flaky device draws the warning
  with an imprecise cause; the "likely a depth or metadata node" hedge is
  the mitigation.
