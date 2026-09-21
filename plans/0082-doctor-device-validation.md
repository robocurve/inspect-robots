# 0082: `doctor` validates configured device slots

Addresses the device-presence portion of item 5 in issue #50. Item 5
also requested a CAN "UP" check, however the codebase currently does
not support this, so that check is out of scope. This plan builds on
the `DEVICE_SLOTS` mechanism from plan 0011 and the `doctor`
runtime-requirement preflight from plan 0010.

## Problem

The registered embodiment plugin declares its device-shaped
constructor arguments via `DEVICE_SLOTS` (plan 0011). The
`inspect-robots setup` probes the host for each device described by
those arguments and records the chosen values back into
`[embodiment.args]`.  However, the embodiment args values can go stale
for various reasons: a device may be unplugged or a CAN adapter might
be renamed by udev for example.

The `doctor` checks the configured embodiment's runtime requirements,
but it never re-checks the current host given the configured device
references. So if the values go stale, issues may first appear when
the embodiment is constructed for a run. This plan uses `doctor` to
re-read the values before the robot is run so that inaccurate device
configuration issues are identified.

## Scope

When `doctor` checks the configured embodiment, before construction it
also:

- reads the embodiment factory's declared `DEVICE_SLOTS`
  (`conformance.device_slots`);
- for each slot whose argument is present in the resolved
  `[embodiment.args]` (config plus `-E`), validates the configured
  value by the slot kind, checking presence the same way `setup` does,
  by:
  - `v4l2` and `serial`: the configured value is an absolute device
    path, valid when that path exists; if `Path.exists()` raises `OSError`,
    it is reported as an error finding for that slot and the remaining slots
    are still checked;
  - `can`: the configured value is a SocketCAN interface name, valid
    when that name is present among the sysfs CAN interfaces
    (`_setup._scan_can`), and the message lists the interfaces that
    are present on a miss;
- reports all device findings together with `doctor`'s other findings.
- returns a nonzero exit when any device finding exists.

Both runtime requirement checks and validation are reported before
construction, so constructor failures do not hide device
findings. Only configured string values are checked.

Note: Presence is the only CAN property `setup` currently
probes. There is currently no notion of interface up/down state
anywhere (`setup` lists interfaces by sysfs type, never operstate).

## Out of scope

- Configuration validation beyond embodiment.
- Identifying the plugin package and install command for unknown or
  uninstalled component names.
- Version compatibility detection.
- CAN interface UP-state checks.
- Policy-server reachability checks.
- Changes to `setup` or wizard behavior.

## Design

New checker in `conformance.py`:

```python
def check_device_slots(
    factory: object,
    configured: Mapping[str, object],
    *,
    sysfs_net: Path | None = None,
) -> list[ConformanceIssue]:
    """Checks for configured device-slot values that no longer resolve on this host."""
```

It reads `device_slots(factory)` and returns one
`ConformanceIssue(severity="error", code="device", ...)` for each
configured value that does not resolve. `configured` is typed as
`Mapping[str, object]` because `[embodiment.args]` values are
processed by `_parse_value` and may be non-string. A slot whose value
is absent or non-string is skipped by this check. Note that
`_scan_can` and `SYSFS_NET` are imported lazily from
`inspect_robots._setup` to avoid a circular import. `sysfs_net` is injectable so
tests inject a fake.

The `_cmd_doctor` fetches the embodiment factory once and runs
`check_device_slots(factory, kvs)` before construction. Each finding
is printed as ` [error] device: <message>` next to the existing
runtime-requirement lines, and factored into the exit code: `return 1
if not report.ok or missing or device_issues else 0`.

Note: `check_device_slots` follows the established convention of being
imported via `from inspect_robots.conformance import ...` and not
added to any `__all__`, so `tests/test_api_snapshot.py` is unchanged.

## Tests


`tests/test_registry_cli.py` adds an embodiment with `DEVICE_SLOTS`
and verifies the `doctor` behavior:

- Multiple invalid device value findings are reported at once and
  `doctor` returns 1.
- Valid configured values yield no findings.

`tests/test_conformance.py` covers `check_device_slots`:
- missing CAN interfaces and device paths are reported in declaration
  order
- a CAN miss lists the interfaces that are present
- existing paths and CAN interfaces pass
- absent and disabled values are skipped
- a factory with no device-slot declaration produces no findings

## Docs

- `docs/guide/cli.md` doctor section: briefly explains that `doctor`
  checks the configured device-slot values before the robot is run,
  reporting a camera or serial path that no longer exists or a CAN
  interface not present in the host's SocketCAN interfaces. Notes that
  the CAN check is presence only.
- `CHANGELOG.md` under `[Unreleased]` / `### Added`, `**Core:**`
  prefix, referencing plan 0082 and #50.
- Module map (`src/inspect_robots/CLAUDE.md`): Updated the
  `conformance.py` row to mention device-slot validation.
