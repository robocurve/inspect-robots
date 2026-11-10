# 0090: Complete the host preflight in `doctor`

Completes the remaining host checks from issue #50, building on plans
0010, 0082 and 0089. No reset, step, camera open, CAN open or policy
construction is needed.

## Problem

Resolution and device presence alone miss stale plugin releases, a viewer
without a display, administratively down CAN links, and unreachable policy
servers. These failures should appear together before a run.

## Design

A private `_doctor_probes.py` module returns `ConformanceIssue` findings.
Small functions accept environment/platform, sysfs roots, distribution
metadata and a TCP connector so tests do not need a rig or a server.
`_cmd_doctor` prints these findings before embodiment construction. Errors
fail the command; warnings and information do not. Existing guided exits
for an unset or unregistered embodiment remain intact.

- Read installed plugin distributions by their `inspect_robots.*` entry
  points, and compare their framework requirements and the framework's
  requirements on installed plugins. An optional `doctor` extra supplies
  lazily imported `packaging` for PEP 440 versions and PEP 508 markers.
  Without it, explain the install command rather than invent a version
  parser. Known yam releases through 0.6.0 lack pose parsing and action
  labels, so flag that independently of dependency bounds. Do not assume
  every plugin must share the framework's version number.
- Share setup's display predicate, including forwarded displays and native
  macOS/Windows desktops. A configured local Rerun viewer without a display
  produces a warning with `config set rerun false` and `--no-rerun`.
- Extend `check_device_slots` through an injectable CAN checker, retaining
  camera/serial path checks. Linux CAN inventory uses sysfs type 280 and
  flags (IFF_UP), since an operational state of UNKNOWN can still be UP.
  Missing names list present interfaces; no adapters suggests cabling/power,
  while present adapters also suggest setup/renaming. A down link gives an
  `ip link set` command. Other platforms report an informational skip.
- Probe configured policy URL arguments (and constructor URL defaults)
  without constructing the policy. Probe configured URLs even when the
  policy is unresolved. Respect args ownership and explicit
  disabled values. HTTP(S), WS(S) and TCP endpoints use a one-second TCP
  connection which is immediately closed. This checks reachability, not
  application readiness, authentication or inference. No HTTP request is
  sent, and URL credentials/query strings are not printed.
- Retain the existing declared runtime-module lookup. It already covers
  every configured factory before construction; it intentionally does not
  execute imports or infer undeclared optional drivers.

All new failure findings include a command to diagnose or repair them.
Known unregistered plugin component names receive an install hint; unknown
providers are not guessed.

## Tests

Add hardware-free unit tests for each probe and an aggregate CLI regression
showing findings appear before a constructor failure. Cover markers,
pre-releases, absent metadata/parser, display/platform cases, CAN flags and
unreadable inventory, path remediation, URL defaults/overrides, malformed
URLs and connection failures. Preserve full line and branch coverage,
strict mypy, lint and formatting gates.

## Docs

Update `docs/guide/cli.md`; add `changelog.d/50.added.1.md`, preserving the
fragment from #568. No public API or snapshot change is required.
