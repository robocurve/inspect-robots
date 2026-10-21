# 0086: `doctor` preflights configured components before embodiment construction

Addresses the registry-resolution and aggregate-reporting parts of
item 1, the policy runtime-requirement part of item 2, and the
aggregation behavior from item 7. It builds on the runtime-requirement
preflight from plan 0010 and the device-slot validation from plan 0082
(#438).

## Problem

`doctor` checked only the configured embodiment. A config naming a
policy, sim embodiment, scorer, or grader that was not registered
could pass `doctor` and fail at runtime instead.

## Scope

Before constructing the embodiment, `doctor` now:

- checks each configured `policy`, `sim_embodiment`, `scorer`, and
  `grader`; unset kinds are skipped and `grader = none` is treated as
  unconfigured;
- looks each name up without constructing the component;
  `sim_embodiment` resolves through the embodiment registry;
- reports every unresolved name together with the registered
  alternatives;
- runs the plan 0010 `RUNTIME_REQUIREMENTS` check on each resolved
  factory.

The existing embodiment checks still run after this preflight. A
completed `doctor` run returns 1 when any preflight or embodiment
finding is present.

## Out of scope

- Identifying the plugin package or install command for an unresolved
  name. This leaves the remaining part of item 1 open.
- Version-skew detection (item 3), the CAN interface UP check (the rest
  of item 5), and the policy-server reachability probe (item 6).
- Changes to `setup`, `run`, or the registry.

## Design

`_cmd_doctor` runs the configured-component preflight before the existing
embodiment block. `_doctor_configured_components` supplies the configured
non-embodiment components, while `registered(kind)` performs the lookup
and `missing_runtime_requirements` performs the existing runtime check.

## Tests

`tests/test_registry_cli.py` covers the behavior introduced by this
change.

## Docs

- `docs/guide/cli.md` and `docs/guide/adapters.md`: document configured
  component resolution and runtime-requirement preflight.
- `doctor` argparse help and `_cmd_doctor` docstring.
- Module map (`src/inspect_robots/CLAUDE.md`): update the `cli.py` row.
- `changelog.d/50.added.md`.
