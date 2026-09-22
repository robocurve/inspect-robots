# inspect-robots-jev — agent guide

Policy `jev`: TypeSafe's Jev (text-only, picks one option from a menu) drives
the YAM arms from AprilTag world state. Spec and task list:
`plans/0076-jev-decision-policy.md` at the repo root. Read §0 (terms) and the
"Global constraints" of the implementation plan before editing.

## Modules

| File | Responsibility |
|---|---|
| `world.py` | `Vec3`, `ObjectView`, `GripperView`, `WorldState`, direction words, `describe_offset` |
| `_decisions.py` | `DecisionsClient` for OpenRouter alpha / TypeSafe direct; injectable `http_post` |
| `menu.py` | `Move`, `build_menu`, `parse_option` (option-id grammar) |
| `phase.py` | `TaskConfig`, `Phase`, `PhaseMachine` (cube-into-bowl incl. `reopen`, `verify`, contact rule) |
| `serializer.py` | `build_state`, `instructions_for` — curated state text in directional words |
| `motion.py` | `MotionMapper`: menu pick → bounded absolute Cartesian `ActionChunk` |
| `calibration.py` | camera→arm transforms from a touched table tag; JSON load/save |
| `perceiver.py` | AprilTag detections → `WorldState` (bundle fusion, depth refinement, occlusion memory) |
| `policy.py` | `JevPolicy`, `jev_policy` registry entry |
| `scorer.py` | `cube_in_bowl` pose-based scorer (reads the last action's `meta["jev"]`, requires the cube re-seen after release) |
| `calibrate.py` | `python -m inspect_robots_jev.calibrate`: corners JSON + `.npy` frame/intrinsics → `calibration.json` |

## Invariants

- Wording is load-bearing: directional words, 0.5 cm rounding, and per-option
  hints are pinned by tests. Rerun `examples/jev_textsim.py` (live API) before
  changing them.
- Code curates what Jev sees: only the current target's geometry, other
  objects by name. Code owns phases and completion; Jev only picks a move.
- One Choice question per request. Confidence is logged, never gated.
- `JevPolicy.settings` holds the plugin config; `PolicyBase.config` is the
  core `PolicyConfig` and must not be shadowed.
- Chunks interpolate from the last *commanded* target, not the measured pose,
  so IK lag never shows as a spurious delta clamp.
- `max_steps` counts control ticks (one pick = 1–4 ticks); a nominal
  cube-into-bowl run is ~150 ticks, budgets use 600.
- Every I/O edge (HTTP, tag detector) is injectable; the whole policy runs in
  tests with no network, camera, or robot.
- Gates: ruff, ruff format, mypy strict (src + tests), pytest 100 % coverage.
