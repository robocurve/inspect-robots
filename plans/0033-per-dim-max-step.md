# 0033 — Per-dimension `max_step`: embodiment-declared speed semantics

## Problem

The agent plugin paces every action dimension at `min(max_speed_frac / hz, 0.05) ×
(high − low)` per control step, and `DeltaLimitApprover` derives its default
per-step limit as `0.05 × (high − low)`. Both treat "fraction of declared range"
as a proxy for safe speed. That heuristic is right for revolute joints (range
2π rad → ~0.6 rad/s at defaults) and badly wrong for a normalized 0–1 gripper
stroke, where the full range is a routine single motion: a full open↔close under
agent control is paced at ~100 steps — 10 s at 10 Hz — while the hardware does
it in well under a second. In fact at 10 Hz defaults a lone full stroke
*exceeds* the 10 s playout cap (`headed_ratio ≈ 100.0001 > _max_steps = 100`),
so the agent gets a "split the move into smaller motions" error and must burn
extra LLM calls to close its own gripper; anything short of the full stroke
crawls. Because interpolation step count is the max over named
dimensions, a combined joints+gripper `move_joints` call drags the *joints* down
to the gripper's crawl too.

Neither layer can fix this alone: the toolset deliberately stays under the
approver's derived ceiling (`native_backstop`), so raising toolset pacing without
teaching the approver would just get clamped (or worse, silently halved).

## Design

One new optional declaration, honored by every layer that derives per-step
limits from the range. Motion knowledge continues to flow from the embodiment
via declarations; the agent plugin stays embodiment-agnostic.

**Absolute modes only.** `max_step` is meaningful only for absolute-target
control (`joint_pos`, `eef_abs_pose`). In displacement/rate modes the box
*already is* the embodiment-authored per-step limit; a second channel for the
same fact would create a toolset/approver asymmetry (the toolset splits deltas
by the box side) and is rejected at construction.

### 1. Core: `ActionSemantics.max_step`

```python
max_step: tuple[float | None, ...] | None = None
```

Per-dimension change limit, in the action space's native units **per control
step**, that the embodiment author considers safe and normal. `None` (the
default) or a `None` entry means "underived — use existing range-based
defaults". Declared entries must be finite and > 0.

Validation is split by what each type can see:

- `ActionSemantics.__post_init__` (new): each entry `None` or finite-positive;
  any non-`None` entry on a non-absolute `control_mode` raises `ValueError`
  ("displacement boxes already declare per-step limits via low/high"). This
  catches invalid declarations at their construction site — the yam builders
  construct semantics detached from any box. The absolute-modes set becomes a
  canonical `ABSOLUTE_CONTROL_MODES` frozenset in `spaces.py` (it cannot live
  in `approver.py`, which imports `spaces`); `approver.py`, `conformance.py`,
  and the agent plugin's `_tools.py` import it instead of keeping their three
  private copies.
- `Box.__post_init__`: length must equal `box.dim` (same as `dim_labels`), and
  a declared entry on a dimension whose bounds pin it (`low == high`) raises
  `ValueError` (contradictory declaration), so every consumer — not just the
  agent policy at bind time — rejects it.

`ActionSemantics` is `frozen` without `eq=False`; a tuple field keeps it
hashable/comparable exactly like `dim_labels` already does. Compatibility
checking (`compat.py`) compares semantics field-by-field (`control_mode`,
`rotation_repr`, `gripper`, `frame`) and must **not** compare `max_step`:
the embodiment declares motion limits; a policy-side semantics never needs to
(see §4). Add a comment there stating this is deliberate.

### 2. Core: `DeltaLimitApprover`

Only default derivation changes; the explicit `max_delta` argument still wins
outright — this matches today's contract where `--max-action-delta` replaces
the derived 5% default entirely, and is a deliberate operator override. Add a
both-present test locking that in.

Absolute modes, no explicit `max_delta`: per-dim default = declared `max_step`
where present, else `0.05 × (high − low)` as today. The finite-bounds
requirement ("deriving a default needs finite low/high") now applies only to
*undeclared* dims: a space whose every dim declares `max_step` needs no finite
bounds for derivation. Arithmetic for undeclared dims stays in the box's native
dtype (unchanged); declared entries are substituted per-dim in float64.

Scope note: this relaxation is approver-only. `conformance.py` and both
plugins' toolset/motion builders still require finite bounds for their own
reasons and are not relaxed to match; only the standalone/CLI-guardrails
approver path gains the capability.

Displacement modes: unchanged (declarations cannot exist there, per §1).

### 3. Agent plugin: toolset pacing (`_tools.py`)

In `build_toolset`, for absolute modes:

- **Undeclared dims**: unchanged — `min(max_speed_frac / hz, 0.05) × range`,
  capped by `native_backstop`.
- **Declared dims**: pacing = `max_step × min(max_speed_frac / _DEFAULT_SPEED_FRAC, 1.0)`
  where `_DEFAULT_SPEED_FRAC = 0.1` (the policy's default `max_speed_frac`).
  Rationale: at the default speed setting a declared dim moves exactly at its
  declared limit; turning the global knob *down* slows declared dims
  proportionally (an operator slowing the robot expects everything slower);
  turning it *up* speeds only range-paced dims — declared dims are already at
  the embodiment's stated ceiling and must not exceed the approver's default,
  which equals the declared value. The existing `_RELATIVE_HEADROOM` keeps
  emitted steps strictly below the limit, so the approver never clamps.
- **Float-spacing guard**: the coarseness check compares grid spacing against
  the per-dim *effective* budget (declared or backstop), not `native_backstop`
  alone, preserving the "no silent truncation" invariant for declared dims.
- **Fixed-dim detection and underflow guards**: declared-dim pacing is folded
  into `step_limits` **before** the existing movable-zero-limit underflow
  guard, which stays as-is and now also covers a declared pacing product that
  underflows to exactly 0.0 (e.g. a subnormal `max_speed_frac` times a tiny
  declared limit) — a zero entry would otherwise misreport a movable dim as
  fixed. `Box` rejecting declarations on pinned (`low == high`) dims means no
  *separate* contradiction check is needed, but the underflow guard must see
  the final per-dim limits.

Step-count cap (`_MAX_DURATION_S`), tool schemas, and result notes are
unchanged.

### 3b. Capx plugin: `MotionQueue` ceiling (`_motion.py`)

Capx reproduces the approver's `0.05 × (high − low)` native-dtype arithmetic so
its emitted chunks are never rewritten by the CLI's default delta backstop.
That mirror must track the new default or its invariant breaks in both
directions (a tighter declaration would clamp capx chunks; a looser one — the
gripper case — would keep capx at the crawl). Capx gets the same two-part
treatment as the agent toolset, since its pacing has the same shape
(`step_limits = min(step_frac × range, ceiling)`, `max_speed_frac` default 0.1
in its policy):

- **Ceiling**: per-dim, declared `max_step` where present, else the existing
  native-dtype 5% backstop (mirrors §2 exactly, so chunks are never rewritten
  by the default approver in either direction).
- **Pacing**: declared dims pace at
  `max_step × min(max_speed_frac / _DEFAULT_SPEED_FRAC, 1.0)` per §3 —
  ceiling-only would leave the capx gripper at the crawl (0.01/step from
  `step_frac × range` regardless of ceiling). Undeclared dims unchanged.
- **Float-spacing guard**: capx's copy of the coarseness guard also checks
  against the per-dim effective budget, keeping the two mirrors symmetric.
- Same guard-ordering rule as §3: pacing folds into the per-dim limits before
  capx's movable-zero-limit underflow check.

Test: `MotionQueue` chunks on a declaring space pass the default approver
unmodified AND traverse a full 0–1 declared dim in the declared number of
steps; undeclaring spaces byte-identical.

### 4. Yam plugin (separate repo/PR): declare the gripper limit

- New `YamConfig` field: `gripper_stroke_s: float = 1.0` — seconds for a full
  0→1 stroke under agent pacing. Validation: finite, > 0.
- Derived per-step limit: `min(1.0, 1.0 / (gripper_stroke_s × hz))` where
  `hz = control_hz if control_hz > 0 else 10.0` — the same fallback the
  embodiment already applies to `control_hz` (which `YamConfig` deliberately
  does not validate). The derivation guards non-finite inputs and products
  (`control_hz = inf`, or an overflowing `gripper_stroke_s × hz`) by declaring
  **no** gripper limit in that case rather than crashing at semantics
  construction — a config whose `embodiment.info` constructs today must keep
  constructing. At defaults this is 0.1/step → 11 interpolation
  steps for a full stroke (headroom ceil, see the table note) → ~1 s. The
  `min` caps at full stroke in one step for absurdly small stroke times.
- The semantics builders (`action_semantics()` and `action_box()`, which the
  embodiment actually calls in `_action_space`) are shared by the embodiment
  **and** the `/act` policy client, whose `ActServerConfig` has neither
  `control_hz` nor the new field. Both builders gain an optional
  `gripper_max_step: float | None = None` parameter, plumbed from `action_box`
  into `action_semantics`: the embodiment passes the derived value
  unconditionally; `action_semantics` applies it only when building an
  **absolute**-mode semantics and ignores it for `joint_delta` (so a
  delta-configured rig cannot trip the new construction-time validation). The
  policy client passes nothing and declares no limits. The two sides' semantics then differ in `max_step` — safe because
  `compat.py` compares field-by-field and deliberately excludes `max_step`
  (§1); soften the builder docstring's "one function guarantees a clean check"
  claim accordingly.
- Declarations go on the two **absolute** interfaces only: `joint_pos`
  (gripper indices 6 and 13 of the 14-D packing: `left_gripper`,
  `right_gripper`) and `eef_abs_pose` (gripper indices 4 and 9 of
  `EEF_DIM_LABELS`), `None` everywhere else. The `joint_delta` interface
  cannot and does not declare (§1).
- `pyproject.toml`: bump `inspect-robots` minimum to the release carrying
  `max_step`.

### Resulting behavior on a stock YAM rig (10 Hz, defaults)

| Motion | Before (agent toolset) | After |
|---|---|---|
| Full gripper close (alone) | playout-cap error; forced split across ≥2 calls, ~10 s total | 11 steps, ~1 s |
| 90% gripper close | 91 steps, ~9 s | 10 steps, ~1 s |
| 1 rad joint reach (alone) | ~16 steps, 1.6 s | unchanged |
| Joints + full gripper close together | playout-cap error | max(joint steps, 11) |

(Counts include the `_RELATIVE_HEADROOM` ceil: a stroke at exactly N× the
per-step limit rounds up one step so every emitted step stays strictly below
the limit — tests assert 11, not 10, for the full stroke. The toolset's
"before" full stroke is a cap error, not a 101-step chunk; the 101-step figure
applies only to capx's `MotionQueue`, which has no duration cap.)

Joints keep byte-identical pacing and approver limits.

## Not doing (YAGNI)

- No displacement-mode declarations (rejected at construction, see Design).
- No per-dim `max_speed_frac` config plumbing in the policy — the declaration
  makes per-rig tuning unnecessary, and `gripper_stroke_s` covers the yam knob.
- No unit-string-driven inference (e.g. special-casing `"normalized"`): units
  describe state, not sanctioned speeds, and inference re-introduces hidden
  motion knowledge into the plugin.
- No changes to `ClampApprover`, collision guardrails, or the CLI flag surface
  (`--max-action-delta` still overrides everything as an explicit scalar).

## Tests (core, 100% coverage per CONTRIBUTING)

- `spaces`: valid declarations round-trip; wrong length, non-positive,
  non-finite entries raise; `None` entries allowed; non-absolute mode with any
  declared entry raises at `ActionSemantics` construction; declaration on a
  pinned (`low == high`) dim raises at `Box` construction; semantics equality/
  hash still work with the tuple field.
- `approver` absolute: mixed declared/underived defaults per-dim; all-declared
  space with non-finite bounds constructs; undeclared dim with non-finite
  bounds still raises; explicit `max_delta` overrides declarations when both
  are present; first-action passthrough and store semantics unchanged.
- `toolset`: declared gripper dim paces at declared limit (11 steps for a full
  stroke at defaults, per the headroom note above); undeclared dims unchanged
  for the same space;
  `max_speed_frac` below default slows declared dims proportionally, above
  default does not raise them; combined move's step count is the per-dim max;
  spacing guard uses the effective per-dim budget; emitted chunks pass a
  default `DeltaLimitApprover` for the same space unmodified (the ceiling
  invariant, as an explicit test).
- `capx`: `MotionQueue` chunks on a declaring space pass the default approver
  unmodified and traverse a declared dim at the declared pace; existing
  behavior on undeclaring spaces byte-identical.
- `compat`: embodiment-declares / policy-doesn't still passes compatibility.
- Yam repo: config validation, derived limit arithmetic (incl. the 1-step cap),
  declaration present on absolute interfaces only and only embodiment-side,
  builder parameter default declares nothing.

## Rollout

1. Core PR (this repo): spaces + approver + agent toolset + capx motion queue
   + tests + CHANGELOG; agent and capx plugin version bumps (follow the repo's
   existing plugin-bump convention from prior PRs). Both plugins' stale
   `inspect-robots>=0.4` dependency floors bump to the core version carrying
   `max_step` — a new plugin wheel importing `ABSOLUTE_CONTROL_MODES` /
   reading `max_step` against an older core would fail at bind time in
   exactly the partial-venv-upgrade scenario rigs hit.
2. Release core + agent (+ capx) to PyPI.
3. Yam PR: config field + builder parameter + declaration + dep bump + tests.
4. Rig enablement (venv upgrade) tracked separately, as with prior features.
