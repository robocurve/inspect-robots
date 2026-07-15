# 0014 — Agent plugin: speed-limited playout replaces `duration_s`

Issue: robocurve/inspect-robots#92. Scope: `plugins/inspect-robots-agent` only;
core is untouched.

## 1. Problem

The agent toolset (plan 0008 §4c) makes the LLM pick `duration_s` for every
motion. Speed is not a quantity the LLM should reason about: the loop is
open-loop per turn (the model re-observes after the chunk plays out) and the
budget it manages is per LLM call, not per second. (Trials also carry a
control-step horizon — `Task.max_steps` — and speed-limited playout consumes
more of it per motion than a short-duration call did; that is the accepted
cost of never truncating, and the step-cap error in §3c bounds it per call.) Worse, a too-short duration does
not fail — the core guardrails clamp the chunk per step (`DeltaLimitApprover`
in absolute modes; `ClampApprover`'s box clamp in displacement modes, where
the delta limiter adds nothing by default), so the arm moves at the limit and
stops short of the requested motion, while the tool result note still claims
the full motion executed. The LLM discovers the divergence only from the next
observation.

## 2. Decision

Drop `duration_s` from both motion tools. The toolset computes the playout
time from a speed cap, so the full requested displacement is emitted;
large motions simply play out over more steps. Guardrails below the plugin
are unchanged (the plugin still contains no safety-critical code path); at the
default speed every emitted per-step change stays strictly within the core
backstop's 5%-of-range default, so at default settings the backstop does not
clamp what the plugin emits.

Honest scoping of that guarantee: `DeltaLimitApprover` measures against the
last *approved command* held in the trial store, while `move_joints`
interpolates from the *observed* state. When the two diverge (hardware
tracking error, a prior clamp, an operator nudge), the first steps of a new
chunk can still exceed `max_delta` relative to the stored reference and get
clamped. That is the backstop doing its job; the plugin's claim is only that
it never *provokes* clamping through its own step sizing.

Resolved knobs (user decisions, 2026-07-14):

- Plugin-only; no core controller middleware.
- `duration_s` is removed, not made optional.
- Default speed: `max_speed_frac = 0.5` — half the joint's range per second
  for absolute modes, which matches the core backstop's implied speed (5% of
  range per step at 10 Hz). `max_speed_frac` applies to absolute modes only;
  see §3b for why `move_by` has no frac.

## 3. Toolset changes (`_tools.py`)

### 3a. `move_joints` (absolute modes: `joint_pos`, `eef_abs_pose`)

- Schema: `targets` only; `required: ["targets"]`. Description explains the
  motion is interpolated at a fixed safe speed and reports its step count.
- Per-dimension speed: `speed = max_speed_frac * (high - low)` per second
  (float64 vector). Zero-width dimensions (`high == low`, common padding in
  VLA action spaces) have `speed == 0` and are legal at bind time.
- Per call, for each *named* dimension: if `speed_i == 0` and
  `target_i != low_i`, return a structured error
  (`"dimension {label} is fixed at {value}"`). The comparison is against the
  *bound*, not the observed state — on hardware the observation carries
  noise, and re-sending the documented fixed value must always succeed.
  Zero-width dimensions contribute nothing to the step count; dimensions
  with zero distance to the observed state likewise contribute nothing.
- Steps: `steps = max(1, ceil(max_i(|target_i - current_i| / speed_i) * hz
  / (1 - 1e-6)))`, the max taken over named dimensions with positive
  distance and positive speed; an empty max (all named targets equal the
  current state) yields `steps = 1`. The `1e-6` relative headroom exists
  because `linspace` interpolation accumulates ~1 ulp of float error per
  step: without it, a step count that divides the distance exactly (e.g.
  range 2.0, distance 1.5, 15 steps, limit 0.1) emits consecutive deltas of
  `0.1 + 1 ulp` and the backstop clamps spuriously at the default boundary.
  `hz` falls back to `_FALLBACK_HZ = 10.0` as today.
- Interpolation is unchanged (`linspace` fractions from current observed
  state).

### 3b. `move_by` (displacement modes: `eef_delta_pos`, `eef_delta_pose`, `joint_delta`)

- Schema: `deltas` only; `required: ["deltas"]`.
- The plugin treats displacement boxes as per-action sized — a plugin-level
  convention (core explicitly refuses to assume this, which is exactly why
  the delta limiter derives no default there). Under that convention, and
  because `ClampApprover` clamps each step to the box, the box side is the
  per-step limit and splitting to fit it is the only non-lossy choice.
- `max_speed_frac` deliberately does not apply here: the box *is* the
  embodiment author's per-step speed statement, and scaling it by a fraction
  of itself has no principled unit. The README documents this asymmetry; a
  test pins that `move_by` ignores the knob.
- Per-step limit per dimension: `high_i` for a positive component, `|low_i|`
  for a negative one.
- Per call, for each named dimension with a nonzero value: if the limit on
  the requested side is zero, return a structured error
  (`"dimension {label} cannot move in that direction"`).
- Steps: `steps = max(1, ceil(max_i(|delta_i| / limit_i) / (1 - 1e-6)))`
  over named dimensions with nonzero values; all-zero deltas (legal today, a
  hold) get `steps = 1`, emitting a single zero action. Each action is
  `vector / steps` (unchanged emission shape). The same `1e-6` headroom as
  §3a applies: `ceil` alone does not bound the per-step magnitude in float
  arithmetic — for some `(delta, limit)` pairs `delta / limit` rounds down
  to an exact integer `n` while `delta / n` rounds 1 ulp *above* the limit,
  and `ClampApprover` would clamp and log a spurious approval event.

### 3c. Chunk cap

`_MAX_DURATION_S = 10.0` stays, reinterpreted: `max_steps = ceil(10 * hz)`
(with `hz` the same fallback-resolved rate used for step computation). A
computed `steps > max_steps` returns a structured `ToolResult(error=...)`
telling the LLM the request exceeds the playout cap and to split the move
into smaller motions — advice that always works, because a smaller distance
or delta strictly reduces the computed steps. At the default speed a
full-range `move_joints` needs 2 s, so in practice only large `move_by`
totals hit this.

### 3d. Bind-time bound requirements (never guess)

`build_toolset` gains the checks; failures raise `ToolsetError` at bind time
with actionable messages (same stance as `DeltaLimitApprover`):

- Absolute modes: every dimension needs finite `low` and `high` (speed is
  derived from the range). `high == low` is allowed (handled per call, §3a).
- Displacement modes: every dimension needs finite `low` and `high` with
  `low <= 0 <= high` (core's `Box` only validates `low <= high`, so a
  displacement box not containing zero would make the §3b side-limits
  negative and the split nonsensical; such spaces also cannot hold still and
  are rejected). A zero bound on one side is allowed (that direction simply
  cannot move; handled per call, §3b).
- Both modes: a declared `control_hz` that is non-finite or `<= 0` is
  rejected at bind time (`None` stays allowed and falls back to
  `_FALLBACK_HZ` for step computation). Core never validates `control_hz`,
  and `hz = 0` would make §3a floor every motion to a single step while §3c
  errors every call — contradictory outcomes this check makes unreachable.
- The "no bounds; move conservatively" `bounds_text` fallback becomes dead
  in both modes (finite bounds are now required everywhere) and is removed
  outright.

Previously such spaces "worked" only because the LLM supplied a duration;
motions in them were never actually speed-safe.

### 3e. Toolset construction

`Toolset.__init__` / `build_toolset` grow `max_speed_frac: float` (default
`0.5`) and precompute the per-dimension speed (absolute) or per-side
per-step limits (displacement). `bounds_text` keeps documenting the box to
the LLM.

## 4. Policy changes (`policy.py`)

- `LLMAgentPolicy.__init__` and `agent_policy` accept
  `max_speed_frac: float = 0.5`; validation: finite and `> 0`
  (`ValueError` otherwise).
- `AgentPolicyConfig` gains `max_speed_frac: float = 0.5` — recorded in
  `EvalSpec.policy_config` for free via `dataclasses.asdict`.
- `bind()` forwards it to `build_toolset`.
- `_SYSTEM_TEMPLATE`: no duration mention today, so only the tool
  descriptions change; keep "small, deliberate motions" guidance.
- CLI passthrough needs no change (`-P max_speed_frac=0.25` already coerces
  via the existing `-P` machinery).

Truncation caveats, documented in the README rather than prevented (the
plugin cannot know the CLI's `--max-action-delta`):

- Absolute modes: `max_speed_frac / hz > 0.05` (e.g. frac above 0.5 at
  10 Hz) means the default backstop clamps again and motions truncate.
- Displacement modes: an explicit `--max-action-delta` tighter than the box
  reintroduces truncation for `move_by` (the delta limiter intersects the
  box with `±max_delta`).

## 5. Reporting

Success note: `executing move_joints over {steps} steps ({steps/hz:.1f}s)`
when the embodiment declares `control_hz`; when `control_hz` is `None` the
note reports only the step count — the chunk plays at the embodiment's
native rate, which the plugin cannot know, so any seconds figure would be a
guess. For the same reason, when `control_hz` is `None` the speed cap and
the §3c playout cap are step-count constructs (computed at the 10 Hz
fallback), not wall-clock guarantees; the README states that the wall-clock
speed cap is only meaningful for declared-rate embodiments. Moves floored at
`steps = 1` run below the cap, not at it — the note makes no "at the speed
cap" claim.

## 6. Docs

- Plugin README: rewrite "How it works" — tool calls state *where* to go;
  the plugin times the motion at a fixed safe speed; `duration_s` is gone;
  document `max_speed_frac` in the knobs list (absolute modes only, §3b),
  both truncation caveats (§4), and the `control_hz=None` scoping (§5).
- Docstrings updated (`_tools.py` module docstring describes the timing
  rule).

## 7. Tests (TDD; plugin's own pytest scope, not the core 100% gate)

`tests/test_tools_motion.py`:

- `move_joints` computes steps from distance and range (range 2.0, frac 0.5
  → speed 1.0/s; distance 1.5 at 10 Hz → 16 steps with the §3a headroom)
  and reaches the exact target in the final action.
- The headroom guarantee itself: for the boundary case above, every
  consecutive per-step delta of the emitted chunk is `<=` the backstop's
  `0.05 * (high - low)` — the spurious-clamp regression test.
- Small move → `steps == 1` floor; all named targets equal to current →
  `steps == 1`.
- Zero-width dimension: targeting the bound value succeeds and contributes
  no steps *even when the observed state differs from the bound* (sensor
  noise); targeting any other value returns the "fixed at" error.
- `move_by` splits by box side: `high = 0.05`, delta `+0.2` → 5 steps of
  0.04 (the §3b headroom pushes the float-exact 4-step split to 5);
  asymmetric `low = -0.1` with delta `-0.2` → 3 steps.
- `move_by` ulp regression: every emitted per-step magnitude is `<=` the box
  side for a non-float-clean pair (e.g. delta `0.29`, limit `0.11287...`-
  style values from a property-style spot check).
- `move_by` with all-zero deltas → 1 zero-action step (hold).
- `move_by` into a zero-bound direction → the "cannot move in that
  direction" error.
- `move_by` ignores `max_speed_frac` (same split at frac 0.5 and 0.1).
- Over-cap request errors with the split-the-move message; boundary exactly
  at `max_steps` succeeds.
- Schemas no longer advertise `duration_s`; a call carrying a stray
  `duration_s` key still succeeds (extra keys were and remain ignored).
- `control_hz=None`: steps computed at the 10 Hz fallback, chunk emitted
  with `control_hz=None`, note contains no seconds figure (replaces the
  existing `test_control_hz_none_falls_back_to_default`, which passes
  `duration_s` and dies with it).
- Declared-rate note: with `control_hz=20`, the note's seconds figure is
  `steps / 20` (pins the units; a `steps * hz` bug must fail).
- Bind-time `ToolsetError` for missing/non-finite bounds per §3d (absolute
  and displacement variants); zero-width and zero-sided bounds bind fine;
  displacement boxes not containing zero are rejected; declared
  `control_hz` of `0`, negative, or non-finite is rejected while `None`
  binds.
- `max_speed_frac` validation errors (0, negative, non-finite).
- Speed derivation honors non-default frac (absolute modes).

`tests/test_policy_e2e.py`: existing flows updated (tool calls no longer
send `duration_s`); config snapshot includes `max_speed_frac`. One test is
redesigned, not updated: `test_wild_swing_is_clamped_by_guardrails_but_not_without`
provokes a clamp by sending `move_by` `dx=5.0`, but under the new surface
the toolset splits that into in-box steps — no default guardrail can bite,
by design, and the split would blow the task's 40-step horizon anyway. It
becomes a test of the §4 caveat instead: build the guardrail chain with an
explicit `max_delta` *tighter than the box*, request a `move_by` sized so
its computed steps fit the horizon, and assert `delta_clamped` approval
events occur with guardrails and the emitted per-step values pass through
untouched without them.

`tests/test_llm.py`, `tests/test_package.py`: untouched unless imports move.

## 8. Versioning / release

Bump plugin `version` in `plugins/inspect-robots-agent/pyproject.toml`
`0.1.0 → 0.2.0` (tool-surface break). Publishes with the next core release
via the existing `publish-agent` job in `release.yml`.

## 9. Out of scope

- A core rate-limiting controller middleware (revisit if a second policy
  plugin needs re-timing).
- Changing core guardrail defaults or `--max-action-delta` semantics.
- Per-dimension `max_speed_frac`.
