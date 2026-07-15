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
are unchanged (the plugin still contains no safety-critical code path); the
§3a per-step ceiling keeps every emitted per-step change strictly within the
core backstop's 5%-of-range default — at any `max_speed_frac` and any
declared control rate — so the default backstop never clamps what the plugin
emits.

Honest scoping of that guarantee: `DeltaLimitApprover` measures against the
last *approved command* held in the trial store, while `move_joints`
interpolates from the *observed* state. When the two diverge (hardware
tracking error, a prior clamp, an operator nudge), the first steps of a new
chunk can still exceed `max_delta` relative to the stored reference and get
clamped. Similarly, if proprioception reads slightly *outside* the box, the
early actions interpolated from that state are out-of-box and
`ClampApprover` trims them. Both are the guardrails doing their job; the
plugin's claim is only that it never provokes clamping through its own step
sizing from in-box state.

Resolved knobs (user decisions, 2026-07-14):

- Plugin-only; no core controller middleware.
- `duration_s` is removed, not made optional.
- Default speed: `max_speed_frac = 0.5` — half the joint's range per second
  for absolute modes, which matches the core backstop's implied speed (5% of
  range per step at 10 Hz); on slower embodiments the §3a per-step ceiling
  wins and the motion is proportionally slower. `max_speed_frac` applies to
  absolute modes only; see §3b for why `move_by` has no frac.

## 3. Toolset changes (`_tools.py`)

### 3a. `move_joints` (absolute modes: `joint_pos`, `eef_abs_pose`)

- Schema: `targets` only; `required: ["targets"]`. Description explains the
  motion is interpolated at a fixed safe speed and reports its step count.
- Per-dimension *per-step* limit:
  `step_frac = min(max_speed_frac / hz, _BACKSTOP_STEP_FRAC)` with
  `_BACKSTOP_STEP_FRAC = 0.05`, giving `step_frac * (high - low)` — then
  elementwise-min'd with `0.05 * (high - low)` computed in the box's
  *native dtype*: `DeltaLimitApprover` derives its default without
  promoting, so a low-precision box (e.g. float16 bounds) rounds the
  backstop *below* the float64 value and the plugin must not outrun it. The wall-clock speed is
  `max_speed_frac * range` per second whenever `hz >= max_speed_frac / 0.05`
  (i.e. ≥ 10 Hz at the default frac); on slower embodiments the per-step cap
  wins and the motion plays proportionally slower rather than provoking the
  backstop. `_BACKSTOP_STEP_FRAC` deliberately mirrors
  `DeltaLimitApprover`'s derived default (5% of range per step) — that is
  the constant being matched, and the guarantee in §2 is per-step, because
  the backstop is per-step. Zero-width dimensions (`high == low`, common
  padding in VLA action spaces) have a zero limit and are legal at bind
  time.
- Per call, for each *named* dimension, validated in this order (zero-width
  first — a zero-width dimension's off-value targets are also out of its
  degenerate `[v, v]` box, so the bounds check would otherwise shadow the
  more instructive error):
  - Zero-width dimensions (`limit == 0`): `target_i != low_i` returns
    `"dimension {label} is fixed at {value}"`. The comparison is against
    the *bound*, not the observed state — on hardware the observation
    carries noise, and re-sending the documented fixed value must always
    succeed. `{value}` is rendered with `repr()` (JSON round-trip exact),
    because the schema's `bounds_text` renders at 4 significant figures
    (`.4g`) and may not match on the first attempt; the error then teaches
    the exact value.
  - Otherwise, `target_i` outside `[low_i, high_i]` returns a structured
    error (`"target for {label} is outside [{low}, {high}]"`). Letting it
    through would recreate §1's silent divergence: the sizing satisfies the
    delta limiter while `ClampApprover` pins every step at the bound and
    the note claims the full motion executed. Bounds are guaranteed finite
    by §3d, so the check is always meaningful.
  - Zero-width dimensions contribute nothing to the step count; dimensions
    with zero distance to the observed state likewise contribute nothing.
- Non-finite observed state: if the proprioceptive reference vector contains
  a non-finite value, `execute` raises `ValueError` (the rollout wraps it as
  `PolicyError` and the trial errors). This check runs before *any* tool
  argument validation (labels, value types, per-dimension checks): a broken
  sensor must end the trial even when the tool call happens to contain its
  own correctable mistake. A broken sensor is not an
  LLM-correctable condition, so it does not use the structured-error
  channel; without this guard the steps formula would poison `ceil`/`max`
  with NaN and crash uncontrolled.
- Steps: `steps = max(1, ceil(max_i(|target_i - current_i| /
  (step_frac * range_i)) / (1 - 1e-6)))`, the max taken over named
  dimensions with positive distance and positive range; an empty max (all
  named targets equal the current state) yields `steps = 1`. The `1e-6` relative headroom exists
  because `linspace` interpolation accumulates ~1 ulp of float error per
  step: without it, a step count that divides the distance exactly (e.g.
  range 2.0, distance 1.5, 15 steps, limit 0.1) emits consecutive deltas of
  `0.1 + 1 ulp` and the backstop clamps spuriously at the default boundary.
  `hz` falls back to `_FALLBACK_HZ = 10.0` as today.
- Interpolation: `linspace` fractions from the current observed state as
  today, with two float repairs. The final action is *snapped* to the exact
  target (`current + (target - current) * 1.0` is not `target` in float
  arithmetic — e.g. `-0.1 + 0.4 = 0.30000000000000004`), and every emitted
  action is then clipped into the box: interpolants can overshoot a bound
  by 1–2 ulp from fully in-box inputs, and `ClampApprover` would flag a
  spurious approval event. Clipping is a per-dimension projection of a
  monotone sequence, so it cannot enlarge consecutive step deltas — the
  §3a headroom guarantee survives it.

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
into smaller motions — advice that always works, because the computed steps
shrink toward 1 as the motion shrinks. The distance/limit ratios are
computed with NumPy float64 (inf-tolerant: a huge finite `move_by` delta
like `1e308` overflows the division to `inf`, never raising) and the cap
comparison runs *before* `ceil` — `math.ceil(inf)` raises `OverflowError`,
which would escape as a trial-killing `PolicyError` for what is an
LLM-correctable mistake, violating the module's errors-not-exceptions
contract. The pre-ceil cap check applies to *both* tools: `move_joints`'
ratio numerator is `|target - current|` with `current` from the
*observation*, which is only checked for finiteness — a grossly-off finite
reading can push the ratio past the cap or to `inf`, and must land in the
structured over-cap error, not an exception. At the default speed a
full-range `move_joints` needs 21 steps (2.1 s at 10 Hz — the §3a headroom
bumps the exact 20-split), so in practice only large `move_by`
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
  and `hz = 0` would make §3c's `max_steps = 0`, turning every call into an
  error — a nonsense configuration this check rejects with a clear message
  instead. A positive-but-tiny declared rate (e.g. 0.1 Hz → `max_steps = 1`)
  binds fine and makes most motions hit the §3c error; that is a truthful
  reflection of an embodiment that genuinely cannot move far in 10 s, not a
  configuration the plugin second-guesses.
- Derived-quantity guards, all `ToolsetError` at bind time: a declared
  `control_hz` whose `10 * hz` playout cap overflows to `inf`; a
  `max_speed_frac` whose `frac / hz` underflows to exactly zero, or whose
  product with any movable dimension's range underflows the derived
  per-step limit to zero (either would misreport dimensions as fixed); a `high - low` range that overflows
  to `inf` despite finite endpoints — checked in *both* float64 and the
  box's native dtype, since `DeltaLimitApprover` subtracts without
  promoting (float32 `[-3e38, 3e38]` overflows only natively); non-1-D
  action-space shapes (the toolset's indexing and bounds text are
  vector-only); and, in absolute modes, bounds whose float spacing at
  their magnitude exceeds `5e-7 *` the native backstop (offset boxes like
  `[1e16, 1e16 + 2]` or ranges whose 5% underflows to zero) — interpolants
  snap to the float grid there, emitted steps jump past the backstop, and
  motions would silently truncate.
- `move_by` per-step underflow: a subnormal requested delta can make
  `vector / steps` exactly zero; a success note over a zero-motion chunk
  would lie, so this returns a structured "too small to split" error.
- Tool-call values: a JSON number is coerced with `float()` before the
  finiteness check — arbitrary-precision integers (`10**400`) overflow
  `float()` and crash `np.isfinite` outright, and both must return the
  standard "must be a finite number" structured error.
- The "no bounds; move conservatively" `bounds_text` fallback becomes dead
  in both modes (finite bounds are now required everywhere) and is removed
  outright.

Previously such spaces "worked" only because the LLM supplied a duration;
motions in them were never actually speed-safe.

### 3e. Toolset construction

`Toolset.__init__` / `build_toolset` grow `max_speed_frac: float` (default
`0.5`) and precompute the per-dimension per-step limits (absolute) or
per-side per-step limits (displacement). `build_toolset` owns frac
validation for direct constructions (`ToolsetError` for non-finite or
`<= 0`); `LLMAgentPolicy.__init__` additionally validates with `ValueError`
so a bad `-P max_speed_frac` fails at construction, before any LLM call
(§4). `bounds_text` keeps documenting the box to the LLM.

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

- Absolute modes: the §3a per-step ceiling means no `max_speed_frac` value
  can provoke the *default* backstop (frac above `0.05 * hz` is simply
  capped — the knob slows motion, never outruns the guardrail). The
  remaining truncation risk is an explicit `--max-action-delta` tighter
  than 5% of range.
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
  and reaches the *bit-exact* target in the final action (valid to assert
  with `==` only because §3a snaps it; plain linspace arithmetic lands
  1 ulp off).
- Interpolation clipping regression: a move from in-box state to a target
  exactly at a bound emits no action outside the box (checks every
  interpolant, not just the final action — e.g. `current=-0.1, high=0.3`,
  which overshoots by 1 ulp without the §3a clip).
- The headroom guarantee itself: for the boundary case above, every
  consecutive per-step delta of the emitted chunk is `<=` the backstop's
  `0.05 * (high - low)` — the spurious-clamp regression test.
- Small move → `steps == 1` floor; all named targets equal to current →
  `steps == 1`.
- Out-of-bounds target returns the "outside [low, high]" error; a target
  exactly at a bound succeeds.
- Zero-width dimension: targeting the bound value succeeds and contributes
  no steps *even when the observed state differs from the bound* (sensor
  noise); targeting any other value returns the "fixed at" error, and the
  error's value renders `repr()`-exact for a non-`.4g`-clean bound (e.g.
  `0.30000000000000004`).
- Per-step ceiling: declared `control_hz=5` at default frac emits 5%-of-range
  steps (not 10%); `max_speed_frac=1.0` at 10 Hz likewise stays at 5%/step
  (the knob cannot outrun the default backstop).
- Non-finite observed state (NaN in the proprioceptive reference) raises
  `ValueError` from `execute` rather than returning a structured error.
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
  at `max_steps` succeeds. Covered for both tools: `move_by` with a large
  total, and `move_joints` with a tiny frac (e.g. `max_speed_frac=0.01` →
  a full-range move needs 1001 steps, 100.1 s).
- Huge finite `move_by` delta (`1e308`) returns the split-the-move error —
  never raises (the §3c inf-tolerant cap check).
- `move_joints` with an absurd finite observed state (e.g. `1e308`) likewise
  returns the over-cap error, never an exception (§3c applies to both
  tools).
- Default-chain integration: a plugin-emitted absolute-mode chunk run
  through `ChainApprover(ClampApprover, DeltaLimitApprover)` with default
  settings — including a target exactly at a bound and a second chunk
  reusing the store — produces zero modified actions (the §2 promise,
  exercised against the real approvers and the store-held reference; unit
  level, since the CubePick e2e world is displacement-mode).
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
- `max_speed_frac` validation errors (0, negative, non-finite) at both
  layers: `build_toolset` (`ToolsetError`) and `LLMAgentPolicy.__init__`
  (`ValueError`; lives in the policy tests).
- Speed derivation honors non-default frac (absolute modes).

`tests/test_policy_e2e.py`: existing flows updated (tool calls no longer
send `duration_s`); config snapshot includes `max_speed_frac`; one e2e run
uses a non-default frac and asserts the resulting step count, pinning that
`bind()` actually forwards the knob to `build_toolset` (the toolset-level
frac tests cannot catch a forgotten forward). One test is
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
