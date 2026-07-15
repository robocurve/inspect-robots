# 0014 — Agent plugin: speed-limited playout replaces `duration_s`

Issue: robocurve/inspect-robots#92. Scope: `plugins/inspect-robots-agent` only;
core is untouched.

## 1. Problem

The agent toolset (plan 0008 §4c) makes the LLM pick `duration_s` for every
motion. Speed is not a quantity the LLM should reason about: the loop is
open-loop per turn (the model re-observes after the chunk plays out) and the
trial budget is per LLM call, not per second. Worse, a too-short duration does
not fail — the core `DeltaLimitApprover` clamps the chunk per step, so the arm
moves at the speed limit and stops short of the target, while the tool result
note still claims the full motion executed. The LLM discovers the divergence
only from the next observation.

## 2. Decision

Drop `duration_s` from both motion tools. The toolset computes the playout
time from a speed cap, so the full requested displacement is always reached;
large motions simply play out over more steps. Guardrails below the plugin
are unchanged (the plugin still contains no safety-critical code path); at the
default speed the per-step change stays within the core backstop, so the
backstop stops biting and tool notes become truthful.

Resolved knobs (user decisions, 2026-07-14):

- Plugin-only; no core controller middleware.
- `duration_s` is removed, not made optional.
- Default speed: `max_speed_frac = 0.5` — half the joint's range per second,
  which equals the core backstop's implied speed (5% of range per step at
  10 Hz).

## 3. Toolset changes (`_tools.py`)

### 3a. `move_joints` (absolute modes: `joint_pos`, `eef_abs_pose`)

- Schema: `targets` only; `required: ["targets"]`. Description explains the
  motion is interpolated at a fixed safe speed and reports its duration.
- Per-dimension speed: `speed = max_speed_frac * (high - low)` per second
  (float64 vector).
- Steps: `steps = max(1, ceil(max_i(|target_i - current_i| / speed_i) * hz))`
  computed only over dimensions named in the call (unnamed dimensions hold
  and contribute zero distance). `hz` falls back to `_FALLBACK_HZ = 10.0` as
  today.
- Interpolation is unchanged (`linspace` fractions from current observed
  state).

### 3b. `move_by` (displacement modes: `eef_delta_pos`, `eef_delta_pose`, `joint_delta`)

- Schema: `deltas` only; `required: ["deltas"]`.
- Displacement boxes are per-action sized by convention, and `ClampApprover`
  clamps each step to the box — so the box side is the per-step limit and
  splitting to fit it is the only non-lossy choice.
- Per-step limit per dimension: `high_i` for a positive component, `|low_i|`
  for a negative one.
- Steps: `steps = max(1, ceil(max_i(|delta_i| / limit_i)))` over named
  dimensions with nonzero values; each action is `vector / steps` (unchanged
  emission shape).

### 3c. Chunk cap

`_MAX_DURATION_S = 10.0` stays, reinterpreted: `max_steps = ceil(10 * hz)`.
A computed `steps > max_steps` returns a structured `ToolResult(error=...)`
telling the LLM the request exceeds the 10 s playout cap and to split the
move. At the default speed a full-range `move_joints` needs 2 s, so in
practice only absurd `move_by` totals hit this.

### 3d. Bind-time bound requirements (never guess)

`build_toolset` gains the checks; failures raise `ToolsetError` at bind time
with actionable messages (same stance as `DeltaLimitApprover`):

- Absolute modes: every dimension needs finite `low` and `high` (speed is
  derived from the range). The existing "no bounds; move conservatively"
  fallback text disappears for absolute modes since unbounded absolute spaces
  are now rejected.
- Displacement modes: every dimension needs a finite bound on each side
  (`low_i < 0 < high_i` finite both sides; a zero bound on a side simply
  means that direction cannot move, which the box already enforces — no
  special-casing, division guards use the bound only for the direction
  requested).

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
- CLI passthrough needs no change (`-P max_speed_frac=0.5` already coerces
  via the existing `-P` machinery).

Values of `max_speed_frac` above the backstop's implied speed (5% of range
per step; e.g. > 0.5 at 10 Hz, or any value whose `frac / hz > 0.05`) mean
the core `DeltaLimitApprover` clamps again and motions truncate. Safe but
lossy; documented in the README, not prevented — the plugin cannot know the
CLI's `--max-action-delta`.

## 5. Reporting

Success note: `executing move_joints over {steps} steps ({steps/hz:.1f}s at
the speed cap)` (analogous for `move_by`). The note is now truthful by
construction at default settings.

## 6. Docs

- Plugin README: rewrite "How it works" — tool calls state *where* to go;
  the plugin times the motion at a fixed safe speed; `duration_s` is gone;
  document `max_speed_frac` in the knobs list and the over-backstop caveat.
- Docstrings updated (`_tools.py` module docstring describes the timing
  rule).

## 7. Tests (TDD; plugin's own pytest scope, not the core 100% gate)

`tests/test_tools_motion.py`:

- `move_joints` computes steps from distance and range (e.g. range 2.0,
  frac 0.5 → speed 1.0/s; distance 1.5 at 10 Hz → 15 steps) and reaches the
  exact target in the final action.
- Small move → `steps == 1` floor.
- `move_by` splits by box side: `high = 0.05`, delta `+0.2` → 4 steps of
  0.05; asymmetric `low = -0.1` with delta `-0.2` → 2 steps.
- Over-cap request errors with the split-the-move message; boundary exactly
  at `max_steps` succeeds.
- `duration_s` provided by the LLM anyway → unknown-argument tolerance:
  extra keys are ignored today (dict lookup); assert schemas no longer
  advertise it and that a call without it succeeds.
- Bind-time `ToolsetError` for missing/non-finite bounds per 3d (absolute
  and displacement variants).
- `max_speed_frac` validation errors.
- Speed derivation honors non-default frac.

`tests/test_policy_e2e.py`: existing flows updated (tool calls no longer
send `duration_s`); config snapshot includes `max_speed_frac`.

`tests/test_llm.py`, `tests/test_package.py`: untouched unless imports move.

## 8. Versioning / release

Bump plugin `version` in `plugins/inspect-robots-agent/pyproject.toml`
`0.1.0 → 0.2.0` (tool-surface break). Publishes with the next core release
via the existing `publish-inspect-robots-agent` job.

## 9. Out of scope

- A core rate-limiting controller middleware (revisit if a second policy
  plugin needs re-timing).
- Changing core guardrail defaults or `--max-action-delta` semantics.
- Per-dimension `max_speed_frac`.
