# 0016 — Embodiment-authored docs for LLM agent policies

Status: draft
Issue: robocurve/inspect-robots#109
Companion: inspect-robots-yam (cheat-sheet content, separate PR after core release)

## 1. Problem

An LLM agent controlling a physical arm through `inspect-robots-agent` sees
only generic dimension labels (`left_j0` … `right_gripper`) and numeric
bounds. It does not know which joint is the shoulder, which direction is
positive, what the zero pose looks like, or that `gripper=1` means open. On
YAM fork-task runs the agent must infer all of this by trial and error inside
a 100-call budget.

Root cause (verified against source): there is no free-text field anywhere in
the embodiment → policy chain. `EmbodimentInfo` carries `name`, spaces,
`control_hz`, `is_simulated`, capability flags, and nothing else;
`ActionSemantics` carries `dim_labels` only; `StateField` carries a short
`unit` string. The agent plugin's `bind()` consumes `action_space`,
`observation_space`, `control_hz`, `name` — there is nothing else to consume.

A second, smaller gap: the per-step observation text renders the proprio
state as an unlabeled rounded list (`state[joint_pos]: [0.01, -0.02, …]`),
forcing the model to count elements to map values to joints.

## 2. Goals / non-goals

Goals:

- G1: an embodiment can publish concise, policy-facing operating notes
  (kinematic layout, sign conventions, units, gripper polarity) that LLM
  agent policies include in their system prompt.
- G2: the proprio state line in each observation is labeled per element using
  the action-space `dim_labels` when shapes line up.
- G3: the channel is generic core API, usable by any embodiment; VLA policies
  ignore it entirely (no compat implications).

Non-goals:

- Structured per-dimension doc objects (a single markdown string is enough;
  the consumer flattens to prose anyway).
- Task- or scene-level docs (that is the instruction's job).
- Any change to compat checking, scoring, or logging schemas. The docs field
  is advisory metadata.

## 3. Design

### 3.1 Core: `EmbodimentInfo.docs`

Add one optional field to the frozen `EmbodimentInfo` dataclass
(`src/inspect_robots/embodiment.py`):

```python
docs: str | None = None
```

Contract (docstring): free-form markdown operating notes for policies that
can read text (LLM agents). Should describe what the spaces cannot: joint
layout and positive directions, zero-pose geometry, gripper polarity, frame
conventions, workspace hints. Concise — this text is injected into a system
prompt verbatim. `None` (default) means the embodiment offers no notes;
consumers must treat absence and empty/whitespace-only strings identically.

Placement rationale: `EmbodimentInfo` rather than the spaces, because the
notes describe the embodiment as a whole (geometry, conventions spanning
action *and* observation) and `bind()` already receives the whole info
object. No new hook is needed.

Ripple effects inside core:

- `__all__` / `tests/test_api_snapshot.py`: `EmbodimentInfo` is already
  exported; the snapshot records field names, so it must be regenerated or
  updated to include `docs`.
- `conformance.py` does not gate on the new field (advisory).
- Mock embodiments: `CubePick` gains a one-paragraph docs string, which
  doubles as the in-tree usage example and exercises the field end to end in
  the existing agent-plugin-against-mock tests (if any) and core tests.
- Nothing in `log.py` persists `EmbodimentInfo`, so no schema bump.

### 3.2 Agent plugin: render docs into the system prompt

`plugins/inspect-robots-agent`:

- `LLMAgentPolicy.bind()` captures `getattr(embodiment_info, "docs", None)`
  (getattr so the plugin keeps working against older cores; the pyproject
  lower bound is still raised — see §5 — but the plugin should not crash if
  someone pins an old core with a new plugin).
- `reset()` appends a section to the system prompt when docs are non-empty
  after `str.strip()`:

  ```text
  {existing _SYSTEM_TEMPLATE text}

  Embodiment notes:
  {docs}
  ```

  No truncation, no reformatting. Whitespace-only docs are treated as absent.
- Tool descriptions are unchanged: bounds and labels already live there, and
  duplicating prose into every tool schema wastes tokens on every call.

### 3.3 Agent plugin: labeled proprio state in observations

In `_observation_content` (policy.py), when rendering the state field that
the toolset selected as the proprio reference (`state_key`) *and* its length
equals the number of action dims, render label/value pairs instead of a bare
list:

```text
state[joint_pos]: left_j0=0.01 left_j1=-0.02 … right_gripper=0.98
```

All other state fields keep the existing unlabeled rendering. Rounding stays
as-is. This is a pure prompt-format change; update the plugin tests that
assert on observation text.

### 3.4 YAM plugin: authored cheat-sheet (companion PR, separate repo)

`inspect_robots_yam` sets `EmbodimentInfo.docs` according to
`control_interface`:

- `joints` mode: the joint cheat-sheet in Appendix A, with labels rewritten
  to the plugin's `{side}_j{i}` naming. Both arms are identical hardware with
  identical sign conventions, so the notes are written once and prefixed
  "each arm".
- `eef_pos` mode: frame identity (+X forward out of the arm base, +Y left,
  +Z up, per-arm base frame), yaw-relative-to-start restated briefly, and
  the validated workspace box from plan 0006 (yam repo) as a reachability
  hint.
- New optional config key `docs_extra: str = ""` (embodiment arg, so
  `-E docs_extra="…"` and config.ini both work): operator-supplied
  rig-specific notes (e.g. how the two arms are mounted relative to the
  table) appended after the built-in text with a blank line. The built-in
  text must stay rig-agnostic because mounting varies.

Implementation must verify against `packing.py`/`config.py` (yam repo), not
this plan, for: label order, gripper normalization direction, and eef dim
labels. Appendix A values were verified numerically against the combined
i2rt MuJoCo model (FK sign probes) on 2026-07-15 and are normative for the
*content*; the label mapping is normative from the code.

### 3.5 What Fable sees afterwards (joints mode, abridged)

```text
You are controlling a real robot embodiment named 'yam_arms' … budget of 100 LLM calls …

Embodiment notes:
Two identical 6-DoF arms ("left_", "right_") with parallel-jaw grippers …
- j0: base yaw, positive swings the arm counterclockwise seen from above …
…
```

## 4. Testing

Core (100% gate):

- `EmbodimentInfo` default `docs is None`; frozen dataclass still hashes/eqs.
- API snapshot updated.
- CubePick publishes non-empty docs.

Agent plugin:

- bind→reset with docs set: system prompt contains the section verbatim.
- docs None / empty / whitespace-only: prompt identical to today's (no
  dangling header).
- getattr fallback: bind against a stub info object without a `docs`
  attribute does not raise.
- labeled proprio line: exact-format test for label=value rendering; a state
  field whose length mismatches action dims keeps the old rendering.

YAM plugin (companion PR):

- joints-mode docs mention every label `left_j0`…`right_gripper` exactly
  once in the bullet list; gripper polarity string matches the normalization
  in `packing.py` (test derives expected polarity from the code, not a
  literal).
- eef-mode docs mention every eef label and the workspace box bounds equal
  to the constants in `config.py`.
- `docs_extra` appended verbatim; empty default adds nothing.

## 5. Rollout

1. Core PR (this plan): core field + agent plugin consumption. Agent plugin
   version → 0.5.0 (feature); core minor release after merge.
2. Yam PR: raise `inspect-robots` lower bound to the new core version, add
   the cheat-sheet + `docs_extra`. Yam minor release after merge.
3. No config migration: absent field defaults preserve today's behavior
   everywhere.

## Appendix A — verified YAM kinematic facts (content source)

Source: combined arm+gripper MuJoCo model from the i2rt repo
(`i2rt/robot_models/arm/yam/yam.xml` + `linear_4310` gripper via
`combine_arm_and_gripper_xml`), signs verified by FK probes; hardware motor
`directions: [1,1,1,1,1,1]` so model signs match hardware.

Per-arm base frame: +X forward (the direction the folded gripper points at
all-zero joints), +Y left, +Z up.

| API label (per side) | physical joint | positive direction | range (rad, XML) |
|---|---|---|---|
| j0 | base yaw (vertical axis) | counterclockwise viewed from above; a forward-pointing gripper swings toward +Y (left) | [-2.618, 3.054] |
| j1 | shoulder pitch | raises the upper arm; 0 = folded horizontally backward, π/2 = straight up, π = horizontal forward | [0, 3.65] |
| j2 | elbow | opens/extends the elbow; 0 = forearm fully folded back against the upper arm | [0, 3.665] |
| j3 | wrist pitch (axis parallel to elbow) | tilts the gripper up | [-1.571, 1.571] |
| j4 | wrist yaw | swings the gripper toward the arm's right viewed from above (opposite sign sense of j0) | [-1.571, 1.571] |
| j5 | wrist roll about the gripper's pointing axis | right-hand rule about the outward axis | [-2.094, 2.094] |
| gripper | parallel jaws, normalized | 0 = fully closed, 1 = fully open (~9.5 cm opening) | [0, 1] |

Geometry: upper arm 0.264 m, forearm 0.252 m, wrist-to-grasp-point 0.247 m
when straight; max reach shoulder→grasp ≈ 0.76 m. At all-zero joints the arm
rests folded with the gripper pointing forward (+X). Runtime software widens
the XML ranges by ±0.15 rad; the plugin's declared bounds already reflect
what the embodiment accepts, so the notes must not restate numeric ranges
(the tool description owns bounds).

Caution for the author of the prose: with the arm folded (near zero), the
*end-effector* motion produced by a joint can be counterintuitive (e.g. +j1
raises the upper arm but initially lowers the grasp point because the
forearm is doubled back). The notes therefore describe joint-level motion,
not end-effector effect.
