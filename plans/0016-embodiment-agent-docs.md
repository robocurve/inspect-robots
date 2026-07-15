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

- `tests/test_api_snapshot.py` needs **no change**: it fences the symbol
  names in `inspect_robots.__all__`, not dataclass fields, and
  `EmbodimentInfo` is already exported. Do not add `"docs"` to `EXPECTED`.
- `conformance.py` does not gate on the new field (advisory).
- `EvalSpec.embodiment_info` (the hand-picked dict projection in `eval.py`)
  deliberately does **not** gain `docs`: the rendered system prompt is
  already captured verbatim in the per-trial policy transcript, which is the
  audit surface that matters. Leave the projection unchanged.
- Mock embodiments: `CubePick` gains a one-paragraph docs string, which
  doubles as the in-tree usage example and exercises the field end to end in
  the existing agent-plugin-against-mock tests (if any) and core tests.
- Nothing in `log.py` persists `EmbodimentInfo`, so no schema bump.

### 3.2 Agent plugin: render docs into the system prompt

`plugins/inspect-robots-agent`:

- `LLMAgentPolicy.bind()` captures `getattr(embodiment_info, "docs", None)`.
  The getattr fallback means the plugin degrades gracefully against older
  cores (docs simply absent), so the `inspect-robots>=0.4` lower bound in
  the plugin's pyproject stays as-is — no bump.
- The docs attribute is initialized to `None` in `__init__` (not only in
  `bind()`): `reset()` must keep working on an unbound policy.
- `reset()` appends a section to the system prompt when docs are non-empty
  after `str.strip()`. Order of operations is normative: run
  `_SYSTEM_TEMPLATE.format(...)` **first**, then concatenate the docs text
  verbatim (`formatted + "\n\nEmbodiment notes:\n" + docs`). Embodiment
  markdown may legally contain `{`/`}` (JSON snippets, code examples);
  passing it through `str.format` would let third-party content crash every
  trial at reset.

  No truncation, no reformatting. Whitespace-only docs are treated as absent
  (no dangling header).
- Tool descriptions are unchanged: bounds and labels already live there, and
  duplicating prose into every tool schema wastes tokens on every call.

### 3.3 Agent plugin: labeled proprio state in observations

Goal: render the proprio state field as label/value pairs instead of a bare
list:

```text
state[joint_pos]: left_j0=0.01 left_j1=-0.02 … right_gripper=0.98
```

Which field gets labeled — normative selection rule, applying in **all**
modes (absolute and displacement; today's `state_key` is only computed for
absolute modes, which would silently skip YAM `joint_delta` runs):

1. If the toolset's existing `state_key` is set (absolute modes), label that
   field.
2. Otherwise (displacement modes), label the state field whose shape equals
   `(action_dim,)` **iff exactly one field matches**; ambiguity (e.g.
   CubePick's two 2-D fields `eef_pos`/`cube_pos` against a 2-D action)
   means no labeling — a mislabeled line is worse than an unlabeled one.
3. Labeling additionally requires real `dim_labels` on the action semantics.
   When `build_toolset` synthesized fallback labels (`"0"`, `"1"`, …),
   render the old unlabeled list — `0=0.01 1=-0.02` is pure noise.
4. At render time, if the runtime vector's length mismatches the labels,
   fall back to the unlabeled rendering (sanity check, never crash).

Plumbing — normative, because `_observation_content` is a module-level
function with no toolset access: `Toolset` gains one public method,
`state_labels() -> tuple[str, tuple[str, ...]] | None`, returning
`(state_key_to_label, labels)` per rules 1–3 or `None`. The policy calls it
once after bind and passes the result into `_observation_content` as an
optional parameter (default `None` keeps the current rendering).

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
  +Z up, per-arm base frame), yaw-relative-to-start restated briefly, arm
  geometry (link lengths, reach), and gripper polarity. **No numeric
  workspace bounds**: in eef mode the action box *is* the workspace box and
  `build_toolset` already renders those numbers into the move tool's
  bounds text; restating them here would violate the no-restated-bounds
  rule and drift whenever an operator overrides `eef_low`/`eef_high`.
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
- No API-snapshot change (see §3.1).
- CubePick publishes non-empty docs.

Agent plugin:

- bind→reset with docs set: system prompt contains the section verbatim,
  **including docs containing `{`/`}` characters** (the format-first rule).
- docs None / empty / whitespace-only: prompt identical to today's (no
  dangling header).
- reset before bind still works (docs attr initialized in `__init__`).
- getattr fallback: bind against a stub info object lacking a `docs`
  attribute does not raise (test will need a `cast`, since `bind()` is
  typed against `EmbodimentInfo`).
- `state_labels()`: absolute mode returns the proprio field labels;
  displacement mode with a unique shape match returns it; ambiguous shapes
  (CubePick-style) return None; synthesized fallback labels return None;
  runtime length mismatch falls back to unlabeled rendering.
- labeled proprio line: exact-format test for label=value rendering.

YAM plugin (companion PR):

- joints-mode docs mention every label `left_j0`…`right_gripper` exactly
  once in the bullet list; the gripper polarity sentence is asserted as a
  literal matching the wire convention hardcoded in the yam
  `embodiment.py` gripper normalization (cmd 1 → open). The convention is
  structural there, not a derivable constant — a literal is the honest test.
- eef-mode docs mention every eef label; assert **no** numeric workspace
  bounds appear (guards the no-restated-bounds rule).
- `docs_extra` appended verbatim (including braces); empty default adds
  nothing.

## 5. Rollout

1. Core PR (this plan): core field + agent plugin consumption. Agent plugin
   version → 0.5.0 (feature); its `inspect-robots>=0.4` bound is unchanged
   (getattr fallback, §3.2). Core minor release after merge.
2. Yam PR: raise the yam plugin's `inspect-robots` lower bound to the new
   core minor (it constructs `EmbodimentInfo(docs=…)`, which raises
   TypeError on older cores — unlike the agent plugin, yam genuinely needs
   the new field), add the cheat-sheet + `docs_extra`. Yam minor release
   after merge.
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
rests folded with the gripper pointing forward (+X).

The notes must not restate numeric ranges: the tool description owns bounds.
This matters doubly because the yam plugin's default joint bounds are
conservative ±π placeholders that do not match the XML ranges above (e.g.
j1 XML [0, 3.65] vs declared [-π, π]) — restated numbers would contradict
the tool text on a default rig. The XML ranges in the table exist to inform
the *prose* (e.g. "j1 and j2 only move one way from zero"), never to be
copied into the docs string.

Caution for the author of the prose: with the arm folded (near zero), the
*end-effector* motion produced by a joint can be counterintuitive (e.g. +j1
raises the upper arm but initially lowers the grasp point because the
forearm is doubled back). The notes therefore describe joint-level motion,
not end-effector effect.
