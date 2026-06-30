# Concepts

RoboInspect factors a robotics evaluation into a few small, orthogonal pieces.

## The two inputs

Unlike LLM evals (one swappable input, the model), a robotics eval has **two**:

- [`Policy`][roboinspect.policy.Policy] — the VLA "brain". Given an
  [`Observation`][roboinspect.types.Observation], returns an
  [`ActionChunk`][roboinspect.types.ActionChunk]: a horizon of actions executed open-loop
  (because VLA inference is slower than the control rate). `H = 1` is the
  degenerate reactive case.
- [`Embodiment`][roboinspect.embodiment.Embodiment] — the "body + world": a real robot or
  a simulator. It produces observations, executes actions, and owns the
  action/observation spaces, the native control rate, and reset/safety machinery.

Both are runtime-checkable Protocols, so you can wrap an existing model or sim
without inheriting anything. Convenience base classes (`PolicyBase`,
`EmbodimentBase`) exist if you prefer.

## Tasks and scenes

A [`Task`][roboinspect.task.Task] is an **embodiment-agnostic** benchmark: a dataset
of [`Scene`][roboinspect.scene.Scene]s plus scorer(s), a step horizon, and an epoch
count. A `Scene` is the robotics analog of Inspect AI's `Sample` — one initial
condition: an instruction, an optional success [`Target`][roboinspect.scene.Target],
and a seed.

## Compatibility

Before any rollout, [`check_compatibility`][roboinspect.compat.check_compatibility] verifies the
`(policy, embodiment)` pair: action dimensions and [`ActionSemantics`][roboinspect.spaces.ActionSemantics]
(control mode, rotation representation, gripper, frame), the observation
cameras/state keys the policy requires (resolving a name remap), the control rate,
and whether each scene is realizable on the embodiment. Hard mismatches fail fast
with a [`CompatibilityError`][roboinspect.errors.CompatibilityError].

## The rollout

[`rollout`][roboinspect.rollout.rollout] runs one trial as a single control-rate loop:

1. A [`Controller`][roboinspect.controller.Controller] decides the next action, internally
   calling `policy.act()` and buffering the chunk (so open-loop execution and
   temporal ensembling compose without forking the loop).
2. An [`Approver`][roboinspect.approver.Approver] reviews the action before it reaches
   the embodiment — pass, clamp, or veto (a safety gate).
3. `embodiment.step(action)` executes it; everything is logged to sinks and
   recorded in an immutable [`TrialRecord`][roboinspect.rollout.TrialRecord] (steps, a typed
   transcript, inference latencies).

Camera frames are streamed to a [`FrameStore`][roboinspect.frames.FrameStore] and the
record keeps lightweight references, so long multi-camera episodes stay
memory-safe.

## Scoring

A [`Scorer`][roboinspect.scorer.Scorer] maps a recorded `TrialRecord` (+ the scene's
`Target`) to a [`Score`][roboinspect.scorer.Score]. Because scorers consume the
*recorded* trajectory (not a live environment), scoring is reproducible from a
saved log. Across the `epochs` of a scene, an **epoch reducer** (`mean`, `max`,
`pass_at_k`, …) collapses scores; metrics then aggregate across scenes.

## Errors and safety

The error taxonomy resolves the "fail fast vs never-crash-overnight" tension:

| Class | Policy |
|---|---|
| [`CompatibilityError`][roboinspect.errors.CompatibilityError], `ConfigError` | fail fast, before any rollout |
| [`PolicyError`][roboinspect.errors.PolicyError] | record the trial, continue (governed by `fail_on_error`) |
| [`EmbodimentFault`][roboinspect.errors.EmbodimentFault], [`SafetyAbort`][roboinspect.errors.SafetyAbort] | **always halt** — a faulted/unsafe robot never auto-advances |

## The eval log

[`eval`][roboinspect.eval.eval] orchestrates scenes × epochs and returns immutable
[`EvalLog`][roboinspect.log.EvalLog]s (status, spec, results, stats, per-scene samples,
error). Logs are written atomically as schema-versioned JSON with a read-back
guarantee.
