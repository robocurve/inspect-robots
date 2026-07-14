# Concepts

Inspect Robots factors a robotics evaluation into a few small, orthogonal pieces.

## The two inputs

Unlike LLM evals (one swappable input, the model), a robotics eval has two:

- [`Policy`][inspect_robots.policy.Policy]: the VLA "brain". Given an
  [`Observation`][inspect_robots.types.Observation], returns an
  [`ActionChunk`][inspect_robots.types.ActionChunk]: a horizon of actions executed open-loop
  (because VLA inference is slower than the control rate). `H = 1` is the
  degenerate reactive case.
- [`Embodiment`][inspect_robots.embodiment.Embodiment]: the "body + world" (a real robot or
  a simulator). It produces observations, executes actions, and owns the
  action/observation spaces, the native control rate, and reset/safety machinery.

Both are runtime-checkable Protocols, so you can wrap an existing model or sim
without inheriting anything. Convenience base classes (`PolicyBase`,
`EmbodimentBase`) exist if you prefer.

## Tasks and scenes

A [`Task`][inspect_robots.task.Task] is an embodiment-agnostic benchmark: a dataset
of [`Scene`][inspect_robots.scene.Scene]s plus scorer(s), a step horizon, and an epoch
count. A `Scene` is the robotics analog of Inspect AI's `Sample`, one initial
condition: an instruction, an optional success [`Target`][inspect_robots.scene.Target],
and a seed.

## Compatibility

Before any rollout, [`check_compatibility`][inspect_robots.compat.check_compatibility] verifies the
`(policy, embodiment)` pair: action dimensions and [`ActionSemantics`][inspect_robots.spaces.ActionSemantics]
(control mode, rotation representation, gripper, frame), the observation
cameras/state keys the policy requires (resolving a name remap), the control rate,
and whether each scene is realizable on the embodiment. Hard mismatches fail fast
with a [`CompatibilityError`][inspect_robots.errors.CompatibilityError].

## The rollout

[`rollout`][inspect_robots.rollout.rollout] runs one trial as a single control-rate loop:

1. A [`Controller`][inspect_robots.controller.Controller] decides the next action, internally
   calling `policy.act()` and buffering the chunk (so open-loop execution and
   temporal ensembling compose without forking the loop).
2. An [`Approver`][inspect_robots.approver.Approver] reviews the action before it reaches
   the embodiment: pass, clamp, or veto (a safety gate).
3. `embodiment.step(action)` executes it; everything is logged to sinks and
   recorded in an immutable [`TrialRecord`][inspect_robots.rollout.TrialRecord] (steps, a typed
   transcript, inference latencies).

Camera frames are streamed to a [`FrameStore`][inspect_robots.frames.FrameStore] and the
record keeps lightweight references, so long multi-camera episodes stay
memory-safe.

## Scoring

A [`Scorer`][inspect_robots.scorer.Scorer] maps a recorded `TrialRecord` (+ the scene's
`Target`) to a [`Score`][inspect_robots.scorer.Score]. Because scorers consume the
*recorded* trajectory (not a live environment), scoring is reproducible from a
saved log. Across the `epochs` of a scene, an epoch reducer (`mean`, `max`,
`pass_at_k`, …) collapses scores; metrics then aggregate across scenes.

## Errors and safety

The error taxonomy resolves the "fail fast vs never-crash-overnight" tension:

| Class | Policy |
|---|---|
| [`CompatibilityError`][inspect_robots.errors.CompatibilityError], `ConfigError` | fail fast, before any rollout |
| [`PolicyError`][inspect_robots.errors.PolicyError] | record the trial, then continue or halt per `fail_on_error` (`True` = first error, `0<x<1` = proportion, `x>1` = count), checked after every trial |
| [`EmbodimentFault`][inspect_robots.errors.EmbodimentFault], [`SafetyAbort`][inspect_robots.errors.SafetyAbort] | **always halt**: a faulted/unsafe robot never auto-advances |

Failures inside a trial (including `reset`) are wrapped into the taxonomy; a
crashing approver becomes a `SafetyAbort` (it can no longer vouch for safety).
Every error raised from inside a trial carries the partial
[`TrialRecord`][inspect_robots.rollout.TrialRecord] on its `record` attribute, so the
steps that did run are delivered to sinks. Errored trials are recorded but
never scored: a failed trial cannot masquerade as data in the metrics.
A finished run in which every trial errored ends with `status: "error"` even
under the default `fail_on_error=False`: a run that scored nothing is not a
success.

## The eval log

[`eval`][inspect_robots.eval.eval] orchestrates scenes × epochs and returns immutable
[`EvalLog`][inspect_robots.log.EvalLog]s (status, spec, results, stats, per-scene samples,
error). Logs are written atomically as schema-versioned JSON with a read-back
guarantee. Once rollouts have started, an `EvalLog` is always produced and
persisted: scorer or reducer failures degrade the run to an error log rather
than crashing away the night's data.

`eval()` also owns what it opens: an embodiment resolved from a registry
name is closed when the run finishes (even on a halt), while a
caller-constructed embodiment object stays open (its lifecycle belongs to the
caller).

Passing `seed=None` draws a fresh seed from the OS and records it in the log,
so an "unseeded" run remains reproducible after the fact (and is distinct from
`seed=0`).
