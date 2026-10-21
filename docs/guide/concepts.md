# Concepts

Inspect Robots factors a robotics evaluation into a few small, orthogonal pieces.

## The two inputs

Unlike LLM evals (one swappable input, the model), a robotics eval has two:

- [`Policy`](../../src/inspect_robots/policy.py): the VLA "brain". Given an
  [`Observation`](../../src/inspect_robots/types.py), returns an
  [`ActionChunk`](../../src/inspect_robots/types.py): a horizon of actions executed open-loop
  (because VLA inference is slower than the control rate). `H = 1` is the
  degenerate reactive case.
- [`Embodiment`](../../src/inspect_robots/embodiment.py): the "body + world" (a real robot or
  a simulator). It produces observations, executes actions, and owns the
  action/observation spaces, the native control rate, and reset/safety machinery.

Both are runtime-checkable Protocols, so you can wrap an existing model or sim
without inheriting anything. Convenience base classes (`PolicyBase`,
`EmbodimentBase`) exist if you prefer.

## Tasks and scenes

A [`Task`](../../src/inspect_robots/task.py) is an embodiment-agnostic benchmark: a dataset
of [`Scene`](../../src/inspect_robots/scene.py)s plus scorer(s), a step- or seconds-based
horizon, and an epoch count. A seconds horizon is resolved to integer steps from
the paired embodiment's declared control rate. A `Scene` is the robotics analog of Inspect AI's `Sample`, one initial
condition: an instruction, an optional success [`Target`](../../src/inspect_robots/scene.py),
and a seed.

## Compatibility

Before any rollout, [`check_compatibility`](../../src/inspect_robots/compat.py) verifies the
`(policy, embodiment)` pair: action dimensions and [`ActionSemantics`](../../src/inspect_robots/spaces.py)
(control mode, rotation representation, gripper, frame), the observation
cameras/state keys the policy requires (resolving a name remap), the control rate,
whether a seconds-based task has a usable control rate, and whether each scene
is realizable on the embodiment. Hard mismatches fail fast
with a [`CompatibilityError`](../../src/inspect_robots/errors.py).

## The rollout

[`rollout`](../../src/inspect_robots/rollout.py) runs one trial as a single control-rate loop:

1. A [`Controller`](../../src/inspect_robots/controller.py) decides the next action, internally
   calling `policy.act()` and buffering the chunk (so open-loop execution and
   temporal ensembling compose without forking the loop).
2. An [`Approver`](../../src/inspect_robots/approver.py) reviews the action before it reaches
   the embodiment: pass, clamp, or veto (a safety gate).
3. `embodiment.step(action)` executes it; everything is logged to sinks and
   recorded in an immutable [`TrialRecord`](../../src/inspect_robots/rollout.py) (steps, a typed
   transcript, inference latencies).

Camera frames are streamed to a [`FrameStore`](../../src/inspect_robots/frames.py) and the
record keeps lightweight references, so long multi-camera episodes stay
memory-safe.

## Scoring

A [`Scorer`](../../src/inspect_robots/scorer.py) maps a recorded `TrialRecord` (+ the scene's
`Target`) to a [`Score`](../../src/inspect_robots/scorer.py). Because scorers consume the
*recorded* trajectory (not a live environment), scoring is reproducible from a
saved log. Across the `epochs` of a scene, an epoch reducer (`mean`, `max`,
`pass_at_k`, …) collapses scores; metrics then aggregate across scenes.

## Errors and safety

The error taxonomy resolves the "fail fast vs never-crash-overnight" tension:

| Class | Policy |
|---|---|
| [`CompatibilityError`](../../src/inspect_robots/errors.py), `ConfigError` | fail fast, before any rollout |
| [`PolicyError`](../../src/inspect_robots/errors.py) | record the trial, then continue or halt per `fail_on_error` (`True` = first error, `0<x<1` = proportion, `x>1` = count), checked after every trial |
| [`EmbodimentFault`](../../src/inspect_robots/errors.py), [`SafetyAbort`](../../src/inspect_robots/errors.py) | **always halt**: a faulted/unsafe robot never auto-advances |

Failures inside a trial (including `reset`) are wrapped into the taxonomy; a
crashing approver becomes a `SafetyAbort` (it can no longer vouch for safety).
A non-finite action from the policy is a `PolicyError`. A non-finite action
introduced by an approver is a `SafetyAbort`.
Every error raised from inside a trial carries the partial
[`TrialRecord`](../../src/inspect_robots/rollout.py) on its `record` attribute, so the
steps that did run are delivered to sinks. Errored trials are recorded but
never scored: a failed trial cannot masquerade as data in the metrics.
A finished run in which every trial errored ends with `status: "error"` even
under the default `fail_on_error=False`: a run that scored nothing is not a
success.

## The eval log

[`eval`](../../src/inspect_robots/eval.py) orchestrates scenes × epochs and returns immutable
[`EvalLog`](../../src/inspect_robots/log.py)s (status, spec, results, stats, per-scene samples,
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
