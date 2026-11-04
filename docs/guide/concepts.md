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

A multimodal language model becomes a policy through a
[`PolicyAdapter`](../../src/inspect_robots/policy_adapter.py): it turns the
embodiment's declarations into control tools, termination tools and operating
instructions, and renders each observation as text. The policy keeps only its
model and conversation, so every language-model policy drives a given
embodiment through the same tools. See
[the policy adapter](policies-and-embodiments.md#a-language-model-policy-the-policy-adapter).

## Tasks and scenes

A [`Task`](../../src/inspect_robots/task.py) is an embodiment-agnostic benchmark: a dataset
of [`Scene`](../../src/inspect_robots/scene.py)s plus scorer(s), a step- or seconds-based
horizon, and an epoch count. A seconds horizon is resolved to integer steps from
the paired embodiment's declared control rate. A `Scene` is the robotics analog of Inspect AI's `Sample`, one initial
condition: an instruction, an optional success [`Target`](../../src/inspect_robots/scene.py),
and a seed. A task measures each trial against one or more
[`Rubric`](../../src/inspect_robots/rubric.py)s, each a named criterion with a
scorer, so capability and safety are scored side by side.

A [`Benchmark`](../../src/inspect_robots/benchmark.py) is a named, versioned
sequence of tasks. Every log it produces records the benchmark's name and
version, so results are citable and different versions are never pooled. See
[Benchmarks](writing-a-benchmark.md#benchmarks-a-versioned-set-of-tasks).

## Compatibility

Before any rollout, [`check_compatibility`](../../src/inspect_robots/compat.py) verifies the
`(policy, embodiment)` pair: action dimensions and [`ActionSemantics`](../../src/inspect_robots/spaces.py)
(control mode, rotation representation, gripper, frame), the observation
cameras/state keys the policy requires (resolving a name remap), the control rate,
whether a seconds-based task has a usable control rate, and whether each scene
is realizable on the embodiment. Hard mismatches fail fast
with a [`CompatibilityError`](../../src/inspect_robots/errors.py).

## Around each trial

Besides `reset()` and `step()`, which the policy drives, an embodiment may
offer an evaluation interface that [`eval`](../../src/inspect_robots/eval.py)
drives between trials:

1. `health()` checks the hardware (motors, cameras, temperatures); a failed
   check stops the run before anything moves.
2. `prepare_trial(scene)` says whether this scene can run now, with a reason
   when it cannot.
3. `home()` moves to the start pose, then `reset()` and the trial run.
4. `park()` moves to a safe rest pose once the trial is scored, so a person can
   reset the scene by hand.

Each method is optional and used only when defined. See
[The evaluation interface](policies-and-embodiments.md#the-evaluation-interface).

## The rollout

[`rollout`](../../src/inspect_robots/rollout.py) runs one trial as a single control-rate loop:

1. An [`ActionManager`](../../src/inspect_robots/action_manager.py) decides the next action, internally
   calling `policy.act()` and buffering the chunk (so open-loop execution and
   temporal ensembling compose without forking the loop). Each time an action
   completes it decides whether to play the next one or rerun the policy on
   the latest observation; see [Replanning on observation](#replanning-on-observation).
2. A [`Guard`](../../src/inspect_robots/guard.py) reviews the action before it reaches
   the embodiment: pass, clamp, or veto (a safety gate). Guards are
   configuration: name registered guards with `--guard` or a `[guards]` config
   section, and each change a guard makes is recorded with the action before
   and after it (see [Guards](cli.md#guards---guard-and-guards)).
3. `embodiment.step(action)` executes it; everything is logged to sinks and
   recorded in an immutable [`Record`](../../src/inspect_robots/rollout.py) (steps, every
   action chunk the policy produced, a typed transcript, inference latencies).

The trial ends at the task's step or time budget, when the embodiment reports
an outcome, or when the policy emits a terminal action, such as `done` or
`give_up`, with its stated reason (see
[Ending a trial](policies-and-embodiments.md#ending-a-trial)). Ending a trial
is never a claim of success; scoring stays with the task.

Camera frames are streamed to a [`FrameStore`](../../src/inspect_robots/frames.py) and the
record keeps lightweight references, so long multi-camera episodes stay
memory-safe.

## Replanning on observation

[`DefaultActionManager`](../../src/inspect_robots/action_manager.py) replans
when a chunk runs out (or after `replan_interval` actions). Give it
[`ReplanTrigger`](../../src/inspect_robots/action_manager.py)s to replan
earlier when the observation calls for it:

```python
from inspect_robots.action_manager import DefaultActionManager, GripperSlip, UnreachableTarget

space = embodiment.info.action_space
manager = DefaultActionManager(
    triggers=[UnreachableTarget(), *GripperSlip.for_embodiment(space, "joint_pos")]
)
eval(task, policy, embodiment, action_manager=manager)
```

Before each buffered action plays, the manager asks every trigger in turn.
When one returns a reason, the rest of the chunk is dropped, the policy is
called on the current observation with `extra["replan"]` set to the trigger
and reason, and the record gets a `replan` event. Built in:

- `UnreachableTarget`: the guard clamped an action of the current chunk to the
  action bounds, so the chunk is heading somewhere the embodiment cannot go.
  Speed-limit clamps alone do not count.
- `GripperSlip`: with an unchanged command, after `settle_steps`, a gripper's
  reading moves more than `threshold` toward its command, which is what a
  closed gripper does when the object leaves it. Absolute control only.

A trigger is any object with a `name` and
`should_replan(observation, t, store) -> str | None`; keep per-trial state in
`store` under a key of your own.

## Scoring

A [`Scorer`](../../src/inspect_robots/scorer.py) maps a recorded `Record` (+ the scene's
`Target`) to a [`Score`](../../src/inspect_robots/scorer.py). Because scorers consume the
*recorded* trajectory (not a live environment), scoring is reproducible from a
saved log. Across the `epochs` of a scene, an epoch reducer (`mean`, `max`,
`pass_at_k`, …) collapses scores; metrics then aggregate across scenes.

## Records, annotations and analysis

The log and each trial's record are raw data that nothing rewrites after the
run. Every record has a unique id and its own directory with the resolved
configuration and the produced action chunks (see
[Trial records](logging-and-rerun.md#trial-records)). Judgements made later,
such as a reviewer correcting a verdict, are
[annotations](../../src/inspect_robots/annotation.py) kept beside the log with
their provenance. [`rescore`](../../src/inspect_robots/analyzer.py) recomputes
results from them, and [analyzers](../../src/inspect_robots/analyzer.py) read
logs and annotations to produce summaries and pages. See
[Editing and analyzing records](editing-and-analysis.md).

## Errors and safety

The error taxonomy resolves the "fail fast vs never-crash-overnight" tension:

| Class | Policy |
|---|---|
| [`CompatibilityError`](../../src/inspect_robots/errors.py), `ConfigError` | fail fast, before any rollout |
| [`PolicyError`](../../src/inspect_robots/errors.py) | record the trial, then continue or halt per `fail_on_error` (`True` = first error, `0<x<1` = proportion, `x>1` = count), checked after every trial |
| [`EmbodimentFault`](../../src/inspect_robots/errors.py), [`SafetyAbort`](../../src/inspect_robots/errors.py) | **always halt**: a faulted/unsafe robot never auto-advances |

Failures inside a trial (including `reset`) are wrapped into the taxonomy; a
crashing guard becomes a `SafetyAbort` (it can no longer vouch for safety).
A non-finite action from the policy is a `PolicyError`. A non-finite action
introduced by a guard is a `SafetyAbort`.
Every error raised from inside a trial carries the partial
[`Record`](../../src/inspect_robots/rollout.py) on its `record` attribute, so the
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
