# Parallel simulation evaluation sets

Issue: https://github.com/robocurve/inspect-robots/issues/361

## Scope

Add `max_workers=1` to `eval_set` and `--max-workers` to `eval-set`.
The existing sequential path remains the default. Parallel execution is
opt-in and restricted to independent simulated tasks. It overlaps inference
waits and native work; it does not promise CPU-bound Python speedups.

## Ownership and dispatch

A bounded thread executor admits at most N tasks, collecting completed work
before admitting replacements. Results retain input order. Ordinary evaluation
errors retain the existing per-task error-log behavior. Escaping halt exceptions
and interrupts stop admission; active tasks finish and close before propagation.
No executor shutdown timeout or forced thread cancellation is promised.

The Python parallel path accepts registry names only. Each task constructs its
own policy, embodiment, grader, controller and default JSON sink. The CLI reuses
its existing configuration and lifecycle code per task, including live logs,
guardrails and epochs. Registry discovery completes before dispatch. Embodiments
must declare `is_simulated`; this is checked before reset. Factories must return
fresh objects and support concurrent independent instances. This does not
isolate plugin globals or external simulator sessions.

Caller-owned objects and mutable hooks are rejected in the parallel Python API.
The CLI requires `--no-prompt` and rejects voice input. Policies with an optional
`close()` hook are released, including when embodiment construction fails.
Embodiments and CLI claims are released through their existing ownership paths.
Every owned cleanup callback runs even if another close fails. Cleanup failures
during an escaping safety halt or interrupt are reported as warnings and never
replace the original halt. Grader construction and preflight happen before each
task opens its components; grading configuration errors stop task admission.
Existing run directories, log schema and seed derivation are retained.

## Verification

Use synchronization barriers to prove overlap and the concurrency bound without
wall-clock assertions. Exercise ordering, ordinary failures, escaping halts,
cleanup on setup failure, real-embodiment rejection before reset, CLI options,
sequential/parallel result equivalence and independent frame/action sidecars.
Run existing sequential lifecycle tests and the full lint, typing and coverage gates.

## Follow-ups

Process execution for CPU-bound Python workloads, cooperative cancellation of
active rollouts, real-device scheduling, caller-provided component factories,
and partial-run resumption need separate contracts. None is required for this
opt-in simulation execution mode. Existing probabilistic run naming is unchanged;
collision-proof run identity remains the separate work in PR #435.
