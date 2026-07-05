# `inspect_robots` package — module map

Read `plans/0001-foundation-design.md` (§9–§11 are binding) before changing core
interfaces. The package is `mypy --strict` clean and ships `py.typed`.

## Modules

| Module | Responsibility |
|--------|----------------|
| `types.py` | `Observation`, `Action`, `ActionChunk`, `StepResult` (frozen, NumPy-native) |
| `spaces.py` | `Box`, `ObservationSpace`, `ActionSemantics`, `StateSpec` + canonical state vocab |
| `policy.py` | `Policy` Protocol + `PolicyBase` ABC, `PolicyInfo`, `PolicyConfig` |
| `embodiment.py` | `Embodiment` Protocol + `EmbodimentBase` ABC, `EmbodimentInfo`, capability flags |
| `scene.py` | `Scene` (the Inspect `Sample` analog), `Target`, `ListSceneDataset` |
| `task.py` | `Task` (scenes + scorer + horizon), `Epochs` |
| `scorer.py` | `Score`/`Scorer`, epoch reducers, builtin scorers (incl. operator/VLM) |
| `controller.py` | `Controller` middleware: `DefaultController` (open-loop chunking), `SmoothingController` |
| `approver.py` | `Approver` safety gate: `AutoApprover`, `ClampApprover` |
| `rollout.py` | `rollout()` closed loop, `TrialRecord`/`StepRecord`, per-trial seeding |
| `frames.py` | `FrameStore`/`FrameRef` — stream camera frames to disk (R5) |
| `transcript.py` | typed event stream (reset/inference/step/approval/operator/error) |
| `compat.py` | `check_compatibility`/`assert_compatible` — fail-fast before rollout |
| `errors.py` | error taxonomy (continue vs halt) |
| `eval.py` | `eval()` / `eval_set()` orchestration |
| `log.py` | immutable, schema-versioned `EvalLog` + `read_eval_log` |
| `logging/` | `LogSink` protocol, `JsonLogSink` (atomic), optional `RerunSink` |
| `registry.py` | decorators + entry-point discovery; `_builtins.py` registers in-tree components |
| `cli.py` | `inspect-robots list` / `inspect-robots run` / `inspect-robots inspect` |
| `mock/` | dependency-free `CubePick` world + scripted/random/noop policies |

## Key invariants

- The rollout loop is **one control-rate loop** calling `Controller.next_action`;
  inference/replanning is controller-internal (so ensembling composes — R3).
- Frames live in a rollout-owned `FrameStore`, never in a sink (R5).
- Action *semantics* live on the action `Box`, not on every `Action` (R8).
- Generic policy/embodiment exceptions are wrapped into `PolicyError` /
  `EmbodimentFault`; `SafetyAbort`/`EmbodimentFault` always halt the eval.
- `mock/` and core must never import `rerun`/`torch` at module top.
