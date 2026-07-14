# `inspect_robots` package — module map

Read `plans/0001-foundation-design.md` (§9–§11 are binding) before changing core
interfaces. The package is `mypy --strict` clean and ships `py.typed`.

## Modules

| Module | Responsibility |
|--------|----------------|
| `types.py` | `Observation`, `Action`, `ActionChunk`, `StepResult` (frozen, NumPy-native) |
| `spaces.py` | `Box`, `ObservationSpace`, `ActionSemantics`, `StateSpec` + canonical state vocab |
| `policy.py` | `Policy` Protocol + `PolicyBase` ABC, `PolicyInfo`, `PolicyConfig`; optional duck-typed `bind(embodiment_info)` hook for embodiment-adaptive policies (called by `eval()` before compat) |
| `embodiment.py` | `Embodiment` Protocol + `EmbodimentBase` ABC, `EmbodimentInfo`, capability flags |
| `scene.py` | `Scene` (the Inspect `Sample` analog), `Target`, `ListSceneDataset` |
| `task.py` | `Task` (scenes + scorer + horizon), `Epochs` |
| `scorer.py` | `Score`/`Scorer`, epoch reducers, builtin scorers (incl. operator/VLM) |
| `controller.py` | `Controller` middleware: `DefaultController` (open-loop chunking), `SmoothingController` |
| `approver.py` | `Approver` safety gate: `AutoApprover`, `ClampApprover`, `DeltaLimitApprover` (semantics-aware no-wild-swings limit), `ChainApprover` |
| `rollout.py` | `rollout()` closed loop, `TrialRecord`/`StepRecord`, per-trial seeding; honors a policy-requested stop via pre-review `action.meta["request_stop"]` (truncation; embodiment termination wins; not preserved under ensembling) |
| `frames.py` | `FrameStore`/`FrameRef` — stream camera frames to disk (R5) |
| `transcript.py` | typed event stream (reset/inference/step/approval/operator/error) |
| `compat.py` | `check_compatibility`/`assert_compatible` — fail-fast before rollout |
| `conformance.py` | adapter conformance kit: `check_embodiment`/`assert_embodiment_conformant` for declarative guardrail/agent readiness; `missing_runtime_requirements` provides runtime-dependency preflight; `DeviceSlot`/`device_slots` declare and defensively read embodiment device slots |
| `errors.py` | error taxonomy (continue vs halt) |
| `eval.py` | `eval()` / `eval_set()` orchestration |
| `log.py` | immutable, schema-versioned `EvalLog` + `read_eval_log` |
| `logging/` | `LogSink` protocol, `JsonLogSink` (atomic), optional `RerunSink` |
| `registry.py` | decorators + entry-point discovery; `_builtins.py` registers in-tree components |
| `cli.py` | `inspect-robots list` / `run` / `inspect` / `config set|show` / `setup` (first-run wizard) / `doctor` (adapter conformance), plus the zero-config form `inspect-robots "<instruction>"` (ad-hoc single-scene task; operator prompt on TTY). Every run wires guardrails (Clamp + DeltaLimit) by default; `--disable-guardrails` is the loud opt-out and the chain degrades per component with stderr warnings |
| `_defaults.py` | user default policy/embodiment (+ `--sim` counterpart) for the zero-config CLI: env vars > `~/.config/inspect-robots/config.ini` (INI — py3.10 has no tomllib; deliberately no project-local file); `set_default` backs `config set` |
| `_setup.py` | the `inspect-robots setup` wizard (plans 0009 and 0011): IO-injected prompts for `[defaults]`, plugin-declared V4L2/CAN/serial device slots with unplug-to-identify and CAN udev guidance, fallback camera discovery, headless-rerun warning; renders config.ini itself (comments survive) and carries unmanaged sections/keys through raw |
| `mock/` | dependency-free `CubePick` world + scripted/random/noop policies |

## Key invariants

- The rollout loop is **one control-rate loop** calling `Controller.next_action`;
  inference/replanning is controller-internal (so ensembling composes — R3).
- Frames live in a rollout-owned `FrameStore`, never in a sink (R5).
- Action *semantics* live on the action `Box`, not on every `Action` (R8).
- Generic policy/embodiment exceptions (incl. from `reset`) are wrapped into
  `PolicyError` / `EmbodimentFault`; a crashing approver becomes `SafetyAbort`;
  `SafetyAbort`/`EmbodimentFault` always halt the eval. Every error raised from
  inside a trial carries the partial `TrialRecord` on `exc.record`.
- `eval()` must always return/persist an `EvalLog` once rollouts have started —
  scorer/reducer failures degrade to an error log, never a crash. Errored
  trials are recorded (and delivered to sinks) but **never scored**.
- `eval()` closes embodiments it resolved from registry names ("close what we
  open"); caller-constructed objects are caller-owned.
- `mock/` and core must never import `rerun`/`torch` at module top.
