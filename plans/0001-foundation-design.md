# RoboInspect — Foundation Design (v2)

> **STATUS (implemented):** The full v0 build order in §10 (M0-lite through OSS
> hygiene + CI matrix) is **complete and green** — all modules, the registry +
> entry-point plugins, the CLI (`list`/`run`/`inspect`), Rerun support, and CI
> across Linux/macOS/Windows × py3.10–3.13 × numpy 1.x/2.x. `eval()`/`eval_set()`
> also accept registry-name strings. Plan 0002 (temporal-ensembling controller)
> is implemented. Still deferred to later plans: concrete sim/VLA adapters,
> vectorized envs, `eval_set` *resume* impl, offline trajectory rescoring
> (`roboinspect score`), and a docs site.

> **Goal:** RoboInspect is the "Inspect AI for robotics" — an open-source evaluation
> framework for **physical AI / VLA (Vision-Language-Action) models**. Define a
> robotics benchmark once and run *any* VLA policy against *any* compatible
> embodiment (real robot or simulator), with first-class logging to
> [Rerun](https://github.com/rerun-io/rerun) and reproducible structured logs.
>
> This repo is the *framework* ("Inspect AI"). Concrete benchmarks
> ("Inspect Evals for robotics") live in a separate repo and are **out of scope**.
> We ship only minimal reference evals (a mock world) to exercise and document it.

**v2 changelog:** rewritten after a 3-way critique loop (Inspect-fidelity,
robotics-domain, OSS-engineering lenses). The structural changes vs v1 — adopted
because they are expensive to retrofit and cheap to specify now:

- **Action chunking is core, not an extension.** Policies emit an `ActionChunk`
  (a horizon of actions played open-loop), because real VLAs (π0, ACT, diffusion
  policies) infer slower than the control rate.
- **A `Scene` dataset replaces `num_trials`.** A benchmark is a dataset of initial
  conditions / instructions / targets (the Inspect `Sample` analog), iterated
  over `epochs` repeats.
- **Action *semantics* are first-class** (control mode, rotation repr, gripper,
  frame, bounds) — `Box` alone cannot say what an action *means*.
- **An explicit error taxonomy** separates "record & continue" from "halt the
  eval" — required for safe unattended overnight runs.
- **Composable controllers** (the rollout middleware layer), **operator/VLM
  scorers**, **entry-point plugins**, **side-car binary logging**, and
  **`eval_set` resumption** are designed in (interfaces reserved even where
  implementation is deferred).

## 0. Design provenance & the central insight

Inspect AI decomposes LLM evals as `Task = Dataset[Sample] + solver + scorer`,
with a pluggable `Model`, an `eval()` entry point, an immutable `EvalLog`, a
registry/decorator extension model, and a great log viewer. The Dataset is the
spine; epochs handle repeats; reducers collapse epochs before metrics aggregate.

**The robotics twist — two orthogonal swappable inputs** (vs one for LLMs):

1. **The VLA / `Policy`** — the "brain". Maps observations + instruction →
   *action chunk*.
2. **The `Embodiment`** — the "body + world". A real robot or simulator. Produces
   observations, executes actions, owns the action/observation *spaces*,
   control rate, and reset/safety machinery.

A benchmark `Task` is defined *independently* of both. The framework: (a)
verifies a `(policy, embodiment)` pair is *compatible* (spaces + semantics +
camera/state key mapping), (b) runs the closed-loop rollout with open-loop chunk
execution, (c) scores recorded trajectories, (d) logs everything reproducibly.
This 2-input factoring is the central architectural commitment.

## 1. Scope of v0 (this foundation)

Ship the framework skeleton with the *hard-to-retrofit interfaces correct*:

- Core types & spaces with **action semantics**; `compat.check_compatibility`.
- `Policy` → `ActionChunk`; `Embodiment`; `Scene`/`SceneDataset`; `Task`.
- `Scorer` (+ `target`) / `Metric` / epoch **reducer** split; builtin scorers
  incl. an operator (human-in-the-loop) scorer stub.
- The **rollout engine** with open-loop chunk execution + a composable
  `Controller` middleware layer + a safety `Approver` gate; the `eval()` entry
  point producing an immutable, schema-versioned `EvalLog`.
- **Error taxonomy** (`CompatibilityError` fail-fast; `PolicyError`
  record-and-continue; `EmbodimentFault`/`SafetyAbort` halt) + circuit breaker.
- **Registry + decorators + entry-point discovery** (`@task/@policy/@embodiment/
  @scorer`), string resolution in `eval()`.
- **Mock world** (`CubePick`, deterministic, dependency-free) + scripted/random
  policies that exercise *chunked* execution — full stack tested in CI w/o
  hardware or sim.
- **Logging:** `JsonLogSink` (canonical, atomic writes, binary side-cars) +
  `RerunSink` (optional, lazy). `LogSink` protocol.
- **CLI** (`roboinspect list|run|inspect|score`), packaging (uv/hatchling +
  hatch-vcs), `mypy --strict`, ruff, pytest, GitHub Actions matrix, docs
  scaffold, and OSS hygiene (CONTRIBUTING, CoC, SECURITY, templates, CHANGELOG).

**Deferred (interfaces reserved, separate plans):** concrete sim adapters
(MuJoCo/ManiSkill/Isaac), concrete VLA adapters (OpenVLA/π0/Octo), vectorized
envs, temporal-ensembling controller, VLM success classifier, `eval_set`
resumption *implementation*, web results viewer, parallel sim sample execution.

## 2. Assumptions (made autonomously — flagged for confirmation)

- **Python 3.10+**; **NumPy** is the obs/action lingua franca; images are raw
  `(H,W,C) uint8` `np.ndarray`. **Torch is not a core dependency.**
- **Sync per-step control** (real robots are sequential). Concurrency across
  *scenes* (sim) and vectorized envs are reserved at the interface, not built.
- **Primary use case = real-world benchmarks.** The `Embodiment` interface
  assumes real-robot reality (no arbitrary-state reset guarantee, human-in-loop
  reset, wall-clock rate, no privileged success oracle); **sim is a stricter
  special case** exposing more via opt-in `capabilities`.
- **Policy owns observation preprocessing** (resize/normalize/history); the
  embodiment emits raw sensor frames. The framework only does **key remapping**
  (policy wants `base_rgb`, robot provides `camera_0`) and presence/space checks.
- **Conventions follow LeRobot** where sensible (camera keys, `observation.state`,
  `action`, fps) to make real-VLA adapters near-trivial.

## 3. Architecture

### 3.1 Module layout

```
roboinspect/
  __init__.py          # public API surface (__all__ fenced)
  types.py             # Observation, Action, ActionChunk, StepResult
  spaces.py            # Space/Box/Dict/Discrete + ActionSemantics + StateSpec
  policy.py            # Policy Protocol + PolicyBase ABC + PolicyConfig, PolicyInfo
  embodiment.py        # Embodiment Protocol + EmbodimentBase ABC + EmbodimentInfo
  scene.py             # Scene (Sample analog), SceneDataset, Target
  task.py              # Task (dataset + scorer + horizon + control_hz + epochs)
  scorer.py            # Scorer/Score/Metric protocols, reducers, builtins
  controller.py        # Controller (rollout middleware): chunk exec, ensembling
  approver.py          # Approver safety gate (auto/operator/policy)
  rollout.py           # rollout() closed loop + TrialRecord, RolloutResult
  eval.py              # eval()/eval_set(), EvalConfig, EvalLog (immutable)
  compat.py            # check_compatibility(policy, embodiment) -> report
  errors.py            # CompatibilityError, PolicyError, EmbodimentFault, SafetyAbort
  registry.py          # decorators + importlib.metadata entry-point discovery
  logging/
    sink.py            # LogSink protocol + event/transcript model
    json_log.py        # EvalLog read/write, schema_version, atomic, side-cars
    rerun_sink.py      # RerunSink (optional import; no-op if rerun missing)
  mock/
    cubepick.py        # CubePickEmbodiment (2D toy world, deterministic)
    policies.py        # ScriptedPolicy, RandomPolicy, NoopPolicy (chunk-aware)
  cli.py               # `roboinspect` CLI
  py.typed
```

Public API is fenced by `__all__`; everything else / `_`-prefixed is private with
no stability guarantee. An API-snapshot test guards accidental surface growth.

### 3.2 Core types (`types.py`)

```python
@dataclass(frozen=True)
class Observation:
    images: Mapping[str, np.ndarray]   # camera key -> (H,W,C) uint8 (raw)
    state: Mapping[str, np.ndarray]    # proprio: joint_pos, eef_pose, gripper...
    instruction: str | None            # language goal (may change across steps)
    image_times: Mapping[str, float]   # per-camera capture time (async sensors)
    state_time: float
    extra: Mapping[str, Any]

@dataclass(frozen=True)
class Action:
    data: np.ndarray
    semantics: ActionSemantics         # full semantics, not a bare string id
    meta: Mapping[str, Any]

@dataclass(frozen=True)
class ActionChunk:                     # VLAs predict H actions, played open-loop
    actions: Sequence[Action]
    control_hz: float | None           # rate to play them at (None = embodiment default)
    inference_latency_s: float | None  # measured; logged as a metric
    meta: Mapping[str, Any]
    # H==1 is the degenerate "reactive policy" case.

@dataclass(frozen=True)
class StepResult:
    observation: Observation           # AFTER applying the action
    reward: float | None               # optional (sims)
    terminated: bool                   # task ended...
    termination_reason: str | None     # ...success | collision | fault | out_of_bounds
    truncated: bool                    # horizon/time cut-off
    info: Mapping[str, Any]            # sims may put privileged success here
```

Frames are large: `TrialRecord` keeps **references** to images streamed to disk
(per-camera mp4 or `.rrd`), not in-RAM arrays, so 1000-step multi-camera
episodes don't blow up memory. The `info` channel is the documented place a sim
puts privileged success.

### 3.3 Spaces & semantics (`spaces.py`)

`Box/Dict/Discrete` describe shapes; **`ActionSemantics`** describes *meaning*
and is required on action spaces:

```python
@dataclass(frozen=True)
class ActionSemantics:
    control_mode: Literal["joint_pos","joint_vel","eef_delta_pose",
                          "eef_abs_pose","eef_delta_pos"]
    rotation_repr: Literal["none","quat_wxyz","quat_xyzw","rot6d",
                           "axis_angle","euler_xyz"]
    gripper: Literal["none","continuous","binary"]
    frame: Literal["base","world","camera"]
    bounds: tuple[np.ndarray, np.ndarray] | None
```

Semantics make compatibility checking real (7-DoF VLA vs 6-DoF arm; delta vs
absolute) and make temporal ensembling *correct* (you cannot average absolute and
delta poses the same way; quaternion/euler averaging needs the repr). `StateSpec`
gives proprioception a controlled vocabulary + units (radians, meters,
normalized gripper) so `state` compatibility isn't illusory.

`compat.check_compatibility(policy, embodiment)` returns a structured report:
hard mismatches (raise `CompatibilityError` before any rollout), soft warnings
(missing optional camera), and the resolved **key remap** (policy ↔ embodiment
camera/state names). It also reconciles `Policy` desired vs `Embodiment` actual
**control rate**.

### 3.4 The two inputs (`policy.py`, `embodiment.py`)

```python
@runtime_checkable
class Policy(Protocol):                # the VLA / "brain"
    info: PolicyInfo                   # name, emitted ActionSemantics, required obs keys
    config: PolicyConfig               # temperature, action_horizon, replan, ensemble — logged
    def reset(self, scene: Scene) -> None: ...
    def act(self, observation: Observation) -> ActionChunk: ...
    # reserved for later: act_batch(list[Observation]) -> list[ActionChunk]

@runtime_checkable
class Embodiment(Protocol):            # the robot or sim / "body + world"
    info: EmbodimentInfo               # spaces, control_hz, is_simulated, capabilities
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def close(self) -> None: ...
    # sim-only (gated by capabilities): render(), set_state(), privileged_state()
```

Both ship as runtime-checkable **Protocols** (wrap existing envs without
inheriting) *plus* optional `PolicyBase`/`EmbodimentBase` ABCs with sane defaults
(`close()` no-op, capability defaults). `capabilities` is an opt-in flag set
(`"seedable"`, `"resettable"`, `"privileged_success"`, `"renderable"`,
`"auto_reset"`). On real hardware `reset()` may drive to home and **block on a
logged human-confirmation prompt**; auto-reset is a sim capability, not assumed.

### 3.5 Scenes & tasks (`scene.py`, `task.py`)

A `Scene` is the **Sample analog** — one initial condition of a benchmark:

```python
@dataclass(frozen=True)
class Scene:
    id: str
    instruction: str                   # per-scene language goal
    target: Target | None              # success spec the scorer reads (goal obj/pose...)
    init_seed: int | None
    setup: str | None                  # registered setup hook name (serializable!)
    subtasks: Sequence[Subtask] = ()   # long-horizon (CALVIN-style) chained goals
    metadata: Mapping[str, Any] = ...

class SceneDataset(Protocol):          # iterable/filterable/sliceable/shuffleable
    def __iter__(self) -> Iterator[Scene]: ...
    def __len__(self) -> int: ...

@dataclass
class Task:
    name: str
    scenes: SceneDataset               # the spine — e.g. 50 object layouts
    scorer: Scorer | Sequence[Scorer]  # normalized to a list internally
    max_steps: int                     # truncation horizon
    epochs: int = 1                    # repeats per scene (was num_trials)
    epoch_reducer: Reducer = mean      # collapse epochs before metrics
    control_hz: float | None = None
    metadata: Mapping[str, Any] = ...
```

A `Task` is embodiment-agnostic. Per-scene `instruction`/`target`/`seed` and
optional `subtasks` (instruction changes mid-episode; scored sequentially) make
real benchmarks (CALVIN/LIBERO/SIMPLER-style) representable. `setup` is a
*registered hook name*, not a raw callable, so it serializes into the log.

### 3.6 Scoring, metrics, reducers (`scorer.py`)

Mirrors Inspect's `@scorer`/`@metric` split and adds an explicit epoch-reducer
stage:

```python
@dataclass(frozen=True)
class Score:
    value: float | bool | str
    explanation: str | None
    metadata: Mapping[str, Any]

class Scorer(Protocol):
    def __call__(self, record: TrialRecord, target: Target | None) -> Score: ...

# epoch reducers (collapse `epochs` per scene): mean, mode, max, pass_at_k
# metrics (aggregate per-scene scores): mean, stderr, success_rate, ...
```

Scorers consume the **recorded trajectory** (`TrialRecord`) + the scene `target`,
never live env access — so scoring is reproducible from a saved log and an
offline `roboinspect score <log>` can re-run scorers (the achievable analog of
Inspect's generation cache). Builtins:

- `success_at_end`, `reached_goal_state`, `min_distance_to_goal`,
  `episode_length`, `progress`/`partial_success` (subtask fraction — key for
  long-horizon), `intervention_rate`, `composite([...])`.
- `OperatorScorer` — **human-in-the-loop**: at episode end, blocks and records
  the operator's success/partial judgement (the dominant real-world method).
- `VLMScorer` — interface reserved: runs a VLM classifier over final frames.

Reducer validity is typed: `mean`/`stderr` over a `str` value raises a clear
error rather than silently coercing.

### 3.7 Rollout: open-loop chunk execution + middleware (`rollout.py`, `controller.py`, `approver.py`)

The heart of the framework. Two nested loops — *inference* (slow VLA) and
*execution* (fast control), connecting a single `act()` to many `step()`s:

```
policy.reset(scene); obs = embodiment.reset(scene, seed=scene.init_seed)
while not done and step_idx < max_steps:
    chunk = controller.infer(policy, obs)        # VLA inference; record latency
    for action in controller.actions_to_execute(chunk):   # open-loop horizon
        action = approver.review(action, state)  # SAFETY GATE before hardware
        result = embodiment.step(action)         # robot/sim executes
        sink.log_step(step_idx, obs, action, result)
        record.append(...); step_idx += 1
        if result.terminated or result.truncated: done = True; break
        obs = result.observation
        pace_to(embodiment control rate)         # paced by EMBODIMENT, not wish
```

- **`Controller`** is the composable middleware layer (Inspect's `@solver`
  analog): default executes the first `replan_interval` actions of each chunk;
  an `EnsemblingController` (deferred) blends overlapping chunks using
  `ActionSemantics`. Observation preprocessing/history, action smoothing, and
  retry-on-stuck are controllers too — so the loop is never forked.
- **`Approver`** intercepts every action before `step()` (auto-clamp / operator
  / policy), logged as events — the robotics analog of Inspect's `ApprovalPolicy`,
  and more safety-critical.
- **Error handling:** a `PolicyError` is recorded as a failed trial and the eval
  continues; an `EmbodimentFault`/`SafetyAbort` **halts the whole eval** (a
  faulted robot must not auto-proceed to scene 2 overnight). A
  `max_consecutive_failures` circuit breaker and `--fail-fast` are config.
- **Transcript:** `TrialRecord` carries a typed **event stream** (reset, infer,
  step, approval decision, intervention, error+traceback, scorer outcome) — the
  data a viewer renders.

### 3.8 eval(), eval_set(), EvalLog (`eval.py`)

```python
def eval(task: str | Task, policy: str | Policy, embodiment: str | Embodiment,
         *, log_dir="logs", sinks=None, seed=None, fail_fast=False,
         max_consecutive_failures=None) -> EvalLog: ...

def eval_set(...) -> EvalLog:          # idempotent; resumes a partial run by
                                       # skipping completed (scene, epoch) keys
```

`eval()` accepts **registry strings** (`policy="openvla/7b"`,
`embodiment="cubepick"`) resolved through the registry with `key=value` args
(CLI `-P k=v`), recording the *resolved spec + args* for reproducibility — the
Inspect ergonomic that makes logs re-runnable. Flow: resolve → compat check
(fail fast) → for each scene × epoch run `rollout()` → score with `target` →
epoch-reduce → aggregate metrics → write immutable `EvalLog`.

`eval_set()` makes overnight runs resumable: a stable run id + per-`(scene,epoch)`
status (`success|error|incomplete`) means a crashed 6-hour run re-invokes and
skips finished work. (Interface in v0; full resume impl is a follow-up plan.)

### 3.9 Logging (`logging/`) & EvalLog schema

`LogSink` protocol: `on_eval_start / on_trial_start / log_step / on_trial_end /
on_eval_end`. Builtins:

- **`JsonLogSink`** (always on, canonical). JSON holds metadata + references;
  large binaries (images→mp4, `.rrd`) are **side-car files** on disk. Writes are
  **atomic** (temp + rename) so an overnight crash never leaves a half log.
- **`RerunSink`** (optional, lazy import). Logs camera images, 3D eef poses,
  joint/action time-series, approval & success markers to a `.rrd`. If
  `rerun-sdk` is absent, warns once and no-ops; `mock/` never imports it.

`EvalLog` (immutable, `schema_version`ed) mirrors Inspect: header (resolved
config, policy/embodiment info + `capabilities`, `PolicyConfig`, git rev, package
versions, timestamps, `EvalStats` incl. actual control rate & inference
latency), per-scene results with `status` + structured `error`/traceback +
`reductions`, aggregate metrics. **Read-back guarantee:** newer RoboInspect always
reads older logs; a JSON Schema / pydantic model + golden-file round-trip test
enforce it.

### 3.10 Registry, plugins, CLI (`registry.py`, `cli.py`)

`@task/@policy/@embodiment/@scorer/@sink` register factories by name. **Out-of-
tree discovery** uses `importlib.metadata` entry-point groups —
`roboinspect.policies`, `roboinspect.embodiments`, `roboinspect.tasks`, `roboinspect.scorers`,
`roboinspect.sinks` — so an installed `roboinspect-maniskill` / `roboinspect-openvla`
package appears in `roboinspect list` without being imported first. CLI: `roboinspect
list [...]`, `roboinspect run --task X --policy Y -P k=v --embodiment Z`, `roboinspect
inspect <log>` (terminal results table — distinct from Rerun trajectory viz),
`roboinspect score <log>` (offline re-scoring).

## 4. Errors, safety & the overnight tension (`errors.py`)

The "fail-fast vs never-crash-overnight" tension, resolved explicitly:

| Class | When | Policy |
|---|---|---|
| `CompatibilityError` | before any rollout | **fail fast**, abort eval |
| `ConfigError` | bad task/registry args | **fail fast** |
| `PolicyError` | VLA raised at inference | record trial `status=error`, **continue** |
| `EmbodimentFault` | robot/sim hardware fault | **halt eval**, require human |
| `SafetyAbort` | approver veto / e-stop | **halt eval**, require human |

Plus `max_consecutive_failures` circuit breaker and `--fail-fast` for dev. A
faulted robot never auto-advances unattended.

## 5. Testing (pytest, TDD)

- **Unit:** spaces + semantics + compat (match/mismatch/subset cameras/key
  remap/rate reconcile); type immutability; scorers on synthetic records incl.
  `target` and subtask progress; reducer type-validity; registry + entry-point
  discovery (fake dist); json-log round-trip + **golden schema** + atomic write.
- **Integration:** full `eval()` on `CubePick` with `ScriptedPolicy` (deterministic
  success) and `RandomPolicy` (mostly failure) — **exercising chunked execution
  (H>1) and replanning** — assert success/metric values, log structure, status
  taxonomy, circuit breaker.
- **Logging:** `RerunSink` behind a skip-guard; a fake sink asserts hook
  ordering + event transcript.
- **Determinism:** scoped to the **mock world only** — same seed ⇒ identical
  `EvalLog` (modulo timestamps). Real-sim determinism is documented as
  backend-dependent/best-effort, *not* a CI gate.
- **Boundary:** a CI job imports core in a **rerun/torch-free** env to enforce the
  dependency boundary.
- **Markers:** `@pytest.mark.hardware/sim/slow` default-deselected in CI.
- **Matrix:** Python 3.10–3.13 × {Linux, macOS, Windows}; numpy 1.x & 2.x floors.
- `ruff` + `mypy --strict` + `pytest` (coverage gate) all green before merge.
  `dict[str, Any]` escape hatches (`info`/`extra`/`meta`) are the deliberate
  typing boundary.

## 6. OSS hygiene & release

- `pyproject.toml` (hatchling + **hatch-vcs** single-source version from git
  tags). Extras: `roboinspect[rerun]`, `[viz]`, `[dev]`, `[all]`; **core depends
  only on numpy + stdlib**.
- SemVer (honest `0.x` = breaking allowed on minor pre-1.0), `CHANGELOG.md` (Keep
  a Changelog), release via tag → CI → PyPI **trusted publishing (OIDC)**.
- `README`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, `SECURITY.md`, issue/PR templates,
  MIT license (already present). Docs: **mkdocs-material** + mkdocstrings (renders
  the typed API), with a "write your first benchmark" tutorial.

## 7. Milestones (each = its own commit/push; plan-per-feature as repo grows)

- **M0** Packaging, CI matrix, OSS hygiene, core-only import test, repo skeleton.
- **M1** `types`, `spaces`+semantics, `errors`, `compat` + tests.
- **M2** `Policy`/`Embodiment` (Protocol+ABC), `Scene`/`SceneDataset`, `Task`,
  `Scorer`/reducers + mock `CubePick` + chunk-aware mock policies + tests.
- **M3** `controller` + `approver` + `rollout` (chunk exec, error taxonomy,
  transcript) + integration tests.
- **M4** `eval()` + `JsonLogSink` + `EvalLog` schema + golden tests; `eval_set`
  interface.
- **M5** `registry` + entry-point discovery + CLI (`list/run/inspect/score`).
- **M6** `RerunSink` + viz docs.
- **M7** Docs site + "write your first benchmark" tutorial.

Later plans (separate files): sim adapter, real VLA adapter, ensembling
controller, vectorized envs, `eval_set` resume impl, web results viewer.

## 8. Remaining open questions (track, don't block)

1. Batched `act_batch`/`VectorEmbodiment` exact signature (reserved, unbuilt).
2. `eval_set` resume granularity (scene vs epoch) & on-disk run-state format.
3. Operator-scorer UX (terminal prompt now; richer later).
4. Whether `Scene` should be the user-facing name vs `Sample` (chose `Scene` for
   robotics intuition; documented as the Inspect Sample analog).

## 9. v3 critique resolutions (BINDING — overrides earlier sections on conflict)

A second critique round (design-verification + scope lenses) found the remaining
issues clustered in the rollout↔controller↔record triad. These resolutions are
**binding** and supersede the prose above wherever they conflict:

- **R1 — `step()` timing contract.** `Embodiment.step()` **returns as soon as the
  command is issued** (it does *not* block until the control period elapses).
  **Reversed 2026-07-14** (see issue tracker: rollout pacing was accepted-but-ignored
  in practice — `_effective_control_hz`/`Task.control_hz` were dead code, and both
  hardware adapters already self-paced as a workaround). The rollout applies **no
  wall-clock pacing of its own**; an embodiment that needs real-time cadence paces
  itself inside `step()` and declares capability `"self_paced"` to document that it
  does. `Task.control_hz` is removed — the framework never had a rate of its own to
  resolve. `compat` still warns when the policy's emitted rate and the embodiment's
  native rate differ by more than a tolerance (unaffected by this reversal).

  Original resolution (superseded above): The **framework owns pacing** to the
  *effective rate* = `chunk.control_hz` → `task.control_hz` →
  `embodiment.info.control_hz` (first non-None). A real-robot adapter whose
  `step()` inherently blocks at the hardware rate declares capability
  `"self_paced"`, and the rollout then skips its own pacing. `compat` warns when
  the policy's emitted rate and the embodiment's native rate differ by more than a
  tolerance.

- **R2 — per-trial seeding.** The seed for scene `s`, epoch `e` is
  `derive_seed(eval_seed, s.init_seed, e)` (a stable hash mix), logged per trial.
  An integration test asserts that epochs of a *stochastic* policy diverge while a
  deterministic policy + fixed seed reproduce bitwise. (Fixes degenerate epochs.)

- **R3 — `Controller` is a single-method, stateful middleware.** Replace the
  two-method shape with:
  `Controller.next_action(policy, obs, t, store) -> Action`. The controller
  *internally* decides when to call `policy.act()` (buffering the returned chunk),
  pops the next action, and — for `EnsemblingController` (deferred) — re-infers
  every step and blends overlapping predictions using the action space's
  semantics. The rollout becomes **one control-rate loop** calling `next_action`
  each step; `replan_interval` and temporal ensembling are controller-internal
  state. `DefaultController` = play `replan_interval` actions of each chunk, then
  re-infer. This is what lets advanced controllers compose without forking the loop.

- **R4 — preprocessing ownership (resolves §2↔§3.7 conflict).** The **policy**
  owns *model-specific spatial* preprocessing (resize / normalize / color order /
  key remap to its native names). The **controller** owns *cross-policy temporal*
  concerns (observation history stacking, action smoothing, ensembling). The
  **embodiment** emits raw sensor frames. No responsibility is claimed twice.

- **R5 — `FrameStore` owns frames, not sinks.** The rollout owns a `FrameStore`
  that streams camera frames to disk (per-camera mp4 / image files) and returns
  lightweight refs. `TrialRecord` stores **only refs** (memory-safe). `JsonLogSink`
  references the same store; `RerunSink` optionally also logs frames. Scorers read
  frames *through the refs*, so scoring is sink-independent and reproducible even
  with all optional sinks off.

- **R6 — operator judgement is a recorded rollout event, not a live scorer.** The
  human success/partial judgement is captured **once during rollout** (an
  `OperatorPrompt` event in `TrialRecord`). The `OperatorScorer` merely **reads**
  that recorded judgement — preserving the "scorers reproducible / offline
  re-runnable from a log" guarantee. In v0 the operator prompt is a **non-blocking
  stub** (no blocking terminal UX; CI-safe, unattended-safe).

- **R7 — scene realizability in `compat`.** An `Embodiment` declares the setup
  hooks and target kinds it supports; `check_compatibility` verifies the task's
  scenes are realizable on that embodiment (a scene's `setup`/`target` resolve in
  the *embodiment's* namespace). Keeps `Task` embodiment-agnostic while catching
  "this scene can't run on this robot" before the rollout.

- **R8 — semantics live on the space, not every `Action`.** `Action` carries
  `data` + `meta` only; `ActionSemantics` lives on the (single) action space known
  to the rollout/controller. Removes per-step redundancy and drift.

- **R9 — namespace reducers vs metrics.** `reducers.mean`/`reducers.pass_at_k`
  (signature `list[Score] -> Score`) are distinct from `metrics.mean`/`stderr`
  (`list[float] -> float`); register under separate namespaces to avoid collision.

- **R10 — YAGNI trims for v0.** `subtasks`/`Subtask`, `progress`/`partial_success`,
  `min_distance_to_goal`, `intervention_rate`, `composite`, and `VLMScorer` are
  **fields/interfaces reserved but unimplemented** in v0 (nothing in the mock
  exercises them). `image_times`/`state_time` remain as fields.

## 10. v0 build order (BINDING — tracer-first, always-green)

Replaces §7's bottom-up milestones with a vertical tracer + thickening, so a
runnable end-to-end demo exists by the *second* commit and the repo stays
shippable all night. Each step is its own commit/push.

1. **M0-lite** — repo skeleton, `pyproject.toml` (hatchling + hatch-vcs),
   ruff + `mypy --strict` + pytest config, `py.typed`, **one Linux CI job** green,
   a **core-only (rerun/torch-free) import test**. *(Full OS/Python/numpy matrix
   deferred to the end as allow-failure.)*
2. **Tracer slice** — `eval(CubePick, ScriptedPolicy)` → one scene → **chunked
   (H>1) open-loop rollout** routed through `DefaultController.next_action` +
   `AutoApprover.review` (trivial pass-throughs, but the seams exist) → scorer
   `success_at_end` + `reducers.mean` → `JsonLogSink` with `schema_version` +
   atomic write. One integration test asserts success==1.0 and log round-trips.
   **From here the repo is always demoable.**
3. **Types/spaces thickening** — full `ActionSemantics` + `StateSpec`, immutability
   tests.
4. **`compat.check_compatibility`** + `CompatibilityError` fail-fast + scene
   realizability (R7) + rate reconcile (R1) + tests.
5. **Scorers** — builtin set, reducer type-validity, `OperatorScorer` reading a
   recorded event (R6, non-blocking stub), reserved interfaces (R10).
6. **Rollout hardening** — `Approver` gate logic, full error taxonomy
   (`PolicyError` continue vs `EmbodimentFault`/`SafetyAbort` halt), circuit
   breaker, transcript event stream + fake-sink hook-ordering test, per-trial
   seeding (R2), `FrameStore` (R5).
7. **Controller middleware** — promote the seam into the real `Controller`
   (`replan_interval`); pure refactor, tests stay green.
8. **EvalLog hardening** — `schema_version` golden read-back, binary side-car refs,
   `EvalStats` (control rate, inference latency), `eval_set` **signature only**.
9. **Registry + entry-point discovery + CLI `list`/`run`** (`-P k=v` string
   resolution). *(CLI `inspect`/`score` deferred.)*
10. **`RerunSink`** (optional, skip-guarded, no-op if missing) — leaf, low risk.
11. **OSS hygiene polish + expand CI matrix LAST** as `continue-on-error`
    (Linux+macOS 3.11/3.12 stay the blocking gate; Windows + 3.10/3.13 + numpy
    1.x/2.x added allow-failure). Docs site + tutorial deferred to a later plan.

## 11. Inspect AI grounding (context7-verified — BINDING for API fidelity)

Verified against Inspect AI's live reference (`/websites/inspect_aisi_uk`).
RoboInspect mirrors these so Inspect users feel at home:

- **`eval()` returns `list[EvalLog]`** (one per task), accepting one task or many.
  `eval_set()` returns `tuple[bool, list[EvalLog]]` (success flag + logs).
- **`epochs: int | Epochs`** where `Epochs(count, reducer)` bundles the repeat
  count with the score reducer (default `"mean"`). Reducers registered via a
  `@score_reducer` decorator (builtins: `mean`, `median`, `mode`, `max`,
  `pass_at_k`/`at_least`).
- **`fail_on_error: bool | float`** is the canonical error-tolerance knob —
  `True`=fail on first error, `False`=never fail, `0<x<1`=fail if that *proportion*
  of trials error, `x>1`=fail if that *count* error. RoboInspect adopts this verbatim
  for `PolicyError`-class failures. **Robotics addition:** `EmbodimentFault` /
  `SafetyAbort` always halt regardless of `fail_on_error` (hardware safety is not
  negotiable) — this is the one place RoboInspect deliberately extends Inspect.
- **`EvalLog` structure** mirrored: `version` (int), `status`
  (`"started"|"success"|"error"`), `eval` (an `EvalSpec`: task/policy/embodiment/
  created/git/versions), `plan` (resolved config incl. `PolicyConfig`,
  controller, approver), `results` (`EvalResults`: aggregate metrics), `stats`
  (`EvalStats`: timing, actual control rate, inference latency), `error`
  (`EvalError` w/ traceback), `tags`, `metadata`, `samples` (per-scene
  `EvalSample`: scene input/target/score/status/transcript), `reductions`
  (per-scene epoch reductions). `samples` is named `scenes`-internally but keeps
  the `EvalSample` shape.
- **`Scene` ≈ `Sample`** field mapping confirmed: Inspect `Sample(input, target,
  id, metadata, setup, files, sandbox)` → RoboInspect `Scene(instruction[=input],
  target, id, metadata, setup, init_seed, subtasks)`. (`files`/`sandbox` are
  LLM-sandbox-specific; the robotics analog is embodiment setup/realizability.)
- **`@scorer(metrics=...)`** returns `score(state, target) -> Score`; RoboInspect
  uses `score(record, target) -> Score` (record replaces the LLM `TaskState`).
- **Per-scene `cleanup` hook** (Inspect `Task.cleanup`) added for embodiment
  teardown after each scene, run even on exception.
```
