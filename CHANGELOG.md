# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is
`0.x`, breaking changes may occur on any minor release.

## [Unreleased]


### Added

- **Policy lifecycle hook: `on_trial_end`** — policies can now hook into the end of a trial to persist state or artifacts. The orchestrator calls `policy.on_trial_end(record, log_dir, run_id)` and any metadata the policy attaches to `record.metadata` is persisted in the final `EvalLog`. Hook failures are caught and logged as trial errors, preventing them from crashing the overall evaluation.
- **Agent plugin transcript persistence** — `LLMAgentPolicy` now implements `on_trial_end` to persist its full conversation transcript (tool calls, observations, system prompts) to a JSONL file per trial under `<log-dir>/transcripts/<run_id>/<scene_id>-e<epoch>.jsonl`. Camera images are stripped from the transcript to save space, as they are already recorded in the frame store. The relative path to the transcript is stored in the trial's metadata for easy post-hoc analysis.
- Live agent-policy transcript rows on the Rerun `step` timeline, with
  best-effort non-blocking streaming and complete eval-log persistence (#124).
- Remote Rerun streaming via `inspect-robots run --rerun-connect [URL]`, so
  headless evaluations can connect over gRPC to a viewer on another machine
  (including through an SSH reverse tunnel) (#86).
- Plugin-declared embodiment device slots for V4L2 cameras, SocketCAN
  interfaces, and serial devices. `inspect-robots setup` probes and interviews
  declared slots, enforces grouped all-or-none assignments, and suggests udev
  serial pinning for order-dependent USB-CAN names (#61).
- Runtime-requirement declarations for registered component factories, with
  missing-import preflight checklists in `inspect-robots setup` and
  `inspect-robots doctor` (#59).
- isaacsim plugin: `_ensure_env`'s cfg-wiring contract (`parse_env_cfg`'s
  args, `gym.make(cfg=...)`, the `headless` → `_disable_debug_vis` gate, and
  the named-obs-terms request) is now exercised in CI via stubbed
  `gymnasium`/`isaaclab_tasks` modules. Previously only the fake-env-injected
  `step()`/`reset()` translation was covered, so a regression in `_ensure_env`
  itself (e.g. #15's missing `cfg=`) would only have failed live (#25).
- **`inspect-robots setup`**: an interactive first-run wizard that prompts
  for the `[defaults]` keys with suggested values, discovers camera devices
  under `/dev/v4l/by-id` (with unplug-to-identify and a `/dev/v4l/by-path`
  fallback for serial-less cameras that collide in by-id), and writes
  `~/.config/inspect-robots/config.ini`. An existing file is backed up to
  `config.ini.bak` and unmanaged sections/keys are carried through
  unchanged. Warns before writing `rerun = true` in a headless session
  (part of #50).
- Public-docstring coverage gate via Ruff's D1 rules, with a full backfill of
  missing public docstrings.

### Fixed

- **Operator scoring no longer prompts twice for self-confirming embodiments**
  (#53). On interactive ad-hoc runs, definitive `success` or `failure`
  termination verdicts are adopted as the operator judgement, announced on the
  terminal, and identified as embodiment-sourced in the in-memory transcript.
- **Literal percent signs in config values now round-trip unchanged** (#54).
  Config reads no longer treat `%` as interpolation syntax, so values such as
  `policy = 50%off` work with `config set`, `config show`, and normal runs.
- **Component argument mistakes now fail cleanly and stale args are flagged**
  (#47). Changing a configured component warns when its non-empty args section
  still belongs to the old name, and invalid constructor kwargs exit with
  guidance to check the config section or CLI args flag instead of a traceback.
- **`inspect-robots run` now surfaces evaluation failures in its summary**:
  top-level errors, per-scene failure context, and a ready-to-run postmortem
  `inspect` command are printed after unsuccessful runs (#57).
- **Config `[*.args]` sections no longer follow a differently-selected
  component** (#44). `[policy.args]` / `[embodiment.args]` /
  `[sim_embodiment.args]` now apply only when the selected component matches
  the `[defaults]` name they were configured alongside; selecting another
  component (by flag or env var) ignores them with a stderr note instead of
  crashing its constructor with foreign kwargs. Selecting the configured
  default explicitly (e.g. `--embodiment` naming the config default) still
  applies its args.

## [0.6.0] - 2026-07-10
### Added

- **New plugin: `inspect-robots-agent`** — frontier LLMs (Claude, GPT,
  anything behind an OpenAI-compatible API) drive any registered embodiment
  through tool calls, as the first-class policy `agent`
  (`--policy agent -P model=anthropic/claude-fable-5`). Each tool call becomes
  one smooth, approver-checked action chunk (`move_joints` with named partial
  targets for absolute control, `move_by` for displacement control;
  `done`/`give_up` end the trial). One `httpx` client speaks the wire format;
  keys resolve from `$ANTHROPIC_API_KEY` / `$OPENAI_API_KEY` /
  `$OPENROUTER_API_KEY` or a custom `base_url` (plan 0008).
- Safety approvers: `DeltaLimitApprover` (semantics-aware "no wild swings"
  per-step limiting) and `ChainApprover` (sequential composition) join
  `ClampApprover` in `inspect_robots.approver`.
- **CLI guardrails on by default**: every `run`/ad-hoc invocation wires
  `ChainApprover(ClampApprover, DeltaLimitApprover)` from the embodiment's
  action space; `--disable-guardrails` is the explicit, loudly-warned opt-out
  and `--max-action-delta` tunes the per-step limit. The chain degrades per
  component with stderr warnings (never blocking, never silent).
- CLI: `inspect-robots config set KEY VALUE` / `config show` persist and
  display `[defaults]` config keys; guided errors now point at `config set`.
- `ActionSemantics.dim_labels` names action dimensions (validated against the
  owning `Box`); `ControlMode` gains `"joint_delta"` for joint-space
  displacement control.
- Policies may define an optional `bind(embodiment_info)` hook — `eval()`
  calls it after resolution and before the compatibility check, so
  embodiment-adaptive policies (like the LLM agent) can adopt the
  embodiment's spaces.
- Adapter conformance kit (`inspect_robots.conformance`):
  `check_embodiment` / `assert_embodiment_conformant` verify an embodiment's
  declared spaces are guardrail-ready and agent-ready (semantics, finite
  bounds, unique `dim_labels`, aligned `StateSpec` for absolute modes,
  limitable rotation reps). Adapter repos enforce it with one CI test; the
  new `inspect-robots doctor --embodiment NAME` command audits installed
  adapters the same way. The `CubePick` mock now labels its dims (`dx`/`dy`)
  and passes its own kit. See the new adapter authoring guide
  (`docs/guide/adapters.md`) for the non-mechanical half (honest control
  modes, per-step delta bounds, hold-behavior verification).
- Rollout honors a policy-requested stop via the pre-review action's
  `meta["request_stop"]` (ends the trial as a truncation; embodiment
  termination wins; not preserved under ensembling).

### Fixed

- The CLI exits with the guided message (not a traceback) when a component
  factory raises `ConfigError` during resolution.

## [0.5.0] - 2026-07-10

Backfilled: this version was released tag-only; the entries were reconstructed
from the merged PRs.

### Added

- CLI: `--rerun` flag and `rerun` config default open a live Rerun viewer
  streaming cameras, state, and actions for each run (#36).
- CLI: `store_frames` config default and per-run frame directories under
  `<log-dir>/frames`; `--store-frames` became tri-state so `--no-store-frames`
  overrides the config (#30).
- CLI: minimal ANSI styling on interactive terminals; plain output when piped
  or `NO_COLOR` is set (#37). `inspect-robot` is accepted as an alias for the
  common typo (#34).

### Fixed

- The CLI closes the embodiment it resolves, even when `eval()` raises: a
  real robot never stays energized after a crashed run (#30).

## [0.4.0] - 2026-07-09

Plugin releases alongside this version: `inspect-robots-xpolicylab` 0.1.0
(first release) and `inspect-robots-isaacsim` 0.1.1 (ships the env-creation
fix below).

### Added

- **New plugin: `inspect-robots-xpolicylab`** — a `Policy` adapter for
  [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) policy servers,
  making its zoo of 40+ served VLAs (π0/π0.5, GR00T, OpenVLA-OFT, RDT-1B,
  SmolVLA, ACT, …) evaluable with any Inspect Robots embodiment
  (`--policy xpolicylab -P url=ws://host:19000`). Speaks XPolicyLab's
  msgpack-over-websocket protocol directly — no `xpolicylab` install needed
  on the eval side.
- CLI: `inspect-robots run` gained `--epochs`, `--fail-on-error`, and
  `--store-frames`; the written log's path is printed at the end of a run.
- Tests are now type-checked under strict mypy (`files = ["src/inspect_robots",
  "tests"]`).
- CI: a blocking `test-rerun` job installs the real `rerun-sdk` and runs
  `test_rerun_sink.py` against it — previously `RerunSink` was only exercised
  against a fake `rerun` module, so a real SDK API change would go unnoticed
  (#6). The `plugin-isaacsim` CI job gained a `ruff format --check` step and
  now reports (but does not gate on) its own test coverage.

### Changed

- **Documentation site moved to a custom domain:** <https://inspectrobots.org/>.

### Fixed

- **`EvalLog` and friends are now actually immutable.** `EvalLog`, `EvalSpec`,
  `EvalStats`, `EvalResults`, and `SceneResult` are frozen dataclasses, and
  `SceneResult.epochs`/`operator_judgements` and `EvalLog.samples` are tuples
  instead of lists — previously nothing stopped e.g. `log.samples.clear()`
  despite the "immutable EvalLog" documentation (#4). `read_eval_log` coerces
  older on-disk logs (whose JSON arrays deserialize as lists) back into tuples,
  so the read-back guarantee is unaffected.
- **isaacsim plugin: real env creation was broken.** `_ensure_env` called
  `gym.make(task_id)` without the mandatory Isaac Lab `cfg` object, so every
  live run failed with `missing 1 required positional argument: 'cfg'`; the
  config is now resolved via Isaac Lab's own `parse_env_cfg`. Alongside it:
  observation groups are requested as *named* dicts (`concatenate_terms=False`
  — a flat tensor left `Observation.state` empty; a warning fires when the
  request can't be honored), and headless runs disable every `debug_vis` flag
  (markers exist for a viewport nobody has, and their material machinery can
  hang env creation on hosts with a broken render stack).
- **Eval logs are strict RFC 8259 JSON.** Non-finite floats (e.g. an inf
  `min_distance_to_goal` when no distance was ever recorded) are mapped to
  `null` at the JSON boundary, so `jq` and other conforming parsers accept the
  file; `json.dump(..., allow_nan=False)` stays on as a regression backstop.
  In-memory scores keep the inf sentinel.
- **`ClampApprover` hardening:** a NaN action raises `SafetyAbort` (a NaN has
  no meaningful clamp and must never reach hardware) while `±inf` clamps to the
  finite bound like any out-of-range value; one-sided boxes (`low`-only /
  `high`-only) are honored instead of ignored; an unmodified action is returned
  as the *same* object so the rollout's identity-based `approval_event` stays
  accurate.
- **Never lose the log.** `eval()` always produces and persists an `EvalLog`
  once rollouts have started: scorer/reducer failures degrade the run to an
  error log instead of crashing; `policy.reset`/`embodiment.reset` failures are
  wrapped into the error taxonomy; every error raised from inside a trial
  carries the partial `TrialRecord` on `exc.record` (recorded and delivered to
  sinks — errored trials are never scored); `on_trial_end` fires for halted
  trials too.
- **A crashing approver now halts the eval as `SafetyAbort`** — an approver that
  crashed cannot vouch for safety — and approved-but-modified actions emit an
  `approval_event`.
- **`eval()` owns what it opens:** an embodiment resolved from a registry name
  is closed when the run finishes (even on a halt); caller-constructed
  embodiments stay caller-owned.
- **`fail_on_error` is checked after every trial** (Inspect semantics:
  `True` = first error, `0<x<1` = proportion, `x>1` = count), not just at the
  end of the run.
- `derive_seed`: `seed=None` no longer aliases `seed=0` — unseeded runs draw a
  fresh OS seed and record it in the log.
- `Task`/`Epochs`/`Box`/`ObservationSpace` validate their configuration at
  construction (`max_steps`/`epochs` must be positive, `Box` bounds must be
  elementwise ordered, `state_keys` must agree with `StateSpec`), raising
  `ConfigError`/`ValueError` instead of failing mid-eval. `Task.scorer` also
  accepts registry names.
- Inference events no longer overstate `chunk_len` when `replan_interval`
  exceeds the chunk; the ensembling no-semantics warning fires per instance
  (at construction) instead of once per process.
- Collision-safe frame-file slugs (camera names and trial ids are fully
  sanitized); broken plugin entry points warn loudly instead of being silently
  skipped.
- Rerun sink: per-trial namespacing, new-SDK (`>=0.23`) compatibility, and a
  correct install hint.

## [0.3.0] - 2026-07-01

### Changed

- **Renamed the framework RoboInspect → Inspect Robots.** The import package is
  now `inspect_robots`, the distribution/CLI `inspect-robots`, the error base
  class `InspectRobotsError`, the log field `inspect_robots_version`, and the
  plugin entry-point groups `inspect_robots.*`. The Isaac Sim plugin follows as
  `inspect-robots-isaacsim` (import package `inspect_robots_isaacsim`, entry
  point group `inspect_robots.embodiments`).

## [0.2.0] - 2026-06-30

### Added

- **Isaac Sim / Isaac Lab plugin** as an in-repo uv-workspace package
  (`plugins/`): an `Embodiment` adapter backed by an Isaac Lab physics
  simulation (default profile: 7-DoF Franka Panda under joint-position control
  with a binary gripper), registered via entry point, with Isaac imported
  lazily so the plugin installs anywhere and the core stays NumPy-only.
  First-party plugins live as their own packages with their own pyproject,
  tests, and coverage scope; `uv sync --all-packages --extra dev` installs
  core + plugins editable.

### Changed

- Renamed the package RoboLens → RoboInspect (superseded by the 0.3.0 rename).

## [0.1.0] - 2026-06-27

### Added

- **Widened the public API for plugin authors.** `inspect_robots.__all__` now exports
  the authoring primitives directly — `Task`/`Epochs`, `Scene`/`Target`,
  `Scorer`/`Score` and the builtin scorers, `Policy`/`PolicyBase`/`PolicyInfo`/
  `PolicyConfig`, `Embodiment`/`EmbodimentBase`/`EmbodimentInfo`, the
  `types`/`spaces` dataclasses, `TrialRecord`, and the `@task`/`@policy`/
  `@embodiment`/`@scorer`/`@sink` registry decorators plus `registered`/`resolve`.
  Out-of-tree benchmarks (e.g. KitchenBench) and adapters can now `from inspect_robots
  import Task, Scene, task, ...` against a stable surface.

- **Core framework foundation.** The two-input model for robotics evals:
  `Policy` (VLA) and `Embodiment` (real robot or simulator), with a benchmark
  `Task` defined independently of both.
- **Types & spaces:** `Observation`, `Action`, `ActionChunk` (open-loop chunked
  execution), `StepResult`; `Box`/`ObservationSpace`, `ActionSemantics`, and a
  canonical proprioception `StateSpec` vocabulary.
- **Scenes & scoring:** `Scene`/`Target` datasets; `Scorer`/`Score` with an
  epoch-reducer split (`mean`/`median`/`max`/`min`/`mode`/`pass_at_k`); builtin
  scorers including `success_at_end`, `min_distance_to_goal`, `reached_goal_state`,
  and an operator-verdict scorer; reserved `VLMScorer` interface.
- **Rollout engine:** open-loop chunk execution via a composable `Controller`
  middleware layer (`DefaultController`, `SmoothingController`,
  `EnsemblingController` for ACT/ALOHA temporal ensembling); an `Approver`
  safety gate (`AutoApprover`, `ClampApprover`); an error taxonomy
  (`PolicyError` continue vs `EmbodimentFault`/`SafetyAbort` halt); a typed
  transcript; per-trial seeding; and a `FrameStore` that streams frames to disk.
- **Compatibility checking:** fail-fast action/observation/semantics checks with
  key remapping, control-rate reconciliation, and scene realizability.
- **`eval()` / `eval_set()`:** Inspect-style orchestration returning immutable,
  schema-versioned `EvalLog`s with `fail_on_error` semantics; atomic JSON logs
  with a read-back guarantee; optional frame side-cars.
- **Registry & plugins:** decorators and `importlib.metadata` entry-point
  discovery so out-of-tree backends register without being imported.
- **Logging sinks:** canonical `JsonLogSink`; optional, lazily-imported
  `RerunSink` for [Rerun](https://github.com/rerun-io/rerun) visualization.
- **CLI:** `inspect-robots list`, `inspect-robots run`, and `inspect-robots inspect <log>`.
- **String resolution:** `eval()`/`eval_set()` accept registry names
  (`eval("cubepick-reach", "scripted", "cubepick")`) in addition to objects.
- Dependency-free `CubePick` mock world and scripted/random/noop policies.
- **Documentation site** (MkDocs + Material + mkdocstrings) auto-generated from
  docstrings, deployed to GitHub Pages, with guides, an API reference, and
  `llms.txt` / `llms-full.txt` for LLM consumers. Homepage-style README.
- **100% test coverage**, enforced by `--cov-fail-under=100` in CI (a blocking PR
  check). Genuinely unexecutable lines (Protocol stubs, `__main__` guards,
  defensive branches) are excluded via `tool.coverage.report`.
- **Pre-commit hooks** (`.pre-commit-config.yaml`): ruff (lint + format) and
  strict mypy on commit, the 100% coverage gate on push. Install with
  `uv run pre-commit install`. Documented in `CONTRIBUTING.md`.

[Unreleased]: https://github.com/robocurve/inspect-robots/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/robocurve/inspect-robots/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robocurve/inspect-robots/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robocurve/inspect-robots/releases/tag/v0.1.0
