# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is
`0.x`, breaking changes may occur on any minor release.

## [Unreleased]

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
