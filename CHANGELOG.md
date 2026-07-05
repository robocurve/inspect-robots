# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is
`0.x`, breaking changes may occur on any minor release.

## [Unreleased]

### Changed

- **Documentation site moved to a custom domain:** <https://inspectrobots.org/>.

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
