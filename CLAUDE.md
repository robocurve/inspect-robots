# RoboInspect — agent guide

RoboInspect is the **"Inspect AI for robotics"**: an open-source evaluation
framework for physical AI / VLA (vision-language-action) models. This repo is the
*framework*; concrete benchmarks and backend adapters live elsewhere (see below).

## The one big idea

LLM evals have one swappable input (the model). Robotics evals have **two**:

- **`Policy`** (the VLA "brain") — observation → `ActionChunk` (open-loop horizon).
- **`Embodiment`** (the robot/sim "body + world") — executes actions, owns spaces.

A **`Task`** (a dataset of `Scene`s + scorers) is defined independently of both.
`eval()` checks a `(policy, embodiment)` pair is compatible, runs the closed-loop
rollout, scores it, and writes an immutable `EvalLog`. Mirrors Inspect AI's
`Task = dataset + solver + scorer`, `eval()`, `EvalLog`, registry/decorator model.

## Layout

- `src/roboinspect/` — the package (see `src/roboinspect/CLAUDE.md` for the module map).
- `tests/` — pytest; the `CubePick` mock world exercises the whole stack with no
  hardware or sim.
- `plans/` — design docs. `plans/0001-foundation-design.md` is the authoritative
  spec (read its §9–§11 "binding resolutions" before changing core interfaces).
- `examples/` — runnable demos (`quickstart.py`).

## Working here

- Dev loop: `uv pip install -e ".[dev]"`, `uv run pre-commit install`, then
  `uv run pytest --cov`.
- Gates that must pass: `ruff check .`, `ruff format --check .`, `mypy` (strict),
  and `pytest --cov` at **100% coverage** (`--cov-fail-under=100`). Pre-commit
  runs ruff+mypy on commit and the coverage gate on push
  (`.pre-commit-config.yaml`). CI runs all gates on Linux+macOS / py3.11-3.12 as
  **blocking, required PR checks**; coverage below 100% fails the build.
- **Core stays NumPy-only.** New deps are optional extras, lazily imported; the
  `core-only-import` CI job enforces this.
- Test-driven; commit/push in small focused steps.
- Public API is fenced by `roboinspect.__all__` and guarded by
  `tests/test_api_snapshot.py` — update both together.

## Out of scope (separate repos / plugins)

Specific benchmarks ("Inspect Evals for robotics"), specific simulators
(ManiSkill/MuJoCo/Isaac), and specific VLA weights (OpenVLA/π0/Octo) ship as
separate plugin packages registered via entry points (`roboinspect.embodiments`,
`roboinspect.policies`, …).
