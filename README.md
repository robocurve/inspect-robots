<div align="center">

# 🤖 Inspect Robots

### An open-source evaluation framework for physical AI and VLA (vision-language-action) models

Define a robotics benchmark once, then run *any* policy against *any* compatible
embodiment — a real robot or a simulator — with reproducible logs and first-class
[Rerun](https://github.com/rerun-io/rerun) visualization.

If you know [Inspect AI](https://inspect.aisi.org.uk/), this is that for robotics.

[![CI](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml)
[![Docs](https://github.com/robocurve/inspect-robots/actions/workflows/docs.yml/badge.svg)](https://inspectrobots.org/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://github.com/robocurve/inspect-robots)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-blue)](https://github.com/robocurve/inspect-robots)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml)

[**Documentation**](https://inspectrobots.org/) ·
[Quickstart](https://inspectrobots.org/guide/quickstart.html) ·
[Concepts](https://inspectrobots.org/guide/concepts.html) ·
[For LLMs](https://inspectrobots.org/llms.txt)

</div>

---

## One framework, two swappable inputs

LLM evaluations have a single swappable input: the model. **Robotics evaluations
have two** — and Inspect Robots makes both first-class and orthogonal:

| | |
|---|---|
| 🧠 **`Policy`** — the VLA | The "brain". Maps an observation + instruction to an **action chunk** (a horizon of actions executed open-loop, as π0 / ACT / diffusion policies do). |
| 🦾 **`Embodiment`** — the robot or sim | The "body + world". Produces observations, executes actions, owns the action/observation spaces and control rate. Real-robot-first; sims are a stricter special case. |

A **`Task`** — a dataset of `Scene`s (initial conditions, instructions, success
targets) plus scorers — is defined *independently* of both. Before any rollout,
Inspect Robots checks the `(policy, embodiment)` pair is **compatible** (action/observation
spaces, semantics, control rate, scene realizability) and fails fast if not.

## Install

```bash
pip install inspect-robots            # core (numpy only)
pip install "inspect-robots[rerun]"   # + Rerun visualization
```

## Quickstart

With a default policy/embodiment configured once in
`~/.config/inspect-robots/config.ini`, just tell the robot what to do:

```bash
inspect-robots "place the spoon on the plate"                # zero-config ad-hoc eval
inspect-robots "place the spoon on the plate" --sim          # same, on your configured sim
```

The full command line resolves any registered task/policy/embodiment
(builtins + installed plugins):

```bash
inspect-robots list                                          # registered components
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
inspect-robots inspect logs/cubepick-reach_*.json            # results table
```

And everything is a Python API. No hardware or simulator needed — the
dependency-free `CubePick` mock world exercises the whole stack:

```python
from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

task = Task(
    name="cubepick-reach",
    scenes=[Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(5)],
    scorer=success_at_end(),
    max_steps=80,
)

# The two swappable inputs: a policy (VLA) and an embodiment (robot/sim).
(log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment())
print(log.status, log.results.metrics)   # success {'success_at_end': 1.0}
```

## Why Inspect Robots

- 🌍 **Real-world first.** Interfaces assume real-robot reality — human-in-the-loop
  reset, no privileged success oracle, wall-clock control rate. Simulators just
  offer more (seeding, privileged success, rendering) via opt-in capabilities.
- 🔁 **Reproducible.** Every run yields an immutable, schema-versioned `EvalLog`
  with the resolved config, git revision, and package versions — re-readable across
  releases, and re-scorable offline.
- 🪶 **Light core.** Depends only on NumPy. Rerun and simulator/VLA backends are
  optional extras and separately installable plugins.
- 🛑 **Safe unattended.** An explicit error taxonomy separates "record and continue"
  from "halt and require a human", so a faulted robot never auto-advances overnight.
- 🎞️ **Rerun visualization.** Stream camera images, 3D poses, joint/action
  time-series, and success markers to a `.rrd` recording.
- 🧩 **Pluggable.** Ship `inspect-robots-maniskill` or `inspect-robots-openvla` as separate
  packages — entry points make them appear in `inspect-robots list` automatically.
- ⚙️ **VLA-native.** Action chunking, open-loop execution, and ACT/ALOHA temporal
  ensembling are built in, with action *semantics* (control mode, rotation
  representation, gripper, frame) that make compatibility and ensembling correct.

## How it maps to Inspect AI

If you know [Inspect AI](https://inspect.aisi.org.uk/), you already know Inspect Robots.

| Inspect AI | Inspect Robots |
|---|---|
| `Model` | `Policy` (VLA) **+** `Embodiment` *(two inputs)* |
| `Task = dataset + solver + scorer` | `Task = scenes + controller + scorer` |
| `Sample` | `Scene` |
| `Solver` chain | `Controller` middleware (chunking, ensembling, smoothing) |
| `eval()` → `EvalLog` | `eval()` → `EvalLog` |
| `@task` / `@solver` / `@scorer` + registry | `@task` / `@policy` / `@embodiment` / `@scorer` + entry points |

This repository is the **framework**. Concrete benchmarks live in
[WorldEvals](https://github.com/robocurve/worldevals), the benchmark catalog,
and backend adapters live in separate plugin packages.

## Documentation

Full guides and an auto-generated API reference live at
**[inspectrobots.org](https://inspectrobots.org/)**.
LLM-friendly versions: [`llms.txt`](https://inspectrobots.org/llms.txt)
and [`llms-full.txt`](https://inspectrobots.org/llms-full.txt).

## Development

> **Dependency changes:** after editing dependencies in `pyproject.toml`, run
> `uv lock` and commit the updated lockfile — CI installs with
> `uv sync --locked` and fails with "the lockfile needs to be updated" if you
> forget. Day-to-day conventions (PR-only `main`, the required `ci-ok` check,
> one-click releases) are documented in [`CLAUDE.md`](CLAUDE.md).

```bash
uv venv && uv pip install -e ".[dev]"
uv run pre-commit install          # ruff + mypy on commit, 100% coverage on push
uv run pytest --cov                 # 100% coverage required
uv run ruff check . && uv run mypy
```

Pre-commit hooks and a blocking CI coverage gate keep `main` green. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the design docs in [`plans/`](plans/).

## Citation

If you use Inspect Robots in your research, please cite it:

```bibtex
@software{inspect-robots,
  author  = {Robocurve},
  title   = {Inspect Robots: The open-source evaluation framework for physical AI},
  year    = {2026},
  url     = {https://github.com/robocurve/inspect-robots},
  version = {0.3.0},
  license = {MIT}
}
```

## License

[MIT](LICENSE)
