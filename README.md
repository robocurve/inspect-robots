# RoboLens

**The [Inspect AI](https://inspect.aisi.org.uk/) for robotics** — an open-source
evaluation framework for **physical AI** and **VLA (vision-language-action)
models**.

Define a robotics benchmark once, then run *any* VLA policy against *any*
compatible embodiment — a real robot or a simulator — with reproducible logs and
first-class [Rerun](https://github.com/rerun-io/rerun) visualization.

> ⚠️ **Status: early alpha.** APIs are unstable and may change on any release
> before `1.0`. This repository is the *framework*. Concrete benchmarks (the
> "Inspect Evals for robotics") live in a separate repository.

## Why RoboLens

LLM evals have one swappable input: the model. **Robotics evals have two** — the
**policy** (the VLA "brain") and the **embodiment** (the robot/sim "body and
world"). RoboLens makes both first-class and orthogonal:

- **`Policy`** — a VLA that maps observations + a language instruction to an
  *action chunk* (a horizon of actions executed open-loop, as π0 / ACT /
  diffusion policies do).
- **`Embodiment`** — a real robot or simulator that produces observations,
  executes actions, and owns the action/observation spaces and control rate.
- **`Task`** — an embodiment-agnostic benchmark: a dataset of `Scene`s (initial
  conditions / instructions / success targets) plus scorers.

Before any rollout, RoboLens checks that a `(policy, embodiment)` pair is
*compatible* (action/observation spaces, semantics, camera/state key mapping,
control rate) and fails fast and loud if not.

## Design principles

- **Real-world first.** The interfaces assume real-robot reality (human-in-the-loop
  reset, no privileged success oracle, wall-clock control rate); simulators are a
  stricter special case that may offer more.
- **Reproducible logs.** Every run produces an immutable, schema-versioned
  `EvalLog` recording the resolved config, git revision, package versions, and
  per-scene results — re-scorable offline.
- **Light core, optional everything.** The core depends only on NumPy and the
  standard library. Rerun and simulator/VLA backends are optional extras and
  separately installable plugins.
- **Safe unattended runs.** An explicit error taxonomy separates "record the
  failure and continue" from "halt the eval and require a human", so a faulted
  robot never auto-advances overnight.

## Install

```bash
pip install robolens            # core (numpy only)
pip install "robolens[rerun]"   # + Rerun visualization
```

Development setup uses [uv](https://github.com/astral-sh/uv):

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

## Status & roadmap

See [`plans/`](plans/) for the design documents. The foundation is being built as
a vertical tracer (one scene → chunked rollout → score → log) that is then
thickened layer by layer.

## License

[MIT](LICENSE)
