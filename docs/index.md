# RoboInspect

<p style="font-size: 1.3rem; font-weight: 500; margin-bottom: 0.25rem;">
The <strong>Inspect AI</strong> for robotics.
</p>

An open-source evaluation framework for **physical AI** and **VLA
(vision-language-action)** models. Define a robotics benchmark once, then run
*any* policy against *any* compatible embodiment — a real robot or a simulator —
with reproducible logs and first-class [Rerun](https://github.com/rerun-io/rerun)
visualization.

[Get started](guide/quickstart.md){ .md-button .md-button--primary }
[Concepts](guide/concepts.md){ .md-button }
[GitHub](https://github.com/robocurve/roboinspect){ .md-button }

---

## One framework, two swappable inputs

LLM evals have a single swappable input: the model. **Robotics evals have two** —
and RoboInspect makes both first-class and orthogonal.

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } __`Policy` — the VLA__

    ---

    The "brain". Maps an observation + a language instruction to an **action
    chunk** (a horizon of actions executed open-loop, as π0 / ACT / diffusion
    policies do).

-   :material-robot-industrial:{ .lg .middle } __`Embodiment` — the robot or sim__

    ---

    The "body + world". Produces observations, executes actions, and owns the
    action/observation spaces and control rate. Real-robot-first; sims are a
    stricter special case.

</div>

A **`Task`** — a dataset of `Scene`s (initial conditions, instructions, success
targets) plus scorers — is defined *independently* of both. Before any rollout,
RoboInspect verifies the `(policy, embodiment)` pair is **compatible** and fails fast
and loud if not.

---

## Quickstart

```bash
pip install roboinspect            # core (numpy only)
pip install "roboinspect[rerun]"   # + Rerun visualization
```

No hardware or simulator required — the dependency-free `CubePick` mock world
exercises the whole stack:

```python
from roboinspect import eval
from roboinspect.mock import CubePickEmbodiment, ScriptedPolicy
from roboinspect.scene import Scene
from roboinspect.scorer import success_at_end
from roboinspect.task import Task

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

…or from the command line:

```bash
roboinspect list                                   # registered components
roboinspect run --task cubepick-reach --policy scripted --embodiment cubepick
roboinspect inspect logs/cubepick-reach_*.json     # results table
```

---

## Why RoboInspect

<div class="grid cards" markdown>

-   :material-earth:{ .lg .middle } __Real-world first__

    ---

    Interfaces assume real-robot reality: human-in-the-loop reset, no privileged
    success oracle, wall-clock control rate. Simulators just offer more.

-   :material-replay:{ .lg .middle } __Reproducible__

    ---

    Every run yields an immutable, schema-versioned `EvalLog` with the resolved
    config, git revision, and package versions — re-readable across releases.

-   :material-feather:{ .lg .middle } __Light core__

    ---

    The core depends only on NumPy. Rerun and simulator/VLA backends are optional
    extras and separately installable plugins.

-   :material-hand-back-right:{ .lg .middle } __Safe unattended__

    ---

    An explicit error taxonomy separates "record and continue" from "halt and
    require a human", so a faulted robot never auto-advances overnight.

-   :material-video-3d:{ .lg .middle } __Rerun visualization__

    ---

    Stream camera images, 3D poses, joint/action time-series, and success markers
    to a [Rerun](https://github.com/rerun-io/rerun) recording.

-   :material-puzzle:{ .lg .middle } __Pluggable__

    ---

    Ship `roboinspect-maniskill` or `roboinspect-openvla` as separate packages — entry
    points make them appear in `roboinspect list` automatically.

</div>

---

## How it maps to Inspect AI

If you know [Inspect AI](https://inspect.aisi.org.uk/), you already know RoboInspect.

| Inspect AI | RoboInspect |
|---|---|
| `Model` | `Policy` (VLA) **+** `Embodiment` *(two inputs)* |
| `Task = dataset + solver + scorer` | `Task = scenes + controller + scorer` |
| `Sample` | `Scene` |
| `Solver` chain | `Controller` middleware (chunking, ensembling, smoothing) |
| `eval()` → `EvalLog` | `eval()` → `EvalLog` |
| `@task`/`@solver`/`@scorer` + registry | `@task`/`@policy`/`@embodiment`/`@scorer` + entry points |

For LLMs: [`llms.txt`](https://robocurve.github.io/roboinspect/llms.txt) ·
[`llms-full.txt`](https://robocurve.github.io/roboinspect/llms-full.txt).
