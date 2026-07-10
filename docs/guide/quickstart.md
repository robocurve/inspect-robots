# Quickstart

## Install

```bash
pip install inspect-robots            # core (numpy only)
pip install "inspect-robots[rerun]"   # + Rerun visualization
```

For development, use [uv](https://github.com/astral-sh/uv):

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

## Run your first evaluation

The dependency-free `CubePick` mock world lets you exercise the whole stack with
no hardware or simulator:

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

(log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment())
print(log.status)                    # "success"
print(log.results.metrics)           # {"success_at_end": 1.0}
```

`eval()` returns a list of [`EvalLog`][inspect_robots.log.EvalLog] (one per task, mirroring
Inspect AI). Each log is immutable, schema-versioned, and written to `log_dir`.

## Use registry names

`task`, `policy`, and `embodiment` may also be registry names, the same
mechanism the CLI uses:

```python
from inspect_robots import eval

(log,) = eval("cubepick-reach", "scripted", "cubepick")
```

## From the command line

```bash
inspect-robots list                                          # all registered components
inspect-robots list policies                                 # just policies
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick -P chunk_size=6
inspect-robots inspect logs/cubepick-reach_*.json            # print a saved log
```

## Next steps

- [Concepts](concepts.md): the core abstractions.
- [Writing A Benchmark](writing-a-benchmark.md): define your own `Task`.
- [Policies And Embodiments](policies-and-embodiments.md): plug in a real VLA or robot/sim.
