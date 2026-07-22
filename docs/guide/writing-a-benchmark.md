# Writing a benchmark

A benchmark is a [`Task`](/api/#inspect_robots.task.Task): a dataset of scenes plus scorer(s).
It is embodiment-agnostic: it describes *what* to evaluate, not *how* the
robot is built.

```python
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Epochs, Task

task = Task(
    name="cubepick-reach",
    scenes=[
        Scene(
            id=f"layout-{i}",
            instruction="reach the cube",
            target=Target(kind="reach_object", spec={"object": "cube"}),
            init_seed=i,
        )
        for i in range(50)
    ],
    scorer=success_at_end(),
    max_steps=200,
    epochs=Epochs(count=3, reducer="mean"),
)
```

## Scenes

Each [`Scene`](/api/#inspect_robots.scene.Scene) is one initial condition (the Inspect `Sample`
analog):

- `id`: unique within the task.
- `instruction`: the language goal handed to the policy.
- `target`: an optional [`Target`](/api/#inspect_robots.scene.Target) the scorer reads; its
  `kind` is resolved in the *embodiment's* namespace (compatibility checking
  verifies the embodiment can realize it).
- `init_seed`: combined with the eval seed and epoch index to seed each trial
  deterministically. (An eval run with `seed=None` draws a fresh OS seed and
  records it in the log, so even "unseeded" runs are reproducible after the
  fact, and distinct from `seed=0`.)

## Epochs and reducers

Repeat each scene `epochs` times to measure stochastic policies. The
[`Epochs`](/api/#inspect_robots.task.Epochs) reducer collapses the per-epoch scores of a scene
before metrics aggregate across scenes. Builtin reducers: `mean`, `median`,
`max`, `min`, `mode`, and `pass_at_<k>` (an unbiased pass@k estimator).

## Multiple scorers

Pass a list to score several dimensions at once:

```python
from inspect_robots.scorer import episode_length, min_distance_to_goal, success_at_end

task = Task(
    name="cubepick-reach",
    scenes=[...],
    scorer=[success_at_end(), episode_length(), min_distance_to_goal()],
    max_steps=200,
)
```

## Horizons

A task declares exactly one rollout horizon. Use `max_steps` when the protocol
is inherently discrete, as in the examples above. Use `max_seconds` when every
embodiment should receive the same physical-time budget:

```python
task = Task(
    name="two-minute-reach",
    scenes=[...],
    scorer=success_at_end(),
    max_seconds=120.0,
)
```

At evaluation time, Inspect Robots resolves the budget as
`ceil(max_seconds * embodiment.info.control_hz)`. A 120-second task therefore
runs for 1,200 steps at 10 Hz and 1,800 steps at 15 Hz. The eval log records
both the declared seconds and the resolved integer step limit.

A seconds-based task is incompatible with an embodiment whose `control_hz` is
missing, non-finite, zero, or negative. Event-driven embodiments should use
`max_steps`. Resolution changes the step budget only: `rollout()` does not add
wall-clock pacing, so real-time cadence remains the embodiment's responsibility.

## Registering for discovery

Wrap a task factory with [`task`](/api/#inspect_robots.registry.task) so it resolves by name in
`eval("my-bench", ...)` and appears in `inspect-robots list`:

```python
from inspect_robots.registry import task

@task("my-bench")
def my_bench(num_scenes: int = 50) -> Task:
    return Task(name="my-bench", scenes=[...], scorer=success_at_end(), max_steps=200)
```

See [Plugins](plugins.md) to ship a benchmark from a separate package.
