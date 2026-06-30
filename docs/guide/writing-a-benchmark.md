# Writing a benchmark

A benchmark is a [`Task`][roboinspect.task.Task]: a dataset of scenes plus scorer(s).
It is **embodiment-agnostic** — it describes *what* to evaluate, not *how* the
robot is built.

```python
from roboinspect.scene import Scene, Target
from roboinspect.scorer import success_at_end
from roboinspect.task import Epochs, Task

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

Each [`Scene`][roboinspect.scene.Scene] is one initial condition (the Inspect `Sample`
analog):

- `id` — unique within the task.
- `instruction` — the language goal handed to the policy.
- `target` — an optional [`Target`][roboinspect.scene.Target] the scorer reads; its
  `kind` is resolved in the *embodiment's* namespace (compatibility checking
  verifies the embodiment can realize it).
- `init_seed` — combined with the eval seed and epoch index to seed each trial
  deterministically.

## Epochs and reducers

Repeat each scene `epochs` times to measure stochastic policies. The
[`Epochs`][roboinspect.task.Epochs] reducer collapses the per-epoch scores of a scene
before metrics aggregate across scenes. Builtin reducers: `mean`, `median`,
`max`, `min`, `mode`, and `pass_at_<k>` (an unbiased pass@k estimator).

## Multiple scorers

Pass a list to score several dimensions at once:

```python
from roboinspect.scorer import episode_length, min_distance_to_goal, success_at_end

task = Task(
    name="cubepick-reach",
    scenes=[...],
    scorer=[success_at_end(), episode_length(), min_distance_to_goal()],
    max_steps=200,
)
```

## Registering for discovery

Wrap a task factory with [`task`][roboinspect.registry.task] so it resolves by name in
`eval("my-bench", ...)` and appears in `roboinspect list`:

```python
from roboinspect.registry import task

@task("my-bench")
def my_bench(num_scenes: int = 50) -> Task:
    return Task(name="my-bench", scenes=[...], scorer=success_at_end(), max_steps=200)
```

See [Plugins](plugins.md) to ship a benchmark from a separate package.
