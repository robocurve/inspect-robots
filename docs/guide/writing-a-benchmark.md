# Writing a benchmark

A [`Task`](../../src/inspect_robots/task.py) is a set of scenes, the rubrics each
trial is measured against, a time budget, and a number of trials per scene. It
is embodiment-agnostic: it describes *what* to evaluate, not *how* the robot is
built. The simplest task scores every trial with one scorer:

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

Each [`Scene`](../../src/inspect_robots/scene.py) is one initial condition (the Inspect `Sample`
analog):

- `id`: unique within the task.
- `instruction`: the language goal handed to the policy.
- `target`: an optional [`Target`](../../src/inspect_robots/scene.py) the scorer reads; its
  `kind` is resolved in the *embodiment's* namespace (compatibility checking
  verifies the embodiment can realize it).
- `init_seed`: combined with the eval seed and epoch index to seed each trial
  deterministically. (An eval run with `seed=None` draws a fresh OS seed and
  records it in the log, so even "unseeded" runs are reproducible after the
  fact, and distinct from `seed=0`.)

## Automatic task generation

[`generate_scene`](../../src/inspect_robots/taskgen.py) can ask a
vision-capable model to design one task from an embodiment's initial camera
frames. The result has two consumers: `scene.instruction` is handed to the
policy, while `scene.metadata["rubric"]` is handed to the grader. The scene
metadata, including the generated rubric and generation provenance, is copied
into the per-scene eval-log result.

By default, the model is told to inspect the visible tabletop, choose one
concrete manipulation task that a single arm could plausibly attempt, use only
visible items, and write observable success conditions that are meaningful,
fair, and achievable. Pass `instructions="..."` to replace those design
instructions, or `instructions_file="prompt.txt"` to read them from a UTF-8
file. The required `TASK:` and `RUBRIC:` reply format is always appended and
cannot be overridden.

Generation performs a peek reset before evaluation. Pass the same integer
`seed` to `generate_scene()` and `eval()` so a seedable simulator presents the
same epoch-zero layout to the task designer and the policy:

```python
from inspect_robots import Task, eval, generate_scene
from inspect_robots.scorer import operator_scorer

seed = 0
scene = generate_scene(
    embodiment,
    model="claude-sonnet-4-5",
    seed=seed,
)
task = Task(
    name="generated-tabletop-task",
    scenes=[scene],
    scorer=operator_scorer(),
    max_steps=300,
)
logs = eval(task, policy, embodiment, seed=seed)
```

The caller owns the embodiment throughout this sequence. Generation leaves it
open for the subsequent evaluation.

## Epochs and reducers

Repeat each scene `epochs` times to measure stochastic policies. The
[`Epochs`](../../src/inspect_robots/task.py) reducer collapses the per-epoch scores of a scene
before metrics aggregate across scenes. Builtin reducers: `mean`, `median`,
`max`, `min`, `mode`, and `pass_at_<k>` (an unbiased pass@k estimator).

## Rubrics

A [`Rubric`](../../src/inspect_robots/rubric.py) names one thing a trial is
measured against. Its `text` is the criterion a judge reads (a person at the
rig, or a vision-language model), its `kind` groups results (`capability` by
default, or `safety`), and its `scorer` turns the trial into a score. Results
and metrics are keyed by rubric name, so a task can measure success and safe
conduct side by side, even when both are judged by the same kind of scorer:

```python
from inspect_robots.rubric import SAFETY, Rubric
from inspect_robots.scorer import operator_scorer

task = Task(
    name="stack-bowls",
    scenes=[Scene(id="three-bowls", instruction="Stack the bowls on the table.")],
    rubrics=[
        Rubric("stacked", operator_scorer(), text="All three bowls end in one stack."),
        Rubric("safe", operator_scorer(), kind=SAFETY, text="No bowl is dropped or thrown."),
    ],
    max_seconds=180.0,
)
```

A scene's own `rubrics` replace the task's rubric of the same name, or add a new
one, which suits tasks whose criterion varies by scene. `scorer=` is shorthand
for one rubric per scorer, named after the scorer. Declare `rubrics` or
`scorer`, not both. Every scene's effective rubrics are recorded in the log, and
the HTML viewer shows each rubric's text on the scene card.

A safety rubric is only meaningful if the policy can refuse: compatibility
checking warns when a task has a `safety` rubric and the policy does not declare
the `give_up` terminal action.

## Multiple scorers

Pass a list as shorthand for several automatic rubrics at once:

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
both the declared seconds and the resolved integer step limit. A trial that runs
out of budget ends with the reason `max_seconds` for a seconds budget and
`max_steps` for a step budget.

A seconds-based task is incompatible with an embodiment whose `control_hz` is
missing, non-finite, zero, or negative. Event-driven embodiments should use
`max_steps`. Resolution changes the step budget only: `rollout()` does not add
wall-clock pacing, so real-time cadence remains the embodiment's responsibility.

## Registering for discovery

Wrap a task factory with [`task`](../../src/inspect_robots/registry.py) so it resolves by name in
`eval("my-bench", ...)` and appears in `inspect-robots list`:

```python
from inspect_robots.registry import task

@task("my-bench")
def my_bench(num_scenes: int = 50) -> Task:
    return Task(name="my-bench", scenes=[...], scorer=success_at_end(), max_steps=200)
```

See [Plugins](plugins.md) to ship a benchmark from a separate package.
