# Scoring

A [`Scorer`][inspect_robots.scorer.Scorer] maps a recorded
[`TrialRecord`][inspect_robots.rollout.TrialRecord] (plus the scene's
[`Target`][inspect_robots.scene.Target]) to a [`Score`][inspect_robots.scorer.Score]. Scorers
read the *recorded* trajectory — never a live environment — so scoring is
**reproducible from a saved log**.

## Builtin scorers

```python
from inspect_robots.scorer import (
    success_at_end,        # 1.0 iff the episode terminated with reason "success"
    episode_length,        # number of steps taken
    min_distance_to_goal,  # closest the effector got (reads StepResult.info["distance"])
    reached_goal_state,    # success iff min distance <= threshold
    operator_scorer,       # reads a human verdict recorded during the rollout
)
```

## Custom scorers

A scorer is any object with a `name` and a `__call__(record, target) -> Score`:

```python
from dataclasses import dataclass
from inspect_robots.scorer import Score

@dataclass(frozen=True)
class SmoothMotion:
    name: str = "smooth_motion"

    def __call__(self, record, target) -> Score:
        deltas = [abs(float(s.action.data).sum()) for s in record.steps]
        return Score(value=-sum(deltas), explanation="negative total command magnitude")
```

Register it with [`scorer`][inspect_robots.registry.scorer] to resolve it by name.

## Epochs and reducers

When a `Task` runs `epochs > 1`, an **epoch reducer** collapses the per-epoch
scores of a scene before metrics aggregate across scenes. Reducers are namespaced
separately from metrics and are selected by name on
[`Epochs`][inspect_robots.task.Epochs]:

| Reducer | Meaning |
|---|---|
| `mean`, `median`, `max`, `min` | numeric reductions (raise on non-numeric strings) |
| `mode` | most common value (works for categorical scores) |
| `pass_at_<k>` | unbiased pass@k estimator (success = value ≥ 0.5) |

```python
from inspect_robots.task import Epochs, Task
Task(..., epochs=Epochs(count=5, reducer="pass_at_2"))
```

## Operator and VLM scoring (real world)

Real robots have no privileged success oracle. The dominant method is a **human
verdict**, captured *once* during the rollout (as a transcript event) and read
back by [`operator_scorer`][inspect_robots.scorer.operator_scorer] — keeping scoring reproducible.
A [`VLMScorer`][inspect_robots.scorer.VLMScorer] interface is reserved for scoring final
frames with a vision-language classifier.
