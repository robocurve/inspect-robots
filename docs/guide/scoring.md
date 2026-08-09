# Scoring

A [`Scorer`](/api/#inspect_robots.scorer.Scorer) maps a recorded
[`TrialRecord`](/api/#inspect_robots.rollout.TrialRecord) (plus the scene's
[`Target`](/api/#inspect_robots.scene.Target)) to a [`Score`](/api/#inspect_robots.scorer.Score). Scorers
read the *recorded* trajectory (never a live environment), so scoring is
reproducible from a saved log.

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
        deltas = [abs(float(s.action.data.sum())) for s in record.steps]
        return Score(value=-sum(deltas), explanation="negative total command magnitude")
```

Register it with [`scorer`](/api/#inspect_robots.registry.scorer) to resolve it by name.

## Epochs and reducers

When a `Task` runs `epochs > 1`, an epoch reducer collapses the per-epoch
scores of a scene before metrics aggregate across scenes. Reducers are namespaced
separately from metrics and are selected by name on
[`Epochs`](/api/#inspect_robots.task.Epochs):

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

Real robots have no privileged success oracle. The dominant method is a human
verdict, captured *once* per trial and read back by
[`operator_scorer`](/api/#inspect_robots.scorer.operator_scorer), keeping scoring reproducible.
Capture is the job of a [`Grader`](/api/#inspect_robots.grader.Grader): a registered
component (`inspect_robots.graders` entry point, `grader` decorator) whose
`grade(record, scene)` runs once per scored trial, after the rollout and
before the scorers, and writes the judgement onto the record. The builtin
`operator` grader prompts the terminal operator; a VLM autograder over final
frames belongs on the same seam (the reserved
[`VLMScorer`](/api/#inspect_robots.scorer.VLMScorer) interface predates it).

Every attended CLI run is graded by default, registered tasks included, so
judgement-reading scorers (the `operator` scorer, or task scorers that fall
back to `operator_judgement`) work with operator-in-the-loop embodiments and
with policies that end their own trials (`done()`/`give_up()`).
`success_at_end` reads only embodiment-detected `"success"` terminations and
scores operator-graded trials as failures; pair attended operator-graded runs
with a judgement-reading scorer instead. From the Python API, pass
`eval(..., grader="operator")` (or any `Grader` object); unattended runs and
`eval()` without a grader stay prompt-free.
