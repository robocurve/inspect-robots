# Scoring

A [`Scorer`](../../src/inspect_robots/scorer.py) maps a recorded
[`Record`](../../src/inspect_robots/rollout.py) (plus the scene's
[`Target`](../../src/inspect_robots/scene.py)) to a [`Score`](../../src/inspect_robots/scorer.py). Scorers
read the *recorded* trajectory (never a live environment), so scoring is
reproducible from a saved log.

A task attaches scorers through [rubrics](writing-a-benchmark.md#rubrics): named
criteria (capability or safety), each with its scorer. Results and metrics are
keyed by rubric name.

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

Register it with [`scorer`](../../src/inspect_robots/registry.py) to resolve it by name.

A scorer with no verdict for a trial can abstain by returning
`Score(value=None)`. The log records the epoch as `null`, which is distinct
from a `0.0` verdict and from an errored trial (an empty epoch entry).
Reducers leave abstained epochs out, a scene where every epoch abstained
reduces to `null`, and a metric averages only the scenes that have a value
(`null` when none do). Reports show an abstention as `n/a`, and
`EvalResults.abstentions` counts abstained trials per scorer so each metric's
denominator stays visible: `inspect` and `view` print it beside the metric.
The `operator` scorer abstains on a trial the grader tried and failed to judge
(it carries `grading_error` in its metadata), so a grading outage is left out
of the metric instead of counting as robot failures; the run then ends in
error (see "Automated grading" in the CLI guide).

## Epochs and reducers

When a `Task` runs `epochs > 1`, an epoch reducer collapses the per-epoch
scores of a scene before metrics aggregate across scenes. Reducers are namespaced
separately from metrics and are selected by name on
[`Epochs`](../../src/inspect_robots/task.py):

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
[`operator_scorer`](../../src/inspect_robots/scorer.py), keeping scoring reproducible.
Benchmarks that read `operator_judgement` directly, instead of delegating to
`operator_scorer`, should call
[`is_affirmative_verdict`](../../src/inspect_robots/scorer.py)
rather than restate the vocabulary. It owns the recognized affirmative words
together with the case-folding and whitespace handling around them, so a change
reaches every consumer at once.
Capture is the job of a [`Grader`](../../src/inspect_robots/grader.py): a registered
component (`inspect_robots.graders` entry point, `grader` decorator) whose
`grade(record, scene)` runs once per scored trial, after the rollout and
before the scorers, and writes the judgement onto the record. The builtin
`operator` grader prompts the terminal operator, and the builtin `vlm`
grader is the autograder on the same seam: a vision model judges the trial's
first and last frames against a rubric (the reserved
[`VLMScorer`](../../src/inspect_robots/scorer.py) interface predates it
and stays a stub, because R6 requires scorers to be pure readers).

Every attended CLI run is graded by default, registered tasks included, so
judgement-reading scorers (the `operator` scorer, or task scorers that fall
back to `operator_judgement`) work with operator-in-the-loop embodiments and
with policies that end their own trials (`done()`/`give_up()`).
`success_at_end` reads only embodiment-detected `"success"` terminations and
scores operator-graded trials as failures; pair attended operator-graded runs
with a judgement-reading scorer instead. From the Python API, pass
`eval(..., grader="operator")` (or any `Grader` object); unattended runs and
`eval()` without a grader stay prompt-free.

## Correcting a verdict after the run

A verdict captured during the run can be corrected later without rewriting the
log: `inspect-robots edit` appends a judgement
[annotation](../../src/inspect_robots/annotation.py) beside the log, and
`inspect-robots score` re-scores from it and compares the result with the
stored metrics.

```bash
inspect-robots edit logs/run.json --trial scene-0-e0 --judgement no \
  --author reviewer --reason "the top bowl slid off after release"
inspect-robots score logs/run.json --out rescored.json
```

A scorer takes part in re-scoring by offering
`score_annotations(annotations, record_id, rubric)`, which scores a trial from
the judgement that counts; the `operator` scorer does. Scorers without it, such
as `success_at_end`, keep their stored scores. See
[Re-scoring](editing-and-analysis.md#re-scoring).
