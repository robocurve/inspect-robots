# Migrating to the paper release

This release renames the framework's core types to the terms used in the Inspect
Robots paper and removes the old names with no aliases. Each section below is
one change; apply them in any order. Saved logs from earlier releases still read
and score without changes.

## Paper vocabulary

| Before | After | Module |
|---|---|---|
| `inspect_robots.controller` | `inspect_robots.action_manager` | |
| `Controller` | [`ActionManager`](../src/inspect_robots/action_manager.py) | `action_manager` |
| `DefaultController` | `DefaultActionManager` | `action_manager` |
| `SmoothingController` | `SmoothingActionManager` | `action_manager` |
| `EnsemblingController` | `EnsemblingActionManager` | `action_manager` |
| `inspect_robots.approver` | `inspect_robots.guard` | |
| `Approver` | [`Guard`](../src/inspect_robots/guard.py) | `guard` |
| `AutoApprover` | `PassThroughGuard` | `guard` |
| `ClampApprover` | `ClampGuard` | `guard` |
| `DeltaLimitApprover` | `DeltaLimitGuard` | `guard` |
| `ChainApprover` | `GuardChain` | `guard` |
| `GuardrailContribution` (field `approvers`) | `GuardContribution` (field `guards`) | `guard` |
| `Embodiment.contribute_guardrails()` | `Embodiment.contribute_guards()` | `embodiment` |
| `TrialRecord` | [`Record`](../src/inspect_robots/rollout.py) | `rollout` |
| `eval(..., controller=, approver=)` | `eval(..., action_manager=, guard=)` | `eval` |

`StepRecord` keeps its name. The command line is unchanged, including
`--disable-guardrails`. What a policy sees is unchanged too: observation text
still reports guard interventions on an `approver:` line, and logs keep their
`approval` events, so earlier logs and transcripts compare directly with new
ones.

A typical update:

```python
# before
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
eval(task, policy, embodiment, approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)))

# after
from inspect_robots.guard import ClampGuard, DeltaLimitGuard, GuardChain
eval(task, policy, embodiment, guard=GuardChain(ClampGuard(space), DeltaLimitGuard(space)))
```

An embodiment that contributed guardrails renames its method and returns the new
type:

```python
def contribute_guards(self, action_space: Box) -> GuardContribution:
    return GuardContribution(guards=(("my-collision", MyCollisionGuard(...)),))
```

## Terminal actions

A policy ends a trial with a terminal action instead of flags in action
metadata. Declare the kinds in `PolicyInfo.terminal_actions` and return a chunk
with `terminal` set:

```python
# before
Action(data=hold, meta={"request_stop": True, "stop_reason": "done", "stop_detail": summary})

# after
ActionChunk(actions=[Action(data=hold)], terminal=Termination(DONE, summary))
```

The old metadata still ends the trial the same way, so existing policies keep
working, but it is deprecated and is not checked against declared kinds. The
record now carries `termination_detail` (the stated reason), logs carry
`termination_details` beside `termination_reasons`, and the transcript has a
`policy_terminal` event. The agent plugin's `done` and `give_up` tools use the
new form. CaP-X's `FINISH` and `GIVE_UP` now end trials with the reasons `done`
and `give_up` instead of `FINISH` and `GIVE_UP`; update any analysis that
filters on the old strings.

## Rubrics and log schema 2

Tasks measure trials against named [rubrics](../src/inspect_robots/rubric.py).
Existing tasks need no change: `scorer=` is shorthand for one rubric per scorer,
named after the scorer, so their results keep the same keys. To add rubric
text, a safety criterion, or per-scene criteria, switch to `rubrics=`:

```python
# before
Task(name="t", scenes=[Scene(id="s", instruction="...", metadata={"rubric": "Bowls stacked."})],
     scorer=operator_scorer(), max_steps=600)

# after
Task(name="t", scenes=[Scene(id="s", instruction="...")],
     rubrics=[Rubric("stacked", operator_scorer(), text="Bowls stacked.")], max_steps=600)
```

`scene.metadata["rubric"]` still works and still takes precedence for the VLM
grader. A grader judges one rubric per trial: the first of the scene's rubrics
with text.

Logs are written as schema version 2, which adds `EvalSpec.rubrics`,
`SceneResult.rubrics` and `SceneResult.termination_details`. Version 1 logs
still read, with empty new fields and unchanged score keys. Tools that parse the
JSON directly should accept `"version": 2`.

A task with a `max_seconds` budget now ends an out-of-budget trial with the
reason `max_seconds` instead of `max_steps`.

## Records and log schema 3

Every trial now has a record with a unique id, the action chunks the policy
produced, and a per-trial directory under `records/` (see
[Trial records](guide/logging-and-rerun.md#trial-records)). Logs are written as
schema version 3, which adds `SceneResult.record_ids`, `SceneResult.record_dirs`,
`SceneResult.events`, `EvalStats.records_dir`, `EvalSpec.reducer` (the epoch
reducer, so a log can be re-scored) and `EvalSpec.benchmark` (the benchmark's
name and version, or null). Versions 1 and 2 still read,
with empty new fields. Tools that parse the JSON directly should accept
`"version": 3`.

`eval.policy_config` now redacts values under credential-like keys. Pass
`store_records=False` to `eval()` or `eval_set()` to skip the record
directories.
