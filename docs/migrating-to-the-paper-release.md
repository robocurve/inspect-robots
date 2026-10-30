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
