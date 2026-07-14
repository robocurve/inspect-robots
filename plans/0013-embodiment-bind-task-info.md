# Embodiment `bind(task_info)`: let the body learn the rollout horizon

Date: 2026-07-14
Status: draft (headed for subagent critique loop; Jay approved the direction:
kill the yam `max_steps_hint` duplication with a real core channel)

## Problem

An embodiment that renders an operator status line cannot show the episode
horizon, because nothing in the core interface tells it the task's
`max_steps`. The `Embodiment` protocol is `reset(scene, seed)` /
`step(action)` / `close()`; the horizon lives on the `Task` and is enforced
by the rollout loop.

The yam plugin worked around this with a `max_steps_hint` embodiment arg
whose own docstring concedes the defect: "Display-only hint of the
framework's episode horizon (the Task/CLI owns the real max_steps) ...
Bounds nothing." Operators must duplicate a value the framework already has;
unset, the countdown silently drops the horizon (`t = 42s` instead of
`t = 42s / 120s`); set, it drifts the moment `--max-steps` changes.

## Design

Mirror the existing policy-side precedent (plan 0008 §3c) exactly.

### `task.py` — `TaskInfo` and `Task.info`

A small frozen dataclass carrying what a component may legitimately learn
about the task it is running:

```python
@dataclass(frozen=True)
class TaskInfo:
    """Identity and rollout envelope of a task, safe to hand to adapters."""

    name: str
    max_steps: int
    control_hz: float | None = None
```

`Task.info` is a property returning it, mirroring `policy.info` /
`embodiment.info`. `control_hz` is the task's own rate (may be `None`); the
embodiment already knows its own rate, and the chunk-level rate of R1's
precedence chain cannot be known before inference, so `TaskInfo` does not
pretend to resolve R1 — it reports the task layer only.

No `epochs`, scenes, or scorers in `TaskInfo` (YAGNI; scoring and dataset
contents are none of the embodiment's business).

### `embodiment.py` — optional duck-typed hook

Exactly like `Policy`: the `Embodiment` Protocol is unchanged; its docstring
documents an optional `bind(task_info)` hook that is not part of the
Protocol, so every existing embodiment stays conformant. `EmbodimentBase`
ships a no-op default:

```python
def bind(self, task_info: TaskInfo) -> None:  # noqa: B027 - no-op default
    """Default: embodiments that don't display or pre-allocate per-task ignore it."""
```

### `eval.py` — call site

Immediately next to the policy bind, before `assert_compatible` (fail fast
before touching hardware), once per eval:

```python
bind_embodiment = getattr(embodiment, "bind", None)
if callable(bind_embodiment):
    bind_embodiment(task.info)
```

Error semantics match the policy hook: an exception from `bind` propagates
and aborts the eval before any rollout starts (no trial exists yet, so
nothing is recorded — same as a failing `assert_compatible`).

### Public API

`TaskInfo` is exported: add to `inspect_robots.__all__` and update
`tests/test_api_snapshot.py` together (the repo rule). `Task` is already
public.

### Out of scope (follow-up in `inspect-robots-yam`)

The yam embodiment implements `bind(task_info)` to drive the countdown from
the real horizon (`max_steps / control_hz`, preferring its own configured
rate as today) and drops the `max_steps_hint` config knob entirely — it
bounds nothing and is now auto-populated. That is a separate PR in the yam
repo, gated the same way.

## Compatibility notes

- Not a Protocol change: `runtime_checkable` `isinstance` checks and all
  existing embodiments (in-tree mock, isaacsim plugin, out-of-tree) are
  untouched.
- Name-collision risk (an existing embodiment with an unrelated `bind`
  attribute) is the same risk the policy side accepted; symmetry wins.
- `conformance.py` needs no change: the hook is optional, and
  `check_embodiment` checks declarative readiness, not optional hooks.
- Binding resolutions: R1 untouched (see above); the hook adds no rate
  authority. Nothing in plans/0001 §9–§11 reserves the embodiment surface.

## Testing

TDD; gates: 100% coverage, mypy strict, ruff (D1 docstrings).

- `Task.info` returns the name/max_steps/control_hz of the task, frozen.
- `eval()` calls `bind` on an embodiment that defines it, with the task's
  `TaskInfo`, before `reset` is ever called (ordering assert), and exactly
  once for multi-scene/multi-epoch tasks.
- `eval()` runs unchanged for embodiments without `bind` (the mock world
  keeps passing untouched — regression suite covers this for free) and for a
  non-callable `bind` attribute.
- A raising `bind` aborts the eval before any rollout (no log written —
  matching a compat failure).
- `EmbodimentBase.bind` default is a no-op (coverage).
- API snapshot updated alongside `__all__`.

## Documentation

- `embodiment.py` Protocol docstring documents the hook (mirroring
  `Policy`'s wording).
- `src/inspect_robots/CLAUDE.md` module-map rows for `task.py`,
  `embodiment.py` (and `policy.py`'s row already mentions its hook — keep
  the two rows parallel).
