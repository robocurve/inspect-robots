# Embodiment `bind_task(envelope)`: let the body learn the rollout horizon

Date: 2026-07-14
Status: revised after subagent critique round 1 (renamed `TaskInfo` →
`TaskEnvelope` to keep the Inspect-faithful name free, hook renamed
`bind` → `bind_task`, dropped `control_hz`, added the optional-input adapter
contract and yam migration notes)

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

Mirror the mechanics of the policy-side precedent (plan 0008 §3c): an
optional duck-typed hook, not part of the Protocol, no-op default on the
base class, called by `eval()` before the compatibility check.

### `task.py` — `TaskEnvelope` and `Task.envelope`

A small frozen dataclass carrying what an adapter may legitimately learn
about the task it is running:

```python
@dataclass(frozen=True)
class TaskEnvelope:
    """Identity and rollout limits of a task, safe to hand to adapters."""

    name: str
    max_steps: int
```

`Task.envelope` is a property returning it (matching `Task`'s existing
derived-view properties `scorers` / `epoch_spec`).

Naming: NOT `TaskInfo` — Inspect AI exports a `TaskInfo` (`file`/`name`/
`attribs`, the `list_tasks()` listing entry), and plan 0001 §11 makes
Inspect API fidelity binding, so that name stays reserved for a future
faithful mirror. "Envelope" says what it is: the identity plus the limits of
the run.

No `control_hz`: the rollout loop does no pacing today (R1's chain is
explicitly unwired, `rollout._effective_control_hz`), a `SELF_PACED`
embodiment like yam paces at its own configured rate, and the chunk-level
rate that tops R1's precedence cannot be known before inference — so a
task-layer rate would only mislead adapters into wrong seconds math. Fields
can be added compatibly later; removing an exported one cannot. No `epochs`,
scenes, or scorers either (scoring and dataset contents are none of the
embodiment's business).

### `embodiment.py` — optional duck-typed hook

The `Embodiment` Protocol is unchanged; its docstring documents an optional
`bind_task(envelope)` hook that is not part of the Protocol, so every
existing embodiment stays conformant. `EmbodimentBase` ships a no-op
default:

```python
def bind_task(self, envelope: TaskEnvelope) -> None:  # noqa: B027 - no-op default
    """Default: embodiments with nothing to display or pre-allocate ignore it."""
```

`TaskEnvelope` is imported under `TYPE_CHECKING` (the same mechanics
`policy.py` uses for `EmbodimentInfo`) — no runtime import edge from
`embodiment.py` to `task.py`.

Naming: NOT bare `bind` — `policy.bind(embodiment_info)` means "receive your
counterpart's info"; an embodiment hook receiving the *task* is a different
relation, and reusing the name would leave it unavailable when policies
later want the envelope too (the agent plugin's `_max_llm_calls` knob is the
same duplication smell on the policy side). `bind_task` is unambiguous, can
be adopted by policies with the identical signature later, and avoids
duck-typing on `bind`, a common method name on transport-ish objects an
embodiment might proxy.

### `eval.py` — call site

Immediately next to the policy bind, before `assert_compatible` (fail fast
before touching hardware), once per `eval()`:

```python
bind_task = getattr(embodiment, "bind_task", None)
if callable(bind_task):
    bind_task(task.envelope)
```

Error semantics match the policy hook (verified): the call sits before sink
construction and `bus.on_eval_start`, so a raising `bind_task` propagates
and aborts with no log written — the same observable behavior as a
`CompatibilityError`, and no conflict with the "always persist a log once
rollouts have started" invariant (none has).

### The adapter contract (goes in the hook docstrings)

- **Optional input, not a guarantee:** the hook never fires on direct
  `rollout()` calls or on older cores that predate it. Adapters must keep a
  graceful fallback (yam: no horizon shown, exactly today's hint-unset
  behavior).
- **Re-bind, latest wins:** the hook fires once per `eval()`, which can be
  several times over an embodiment's lifetime (`eval_set`, or a caller
  reusing one instance across `eval()` calls). Each call replaces the
  previous envelope.

### Public API

`TaskEnvelope` is exported: `__init__.py` import + `__all__` +
`tests/test_api_snapshot.py` `EXPECTED` move together (any one missing fails
the suite). Docs self-render via mkdocstrings (`docs/api/index.md` already
includes `::: inspect_robots.task`).

### Out of scope (follow-up in `inspect-robots-yam`)

The yam embodiment implements `bind_task` and computes the countdown as
`envelope.max_steps / cfg.control_hz` — its own configured rate, which is
correct because yam is `SELF_PACED` and that rate is the one it actually
sleeps to. Migration constraints for that PR, recorded here because the core
design creates them:

- `max_steps_hint` cannot be deleted outright: `YamConfig.from_kwargs`
  raises `TypeError` on unknown keys, so operators with the key in
  `config.ini` would crash on upgrade. Deprecate first: accept the key,
  warn, use it only as the fallback when `bind_task` never fires; delete in
  a later release.
- Version coupling: yam pins `inspect-robots>=0.8`. Either bump the floor to
  the core release that ships `TaskEnvelope`, or keep the import under
  `TYPE_CHECKING` (yam already uses `from __future__ import annotations`) so
  the module still imports on older cores.

## Compatibility notes

- Not a Protocol change: `runtime_checkable` `isinstance` checks and all
  existing embodiments (in-tree mock, isaacsim plugin, out-of-tree) are
  untouched; none defines `bind_task`.
- `conformance.py` needs no change: `check_embodiment` checks declarative
  readiness from `EmbodimentInfo`, and the attribute-scanning helpers read
  only `DEVICE_SLOTS`/`RUNTIME_REQUIREMENTS`.
- Binding resolutions: R1 untouched (the envelope carries no rate
  authority); §11's Inspect-name fidelity is respected by avoiding
  `TaskInfo`.

## Testing

TDD; gates: 100% coverage, mypy strict, ruff (D1 docstrings).

- `Task.envelope` returns the task's name and `max_steps`, frozen.
- `eval()` calls `bind_task` on an embodiment that defines it, with the
  task's envelope, before `reset` is ever called (ordering assert), and
  exactly once per eval for multi-scene/multi-epoch tasks.
- Two sequential `eval()` calls against one embodiment instance re-bind with
  each task's envelope (latest wins).
- `eval()` runs unchanged for embodiments without the hook (the existing
  suite covers this for free) and for a non-callable `bind_task` attribute.
- A raising `bind_task` aborts the eval before any rollout, with no log
  written (matching a compat failure).
- `EmbodimentBase.bind_task` default is a no-op (coverage).
- API snapshot updated alongside `__all__`.

## Documentation

- `embodiment.py` Protocol docstring documents the hook and the adapter
  contract above (mirroring the wording style of `Policy`'s).
- `src/inspect_robots/CLAUDE.md` module-map rows for `task.py` and
  `embodiment.py` mention the envelope/hook, keeping parity with
  `policy.py`'s row.
