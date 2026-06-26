"""Controllers — the rollout middleware layer (Inspect's ``@solver`` analog).

A :class:`Controller` owns the per-control-step decision of which action to send
to the embodiment. It internally decides *when* to call ``policy.act()`` (a slow
VLA inference returning an :class:`~robolens.types.ActionChunk`), buffers the
returned chunk, and pops the next action each step. This single-method, stateful
shape (R3) is what lets advanced controllers — e.g. a temporal-ensembling
controller that re-infers every step and blends overlapping predictions —
compose without forking the rollout loop.

``DefaultController`` plays the first ``replan_interval`` actions of each chunk,
then re-infers (``replan_interval=None`` ⇒ play the whole chunk before replanning).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol, runtime_checkable

from robolens.policy import Policy
from robolens.types import Action, Observation

_BUFFER_KEY = "_controller_action_buffer"
_LATENCY_KEY = "_controller_inference_latencies"


@runtime_checkable
class Controller(Protocol):
    """Decides the next action to execute, calling the policy as needed."""

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action: ...


class DefaultController:
    """Open-loop chunk execution with periodic replanning."""

    def __init__(self, replan_interval: int | None = None):
        if replan_interval is not None and replan_interval < 1:
            raise ValueError("replan_interval must be >= 1 or None")
        self.replan_interval = replan_interval

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        buffer: deque[Action] = store.setdefault(_BUFFER_KEY, deque())
        if not buffer:
            chunk = policy.act(observation)
            take = self.replan_interval or len(chunk)
            buffer.extend(list(chunk.actions)[:take])
            if chunk.inference_latency_s is not None:
                store.setdefault(_LATENCY_KEY, []).append(chunk.inference_latency_s)
        return buffer.popleft()
