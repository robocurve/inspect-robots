"""A typed transcript of rollout events.

Each trial records an ordered stream of events (reset, inference, step, approval,
operator judgement, error). This is the robotics analog of Inspect AI's
transcript and is the data a results viewer renders. Events are deliberately
lightweight: a ``kind``, the step index ``t`` (``-1`` for pre-loop events), and a
small data payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

EventKind = str  # "reset" | "inference" | "step" | "approval" | "operator" | "error"


@dataclass(frozen=True)
class Event:
    """One entry in a trial's transcript."""

    kind: EventKind
    t: int
    data: Mapping[str, Any] = field(default_factory=dict)


def reset_event(seed: int | None) -> Event:
    """Record the seed used before control step zero."""
    return Event(kind="reset", t=-1, data={"seed": seed})


def inference_event(t: int, latency_s: float | None, chunk_len: int) -> Event:
    """Record one policy call's timing and buffered action count at step ``t``."""
    return Event(kind="inference", t=t, data={"latency_s": latency_s, "chunk_len": chunk_len})


def step_event(t: int, terminated: bool, truncated: bool, reason: str | None) -> Event:
    """Record termination state after executing control step ``t``."""
    return Event(
        kind="step",
        t=t,
        data={"terminated": terminated, "truncated": truncated, "reason": reason},
    )


def approval_event(t: int, modified: bool, detail: str | None = None) -> Event:
    """Record whether the safety gate changed the proposed action at step ``t``."""
    return Event(kind="approval", t=t, data={"modified": modified, "detail": detail})


def operator_event(t: int, verdict: str) -> Event:
    """Record the operator's verdict after the rollout ends."""
    return Event(kind="operator", t=t, data={"verdict": verdict})


def error_event(t: int, error_type: str, message: str) -> Event:
    """Record an exception's type and message at the failing control step."""
    return Event(kind="error", t=t, data={"type": error_type, "message": message})
