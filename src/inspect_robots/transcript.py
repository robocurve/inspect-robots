"""A typed transcript of rollout events.

Each trial records an ordered stream of events (reset, inference, step, approval,
operator feedback, operator judgement, error). This is the robotics analog of Inspect AI's
transcript and is the data a results viewer renders. Events are deliberately
lightweight: a ``kind``, the step index ``t`` (``-1`` for pre-loop events), and a
small data payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord

EventKind = str  # Includes reset, inference, step, approval, operator_message, operator, error.


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


def operator_message_event(t: int, text: str, source: str = "console") -> Event:
    """Record live feedback typed at the console or spoken in voice mode.

    This is distinct from the post-hoc operator verdict event.
    """
    return Event(kind="operator_message", t=t, data={"text": text, "source": source})


def operator_event(t: int, verdict: str, source: str = "prompt", note: str | None = None) -> Event:
    """Record the operator's verdict, optional note, and source after the rollout ends.

    ``verdict`` carries whatever the operator answered, which includes the
    non-verdict ``"skip"`` when they declined to grade a trial but still left a
    note. Consumers deriving an outcome from this event must treat ``"skip"``
    as "no judgement", never as a result.
    """
    return Event(
        kind="operator",
        t=t,
        data={"verdict": verdict, "source": source, "note": note},
    )


def judgement_source(record: TrialRecord) -> str | None:
    """Return the last operator-event source for a recorded judgement, if any."""
    if record.operator_judgement is None:
        return None
    for event in reversed(record.events):
        if event.kind == "operator":
            return event.data.get("source")
    return None


def error_event(t: int, error_type: str, message: str) -> Event:
    """Record an exception's type and message at the failing control step."""
    return Event(kind="error", t=t, data={"type": error_type, "message": message})
