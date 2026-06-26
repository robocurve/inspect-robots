"""The Approver — a safety gate between policy output and the embodiment.

Every action passes through ``Approver.review`` before ``embodiment.step``. This
is the robotics analog of Inspect AI's ``ApprovalPolicy`` and is more
safety-critical: an approver may pass, clamp, or veto an action (a veto raises
:class:`~robolens.errors.SafetyAbort`). In the tracer slice the default approver
passes everything through; clamping/operator approval land in rollout hardening.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from robolens.types import Action


@runtime_checkable
class Approver(Protocol):
    """Reviews an action before it reaches the embodiment."""

    def review(self, action: Action, store: dict[str, Any]) -> Action: ...


class AutoApprover:
    """Approve every action unchanged (the permissive default)."""

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        return action
