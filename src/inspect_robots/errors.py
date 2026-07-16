"""Inspect Robots error taxonomy.

The split below resolves the "fail fast vs never-crash-overnight" tension:

- [`ConfigError`][inspect_robots.errors.ConfigError] /
[`CompatibilityError`][inspect_robots.errors.CompatibilityError] are raised *before* any
  rollout — bad configuration should fail loudly and immediately.
- [`PolicyError`][inspect_robots.errors.PolicyError] is recorded as a failed trial; whether it
aborts the eval
  is governed by ``fail_on_error`` (Inspect semantics).
- [`EmbodimentFault`][inspect_robots.errors.EmbodimentFault] and
[`SafetyAbort`][inspect_robots.errors.SafetyAbort] *always* halt the eval
  regardless of ``fail_on_error`` — a faulted or unsafe robot must never
  auto-advance to the next scene unattended.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord


class _CancelledTrial(KeyboardInterrupt):
    """Ctrl-C during a trial, carrying the partial record (mirrors exc.record)."""

    def __init__(self, message: str, record: TrialRecord) -> None:
        super().__init__(message)
        self.record = record


class InspectRobotsError(Exception):
    """Base class for all Inspect Robots errors.

    When an error is raised from inside a running trial, the rollout engine
    attaches the partial [`TrialRecord`][inspect_robots.rollout.TrialRecord] — the
    steps walked and the transcript events up to the failure — as ``record``,
    so the orchestrator can preserve it in logs. ``record`` is ``None`` for
    errors raised outside a rollout (configuration, compatibility, ...).
    """

    record: TrialRecord | None = None


class ConfigError(InspectRobotsError):
    """Invalid task / policy / embodiment configuration. Fail fast."""


class CompatibilityError(InspectRobotsError):
    """A policy and embodiment are not compatible. Fail fast, before any rollout."""


class PolicyError(InspectRobotsError):
    """The policy raised during inference. Recorded as a failed trial."""


class EmbodimentFault(InspectRobotsError):
    """The embodiment/robot faulted. Always halts the eval and requires a human."""


class SafetyAbort(InspectRobotsError):
    """An approver vetoed an action / e-stop. Always halts the eval."""
