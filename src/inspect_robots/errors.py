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


class InspectRobotsError(Exception):
    """Base class for all Inspect Robots errors."""


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
