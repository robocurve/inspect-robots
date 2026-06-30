"""RoboInspect error taxonomy.

The split below resolves the "fail fast vs never-crash-overnight" tension:

- [`ConfigError`][roboinspect.errors.ConfigError] /
[`CompatibilityError`][roboinspect.errors.CompatibilityError] are raised *before* any
  rollout — bad configuration should fail loudly and immediately.
- [`PolicyError`][roboinspect.errors.PolicyError] is recorded as a failed trial; whether it
aborts the eval
  is governed by ``fail_on_error`` (Inspect semantics).
- [`EmbodimentFault`][roboinspect.errors.EmbodimentFault] and
[`SafetyAbort`][roboinspect.errors.SafetyAbort] *always* halt the eval
  regardless of ``fail_on_error`` — a faulted or unsafe robot must never
  auto-advance to the next scene unattended.
"""

from __future__ import annotations


class RoboInspectError(Exception):
    """Base class for all RoboInspect errors."""


class ConfigError(RoboInspectError):
    """Invalid task / policy / embodiment configuration. Fail fast."""


class CompatibilityError(RoboInspectError):
    """A policy and embodiment are not compatible. Fail fast, before any rollout."""


class PolicyError(RoboInspectError):
    """The policy raised during inference. Recorded as a failed trial."""


class EmbodimentFault(RoboInspectError):
    """The embodiment/robot faulted. Always halts the eval and requires a human."""


class SafetyAbort(RoboInspectError):
    """An approver vetoed an action / e-stop. Always halts the eval."""
