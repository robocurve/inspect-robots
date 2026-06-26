"""RoboLens error taxonomy.

The split below resolves the "fail fast vs never-crash-overnight" tension:

- :class:`ConfigError` / :class:`CompatibilityError` are raised *before* any
  rollout — bad configuration should fail loudly and immediately.
- :class:`PolicyError` is recorded as a failed trial; whether it aborts the eval
  is governed by ``fail_on_error`` (Inspect semantics).
- :class:`EmbodimentFault` and :class:`SafetyAbort` *always* halt the eval
  regardless of ``fail_on_error`` — a faulted or unsafe robot must never
  auto-advance to the next scene unattended.
"""

from __future__ import annotations


class RoboLensError(Exception):
    """Base class for all RoboLens errors."""


class ConfigError(RoboLensError):
    """Invalid task / policy / embodiment configuration. Fail fast."""


class CompatibilityError(RoboLensError):
    """A policy and embodiment are not compatible. Fail fast, before any rollout."""


class PolicyError(RoboLensError):
    """The policy raised during inference. Recorded as a failed trial."""


class EmbodimentFault(RoboLensError):
    """The embodiment/robot faulted. Always halts the eval and requires a human."""


class SafetyAbort(RoboLensError):
    """An approver vetoed an action / e-stop. Always halts the eval."""
