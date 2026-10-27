"""inspect-robots-openvla — evaluate base OpenVLA from Inspect Robots.

Install, then use it by registry name from the CLI::

    inspect-robots list policies      # -> includes "openvla_local"
    inspect-robots run --task my-task --policy openvla_local --embodiment my-robot \
        -P camera=agentview -P unnorm_key=bridge_orig

or programmatically::

    from inspect_robots import eval
    from inspect_robots_openvla import OpenVLAPolicy

    with OpenVLAPolicy(unnorm_key="bridge_orig") as policy:
        eval("my-task", policy, "my-robot")

The adapter is discovered through the ``inspect_robots.policies`` entry point,
so it shows up without being imported first.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_openvla.policy import OpenVLAPolicy

__all__ = ["OpenVLAPolicy", "openvla_policy"]

__version__ = "0.1.0"


def openvla_policy(**kwargs: Any) -> OpenVLAPolicy:
    """Factory the Inspect Robots registry calls (entry point ``openvla_local``).

    Accepts the same keyword arguments as :class:`OpenVLAPolicy`; the CLI
    forwards ``-P key=value`` pairs here.
    """
    return OpenVLAPolicy(**kwargs)
