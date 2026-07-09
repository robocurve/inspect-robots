"""inspect-robots-xpolicylab — drive XPolicyLab-served policies from Inspect Robots.

`XPolicyLab <https://github.com/XPolicyLab/XPolicyLab>`_ unifies 40+ VLA and
imitation-learning policies behind one websocket policy-server contract. Start
any of its policy servers, then evaluate it like any other Inspect Robots
policy::

    inspect-robots list policies           # -> includes "xpolicylab"
    inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
        -P url=ws://gpu-box:19000 -P cameras=cam_head:base_rgb

or programmatically::

    from inspect_robots import eval
    from inspect_robots_xpolicylab import XPolicyLabPolicy

    with XPolicyLabPolicy(url="ws://gpu-box:19000") as policy:
        eval("my-task", policy, "isaacsim")

The policy is discovered via the ``inspect_robots.policies`` entry point, so it
shows up without being imported first.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_xpolicylab.policy import XPolicyLabPolicy

__all__ = ["XPolicyLabPolicy", "xpolicylab_policy"]

__version__ = "0.1.0"


def xpolicylab_policy(**kwargs: Any) -> XPolicyLabPolicy:
    """Factory the Inspect Robots registry calls (entry point ``xpolicylab``).

    Accepts the same keyword arguments as :class:`XPolicyLabPolicy`; the CLI
    forwards ``-P key=value`` pairs here (mapping-valued args accept the
    compact ``"key:value,key:value"`` string form).
    """
    return XPolicyLabPolicy(**kwargs)
