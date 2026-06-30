"""robolens-isaacsim — an Isaac Lab (Isaac Sim) embodiment for RoboInspect.

Install this alongside a working Isaac Lab environment and the ``isaacsim``
embodiment becomes available to RoboInspect::

    roboinspect list embodiments        # -> includes "isaacsim"
    roboinspect run --task my-task --policy my-vla --embodiment isaacsim \
        -E task_id=Isaac-Lift-Cube-Franka-v0

or programmatically::

    from roboinspect import eval
    eval("my-task", "my-vla", "isaacsim")

The embodiment is discovered via the ``roboinspect.embodiments`` entry point, so it
shows up without being imported first.
"""

from __future__ import annotations

from typing import Any

from robolens_isaacsim.embodiment import IsaacSimEmbodiment

__all__ = ["IsaacSimEmbodiment", "isaacsim_embodiment"]

__version__ = "0.1.0"


def isaacsim_embodiment(**kwargs: Any) -> IsaacSimEmbodiment:
    """Factory the RoboInspect registry calls (entry point ``isaacsim``).

    Accepts the same keyword arguments as :class:`IsaacSimEmbodiment`; the CLI
    forwards ``-E key=value`` pairs here.
    """
    return IsaacSimEmbodiment(**kwargs)
