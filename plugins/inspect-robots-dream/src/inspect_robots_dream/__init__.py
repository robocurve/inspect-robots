"""Dream Labs' Dream-1 as an Inspect Robots policy for the bimanual YAM.

::

    inspect-robots run --embodiment yam_sim --policy dream -E control_hz=20 \\
        --instruction "put the red cube on the plate"

The policy is registered as ``dream`` through the ``inspect_robots.policies`` entry point.
Where the Dream client cannot be installed next to the embodiment (Isaac Sim), run
``inspect-robots-dream-bridge`` in its own environment and pass ``-P server_url=...``.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

from inspect_robots_dream.policy import DreamPolicy

__all__ = ["DreamPolicy", "dream_policy"]

__version__ = version("inspect-robots-dream")


def dream_policy(**kwargs: Any) -> DreamPolicy:
    """Build the ``dream`` policy; the CLI's ``-P key=value`` pairs arrive as keyword arguments."""
    return DreamPolicy(**kwargs)
