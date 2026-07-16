"""Run Inspect Robots evaluations on ROS 1 and ROS 2 robots through rosbridge.

The ``ros`` embodiment is discovered through the
``inspect_robots.embodiments`` entry-point group. Construction and ``.info``
remain network-free; the websocket connects on the first reset.
"""

from __future__ import annotations

from inspect_robots_ros.embodiment import RosEmbodiment, ros_embodiment

__all__ = ["RosEmbodiment", "ros_embodiment"]

__version__ = "0.1.0"
