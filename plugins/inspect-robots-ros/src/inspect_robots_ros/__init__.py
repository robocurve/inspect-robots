"""Run Inspect Robots evaluations on ROS robots through rosbridge.

The ``ros`` embodiment is discovered through the
``inspect_robots.embodiments`` entry-point group. Its implementation is added
in the subsequent focused commits described by plan 0016.
"""

from __future__ import annotations

from inspect_robots_ros.embodiment import RosEmbodiment, ros_embodiment

__all__ = ["RosEmbodiment", "ros_embodiment"]

__version__ = "0.1.0"
