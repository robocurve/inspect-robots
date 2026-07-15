"""Expose the ROS robot and its rosbridge streams through the embodiment contract."""

from __future__ import annotations

from typing import Any


class RosEmbodiment:
    """Placeholder for the ROS embodiment implemented by plan 0016."""

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError("RosEmbodiment is added in plan 0016 implementation step 5")


def ros_embodiment(**kwargs: Any) -> RosEmbodiment:
    """Construct the registry-facing ROS embodiment placeholder."""
    return RosEmbodiment(**kwargs)
