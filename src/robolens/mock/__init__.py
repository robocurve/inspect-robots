"""Dependency-free mock world and policies for testing and demos.

``CubePick`` is a deterministic 2D toy world: an end-effector must reach a cube.
It exercises the full stack — chunked open-loop execution, scoring, logging — in
CI with no hardware or simulator. This module must never import optional
dependencies (rerun, torch).
"""

from __future__ import annotations

from robolens.mock.cubepick import CubePickEmbodiment
from robolens.mock.policies import NoopPolicy, RandomPolicy, ScriptedPolicy

__all__ = ["CubePickEmbodiment", "NoopPolicy", "RandomPolicy", "ScriptedPolicy"]
