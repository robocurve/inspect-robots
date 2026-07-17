"""CaP-X code-as-policy integration for Inspect Robots."""

from __future__ import annotations

from importlib.metadata import version

from inspect_robots_capx.policy import CapxPolicy, CapxPolicyConfig, capx_policy

__all__ = ["CapxPolicy", "CapxPolicyConfig", "capx_policy"]

__version__ = version("inspect-robots-capx")
