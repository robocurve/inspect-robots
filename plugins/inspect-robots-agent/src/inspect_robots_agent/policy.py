"""LLMAgentPolicy — scaffold stub; the real implementation lands with the
tool/motion/client modules (plan 0008 §4b)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inspect_robots.policy import PolicyBase, PolicyConfig
from inspect_robots.types import ActionChunk, Observation


@dataclass(frozen=True)
class AgentPolicyConfig(PolicyConfig):
    """Inference-time configuration recorded in the eval log."""

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_llm_calls: int = 50


class LLMAgentPolicy(PolicyBase):
    """Placeholder that fails loudly until the agent loop lands."""

    def __init__(self, **kwargs: Any):
        raise NotImplementedError("inspect-robots-agent scaffold: policy lands in plan 0008 §4b")

    def act(self, observation: Observation) -> ActionChunk:  # pragma: no cover
        raise NotImplementedError


def agent_policy(**kwargs: Any) -> LLMAgentPolicy:
    """Factory the Inspect Robots registry calls (entry point ``agent``)."""
    return LLMAgentPolicy(**kwargs)
