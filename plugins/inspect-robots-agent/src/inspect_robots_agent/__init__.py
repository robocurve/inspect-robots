"""inspect-robots-agent — frontier LLMs as first-class Inspect Robots policies.

An LLM behind any OpenAI-compatible API (OpenRouter, OpenAI, local
vLLM/Ollama, Anthropic's compat endpoint) drives whatever embodiment it is
paired with: each tool call becomes one smooth, approver-checked action
chunk. Registered as the policy ``agent``::

    inspect-robots "pick up the cube" --policy agent \
        -P model=anthropic/claude-fable-5 --embodiment cubepick

or programmatically::

    from inspect_robots import eval
    from inspect_robots_agent import LLMAgentPolicy

    eval("my-task", LLMAgentPolicy(model="openai/gpt-5.2"), "cubepick")

The policy is discovered via the ``inspect_robots.policies`` entry point, so
it shows up in ``inspect-robots list policies`` without being imported first.
"""

from __future__ import annotations

from importlib.metadata import version

from inspect_robots_agent._llm import (
    ENV_MODEL,
    AssistantMessage,
    ChatClient,
    Provider,
    resolve_provider,
)
from inspect_robots_agent._png import encode_png, png_data_url
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent.policy import AgentPolicyConfig, LLMAgentPolicy, agent_policy

__all__ = [
    "ENV_MODEL",
    "AgentPolicyConfig",
    "AssistantMessage",
    "ChatClient",
    "LLMAgentPolicy",
    "Provider",
    "ResponsesClient",
    "agent_policy",
    "encode_png",
    "png_data_url",
    "resolve_provider",
]

# Derived from package metadata so it can never drift from pyproject again
# (0.2.0 and 0.2.1 shipped with a stale hardcoded "0.1.0").
__version__ = version("inspect-robots-agent")
