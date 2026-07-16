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

from inspect_robots_agent.policy import AgentPolicyConfig, LLMAgentPolicy, agent_policy

__all__ = ["AgentPolicyConfig", "LLMAgentPolicy", "agent_policy"]

# Derived from package metadata so it can never drift from pyproject again
# (0.2.0 and 0.2.1 shipped with a stale hardcoded "0.1.0").
__version__ = version("inspect-robots-agent")
