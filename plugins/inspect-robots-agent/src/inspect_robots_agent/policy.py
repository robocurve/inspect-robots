"""LLMAgentPolicy — a frontier LLM as a first-class Inspect Robots policy.

The conversation loop lives inside ``act()``: observation in (labeled state
text + camera frames), one validated tool call out, synthesized into an
open-loop ``ActionChunk`` by the motion layer. The LLM never sees raw
actuation, and every emitted action still passes the rollout's approver
chain — this module contains no safety-critical code path of its own
(plan 0008 §4b).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_agent._llm import ENV_MODEL, ChatClient, ToolCall, resolve_provider
from inspect_robots_agent._png import png_data_url
from inspect_robots_agent._tools import Toolset, build_toolset

_MAX_CONSECUTIVE_FAILURES = 3

# reasoning_effort values accepted across OpenAI-compatible endpoints
# (Anthropic compat maps these to thinking effort; OpenRouter forwards them).
_EFFORT_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})

_SYSTEM_TEMPLATE = """You are controlling a real robot embodiment named {name!r} \
through tool calls. Each observation message gives you the current \
proprioceptive state and camera images. Work toward the user's goal in \
small, deliberate motions; re-check the observation after every motion. \
Safety approvers clamp out-of-bounds and too-fast actions below you. \
Respond with exactly one tool call per turn. When the goal is achieved call \
done; if it cannot be achieved call give_up. You have a budget of \
{budget} LLM calls for the whole trial."""


@dataclass(frozen=True)
class AgentPolicyConfig(PolicyConfig):
    """Inference-time configuration recorded in the eval log.

    Extends the core ``PolicyConfig``; ``eval()`` serializes configs with
    ``dataclasses.asdict``, so these fields land in ``EvalSpec.policy_config``
    for free.
    """

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_llm_calls: int = 50
    effort: str | None = "low"


class LLMAgentPolicy(PolicyBase):
    """Drives whatever embodiment it is bound to via LLM tool calls.

    Embodiment-adaptive: ``bind()`` (called by ``eval()`` before the
    compatibility check) adopts the embodiment's spaces and builds the tool
    surface from them. Conversation state is per-trial (``reset``).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        max_llm_calls: int = 50,
        temperature: float | None = None,
        effort: str | None = "low",
        transport: httpx.BaseTransport | None = None,
        env: dict[str, str] | None = None,
    ):
        environ = dict(os.environ) if env is None else env
        provider = resolve_provider(
            model=model or environ.get(ENV_MODEL),
            base_url=base_url,
            api_key_env=api_key_env,
            env=environ,
        )
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be >= 1")
        if effort is not None and effort not in _EFFORT_LEVELS:
            raise ValueError(
                f"effort must be one of {sorted(_EFFORT_LEVELS)} or none, got {effort!r}"
            )
        self._client = ChatClient(provider, transport=transport)
        self._max_llm_calls = max_llm_calls
        self._temperature = temperature
        # Robot control is latency-sensitive: default to low reasoning effort
        # (the arm stands still while the model thinks; safety guardrails sit
        # below the model, so effort trades thinking time, not safety).
        self._effort = effort
        self.config = AgentPolicyConfig(
            temperature=temperature,
            model=provider.model,
            base_url=provider.base_url,
            api_key_env=api_key_env,
            max_llm_calls=max_llm_calls,
            effort=effort,
        )
        # Placeholder until bind(); eval() always binds before compat/rollout.
        self.info = PolicyInfo(name="agent", action_space=Box(shape=(1,)))
        self._toolset: Toolset | None = None
        self._embodiment_name = "(unbound)"
        self._messages: list[dict[str, Any]] = []
        self._calls_used = 0

    # -- lifecycle ---------------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment's spaces and build the tool surface from them."""
        self._toolset = build_toolset(
            embodiment_info.action_space,
            embodiment_info.observation_space,
            embodiment_info.control_hz,
        )
        self._embodiment_name = embodiment_info.name
        self.info = PolicyInfo(
            name="agent",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        self._messages = [
            {
                "role": "system",
                "content": _SYSTEM_TEMPLATE.format(
                    name=self._embodiment_name, budget=self._max_llm_calls
                ),
            },
            {"role": "user", "content": f"Goal: {scene.instruction}"},
        ]
        self._calls_used = 0

    # -- the loop ------------------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        toolset = self._toolset
        if toolset is None:
            raise RuntimeError(
                "LLMAgentPolicy.act() before bind(); run it through eval() or call "
                "policy.bind(embodiment.info) first"
            )
        self._messages.append({"role": "user", "content": _observation_content(observation)})
        failures = 0
        while True:
            if self._calls_used >= self._max_llm_calls:
                return self._forced_give_up(toolset, observation, "LLM call budget exhausted")
            message = self._client.complete(
                self._messages,
                toolset.schemas(),
                temperature=self._temperature,
                reasoning_effort=self._effort,
            )
            self._calls_used += 1
            self._messages.append(message.raw())

            if not message.tool_calls:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"LLM produced no tool call in {failures} consecutive turns")
                self._messages.append(
                    {"role": "user", "content": "Respond with exactly one tool call."}
                )
                continue

            call, *extras = message.tool_calls
            for extra in extras:
                # Every tool_call id needs an answer on the wire; only the
                # first call per turn is executed.
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": extra.id,
                        "content": "ignored: one tool call per turn",
                    }
                )
            result = toolset.execute(call, observation)
            self._messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result.error or result.note}
            )
            if result.error:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"LLM tool calls kept failing; last error: {result.error}")
                continue
            assert result.chunk is not None  # execute() sets exactly one of chunk/error
            return result.chunk

    def _forced_give_up(self, toolset: Toolset, observation: Observation, why: str) -> ActionChunk:
        synthetic = ToolCall(id="budget", name="give_up", arguments=json.dumps({"reason": why}))
        result = toolset.execute(synthetic, observation)
        assert result.chunk is not None
        return result.chunk


def _observation_content(observation: Observation) -> list[dict[str, Any]]:
    """State as readable text plus camera frames as inline PNG data URLs."""
    lines = ["Current observation."]
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    for key, value in observation.state.items():
        rounded = np.round(np.asarray(value, dtype=np.float64), 4).tolist()
        lines.append(f"state[{key}]: {rounded}")
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    for name, image in observation.images.items():
        parts.append({"type": "text", "text": f"camera {name!r}:"})
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
    return parts


def agent_policy(**kwargs: Any) -> LLMAgentPolicy:
    """Factory the Inspect Robots registry calls (entry point ``agent``).

    Accepts the same keyword arguments as
    [`LLMAgentPolicy`][inspect_robots_agent.policy.LLMAgentPolicy]; the CLI
    forwards ``-P key=value`` pairs here.
    """
    return LLMAgentPolicy(**kwargs)
