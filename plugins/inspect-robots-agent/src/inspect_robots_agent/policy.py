"""LLMAgentPolicy — a frontier LLM as a first-class Inspect Robots policy.

The conversation loop lives inside ``act()``: observation in (labeled state
text + camera frames), one validated tool call out, synthesized into an
open-loop ``ActionChunk`` by the motion layer. The LLM never sees raw
actuation, and every emitted action still passes the rollout's approver
chain — this module contains no safety-critical code path of its own
(plan 0008 §4b).
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord
from inspect_robots_agent._llm import ENV_MODEL, ChatClient, ToolCall, resolve_provider
from inspect_robots_agent._png import png_data_url
from inspect_robots_agent._tools import Toolset, build_toolset

_MAX_CONSECUTIVE_FAILURES = 3

# reasoning_effort values accepted across OpenAI-compatible endpoints
# (Anthropic compat maps these to thinking effort; OpenRouter forwards them).
_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})

_SYSTEM_TEMPLATE = """You are controlling a real robot embodiment named {name!r} \
through tool calls. Each observation message gives you the current \
proprioceptive state and camera images. Work toward the user's goal in \
small, deliberate motions; re-check the observation after every motion. \
Safety approvers clamp out-of-bounds and too-fast actions below you. \
Respond with exactly one tool call per turn. When the goal is achieved call \
done; if it cannot be achieved call give_up. You have a budget of \
{budget} LLM calls for the whole trial."""


def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an image-free deep copy suitable for persistence or visualization."""
    sanitized = copy.deepcopy(messages)
    for message in sanitized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                content[index] = {
                    "type": "text",
                    "text": "[image omitted: streamed camera frame]",
                }
    return sanitized


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
    max_llm_calls: int = 100
    effort: str | None = "low"
    max_speed_frac: float = 0.1
    transcript_echo: bool = False


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
        max_llm_calls: int = 100,
        temperature: float | None = None,
        effort: str | None = "low",
        max_speed_frac: float = 0.1,
        transcript_echo: bool = False,
        transport: httpx.BaseTransport | None = None,
        env: dict[str, str] | None = None,
    ):
        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ValueError("max_speed_frac must be finite and > 0")
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
                f"effort must be one of {sorted(_EFFORT_LEVELS)}, or None to omit "
                f"the field, got {effort!r}"
            )
        self._client = ChatClient(provider, transport=transport)
        self._max_llm_calls = max_llm_calls
        self._temperature = temperature
        # Robot control is latency-sensitive: default to low reasoning effort
        # (the arm stands still while the model thinks; safety guardrails sit
        # below the model, so effort trades thinking time, not safety).
        self._effort = effort
        self._max_speed_frac = max_speed_frac
        self._transcript_echo = transcript_echo
        self.config = AgentPolicyConfig(
            temperature=temperature,
            model=provider.model,
            base_url=provider.base_url,
            api_key_env=api_key_env,
            max_llm_calls=max_llm_calls,
            effort=effort,
            max_speed_frac=max_speed_frac,
            transcript_echo=transcript_echo,
        )
        # Placeholder until bind(); eval() always binds before compat/rollout.
        self.info = PolicyInfo(name="agent", action_space=Box(shape=(1,)))
        self._toolset: Toolset | None = None
        self._embodiment_name = "(unbound)"
        self._embodiment_docs: str | None = None
        self._state_labels: tuple[str, tuple[str, ...]] | None = None
        self._messages: list[dict[str, Any]] = []
        self._delta_cursor = 0
        self._calls_used = 0

    # -- lifecycle ---------------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment's spaces and build the tool surface from them."""
        toolset = build_toolset(
            embodiment_info.action_space,
            embodiment_info.observation_space,
            embodiment_info.control_hz,
            self._max_speed_frac,
        )
        self._toolset = toolset
        self._state_labels = toolset.state_labels()
        self._embodiment_name = embodiment_info.name
        self._embodiment_docs = getattr(embodiment_info, "docs", None)
        self.info = PolicyInfo(
            name="agent",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        """Start a fresh per-trial conversation with the scene goal and call budget."""
        formatted = _SYSTEM_TEMPLATE.format(name=self._embodiment_name, budget=self._max_llm_calls)
        docs = self._embodiment_docs
        if docs is not None and docs.strip():
            formatted = formatted + "\n\nEmbodiment notes:\n" + docs.strip()
        self._messages = [
            {
                "role": "system",
                "content": formatted,
            },
            {"role": "user", "content": f"Goal: {scene.instruction}"},
        ]
        self._echo(f"[agent] goal: {scene.instruction}")
        self._delta_cursor = 0
        self._calls_used = 0

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Persist the transcript at the end of the trial."""
        if not self._messages:
            return

        transcript_dir = Path(log_dir) / "transcripts" / run_id
        transcript_dir.mkdir(parents=True, exist_ok=True)

        trial_id = f"{record.scene_id}-e{record.epoch}"
        path = transcript_dir / f"{trial_id}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for msg in self._messages:
                clean_msg = dict(msg)
                if isinstance(clean_msg.get("content"), list):
                    clean_content = [
                        part for part in clean_msg["content"] if part.get("type") != "image_url"
                    ]
                    clean_msg["content"] = clean_content
                f.write(json.dumps(clean_msg) + "\n")

        # Make path relative to log_dir for portability
        record.metadata["transcript"] = f"transcripts/{run_id}/{trial_id}.jsonl"

    def transcript(self) -> list[dict[str, Any]] | None:
        """Return an image-free deep copy of the current trial's conversation."""
        if not self._messages:
            return None
        return _sanitize(self._messages)

    def transcript_delta(self) -> list[dict[str, Any]] | None:
        """Sanitized messages appended since the previous call (core live-stream hook)."""
        new = self._messages[self._delta_cursor :]
        self._delta_cursor = len(self._messages)
        return _sanitize(new) if new else None

    # -- the loop ------------------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        """Run LLM turns until one validated tool call yields an action chunk."""
        toolset = self._toolset
        if toolset is None:
            raise RuntimeError(
                "LLMAgentPolicy.act() before bind(); run it through eval() or call "
                "policy.bind(embodiment.info) first"
            )
        self._messages.append(
            {
                "role": "user",
                "content": _observation_content(observation, self._state_labels),
            }
        )
        summary = f"{len(observation.images)} camera(s)"
        state_summary = " | ".join(_state_lines(observation, self._state_labels))
        if state_summary:
            summary = f"{summary}, {state_summary}"
        step_label = _step_label(observation)
        if step_label:
            self._echo(f"[agent] >> {step_label}: {summary}")
        else:
            self._echo(f"[agent] >> observation: {summary}")
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
            raw_message = message.raw()
            self._messages.append(raw_message)
            content = raw_message.get("content")
            if isinstance(content, str) and content:
                self._echo(f"[agent] << {content}")
            for tool_call in message.tool_calls:
                self._echo(f"[agent] << tool_call {tool_call.name}({tool_call.arguments})")

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
                self._echo("[agent] -- ignored: one tool call per turn")
            result = toolset.execute(call, observation)
            self._messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result.error or result.note}
            )
            self._echo(f"[agent] -- {result.error or result.note}")
            if result.error:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"LLM tool calls kept failing; last error: {result.error}")
                continue
            assert result.chunk is not None  # execute() sets exactly one of chunk/error
            return result.chunk

    def _forced_give_up(self, toolset: Toolset, observation: Observation, why: str) -> ActionChunk:
        # Echoed but never appended to _messages: the synthetic call is not
        # model output, so it stays out of the transcript.
        self._echo(f"[agent] -- {why}; forcing give_up")
        synthetic = ToolCall(id="budget", name="give_up", arguments=json.dumps({"reason": why}))
        result = toolset.execute(synthetic, observation)
        assert result.chunk is not None
        return result.chunk

    def _echo(self, text: str) -> None:
        if self._transcript_echo:
            print(text, file=sys.stderr, flush=True)


def _state_lines(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """One state[key] line per state entry, shared by prompt and echo so they never drift."""
    lines: list[str] = []
    for key, value in observation.state.items():
        array = np.asarray(value, dtype=np.float64)
        rounded = np.round(array, 4).tolist()
        if state_labels is not None and key == state_labels[0]:
            labels = state_labels[1]
            if array.shape == (len(labels),):
                labeled = " ".join(
                    f"{label}={item}" for label, item in zip(labels, rounded, strict=True)
                )
                lines.append(f"state[{key}]: {labeled}")
                continue
        lines.append(f"state[{key}]: {rounded}")
    return lines


def _step_label(observation: Observation) -> str:
    """Shared prompt/echo step gate: "step {n}" for int env_step (bool included), else ""."""
    step = observation.extra.get("env_step")
    return f"step {step}" if isinstance(step, int) else ""


def _observation_content(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """State as readable text plus camera frames as inline PNG data URLs."""
    lines = ["Current observation."]
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    lines.extend(_state_lines(observation, state_labels))
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    step_label = _step_label(observation)
    suffix = f" ({step_label})" if step_label else ""
    for name, image in observation.images.items():
        parts.append({"type": "text", "text": f"camera {name!r}{suffix}:"})
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
    return parts


def agent_policy(**kwargs: Any) -> LLMAgentPolicy:
    """Factory the Inspect Robots registry calls (entry point ``agent``).

    Accepts the same keyword arguments as
    [`LLMAgentPolicy`][inspect_robots_agent.policy.LLMAgentPolicy]; the CLI
    forwards ``-P key=value`` pairs here.
    """
    return LLMAgentPolicy(**kwargs)
