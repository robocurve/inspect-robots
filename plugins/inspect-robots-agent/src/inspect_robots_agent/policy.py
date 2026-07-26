"""LLMAgentPolicy — a frontier LLM as a first-class Inspect Robots policy.

The conversation loop lives inside ``act()``: observation in (labeled state
text + camera frames), one validated tool call out, synthesized into an
open-loop ``ActionChunk`` by the motion layer. The LLM never sees raw
actuation, and every emitted action still passes the rollout's approver
chain — this module contains no safety-critical code path of its own
(plan 0008 §4b). Chat-format history stays canonical while the selected
client translates it to the configured API wire format.
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
import numpy.typing as npt

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord
from inspect_robots_agent._anthropic import _DEFAULT_MAX_OUTPUT_TOKENS, AnthropicClient
from inspect_robots_agent._llm import (
    _DIRECT_PROVIDERS,
    _OPENROUTER_BASE,
    ENV_MODEL,
    ChatClient,
    ToolCall,
    _has_openrouter_variant,
    resolve_provider,
)
from inspect_robots_agent._png import png_data_url
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent._tools import Toolset, build_toolset

_MAX_CONSECUTIVE_FAILURES = 3

#: The only endpoint that serves /v1/messages without an explicit base_url.
_ANTHROPIC_BASE = _DIRECT_PROVIDERS["anthropic"].base_url

# reasoning_effort values accepted across OpenAI-compatible endpoints
# (Anthropic compat maps these to thinking effort; OpenRouter forwards them).
# The native wire reuses the set as output_config.effort, where "none" and
# "minimal" are rejected and xhigh/max need a cap this client cannot stream;
# both surface as a guided 400 rather than a per-wire allowlist (plan 0026).
_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_WIRE_FORMATS = frozenset({"chat", "responses", "anthropic"})
_SPEEDS = frozenset({"fast"})
_IMAGE_MODES = frozenset({"always", "on_demand"})

_SYSTEM_TEMPLATE = """You are controlling a real robot embodiment named {name!r} \
through tool calls. Each observation message gives you the current \
proprioceptive state and camera images. Work toward the user's goal in \
small, deliberate motions; re-check the observation after every motion. \
Every move tool call must include a `note`: in one or two sentences, say what \
you observe in the current observation and why you chose this motion. The user \
is watching these notes to see what you see and what you decide, so write them \
for a human reader. \
Safety approvers clamp out-of-bounds and too-fast actions below you. \
Respond with exactly one tool call per turn. When the goal is achieved call \
done; if it cannot be achieved call give_up. You have a budget of \
{budget} LLM calls for the whole trial."""

_ON_DEMAND_SYSTEM_TEMPLATE = """You are controlling a real robot embodiment named {name!r} \
through tool calls. Each observation message gives you the current \
proprioceptive state. Camera images are not attached automatically; call \
`take_pic` to see them. A camera already shown for the current observation \
cannot be re-taken until the robot moves. Work toward the user's goal in \
small, deliberate motions; re-check the observation after every motion. \
Every move tool call must include a `note`: in one or two sentences, say what \
you observe in the current observation and why you chose this motion. The user \
is watching these notes to see what you see and what you decide, so write them \
for a human reader. \
Safety approvers clamp out-of-bounds and too-fast actions below you. \
Respond with exactly one motion tool call per turn; `take_pic` may be chained \
in the same turn. Placed after a motion, it returns frames once the motion has \
finished playing; placed alone, it looks before you decide what motion to make. \
When the goal is achieved call done; if it cannot be achieved call give_up. You \
have a budget of {budget} LLM calls for the whole trial."""

_ON_DEMAND_NUDGE = (
    "Respond with one motion tool call, one motion followed by take_pic, or take_pic alone."
)


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
    wire: str = "chat"
    speed: str | None = None
    #: Effective per-response cap on ``wire=anthropic``; ``None`` on the other
    #: wires, where nothing constrained the output.
    max_output_tokens: int | None = None
    max_llm_calls: int = 100
    effort: str | None = "low"
    max_speed_frac: float = 0.1
    transcript_echo: bool = False
    images: str = "always"


@dataclass(frozen=True, eq=False)
class _PendingCapture:
    """A capture request waiting for the observation produced after motion playout."""

    requested: tuple[str, ...] | None
    issued_step: object
    chunk_len: int
    target: npt.NDArray[np.float64] | None


class LLMAgentPolicy(PolicyBase):
    """Drives whatever embodiment it is bound to via LLM tool calls.

    Embodiment-adaptive: ``bind()`` (called by ``eval()`` before the
    compatibility check) adopts the embodiment's spaces and builds the tool
    surface from them. Conversation state is per-trial (``reset``), and the
    selected wire client translates that state without changing the loop.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        wire: str = "chat",
        speed: str | None = None,
        max_output_tokens: int | None = None,
        max_llm_calls: int = 100,
        temperature: float | None = None,
        effort: str | None = "low",
        max_speed_frac: float = 0.1,
        transcript_echo: bool = False,
        images: str = "always",
        transport: httpx.BaseTransport | None = None,
        env: dict[str, str] | None = None,
    ):
        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ValueError("max_speed_frac must be finite and > 0")
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be >= 1")
        if effort is not None and effort not in _EFFORT_LEVELS:
            raise ValueError(
                f"effort must be one of {sorted(_EFFORT_LEVELS)}, or None to omit "
                f"the field, got {effort!r}"
            )
        # Order matters from here down (plan 0026): wire is validated before
        # the params that are only legal on one wire, the api_key_env default
        # is applied before resolution, and the OpenRouter check after it.
        # The new wire-gated checks raise ConfigError so the CLI renders them;
        # the older ValueErrors above are inconsistent with that and tracked
        # separately in #168, since converting them changes existing behavior.
        if wire not in _WIRE_FORMATS:
            raise ValueError(f"wire must be one of {sorted(_WIRE_FORMATS)}, got {wire!r}")
        if speed is not None and speed not in _SPEEDS:
            raise ValueError(f"speed must be one of {sorted(_SPEEDS)}, or None, got {speed!r}")
        if images not in _IMAGE_MODES:
            raise ConfigError(
                f"images must be one of {sorted(_IMAGE_MODES)}, got {images!r}.\n"
                "fix: pass -P images=always or -P images=on_demand"
            )
        if wire != "anthropic":
            # A dropped speed would bill at standard rates while the user
            # believes fast mode is on; a dropped cap is a limit that never
            # applied. Both fail loudly rather than silently.
            if speed is not None:
                raise ConfigError(
                    f"speed is only supported on wire='anthropic', got wire={wire!r}.\n"
                    "fix: pass -P wire=anthropic, or drop -P speed="
                )
            if max_output_tokens is not None:
                raise ConfigError(
                    "max_output_tokens is only supported on wire='anthropic', got "
                    f"wire={wire!r}.\nfix: pass -P wire=anthropic, or drop "
                    "-P max_output_tokens="
                )
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be an int >= 1")

        environ = dict(os.environ) if env is None else env
        # resolve_provider reads api_key_env only when base_url is set, where
        # it otherwise defaults to OPENROUTER_API_KEY and would send an
        # OpenRouter key as x-api-key to a third-party gateway.
        # `not api_key_env` matches resolve_provider's own `api_key_env or
        # _OPENROUTER_KEY`, because `-P api_key_env=` parses to "": an
        # `is None` spelling here would let the empty form slip past and hand
        # that gateway the wrong secret. The base_url test is truthy for
        # consistency only, since resolve_provider ignores api_key_env when
        # base_url is falsy; it is load-bearing in the guard below.
        effective_key_env = api_key_env
        if wire == "anthropic" and base_url and not api_key_env:
            effective_key_env = "ANTHROPIC_API_KEY"
        requested_model = model or environ.get(ENV_MODEL)
        provider = resolve_provider(
            model=requested_model,
            base_url=base_url,
            api_key_env=effective_key_env,
            env=environ,
        )
        if wire == "anthropic" and not base_url and provider.base_url != _ANTHROPIC_BASE:
            # Only Anthropic's own endpoint serves /v1/messages. Resolution can
            # land elsewhere two ways: the OpenRouter fallback, or another
            # direct provider whose key happens to be set (openai/* with
            # $OPENAI_API_KEY). An explicit -P base_url= is the user's call.
            # Branch on the requested id, not provider.model: a direct
            # provider strips its own prefix, so the resolved id would read
            # as bare and draw a nonsense 'anthropic/gpt-5.6' suggestion.
            # The likeliest cause is a missing model prefix, not a missing
            # key: a bare id misses the direct-provider table and falls
            # through to OpenRouter even when $ANTHROPIC_API_KEY is set.
            asked = requested_model or ""
            stripped = asked.rpartition(":")[0] if _has_openrouter_variant(asked) else asked
            if not stripped.removeprefix("anthropic/"):
                # ':free', 'anthropic/', 'anthropic/:free': nothing usable is
                # left once the suffix comes off, so echoing the remainder
                # would name an empty id. Give a whole command instead.
                fix = "fix: pass a full model id (-P model=anthropic/claude-opus-5)"
            elif "/" in stripped and not stripped.startswith("anthropic/"):
                # A foreign prefix can never resolve to the Messages API, so
                # this is terminal whatever the suffix says. Decided before
                # the variant branch, which would otherwise spend a refusal
                # removing a suffix that was never the real problem.
                fix = "fix: use an anthropic/ model id"
            elif stripped != asked:
                # A :variant id routes here whatever keys are set, so naming
                # the key would send the user to fix something already right.
                # Repair the prefix in the same breath when both are wrong;
                # advice that earns a second refusal is worse than none.
                fix = f"fix: drop the OpenRouter variant suffix (-P model={stripped})"
                if "/" not in stripped:
                    fix = (
                        f"fix: use -P model=anthropic/{stripped} "
                        "(the :variant suffix routes to OpenRouter)"
                    )
            elif "/" not in asked:
                fix = f"fix: prefix the model id (-P model=anthropic/{asked})"
            else:
                # Anthropic-prefixed with a usable body: only the key is left.
                # Unreachable with the key set, since such an id resolves to
                # Anthropic's own endpoint and never reaches this guard.
                fix = "fix: set $ANTHROPIC_API_KEY"
            where = "OpenRouter" if provider.base_url == _OPENROUTER_BASE else provider.base_url
            raise ConfigError(
                "wire='anthropic' needs a Messages API endpoint, but the model "
                f"{asked!r} resolved to {where}, which does not serve one.\n"
                f"{fix}, or pass -P base_url=... for a gateway that serves /v1/messages"
            )

        resolved_max_output_tokens = (
            (max_output_tokens if max_output_tokens is not None else _DEFAULT_MAX_OUTPUT_TOKENS)
            if wire == "anthropic"
            else None
        )
        self._client: ChatClient | ResponsesClient | AnthropicClient
        if wire == "anthropic":
            assert resolved_max_output_tokens is not None
            self._client = AnthropicClient(
                provider,
                max_output_tokens=resolved_max_output_tokens,
                speed=speed,
                transport=transport,
            )
        elif wire == "responses":
            self._client = ResponsesClient(provider, transport=transport)
        else:
            self._client = ChatClient(provider, transport=transport)
        self._max_llm_calls = max_llm_calls
        self._temperature = temperature
        # Robot control is latency-sensitive: default to low reasoning effort
        # (the arm stands still while the model thinks; safety guardrails sit
        # below the model, so effort trades thinking time, not safety).
        self._effort = effort
        self._max_speed_frac = max_speed_frac
        self._transcript_echo = transcript_echo
        self._images = images
        self.config = AgentPolicyConfig(
            temperature=temperature,
            model=provider.model,
            base_url=provider.base_url,
            api_key_env=api_key_env,
            wire=wire,
            speed=speed,
            max_output_tokens=resolved_max_output_tokens,
            max_llm_calls=max_llm_calls,
            effort=effort,
            max_speed_frac=max_speed_frac,
            transcript_echo=transcript_echo,
            images=images,
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
        self._pending: _PendingCapture | None = None
        self._revealed: set[str] = set()

    # -- lifecycle ---------------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment's spaces and build the tool surface from them."""
        toolset = build_toolset(
            embodiment_info.action_space,
            embodiment_info.observation_space,
            embodiment_info.control_hz,
            self._max_speed_frac,
            images=self._images,
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
        template = _ON_DEMAND_SYSTEM_TEMPLATE if self._images == "on_demand" else _SYSTEM_TEMPLATE
        formatted = template.format(name=self._embodiment_name, budget=self._max_llm_calls)
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
        self._pending = None
        self._revealed.clear()

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Persist the transcript at the end of the trial."""
        messages = self.transcript()
        if not messages:
            return

        transcript_dir = Path(log_dir) / "transcripts" / run_id
        transcript_dir.mkdir(parents=True, exist_ok=True)

        trial_id = f"{record.scene_id}-e{record.epoch}"
        path = transcript_dir / f"{trial_id}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

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
        self._revealed.clear()
        reveal: tuple[str, ...] | None = None
        narration: str | None = None
        if self._images == "on_demand":
            reveal = ()
            pending = self._pending
            self._pending = None
            if pending is not None:
                requested = (
                    tuple(observation.images)
                    if pending.requested is None
                    else _unique_names(pending.requested)
                )
                reveal = tuple(name for name in requested if name in observation.images)
                missing = tuple(name for name in requested if name not in observation.images)
                self._revealed.update(reveal)
                narration = self._pending_narration(pending, observation, missing)
        observation_content = _observation_content(
            observation,
            self._state_labels,
            reveal=reveal,
            narration=narration,
        )
        if self._images == "on_demand":
            available = _quoted_names(tuple(observation.images))
            camera_line = (
                f"Cameras available to take_pic: {available}."
                if available
                else "No camera images are available to take_pic in this observation."
            )
            observation_content[0]["text"] += f"\n{camera_line}"
        self._messages.append({"role": "user", "content": observation_content})
        summary = f"{len(observation.images)} camera(s)"
        if self._images == "on_demand":
            summary += " available"
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
                    {
                        "role": "user",
                        "content": (
                            _ON_DEMAND_NUDGE
                            if self._images == "on_demand"
                            else "Respond with exactly one tool call."
                        ),
                    }
                )
                continue

            chunk: ActionChunk | None = None
            target: npt.NDArray[np.float64] | None = None
            immediate_frames: tuple[str, ...] = ()
            closed = False
            stopped = False
            last_error: str | None = None
            for call in message.tool_calls:
                result_text: str
                echo_text: str | None = None
                is_capture = self._images == "on_demand" and call.name == "take_pic"
                if not closed:
                    result = toolset.execute(call, observation)
                    if is_capture:
                        closed = True
                        if result.error is not None:
                            result_text = result.error
                            failures += 1
                            last_error = result.error
                        else:
                            requested = (
                                tuple(observation.images)
                                if result.capture is None
                                else _unique_names(result.capture)
                            )
                            skipped = tuple(name for name in requested if name in self._revealed)
                            immediate_frames = tuple(
                                name for name in requested if name not in self._revealed
                            )
                            if immediate_frames:
                                self._revealed.update(immediate_frames)
                                result_text = (
                                    f"captured {len(immediate_frames)} frame(s): "
                                    f"{_quoted_names(immediate_frames)}"
                                )
                                if skipped:
                                    result_text += f" (already shown: {_quoted_names(skipped)})"
                                failures = 0
                                echo_text = f"captured {len(immediate_frames)} frame(s)"
                            else:
                                result_text = (
                                    "already shown for this observation: "
                                    f"{_quoted_names(skipped)}; the view cannot change until "
                                    "the robot moves"
                                )
                    else:
                        if result.error is not None:
                            result_text = result.error
                            failures += 1
                            last_error = result.error
                            closed = True
                        else:
                            result_text = result.note
                            if result.chunk is not None:
                                chunk = result.chunk
                                target = result.target
                                stopped = bool(chunk.actions[0].meta.get("request_stop"))
                                closed = True
                elif is_capture and chunk is not None and not stopped and self._pending is None:
                    # The closed-state exception validates only the capture
                    # that may ride behind a motion. A failed validation keeps
                    # the queue slot open for a later call in this same turn.
                    result = toolset.execute(call, observation)
                    if result.error is not None:
                        result_text = result.error
                    else:
                        self._pending = _PendingCapture(
                            requested=result.capture,
                            issued_step=observation.extra.get("env_step"),
                            chunk_len=len(chunk),
                            target=target,
                        )
                        result_text = (
                            "queued: frames arrive with the next observation, "
                            "once the motion has finished playing"
                        )
                        requested = (
                            tuple(observation.images) if result.capture is None else result.capture
                        )
                        echo_text = f"queued capture: {_quoted_names(requested)}"
                elif is_capture and chunk is not None and stopped:
                    result_text = "ignored: the trial ends with this call"
                else:
                    result_text = "ignored: one tool call per turn"

                self._messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result_text}
                )
                self._echo(f"[agent] -- {echo_text or result_text}")

            if immediate_frames:
                self._messages.append(
                    {
                        "role": "user",
                        "content": _image_parts(observation, reveal=immediate_frames),
                    }
                )
            if chunk is not None:
                return chunk
            if failures >= _MAX_CONSECUTIVE_FAILURES:
                assert last_error is not None
                raise RuntimeError(f"LLM tool calls kept failing; last error: {last_error}")

    def _forced_give_up(self, toolset: Toolset, observation: Observation, why: str) -> ActionChunk:
        # Echoed but never appended to _messages: the synthetic call is not
        # model output, so it stays out of the transcript.
        self._echo(f"[agent] -- {why}; forcing give_up")
        synthetic = ToolCall(id="budget", name="give_up", arguments=json.dumps({"reason": why}))
        result = toolset.execute(synthetic, observation)
        if result.chunk is None:  # pragma: no cover - synthetic stop is structurally valid
            raise RuntimeError(f"forced give_up failed: {result.error}")
        return result.chunk

    def _pending_narration(
        self,
        pending: _PendingCapture,
        observation: Observation,
        missing: tuple[str, ...],
    ) -> str:
        """Describe observed playout, residual, and any cameras missing at delivery."""
        arriving_step = observation.extra.get("env_step")
        if (
            isinstance(pending.issued_step, int)
            and isinstance(arriving_step, int)
            and (advanced := arriving_step - pending.issued_step) > 0
        ):
            if advanced >= pending.chunk_len:
                narration = (
                    f"the motion finished playing ({pending.chunk_len} of "
                    f"{pending.chunk_len} steps)."
                )
            else:
                narration = (
                    f"the motion played {advanced} of {pending.chunk_len} steps before "
                    "this observation; it did not run to the end."
                )
        else:
            narration = "these frames follow the motion."

        toolset = self._toolset
        if pending.target is not None and toolset is not None:
            residual = toolset.residual(pending.target, observation)
            if residual is not None:
                label, magnitude = residual
                narration += (
                    " largest remaining offset from the requested target is "
                    f"{magnitude:.4g} on {label}."
                )
        if missing:
            narration += f" missing camera(s) in this observation: {_quoted_names(missing)}."
        return narration

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
    *,
    reveal: tuple[str, ...] | None = None,
    narration: str | None = None,
) -> list[dict[str, Any]]:
    """State as readable text plus camera frames as inline PNG data URLs."""
    lines = ["Current observation."]
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    lines.extend(_state_lines(observation, state_labels))
    if narration is not None:
        lines.append(narration)
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    parts.extend(_image_parts(observation, reveal=reveal))
    return parts


def _image_parts(
    observation: Observation,
    *,
    reveal: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Build frame labels and payloads without changing the report's label join key."""
    parts: list[dict[str, Any]] = []
    step_label = _step_label(observation)
    suffix = f" ({step_label})" if step_label else ""
    names = tuple(observation.images) if reveal is None else reveal
    for name in names:
        image = observation.images.get(name)
        if image is None:
            continue
        parts.append({"type": "text", "text": f"camera {name!r}{suffix}:"})
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
    return parts


def _quoted_names(names: tuple[str, ...]) -> str:
    """Render camera names in stable request order for model-facing narration."""
    return ", ".join(repr(name) for name in names)


def _unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate requested cameras without changing their first-seen order."""
    return tuple(dict.fromkeys(names))


def agent_policy(**kwargs: Any) -> LLMAgentPolicy:
    """Factory the Inspect Robots registry calls (entry point ``agent``).

    Accepts the same keyword arguments as
    [`LLMAgentPolicy`][inspect_robots_agent.policy.LLMAgentPolicy]; the CLI
    forwards ``-P key=value`` pairs here.
    """
    return LLMAgentPolicy(**kwargs)
