"""LLMAgentPolicy — a frontier LLM as a first-class Inspect Robots policy.

The conversation loop lives inside ``act()``: observation in (labeled state
text, plus camera frames unless ``images=on_demand``), one validated tool call
out, synthesized into an open-loop ``ActionChunk`` by the motion layer. The
LLM never sees raw actuation, and every emitted action still passes the
rollout's approver chain — this module contains no safety-critical code path
of its own (plan 0008 §4b). Chat-format history stays canonical while the
selected client translates it to the configured API wire format.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

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
from inspect_robots_agent._depth import depth_parts, resolve_depth
from inspect_robots_agent._gemini_live import GeminiLiveClient
from inspect_robots_agent._interactions import InteractionsClient
from inspect_robots_agent._llm import (
    _DIRECT_PROVIDERS,
    _OPENROUTER_BASE,
    ENV_MODEL,
    ChatClient,
    Provider,
    ToolCall,
    _direct_claim,
    _has_openrouter_variant,
    resolve_provider,
)
from inspect_robots_agent._png import png_data_url
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent._tools import PreCheck, Toolset, build_toolset

from ._capture import WireCapture, _safe

_MAX_CONSECUTIVE_FAILURES = 3
# Shared by the camera label writer and the reader that recovers revealed
# camera names from it, so the two cannot drift apart again.
_CAMERA_LABEL_PREFIX = "camera "

#: Anthropic's base URL, allowlisted by the Messages-endpoint guard below.
_ANTHROPIC_BASE = _DIRECT_PROVIDERS["anthropic"].base_url

#: The HTTP endpoint resolution must land on before it can be upgraded to Live.
_GOOGLE_BASE = _DIRECT_PROVIDERS["google"].base_url

#: Key-free endpoint recorded in policy configuration and capture rows.
_GEMINI_LIVE_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

#: Google's native stateful HTTP endpoint, after direct-provider resolution.
_INTERACTIONS_BASE = "https://generativelanguage.googleapis.com/v1beta"

# reasoning_effort values accepted across OpenAI-compatible endpoints
# (Anthropic compat maps these to thinking effort; OpenRouter forwards them).
# The Messages wire reuses the set: "none" becomes thinking-disabled client-side
# and the rest go out as output_config.effort, of which Anthropic rejects
# "minimal", and its endpoint requires streaming for the cap xhigh/max
# need; Tinker accepts xhigh/max without streaming. Rejections get guided errors.
_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
# Some servers also take a continuous effort alongside the named levels: Tinker's
# OpenAI-compatible endpoint accepts a fraction (probed 2026-08-06 — 0.0 through
# 0.99 return 200, 0.995 and above 422). A fraction is passed through untouched
# rather than quantized into a level, so a sweep keeps the resolution the server
# offers. The range is half-open by construction: 1.0 means "past max effort",
# and servers that cap lower reject it with a guided 4xx.
_EFFORT_FRACTION_LIMIT = 1.0
_WIRE_FORMATS = frozenset({"chat", "responses", "messages", "gemini-live", "interactions"})
_WIRE_ALIASES = {"anthropic": "messages"}
_AGENT_NATIVE_WIRES = frozenset({"chat", "messages"})
_MESSAGES_CAPABLE_PREFIXES = frozenset(
    {"anthropic"}
    | {prefix for prefix, direct in _DIRECT_PROVIDERS.items() if direct.wire == "messages"}
)
_SPEEDS = frozenset({"fast"})
_IMAGE_MODES = frozenset({"always", "on_demand"})
_DEPTH_MODES = frozenset({"render", "off"})


class _Unset:
    """Marker type for constructor defaults resolved per wire."""


_UNSET: Final = _Unset()

# Duplicated in inspect_robots_capx/policy.py; keep both limits in sync.
_PRIOR_LEARNINGS_TEXT_LIMIT = 32 * 1024

_SYSTEM_TEMPLATE = """You are controlling a real robot embodiment named {name!r} \
through tool calls. Each observation message gives you the current \
proprioceptive state and camera images. Work toward the user's goal in \
small, deliberate motions; re-check the observation after every motion. \
Every move tool call must include a `note`: in one or two sentences, say what \
you observe in the current observation and why you chose this motion. The user \
is watching these notes to see what you see and what you decide, so write them \
for a human reader. \
Safety approvers clamp out-of-bounds and too-fast actions below you. \
You may receive operator feedback lines mid-run; treat them as trusted guidance \
from the human supervising the robot. \
Respond with exactly one tool call per turn. When the goal is achieved call \
done; if it cannot be achieved call give_up. Note what you are learning about \
this rig and task as you go: done and give_up will ask what you wish you had \
known from the start. You have a budget of \
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
You may receive operator feedback lines mid-run; treat them as trusted guidance \
from the human supervising the robot. \
Respond with exactly one motion tool call per turn; `take_pic` may be chained \
in the same turn. Placed after a motion, its frames arrive with the next \
observation, after the controller has played the motion; the narration reports \
how much actually played. Placed alone, it looks before you decide what motion \
to make. When the goal is achieved call done; if it cannot be achieved call \
give_up. Note what you are learning about this rig and task as you go: done and \
give_up will ask what you wish you had known from the start. You have a budget \
of {budget} LLM calls for the whole trial."""

_PRE_CHECK_PROMPT_CLAUSE = (
    " A motion pre-check may reject a move with a stated reason. Adjust the target "
    "rather than repeating a rejected move."
)

_ON_DEMAND_NUDGE = (
    "Respond with one motion tool call, one motion followed by take_pic, or take_pic alone."
)


def _validated_effort(effort: object) -> str | float:
    """Return the wire value for an accepted effort, else raise ``ConfigError``.

    Accepts a named level verbatim, or a number in ``[0.0, 1.0)`` normalized to
    ``float`` for the servers that read effort as a fraction. ``bool`` is not a
    number here: ``-P effort=false`` parses to ``False``, which would otherwise
    silently mean zero effort instead of failing as the typo it is.
    """
    if isinstance(effort, str):
        if effort in _EFFORT_LEVELS:
            return effort
    # The range check rejects nan and inf too: every nan comparison is False, and
    # inf fails the upper bound.
    elif (
        isinstance(effort, int | float)
        and not isinstance(effort, bool)
        and 0.0 <= effort < _EFFORT_FRACTION_LIMIT
    ):
        return float(effort)
    raise ConfigError(
        f"effort must be one of {sorted(_EFFORT_LEVELS)}, or a number in "
        f"[0.0, {_EFFORT_FRACTION_LIMIT}) on servers that take a fractional "
        f"effort, got {effort!r}.\n"
        "fix: omit -P effort= to use the provider default"
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


def _evicted_view(
    messages: list[dict[str, Any]],
    horizon: int,
    *,
    mark_anchor: bool = False,
) -> list[dict[str, Any]]:
    """Return a view with camera frames older than the image horizon stubbed."""
    image_message_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance((content := message.get("content")), list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
    ]
    stubbed_indices = image_message_indices[:-horizon]
    if not stubbed_indices:
        return list(messages)

    newest_stubbed = stubbed_indices[-1]
    view = list(messages)
    for message_index in stubbed_indices:
        message = messages[message_index]
        content = message["content"]
        assert isinstance(content, list)
        image_indices = [
            index
            for index, part in enumerate(content)
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        removed_indices = set(image_indices)
        for image_index in image_indices:
            if image_index == 0:
                continue
            label = content[image_index - 1]
            if isinstance(label, dict) and label.get("type") == "text":
                removed_indices.add(image_index - 1)
        stubbed_content = [
            part for index, part in enumerate(content) if index not in removed_indices
        ]
        stubbed_content.append(
            {
                "type": "text",
                "text": f"[{len(image_indices)} camera frame(s) elided]",
            }
        )
        stubbed_message = {**message, "content": stubbed_content}
        if mark_anchor and message_index == newest_stubbed:
            stubbed_message["cache_anchor"] = True
        view[message_index] = stubbed_message
    return view


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
    #: Canonical wire name; constructor input ``anthropic`` aliases ``messages``.
    wire: str = "chat"
    wire_capture: bool = True
    speed: str | None = None
    #: Effective per-response cap on ``wire=messages``; ``None`` on the other
    #: wires, where nothing constrained the output.
    max_output_tokens: int | None = None
    max_llm_calls: int = 100
    #: Resolved effort level; ``None`` means the field is omitted and the
    #: provider default applies. A number is a fractional effort, recorded as
    #: sent rather than snapped to a level.
    effort: str | float | None = None
    max_speed_frac: float = 0.1
    transcript_echo: bool = False
    images: str = "always"
    depth: str = "render"
    image_horizon: int | None = 2
    #: Resolved absolute path to the injected prior-learnings file.
    prior_learnings: str | None = None
    #: SHA-256 hexdigest of the injected prior-learnings text.
    prior_learnings_sha256: str | None = None
    #: Best-effort module and qualified-name identity of the motion pre-check.
    pre_check: str | None = None


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
    Metric camera depth is rendered by default; pass ``depth="off"`` to omit
    it without resolving any depth entries.
    ``prior_learnings`` optionally loads a UTF-8 notes file once at construction
    and appends its text to every trial's system prompt.
    """

    #: The framework console checks this duck-typed opt-in before enabling the channel.
    accepts_operator_messages: bool = True

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        wire: str | _Unset = _UNSET,
        wire_capture: bool = True,
        speed: str | None = None,
        max_output_tokens: int | None = None,
        max_llm_calls: int = 100,
        temperature: float | None = None,
        effort: str | float | None | _Unset = _UNSET,
        max_speed_frac: float = 0.1,
        transcript_echo: bool = False,
        images: str = "always",
        depth: str = "render",
        image_horizon: int | None | _Unset = _UNSET,
        prior_learnings: str | None = None,
        transport: httpx.BaseTransport | None = None,
        env: dict[str, str] | None = None,
        pre_check: PreCheck | None = None,
    ) -> None:
        prior_learnings_path: str | None = None
        prior_learnings_text: str | None = None
        prior_learnings_sha256: str | None = None
        if prior_learnings is not None:
            if not isinstance(prior_learnings, str) or not prior_learnings:
                raise ConfigError(
                    "prior_learnings must be a non-empty filesystem path string, "
                    f"got {prior_learnings!r}.\n"
                    "fix: the -P parser coerces unquoted values; pass "
                    "-P 'prior_learnings=\"path/to/learnings.md\"'"
                )
            path = Path(prior_learnings)
            try:
                prior_learnings_text = path.read_text(encoding="utf-8")
                prior_learnings_path = str(path.resolve())
            except (OSError, UnicodeDecodeError) as exc:
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} could not be read as UTF-8 "
                    f"({exc}).\n"
                    "fix: pass the path to a readable UTF-8 learnings file"
                ) from exc
            if not prior_learnings_text.strip():
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} is empty or whitespace-only.\n"
                    "fix: add concise notes to the file or omit -P prior_learnings="
                )
            if len(prior_learnings_text) > _PRIOR_LEARNINGS_TEXT_LIMIT:
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} has "
                    f"{len(prior_learnings_text)} characters; the limit is "
                    f"{_PRIOR_LEARNINGS_TEXT_LIMIT}.\n"
                    "fix: summarize it first and pass the path to the shorter learnings file"
                )
            prior_learnings_sha256 = hashlib.sha256(
                prior_learnings_text.encode("utf-8")
            ).hexdigest()

        pre_check_identity: str | None = None
        if pre_check is not None:
            if not callable(pre_check):
                raise ConfigError(
                    f"pre_check must be callable or None, got {pre_check!r}.\n"
                    "fix: -P CLI flags cannot carry callables; the pre_check hook is "
                    "programmatic-only"
                )
            module = getattr(pre_check, "__module__", None)
            qualname = getattr(pre_check, "__qualname__", None)
            if module is not None and qualname is not None:
                pre_check_identity = f"{module}.{qualname}"
            else:
                hook_type = type(pre_check)
                type_module = getattr(hook_type, "__module__", None)
                type_qualname = getattr(hook_type, "__qualname__", None)
                pre_check_identity = f"{type_module}.{type_qualname}"

        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ConfigError("max_speed_frac must be finite and > 0")
        if max_llm_calls < 1:
            raise ConfigError("max_llm_calls must be >= 1")
        environ = dict(os.environ) if env is None else env
        requested_model = model or environ.get(ENV_MODEL)
        direct_claim = (
            _direct_claim(requested_model, environ, native_wires=_AGENT_NATIVE_WIRES)
            if not base_url
            else None
        )
        # Order matters from here down (plan 0026): wire is validated before
        # the params that are only legal on one wire, the api_key_env default
        # is applied before resolution, and the OpenRouter check after it.
        # Every construction check raises ConfigError so the CLI renders a
        # guided message instead of a traceback (#168).
        wire_was_explicit = not isinstance(wire, _Unset)
        if isinstance(wire, _Unset):
            wire = direct_claim[1].wire if direct_claim is not None else "chat"
        else:
            wire = _WIRE_ALIASES.get(wire, wire)
            if wire not in _WIRE_FORMATS:
                raise ConfigError(f"wire must be one of {sorted(_WIRE_FORMATS)}, got {wire!r}")
        resolved_effort: str | float | None = None
        if not isinstance(effort, _Unset):
            resolved_effort = _validated_effort("none" if effort is None else effort)
        if wire == "gemini-live" and effort is not _UNSET:
            raise ConfigError(
                "effort is not supported on wire='gemini-live'.\nfix: drop -P effort="
            )
        if wire == "interactions" and resolved_effort not in {
            None,
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ConfigError(
                "effort on wire='interactions' must be minimal, low, medium, or high, "
                f"got {resolved_effort!r}.\n"
                "fix: pass -P effort=minimal|low|medium|high (maps to thinking_level), "
                "or drop -P effort="
            )
        if speed is not None and speed not in _SPEEDS:
            raise ConfigError(f"speed must be one of {sorted(_SPEEDS)}, or None, got {speed!r}")
        if images not in _IMAGE_MODES:
            raise ConfigError(
                f"images must be one of {sorted(_IMAGE_MODES)}, got {images!r}.\n"
                "fix: pass -P images=always or -P images=on_demand"
            )
        if depth not in _DEPTH_MODES:
            raise ConfigError(
                f"depth must be one of {sorted(_DEPTH_MODES)}, got {depth!r}.\n"
                "fix: pass -P depth=render or -P depth=off"
            )
        if wire != "messages":
            # Claude fast mode and the Messages output cap cannot apply on
            # other wires, so reject those mismatches during construction.
            # A Messages server may still ignore speed itself, as Tinker does.
            if speed is not None:
                raise ConfigError(
                    f"speed is only supported on wire='messages', got wire={wire!r}.\n"
                    "fix: pass -P wire=messages, or drop -P speed="
                )
            if max_output_tokens is not None:
                raise ConfigError(
                    "max_output_tokens is only supported on wire='messages', got "
                    f"wire={wire!r}.\nfix: pass -P wire=messages, or drop "
                    "-P max_output_tokens="
                )
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens < 1
        ):
            raise ConfigError("max_output_tokens must be an int >= 1")
        if (
            image_horizon is not _UNSET
            and image_horizon is not None
            and (
                isinstance(image_horizon, bool)
                or not isinstance(image_horizon, int)
                or image_horizon < 1
            )
        ):
            raise ConfigError(
                "image_horizon must be an int >= 1, or None to send full image history.\n"
                "fix: pass -P image_horizon=N or -P image_horizon=none"
            )
        resolved_image_horizon: int | None = None if wire in {"gemini-live", "interactions"} else 2
        if not isinstance(image_horizon, _Unset):
            resolved_image_horizon = image_horizon
        if wire == "gemini-live" and image_horizon is not _UNSET and image_horizon is not None:
            raise ConfigError(
                "image_horizon is not supported on wire='gemini-live'.\n"
                "fix: drop -P image_horizon=; the Live API's own context-window "
                "compression is the equivalent mechanism because already-streamed "
                "frames cannot be evicted"
            )
        if wire == "interactions" and image_horizon is not _UNSET and image_horizon is not None:
            raise ConfigError(
                "image_horizon is not supported on wire='interactions'.\n"
                "fix: drop -P image_horizon=; the Interactions API's server-side history "
                "is the equivalent mechanism because frames already absorbed by the chain "
                "cannot be evicted client-side"
            )

        if wire == "gemini-live" and base_url and not base_url.startswith(("ws://", "wss://")):
            raise ConfigError(
                "wire='gemini-live' requires a ws:// or wss:// base_url, got "
                f"{base_url!r}.\nfix: drop -P base_url= to use Google's Live API, "
                "or pass a websocket endpoint"
            )
        if wire == "interactions" and base_url and not base_url.startswith(("http://", "https://")):
            raise ConfigError(
                "wire='interactions' requires an http:// or https:// base_url, got "
                f"{base_url!r}.\nfix: wire='interactions' is HTTP; drop -P base_url= "
                "or pass the Live wire a websocket endpoint via -P wire=gemini-live"
            )

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
        if wire == "messages" and base_url and not api_key_env:
            effective_key_env = "ANTHROPIC_API_KEY"
        if wire == "gemini-live" and base_url and not api_key_env:
            effective_key_env = "GEMINI_API_KEY"
        if wire == "interactions" and base_url and not api_key_env:
            effective_key_env = "GEMINI_API_KEY"
        try:
            provider = resolve_provider(
                model=requested_model,
                base_url=base_url,
                api_key_env=effective_key_env,
                env=environ,
                native_wires=_AGENT_NATIVE_WIRES,
            )
        except ConfigError as exc:
            if wire not in {"gemini-live", "interactions"} or base_url:
                raise
            if wire == "interactions":
                raise ConfigError(
                    "wire='interactions' needs Google's direct provider.\n"
                    "fix: use -P model=google/... and set $GEMINI_API_KEY"
                ) from exc
            raise ConfigError(
                "wire='gemini-live' needs Google's direct Live API provider.\n"
                "fix: use -P model=google/... and set $GEMINI_API_KEY"
            ) from exc
        if (
            direct_claim is not None
            and direct_claim[1].wire != "chat"
            and wire_was_explicit
            and wire != direct_claim[1].wire
        ):
            prefix, direct = direct_claim
            fix = f"fix: drop -P wire= ({prefix}/* defaults to wire={direct.wire})"
            if wire in {"chat", "responses"}:
                fix += (
                    ", or pass -P base_url=... (+ -P api_key_env=NAME) to route this wire "
                    "through a gateway such as OpenRouter deliberately"
                )
            raise ConfigError(
                f"wire={wire!r} cannot drive {prefix}/* — the provider's direct "
                "endpoint serves only the Messages API.\n"
                f"{fix}"
            )
        if (
            wire == "messages"
            and not base_url
            and provider.wire != "messages"
            and provider.base_url != _ANTHROPIC_BASE
        ):
            # Resolution can land elsewhere through OpenRouter or another
            # direct provider. An explicit -P base_url= is the user's call.
            # Branch on the requested id, not provider.model: a direct
            # provider strips its own prefix, so the resolved id would read
            # as bare and draw a nonsense 'anthropic/gpt-5.6' suggestion.
            # The likeliest cause is a missing model prefix, not a missing
            # key: a bare id misses the direct-provider table and falls
            # through to OpenRouter even when $ANTHROPIC_API_KEY is set.
            asked = requested_model or ""
            stripped = asked.rpartition(":")[0] if _has_openrouter_variant(asked) else asked
            prefix, separator, body = stripped.partition("/")
            domestic = prefix in _MESSAGES_CAPABLE_PREFIXES
            stripped_body = body if domestic and separator else stripped
            example = (
                "thinkingmachines/Inkling"
                if prefix == "thinkingmachines"
                else "anthropic/claude-opus-5"
            )
            if not stripped_body:
                # ':free', 'anthropic/', 'anthropic/:free': nothing usable is
                # left once the suffix comes off, so echoing the remainder
                # would name an empty id. Give a whole command instead.
                fix = f"fix: pass a full model id (-P model={example})"
            elif "/" in stripped and not domestic:
                # A foreign prefix cannot resolve to a Messages endpoint.
                # Decide this before the variant branch, which would otherwise
                # remove a suffix that was never the real problem.
                fix = "fix: use an anthropic/ or thinkingmachines/ model id"
            elif stripped != asked:
                # A :variant id routes here whatever keys are set, so naming
                # the key would send the user to fix something already right.
                # Repair the prefix in the same breath when both are wrong;
                # advice that earns a second refusal is worse than none.
                fix = f"fix: drop the OpenRouter variant suffix (-P model={stripped})"
                if "/" not in stripped:
                    # A bare provider prefix must not be prefixed again:
                    # 'anthropic/thinkingmachines' would name no model.
                    fix = (
                        f"fix: pass a full model id (-P model={example})"
                        if stripped in _MESSAGES_CAPABLE_PREFIXES
                        else (
                            f"fix: use -P model=anthropic/{stripped} "
                            "(the :variant suffix routes to OpenRouter)"
                        )
                    )
            elif "/" not in asked:
                fix = (
                    f"fix: pass a full model id (-P model={example})"
                    if asked in _MESSAGES_CAPABLE_PREFIXES
                    else f"fix: prefix the model id (-P model=anthropic/{asked})"
                )
            else:
                # A Messages-capable prefix with a usable body has only its
                # direct-provider key left to fix. Reaching here implies
                # `domestic` (a foreign prefix took the branch above), so the
                # prefix is always a table entry.
                fix = f"fix: set ${_DIRECT_PROVIDERS[prefix].key_env}"
            where = "OpenRouter" if provider.base_url == _OPENROUTER_BASE else provider.base_url
            raise ConfigError(
                "wire='messages' needs a Messages API endpoint, but the model "
                f"{asked!r} resolved to {where}, which does not serve one.\n"
                f"{fix}, or pass -P base_url=... for a gateway that serves /v1/messages"
            )
        if wire == "gemini-live" and not base_url:
            if provider.base_url != _GOOGLE_BASE:
                raise ConfigError(
                    "wire='gemini-live' needs Google's direct Live API provider.\n"
                    "fix: use -P model=google/... and set $GEMINI_API_KEY"
                )
            provider = Provider(
                base_url=_GEMINI_LIVE_BASE,
                api_key=provider.api_key,
                model=provider.model,
            )
        if wire == "interactions" and not base_url:
            if provider.base_url != _GOOGLE_BASE:
                raise ConfigError(
                    "wire='interactions' needs Google's direct provider.\n"
                    "fix: use -P model=google/... and set $GEMINI_API_KEY"
                )
            provider = Provider(
                base_url=_INTERACTIONS_BASE,
                api_key=provider.api_key,
                model=provider.model,
            )

        resolved_max_output_tokens = (
            (max_output_tokens if max_output_tokens is not None else _DEFAULT_MAX_OUTPUT_TOKENS)
            if wire == "messages"
            else None
        )
        self._capture = WireCapture() if wire_capture else None
        self._client: (
            ChatClient | ResponsesClient | AnthropicClient | GeminiLiveClient | InteractionsClient
        )
        if wire == "messages":
            assert resolved_max_output_tokens is not None
            self._client = AnthropicClient(
                provider,
                max_output_tokens=resolved_max_output_tokens,
                speed=speed,
                transport=transport,
                capture=self._capture,
            )
        elif wire == "responses":
            self._client = ResponsesClient(provider, transport=transport, capture=self._capture)
        elif wire == "gemini-live":
            self._client = GeminiLiveClient(provider, capture=self._capture)
        elif wire == "interactions":
            self._client = InteractionsClient(provider, transport=transport, capture=self._capture)
        else:
            self._client = ChatClient(provider, transport=transport, capture=self._capture)
        self._max_llm_calls = max_llm_calls
        self._temperature = temperature
        # Preserve the operator's requested effort exactly; when it is unset,
        # omit the field so the provider's own default applies.
        self._effort = resolved_effort
        self._max_speed_frac = max_speed_frac
        self._transcript_echo = transcript_echo
        self._images = images
        self._depth = depth
        self._image_horizon = resolved_image_horizon
        self._prior_learnings_text = prior_learnings_text
        self._pre_check = pre_check
        self._explicit_base_url = bool(base_url)
        self._wire = wire
        self.config = AgentPolicyConfig(
            temperature=temperature,
            model=provider.model,
            base_url=provider.base_url,
            api_key_env=api_key_env,
            wire=wire,
            wire_capture=wire_capture,
            speed=speed,
            max_output_tokens=resolved_max_output_tokens,
            max_llm_calls=max_llm_calls,
            effort=resolved_effort,
            max_speed_frac=max_speed_frac,
            transcript_echo=transcript_echo,
            images=images,
            depth=depth,
            image_horizon=resolved_image_horizon,
            prior_learnings=prior_learnings_path,
            prior_learnings_sha256=prior_learnings_sha256,
            pre_check=pre_check_identity,
        )
        # Placeholder until bind(); eval() always binds before compat/rollout.
        self.info = PolicyInfo(name="agent", action_space=Box(shape=(1,)))
        self._toolset: Toolset | None = None
        self._embodiment_name = "(unbound)"
        self._embodiment_docs: str | None = None
        self._state_labels: tuple[str, tuple[str, ...]] | None = None
        self._hindsight: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._delta_cursor = 0
        self._calls_used = 0
        self._usage_totals: dict[str, int] = {}
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
            pre_check=self._pre_check,
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
        self._hindsight = None
        template = _ON_DEMAND_SYSTEM_TEMPLATE if self._images == "on_demand" else _SYSTEM_TEMPLATE
        formatted = template.format(name=self._embodiment_name, budget=self._max_llm_calls)
        if self._pre_check is not None:
            formatted += _PRE_CHECK_PROMPT_CLAUSE
        docs = self._embodiment_docs
        if docs is not None and docs.strip():
            formatted = formatted + "\n\nEmbodiment notes:\n" + docs.strip()
        if self._prior_learnings_text is not None:
            formatted = (
                formatted
                + "\n\nNotes from a previous attempt at tasks like this one. They may "
                + "be wrong or stale; the current observation always wins:\n"
                + self._prior_learnings_text
            )
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
        self._usage_totals.clear()
        self._pending = None
        self._revealed.clear()

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
        """Begin streaming wire attempts for the next trial when enabled."""
        if self._capture is not None:
            self._capture.begin_trial(log_dir, run_id, f"{_safe(scene_id)}-e{epoch}")

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Persist wire capture, hindsight, usage, and the transcript at trial end."""
        if isinstance(self._client, GeminiLiveClient):
            # Trial finalization must stay successful even if a half-dead Live
            # transport violates close()'s own best-effort contract.
            with suppress(Exception):
                self._client.close()
        if self._capture is not None:
            capture_path = self._capture.end_trial()
            if capture_path is not None:
                record.metadata["wire_capture"] = capture_path
            self._capture.warn_if_never_began()

        hindsight = self._hindsight.strip() if isinstance(self._hindsight, str) else ""
        if hindsight:
            record.metadata["hindsight"] = hindsight

        messages = self.transcript()
        if not messages:
            return

        transcript_dir = Path(log_dir) / "transcripts" / run_id
        transcript_dir.mkdir(parents=True, exist_ok=True)

        trial_id = f"{_safe(record.scene_id)}-e{record.epoch}"
        path = transcript_dir / f"{trial_id}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        # Make path relative to log_dir for portability
        record.metadata["transcript"] = f"transcripts/{run_id}/{trial_id}.jsonl"
        if self._calls_used:
            record.metadata["llm_usage"] = {
                "llm_calls": self._calls_used,
                **{key: value for key, value in self._usage_totals.items() if key != "llm_calls"},
            }

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
        depth = resolve_depth(observation) if self._depth == "render" else {}
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
            depth=depth,
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
        rejections = 0
        while True:
            if self._calls_used >= self._max_llm_calls:
                return self._forced_give_up(toolset, observation, "LLM call budget exhausted")
            outgoing = self._messages
            if self._image_horizon is not None:
                outgoing = _evicted_view(
                    self._messages,
                    self._image_horizon,
                    mark_anchor=isinstance(self._client, AnthropicClient),
                )
            message = self._client.complete(
                outgoing,
                toolset.schemas(),
                temperature=self._temperature,
                reasoning_effort=self._effort,
            )
            self._calls_used += 1
            if message.usage is not None:
                for key, value in message.usage.items():
                    self._usage_totals[key] = self._usage_totals.get(key, 0) + value
                self._echo(
                    "[agent] -- usage: "
                    f"in={message.usage.get('input_tokens', 0)} "
                    f"cache_read={message.usage.get('cache_read_input_tokens', 0)} "
                    f"out={message.usage.get('output_tokens', 0)}"
                )
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
                    error = f"LLM produced no tool call in {failures} consecutive turns"
                    if self._wire == "chat" and self._explicit_base_url:
                        error += (
                            "\nnote: some OpenAI-compatible endpoints accept `tools` but "
                            "silently ignore them (Tinker's OpenAI-compatible API is one). "
                            "If the provider serves the Messages API, retry with "
                            "-P wire=messages and its Messages base_url."
                        )
                    raise RuntimeError(error)
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
            closed_after_failure = False
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
                            closed_after_failure = True
                        elif result.capture == ():
                            result_text = "no camera images are available in this observation"
                            rejections += 1
                            if rejections > 1:
                                failures += 1
                                last_error = result_text
                            closed_after_failure = True
                        else:
                            requested = (
                                tuple(observation.images)
                                if result.capture is None
                                else _unique_names(result.capture)
                            )
                            # Exclude cameras from self._revealed if their image-bearing message
                            # was evicted by image_horizon
                            if self._image_horizon is not None:
                                active_camera_names: set[str] = set()
                                for msg in outgoing:
                                    content_list = msg.get("content")
                                    if isinstance(content_list, list):
                                        for part in content_list:
                                            if (
                                                isinstance(part, dict)
                                                and part.get("type") == "text"
                                            ):
                                                text = part.get("text", "")
                                                # The label is written with !r, and
                                                # repr() uses double quotes for a
                                                # name holding an apostrophe, so
                                                # accept whichever quote it chose.
                                                start = len(_CAMERA_LABEL_PREFIX) + 1
                                                quote = text[start - 1 : start]
                                                if (
                                                    text.startswith(_CAMERA_LABEL_PREFIX)
                                                    and quote in ("'", '"')
                                                    and (end := text.find(quote, start)) != -1
                                                ):
                                                    active_camera_names.add(text[start:end])
                                self._revealed.intersection_update(active_camera_names)

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
                                rejections += 1
                                if rejections > 1:
                                    failures += 1
                                    last_error = result_text
                                closed_after_failure = True
                    else:
                        if result.error is not None:
                            result_text = result.error
                            failures += 1
                            last_error = result.error
                            closed = True
                            closed_after_failure = True
                        else:
                            result_text = result.note
                            if result.chunk is not None:
                                chunk = result.chunk
                                target = result.target
                                stopped = bool(chunk.actions[0].meta.get("request_stop"))
                                if stopped:
                                    self._hindsight = chunk.actions[0].meta.get("stop_hindsight")
                                closed = True
                elif is_capture and chunk is not None and not stopped and self._pending is None:
                    # The closed-state exception validates only the capture
                    # that may ride behind a motion. A failed validation keeps
                    # the queue slot open for a later call in this same turn.
                    result = toolset.execute(call, observation)
                    if result.error is not None:
                        result_text = result.error
                    elif result.capture == ():
                        result_text = "no camera images are available in this observation"
                    else:
                        self._pending = _PendingCapture(
                            requested=result.capture,
                            issued_step=observation.extra.get("env_step"),
                            chunk_len=len(chunk),
                            target=target,
                        )
                        result_text = (
                            "queued: frames arrive with the next observation, "
                            "after the motion plays"
                        )
                        requested = (
                            tuple(observation.images) if result.capture is None else result.capture
                        )
                        echo_text = f"queued capture: {_quoted_names(requested)}"
                elif is_capture and chunk is not None and stopped:
                    result_text = "ignored: the trial ends with this call"
                elif closed_after_failure:
                    result_text = "ignored: an earlier call in this turn failed"
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
                        "content": _image_parts(
                            observation,
                            reveal=immediate_frames,
                            depth=depth,
                        ),
                    }
                )
            if chunk is not None:
                return chunk
            if failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"LLM tool calls kept failing; last error: {last_error or 'unknown'}"
                )

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
                    f"The motion finished playing ({pending.chunk_len} of "
                    f"{pending.chunk_len} steps)."
                )
            else:
                narration = (
                    f"The motion played {advanced} of {pending.chunk_len} steps before "
                    "this observation; it did not run to the end."
                )
        else:
            narration = "These frames follow the motion."

        toolset = self._toolset
        if pending.target is not None and toolset is not None:
            residual = toolset.residual(pending.target, observation)
            if residual is not None:
                label, magnitude = residual
                narration += (
                    " Largest remaining offset from the requested target is "
                    f"{magnitude:.4g} on {label}."
                )
        if missing:
            narration += f" Missing camera(s) in this observation: {_quoted_names(missing)}."
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


def _approvals_line(observation: Observation) -> str | None:
    approvals = observation.extra.get("approvals")
    if not isinstance(approvals, list) or not approvals:
        return None
    total = len(approvals)
    details = [str(a.get("detail")) for a in approvals if isinstance(a, dict) and a.get("detail")]
    if not details:
        return f"approver: {total} step(s) modified."
    counts = Counter(details)
    formatted: list[str] = []
    for flag, count in counts.items():
        if count > 1:
            formatted.append(f"{flag} \u00d7{count}")
        else:
            formatted.append(flag)
    detail_str = f" ({', '.join(formatted)})"
    return f"approver: {total} step(s) modified{detail_str}."


def _operator_lines(observation: Observation) -> list[str]:
    """Render only well-formed feedback entries from the framework-reserved channel."""
    messages = observation.extra.get("operator_messages")
    if not isinstance(messages, list):
        return []
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        step = message.get("t")
        text = message.get("text")
        if not isinstance(step, int) or not isinstance(text, str):
            continue
        lines.append(f"operator feedback (step {step}): {text}")
    return lines


def _observation_content(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None = None,
    *,
    reveal: tuple[str, ...] | None = None,
    narration: str | None = None,
    depth: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """State as readable text plus camera frames as inline PNG data URLs."""
    lines = ["Current observation."]
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    lines.extend(_state_lines(observation, state_labels))
    app_line = _approvals_line(observation)
    if app_line is not None:
        lines.append(app_line)
    lines.extend(_operator_lines(observation))
    if narration is not None:
        lines.append(narration)
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    parts.extend(_image_parts(observation, reveal=reveal, depth=depth))
    return parts


def _image_parts(
    observation: Observation,
    *,
    reveal: tuple[str, ...] | None = None,
    depth: Mapping[str, Any] | None = None,
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
        parts.append({"type": "text", "text": f"{_CAMERA_LABEL_PREFIX}{name!r}{suffix}:"})
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
        if depth is not None and name in depth:
            parts.extend(depth_parts(name, depth[name], step_label))
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
