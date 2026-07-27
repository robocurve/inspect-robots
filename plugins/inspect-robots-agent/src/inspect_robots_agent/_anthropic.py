"""Native Anthropic Messages client with thinking-block replay (plan 0026).

The chat-completions message format stays the canonical transcript; this
boundary translates it to `/v1/messages` and back. Unlike the OpenAI-compat
shim, the native wire can carry fast mode (``speed="fast"`` plus its beta
header), which is the reason this client exists: robot control is
latency-sensitive, and fast mode buys latency without trading reasoning depth.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from inspect_robots_agent._llm import AssistantMessage, Provider, ToolCall

_ANTHROPIC_VERSION = "2023-06-01"
_FAST_MODE_BETA = "fast-mode-2026-02-01"

#: Applied by ``policy.py`` when the user leaves ``max_output_tokens`` unset.
#: Lives here so the number has one home, but the policy owns resolution: the
#: eval log records the effective value, and a second default would drift.
_DEFAULT_MAX_OUTPUT_TOKENS = 16000

_PNG_DATA_URL_PREFIX = "data:image/png;base64,"

# stop_reason values that carry a normal assistant turn. Anything else is
# terminal: default-deny, because enumerating the failures misses the ones
# added after this was written (model_context_window_exceeded is already one).
_CONTINUING_STOP_REASONS = frozenset({"end_turn", "tool_use", "stop_sequence"})

# Retried alongside 5xx and transport errors, matching the Anthropic SDKs.
_RETRYABLE_STATUSES = frozenset({408, 409, 429})

_CONNECT_TIMEOUT_S = 10.0
_MAX_READ_TIMEOUT_S = 600.0


class AnthropicClient:
    """Blocking Messages-API client with thinking-block replay and bounded retry.

    Retries 429/5xx and transport errors with exponential backoff; other 4xx
    fail immediately. Terminal ``stop_reason`` values raise rather than
    returning an empty turn, so a refusal or a truncation surfaces its own
    cause instead of being mistaken for "the model produced no tool call".
    """

    def __init__(
        self,
        provider: Provider,
        *,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        speed: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._provider = provider
        self._max_output_tokens = max_output_tokens
        # ~8s per 1k output tokens, floored at the 120s the other clients use
        # and capped at 600s (the SDKs' own ceiling; past that, streaming is
        # the real answer). A raised cap must not turn truncation into a
        # silent stall, but it must not stretch connect timeouts either.
        resolved_timeout = (
            timeout_s
            if timeout_s is not None
            else min(_MAX_READ_TIMEOUT_S, max(120.0, max_output_tokens / 125.0))
        )
        self._speed = speed
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        # tool_use id -> the verbatim content array of the response it came in.
        # Keyed on id alone because the API guarantees tool_use ids are unique
        # within a conversation; a gateway that recycles them would replay the
        # later turn's blocks for both turns.
        self._raw_blocks_by_tool_use_id: dict[str, list[dict[str, Any]]] = {}
        headers = {"anthropic-version": _ANTHROPIC_VERSION}
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        if speed == "fast":
            # The beta header is what makes POST /v1/messages the beta
            # endpoint over raw HTTP; there is no separate URL.
            headers["anthropic-beta"] = _FAST_MODE_BETA
        self._http = httpx.Client(
            base_url=provider.base_url,
            headers=headers,
            timeout=httpx.Timeout(resolved_timeout, connect=_CONNECT_TIMEOUT_S),
            transport=transport,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> AssistantMessage:
        """Return one assistant turn for the translated chat-format history."""
        # Prune before storing: the reverse order would evict every fresh
        # entry and make the cache always miss.
        history_ids = _history_tool_use_ids(messages)
        self._raw_blocks_by_tool_use_id = {
            tool_use_id: blocks
            for tool_use_id, blocks in self._raw_blocks_by_tool_use_id.items()
            if tool_use_id in history_ids
        }

        system, translated = _translate_messages(messages, self._raw_blocks_by_tool_use_id)
        body: dict[str, Any] = {
            "model": self._provider.model,
            "max_tokens": self._max_output_tokens,
            # Explicit, not omitted: omitting only means adaptive on Opus 5.
            # On Opus 4.8 and 4.7 it means thinking off, which would silently
            # halve this wire's target set and make the replay cache dead code.
            "thinking": {"type": "adaptive"},
            "messages": translated,
        }
        if system is not None:
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            body["tools"] = _translate_tools(tools)
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["output_config"] = {"effort": reasoning_effort}
        if self._speed is not None:
            body["speed"] = self._speed

        last_error = "unknown error"
        last_status: int | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._http.post("/messages", json=body)
            except httpx.TransportError as exc:
                last_error = str(exc)
                # Reset, or a 429 followed by a connection failure would emit
                # the fast-mode rate-limit guidance for the wrong cause.
                last_status = None
            else:
                last_status = response.status_code
                if response.status_code == 200:
                    payload = response.json()
                    message = _parse_response(payload)
                    for block in payload.get("content") or []:
                        if block.get("type") == "tool_use":
                            self._raw_blocks_by_tool_use_id[str(block["id"])] = list(
                                payload["content"]
                            )
                    return message
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in _RETRYABLE_STATUSES and response.status_code < 500:
                    raise RuntimeError(
                        f"LLM request rejected — {last_error}"
                        f"{self._rejection_guidance(response.text, temperature)}"
                    )
            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)

        guidance = ""
        # 429 only: _RETRYABLE_STATUSES also covers 408/409, which are not
        # rate limits and must not be told to drop fast mode.
        if last_status == 429 and self._speed == "fast":
            guidance = (
                "\nfix: fast mode has its own rate limit; retry later or drop "
                "-P speed=fast to fall back to standard speed"
            )
        raise RuntimeError(
            f"LLM request failed after {self._max_retries} attempts — {last_error}{guidance}"
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def _rejection_guidance(self, body: str, temperature: float | None) -> str:
        """Name the fix for the 4xx bodies this wire provokes, else say nothing."""
        lowered = body.lower()
        if self._speed == "fast" and "speed" in lowered:
            return (
                "\nfix: fast mode needs Claude Opus 5 or Opus 4.8 on the Claude API "
                "(not Bedrock, Vertex, Foundry, or Claude Platform on AWS), or drop "
                "-P speed=fast"
            )
        if temperature is not None and "temperature" in lowered:
            return (
                "\nfix: this model rejects temperature (Opus 5, 4.8, 4.7, Sonnet 5, and "
                "Fable 5 all do; Opus 4.6 and Sonnet 4.6 accept it); drop -P temperature="
            )
        if "effort" in lowered:
            return (
                "\nfix: the Messages API accepts -P effort= low, medium, high, "
                "xhigh, or max; none and minimal are OpenAI-only values"
            )
        return ""


def _history_tool_use_ids(messages: list[dict[str, Any]]) -> set[str]:
    """Collect every tool-use id still referenced by the submitted history."""
    tool_use_ids: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            tool_use_ids.add(str(call["id"]))
        if message.get("role") == "tool" and message.get("tool_call_id") is not None:
            tool_use_ids.add(str(message["tool_call_id"]))
    return tool_use_ids


def _translate_content_part(part: dict[str, Any]) -> dict[str, Any]:
    """Translate one user content part; unknown shapes raise rather than vanish."""
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": part["text"]}
    if part_type == "image_url":
        url = str(part.get("image_url", {}).get("url", ""))
        if not url.startswith(_PNG_DATA_URL_PREFIX):
            raise RuntimeError(
                f"image content part is not a PNG data URL: {url[:60]!r}; "
                "the agent policy emits png_data_url() output only"
            )
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": url[len(_PNG_DATA_URL_PREFIX) :],
            },
        }
    # A silently dropped camera frame is an invisible capability loss
    # mid-trial, so this wire raises where the Responses wire drops.
    raise RuntimeError(f"unsupported content part type {part_type!r}")


def _translate_content(content: Any) -> Any:
    """Keep plain-string content as-is; translate multipart lists part by part."""
    if isinstance(content, list):
        return [_translate_content_part(part) for part in content]
    return content


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthesize the content blocks for one assistant turn (cache-miss path)."""
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    for call in message.get("tool_calls") or []:
        name = call["function"]["name"]
        raw_arguments = call["function"]["arguments"]
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"tool call {name!r} has unparseable arguments: {exc}") from exc
        if not isinstance(parsed, dict):
            # A non-object input 400s with a message that does not name the
            # offending call, so name it here.
            raise RuntimeError(
                f"tool call {name!r} arguments must be a JSON object, got {type(parsed).__name__}"
            )
        blocks.append({"type": "tool_use", "id": call["id"], "name": name, "input": parsed})
    return blocks


def _with_cache_breakpoint(turn: dict[str, Any]) -> dict[str, Any]:
    """Return a turn with ephemeral caching on its last eligible content block."""
    content = turn.get("content")
    if isinstance(content, str):
        return {
            **turn,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    if not isinstance(content, list):
        return turn
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if block.get("type") in {"thinking", "redacted_thinking"}:
            continue
        copied_content = list(content)
        copied_content[index] = {
            **block,
            "cache_control": {"type": "ephemeral"},
        }
        return {**turn, "content": copied_content}
    return turn


def _translate_messages(
    messages: list[dict[str, Any]],
    raw_blocks_by_tool_use_id: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate chat history to (system, messages) for the Messages API.

    A leading system message is hoisted to the top-level ``system`` field; a
    system message anywhere else raises, since this loop never produces one.
    Consecutive tool messages merge into a single user turn: splitting
    ``tool_result`` blocks across messages is legal but silently trains the
    model out of parallel tool calls.
    """
    system: str | None = None
    translated: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []
    pending_anchor = False
    anchor_turn: dict[str, Any] | None = None

    def flush_results() -> None:
        nonlocal anchor_turn, pending_anchor
        if pending_results:
            turn = {"role": "user", "content": list(pending_results)}
            if pending_anchor:
                turn = _with_cache_breakpoint(turn)
                anchor_turn = turn
            translated.append(turn)
            pending_results.clear()
            pending_anchor = False

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            if index != 0:
                raise RuntimeError(
                    f"system message at index {index}; the Messages API takes a single "
                    "top-level system prompt"
                )
            content = message["content"]
            if not isinstance(content, str):
                raise RuntimeError(
                    "system message content must be a string; the Messages API takes "
                    "a single top-level system prompt"
                )
            system = content
            continue
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    # No is_error flag: act() folds tool failures into the
                    # content string, and the shared chat history has nowhere
                    # to carry the distinction (plan 0026).
                    "content": message["content"],
                }
            )
            pending_anchor = pending_anchor or message.get("cache_anchor") is True
            continue
        flush_results()
        if role == "assistant":
            calls = message.get("tool_calls") or []
            cached = raw_blocks_by_tool_use_id.get(str(calls[0]["id"])) if calls else None
            # A cache hit replaces every synthesized block for this message,
            # text included: emitting both would show the model its own text
            # twice and drop the thinking block that must be replayed verbatim.
            # A miss synthesizes text + tool_use with no thinking block, which
            # is lossy for a tool-use turn. No tool-use turn misses within a
            # trial (the cache is populated on every 200 that carries a tool
            # call, and the rollout is serial). A text-only turn does miss,
            # via the nudge path, but carries no tool_use and so no signature
            # or ordering constraint. The miss path also serves foreign
            # histories and non-thinking models.
            blocks = cached if cached is not None else _assistant_blocks(message)
            if not blocks:
                # The nudge retry path appends exactly this; an assistant
                # message with an empty content array is a 400.
                continue
            turn = {"role": "assistant", "content": blocks}
            if message.get("cache_anchor") is True:
                turn = _with_cache_breakpoint(turn)
                anchor_turn = turn
            translated.append(turn)
            continue
        if role != "user":
            raise RuntimeError(f"unsupported message role {role!r}")
        turn = {"role": role, "content": _translate_content(message["content"])}
        if message.get("cache_anchor") is True:
            turn = _with_cache_breakpoint(turn)
            anchor_turn = turn
        translated.append(turn)

    flush_results()
    if translated and translated[-1] is not anchor_turn:
        # Anthropic's 20-block lookback can miss after unusually heavy retry
        # churn, causing one harmless full-prefix write. The nudge string also
        # changes from a wrapped final block to a bare string once superseded,
        # so that final-breakpoint entry can miss once; the anchor still hits
        # and both cases self-heal on the next ordinary cycle.
        translated[-1] = _with_cache_breakpoint(translated[-1])
    return system, translated


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten OpenAI-nested tool schemas to the Messages API's flat shape."""
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"]["parameters"],
        }
        for tool in tools
    ]


def _parse_response(payload: dict[str, Any]) -> AssistantMessage:
    """Parse one Messages response, raising on any terminal ``stop_reason``."""
    stop_reason = payload.get("stop_reason")
    if stop_reason not in _CONTINUING_STOP_REASONS:
        raise RuntimeError(_terminal_message(payload, stop_reason))

    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in payload.get("content") or []:
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block["text"]))
        elif block_type == "tool_use":
            calls.append(
                ToolCall(
                    id=str(block["id"]),
                    name=str(block["name"]),
                    arguments=json.dumps(block["input"]),
                )
            )
        # thinking blocks matter only to the replay cache, not to the turn.
    joined = "".join(texts)
    # Never "": it would round-trip into the next request as an empty text
    # block, which is a 400.
    return AssistantMessage(content=joined or None, tool_calls=tuple(calls))


def _terminal_message(payload: dict[str, Any], stop_reason: Any) -> str:
    """Build the guided error for a stop_reason that carries no usable turn."""
    if stop_reason == "refusal":
        # stop_details is informational and may be null even on a refusal.
        category = (payload.get("stop_details") or {}).get("category")
        suffix = f" (category: {category})" if category else ""
        return f"LLM refused the request{suffix}"
    if stop_reason == "max_tokens":
        return "LLM response was truncated at the output limit\nfix: raise -P max_output_tokens="
    return f"LLM stopped with stop_reason {stop_reason!r}, which carries no usable turn"
