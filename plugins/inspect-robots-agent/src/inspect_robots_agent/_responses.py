"""Stateless OpenAI Responses client with chat-history translation."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from typing import Any, cast

import httpx

from inspect_robots_agent._llm import AssistantMessage, Provider, ToolCall, _retry_delay

from ._capture import WireCapture


class ResponsesClient:
    """Blocking Responses client with raw reasoning-item replay and bounded retries.

    Chat-format messages remain the policy transcript. This boundary translates
    them to Responses input items and preserves raw output items required to
    replay reasoning-backed function calls. Provider ``Retry-After`` headers
    take precedence over exponential backoff.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        service_tier: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        capture: WireCapture | None = None,
    ):
        self._provider = provider
        self._service_tier = service_tier
        # Earlier and unknown models may reject the explicit-breakpoint fields.
        model = provider.model.removeprefix("openai/")
        self._cache_anchors = any(
            model == family or model.startswith(f"{family}-") for family in ("gpt-5.6", "gpt-6")
        )
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._capture = capture
        self._raw_items_by_call_id: dict[str, list[dict[str, Any]]] = {}
        self._cache_prefixes: set[str] = set()
        headers = {}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        self._http = httpx.Client(
            base_url=provider.base_url,
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """Return one assistant turn for the translated chat-format history."""
        history_call_ids = _history_call_ids(messages)
        self._raw_items_by_call_id = {
            call_id: items
            for call_id, items in self._raw_items_by_call_id.items()
            if call_id in history_call_ids
        }
        body: dict[str, Any] = {
            "model": self._provider.model,
            "input": _translate_messages(
                messages, self._raw_items_by_call_id, cache_anchors=self._cache_anchors
            ),
            "tools": _translate_tools(tools),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if self._cache_anchors:
            body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if self._service_tier is not None:
            body["service_tier"] = self._service_tier
        pending_prefixes = self._prepare_cache_reuse(body) if self._cache_anchors else set()

        last_error = "unknown error"
        for attempt in range(self._max_retries):
            retry_response: httpx.Response | None = None
            t_start = time.time() if self._capture is not None else 0.0
            try:
                response = self._http.post("/responses", json=body)
            except httpx.TransportError as exc:
                if self._capture is not None:
                    self._capture.record(
                        attempt=attempt,
                        endpoint="/responses",
                        request=body,
                        status=None,
                        response_text=None,
                        error=str(exc),
                        t_start=t_start,
                        duration_s=time.time() - t_start,
                    )
                last_error = str(exc)
            else:
                retry_response = response
                if self._capture is not None:
                    self._capture.record(
                        attempt=attempt,
                        endpoint="/responses",
                        request=body,
                        status=response.status_code,
                        response_text=response.text,
                        error=None,
                        t_start=t_start,
                        duration_s=time.time() - t_start,
                    )
                if response.status_code == 200:
                    payload = response.json()
                    message, output = _parse_response(payload)
                    if payload.get("status") == "completed":
                        self._cache_prefixes = pending_prefixes
                    for item in output:
                        if item.get("type") == "function_call":
                            self._raw_items_by_call_id[str(item["call_id"])] = output
                    return message
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in (429,) and response.status_code < 500:
                    raise RuntimeError(f"LLM request rejected — {last_error}")
            if attempt + 1 < self._max_retries:
                time.sleep(_retry_delay(retry_response, backoff_s=self._backoff_s, attempt=attempt))
        raise RuntimeError(f"LLM request failed after {self._max_retries} attempts — {last_error}")

    def _reset_cache_tracking(self) -> None:
        """Forget locally tracked prefix candidates at the start of a trial."""
        self._cache_prefixes.clear()

    def _prepare_cache_reuse(self, body: dict[str, Any]) -> set[str]:
        """Retain one earlier lookup endpoint, without assuming a server cache hit."""
        settings = {key: value for key, value in body.items() if key != "input"}
        prefix = hashlib.sha256(json.dumps(settings, separators=(",", ":")).encode())
        available: dict[str, dict[str, Any]] = {}
        logical: set[str] = set()
        for item in body["input"]:
            block = _cache_end_block(item)
            selected = block is not None and block.pop("prompt_cache_breakpoint", None) is not None
            # Include all intervening items (including raw reasoning), not just
            # the final block. A changed image invalidates every later prefix.
            prefix.update(b"\n")
            prefix.update(json.dumps(item, separators=(",", ":")).encode())
            if block is not None:
                fingerprint = prefix.hexdigest()
                available[fingerprint] = block
                if selected:
                    logical.add(fingerprint)

        surviving = self._cache_prefixes.intersection(available)
        selected_prefixes = set(logical)
        # Insertion order follows the prompt, so the last surviving endpoint
        # is the longest. Three logical targets plus this lookup use <=4 slots.
        previous = next((key for key in reversed(available) if key in surviving), None)
        if previous is not None:
            selected_prefixes.add(previous)
        for fingerprint in selected_prefixes:
            available[fingerprint]["prompt_cache_breakpoint"] = {"mode": "explicit"}
        # Only logical write targets enter history; the extra lookup marker
        # never invents a new target. Commit this set only after completion.
        return surviving | logical

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()


def _history_call_ids(messages: list[dict[str, Any]]) -> set[str]:
    """Collect every function call id still referenced by the submitted history."""
    call_ids: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_ids.add(str(call["id"]))
        if message.get("role") == "tool" and message.get("tool_call_id") is not None:
            call_ids.add(str(message["tool_call_id"]))
    return call_ids


def _translate_messages(
    messages: list[dict[str, Any]],
    raw_items_by_call_id: dict[str, list[dict[str, Any]]],
    *,
    cache_anchors: bool = False,
) -> list[dict[str, Any]]:
    """Translate canonical chat messages to stateless Responses input items."""
    items: list[dict[str, Any]] = []
    anchor_item: dict[str, Any] | None = None
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "tool":
            output = message["content"]
            if cache_anchors:
                output = (
                    [{"type": "input_text", "text": output}]
                    if isinstance(output, str)
                    else deepcopy(output)
                )
                _append_image_cache_anchor(output)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": output,
                }
            )
            continue

        tool_calls = message.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            cached = raw_items_by_call_id.get(str(tool_calls[0]["id"]))
            if cached is not None:
                items.extend(cached)
                continue

        content = message.get("content")
        if isinstance(content, list):
            content = _translate_content_parts(content)
        elif cache_anchors and role in {"system", "developer", "user"} and isinstance(content, str):
            content = [{"type": "input_text", "text": content}]
        if cache_anchors and role in {"system", "developer", "user"}:
            _append_image_cache_anchor(content)
        if isinstance(content, list) or role != "assistant" or content:
            item = {"role": role, "content": content}
            items.append(item)
            if cache_anchors:
                if index == 0 and role in {"system", "developer"}:
                    _mark_cache_breakpoint(item)
                if role == "user" and message.get("cache_anchor") is True:
                    anchor_item = item

        if role == "assistant":
            for call in tool_calls:
                function = call["function"]
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                )
    if cache_anchors:
        if anchor_item is not None:
            _mark_cache_breakpoint(anchor_item)
        if items:
            _mark_cache_breakpoint(items[-1])
    return items


def _append_image_cache_anchor(content: Any) -> None:
    """Keep a stable text endpoint after images; gateways may strip image markers."""
    if isinstance(content, list) and content and content[-1].get("type") == "input_image":
        # Keep this empty block even when unmarked so older prefixes stay equal.
        # This list is already a translated copy, never canonical or raw replay.
        content.append({"type": "input_text", "text": ""})


def _cache_end_block(item: dict[str, Any]) -> dict[str, Any] | None:
    """Find a supported tail block without touching assistant or reasoning replay."""
    if item.get("type") == "function_call_output":
        content = item.get("output")
    elif item.get("role") in {"system", "developer", "user"}:
        content = item.get("content")
    else:
        return None
    if isinstance(content, list) and content:
        block = content[-1]
        if isinstance(block, dict) and block.get("type") == "input_text":
            return cast(dict[str, Any], block)
    return None


def _mark_cache_breakpoint(item: dict[str, Any]) -> None:
    """Mark the exact end of an eligible item, including image-ending observations."""
    block = _cache_end_block(item)
    if block is not None:
        block["prompt_cache_breakpoint"] = {"mode": "explicit"}


def _translate_content_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate chat text and image parts to Responses input content parts."""
    translated: list[dict[str, Any]] = []
    for part in parts:
        if part["type"] == "text":
            translated.append({"type": "input_text", "text": part["text"]})
        elif part["type"] == "image_url":
            translated.append({"type": "input_image", "image_url": part["image_url"]["url"]})
    return translated


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten chat function schemas and explicitly retain non-strict behavior."""
    translated: list[dict[str, Any]] = []
    for tool in tools:
        function = tool["function"]
        translated.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
                "strict": False,
            }
        )
    return translated


def _parse_response(payload: dict[str, Any]) -> tuple[AssistantMessage, list[dict[str, Any]]]:
    """Parse usable assistant output while retaining the exact raw item list."""
    if payload.get("status") == "failed":
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(f"LLM response failed — {message or 'unknown error'}")

    output = cast(list[dict[str, Any]], payload["output"])
    texts: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(str(part["text"]))
        elif item.get("type") == "function_call":
            calls.append(
                ToolCall(
                    id=str(item["call_id"]),
                    name=str(item["name"]),
                    arguments=str(item["arguments"]),
                )
            )
    return (
        AssistantMessage(content="".join(texts) if texts else None, tool_calls=tuple(calls)),
        output,
    )
