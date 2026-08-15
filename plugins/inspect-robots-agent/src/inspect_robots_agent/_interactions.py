"""Stateful Gemini Interactions HTTP client with bounded fold recovery.

The policy's append-only chat-format list remains the conversation source of
truth. This boundary sends only new user and tool entries while Google's
interaction chain is intact, then folds the authoritative history into a
fresh unchained request when the remote chain is lost.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from inspect_robots_agent._gemini_live import _RECOVERY_CONTINUATION, _RECOVERY_PROLOGUE
from inspect_robots_agent._llm import AssistantMessage, Provider, ToolCall

from ._capture import WireCapture

_SUCCESS_STATUSES = frozenset({"completed", "requires_action", "incomplete"})
_USAGE_KEYS = {
    "total_input_tokens": "input_tokens",
    "total_output_tokens": "output_tokens",
    "total_cached_tokens": "cache_read_input_tokens",
    "total_thought_tokens": "thought_tokens",
    "total_tokens": "total_tokens",
}


class InteractionsClient:
    """Send append-only chat deltas over Google's stateful Interactions API.

    The streamed cursor contains references, not copied values. A changed
    prefix therefore starts a new chain even when two trials have
    byte-identical prompts. Lost server chains recover by folding the full
    authoritative view into one unchained request.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        capture: WireCapture | None = None,
    ) -> None:
        self._provider = provider
        self._model = provider.model.removeprefix("google/")
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._capture = capture
        headers = {}
        if provider.api_key:
            headers["x-goog-api-key"] = provider.api_key
        self._http = httpx.Client(
            base_url=provider.base_url,
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )
        self._streamed: list[dict[str, Any]] = []
        self._last_interaction_id: str | None = None
        self._needs_fold = False
        # The live endpoint requires the function name on every
        # function_result step (probed 2026-08-14; a call_id-only step gets
        # an opaque 400), and the chat-format tool message carries only the
        # id, so names are recorded at parse time as on the Live wire.
        self._call_names: dict[str, str] = {}

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """Return one assistant turn after sending only the unseen suffix."""
        self._check_identity_prefix(messages)
        suffix = messages[len(self._streamed) :]
        last_error = "unknown error"

        for attempt in range(self._max_retries):
            folding = self._needs_fold
            input_items = (
                self._fold_input(messages)
                if folding
                else _translate_suffix(suffix, self._call_names)
            )
            body = self._request_body(
                messages,
                tools,
                input_items,
                temperature,
                reasoning_effort,
                include_previous=not folding,
            )
            sent_previous = "previous_interaction_id" in body
            t_start = time.time() if self._capture is not None else 0.0
            try:
                response = self._http.post("/interactions", json=body)
            except httpx.TransportError as exc:
                last_error = str(exc)
                self._record_attempt(
                    attempt,
                    body,
                    None,
                    None,
                    last_error,
                    t_start,
                    time.time() - t_start if self._capture is not None else 0.0,
                )
            else:
                self._record_attempt(
                    attempt,
                    body,
                    response.status_code,
                    response.text,
                    None,
                    t_start,
                    time.time() - t_start if self._capture is not None else 0.0,
                )
                if response.status_code == 200:
                    result, interaction_id = _parse_response(response)
                    self._last_interaction_id = interaction_id
                    if folding:
                        self._streamed = list(messages)
                        self._needs_fold = False
                        # Pre-fold calls live inside the sanitized prologue
                        # now; only calls parsed from here on can still
                        # receive a result as a delta.
                        self._call_names.clear()
                    else:
                        self._streamed.extend(suffix)
                    for call in result.tool_calls:
                        self._call_names[call.id] = call.name
                    return result

                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                chain_lost = sent_previous and (
                    response.status_code == 404
                    or (
                        response.status_code == 400
                        and "previous_interaction" in response.text.lower()
                    )
                )
                if chain_lost:
                    self._needs_fold = True
                elif response.status_code not in (429,) and response.status_code < 500:
                    guidance = ""
                    if "thinking_level" in response.text.lower():
                        guidance = (
                            "\nfix: supported thinking_level values vary by model; "
                            "gemini-3.7-flash accepts low|medium|high"
                        )
                    raise RuntimeError(f"LLM request rejected — {last_error}{guidance}")

            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)

        raise RuntimeError(f"LLM request failed after {self._max_retries} attempts — {last_error}")

    def close(self) -> None:
        """Close the underlying blocking HTTP client."""
        self._http.close()

    def _check_identity_prefix(self, messages: list[dict[str, Any]]) -> None:
        prefix_matches = len(messages) >= len(self._streamed) and all(
            messages[index] is streamed for index, streamed in enumerate(self._streamed)
        )
        if prefix_matches:
            return
        if messages and self._streamed and messages[0] is self._streamed[0]:
            raise RuntimeError(
                "Interactions received a rewritten conversation view without a trial reset"
            )
        self._streamed.clear()
        self._last_interaction_id = None
        self._needs_fold = False
        self._call_names.clear()

    def _request_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        temperature: float | None,
        reasoning_effort: str | float | None,
        *,
        include_previous: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "store": True,
            "tools": [_translate_tool(tool) for tool in tools],
        }
        if messages and messages[0].get("role") == "system":
            body["system_instruction"] = str(messages[0].get("content", ""))
        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if reasoning_effort is not None:
            generation_config["thinking_level"] = reasoning_effort
        if generation_config:
            body["generation_config"] = generation_config
        if include_previous and self._last_interaction_id is not None:
            body["previous_interaction_id"] = self._last_interaction_id
        return body

    def _fold_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchor_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        if anchor_index is None:
            raise RuntimeError("Interactions fold recovery requires at least one user message")

        # Deferred to avoid policy.py -> _interactions.py -> policy.py at import.
        from inspect_robots_agent.policy import _sanitize

        start = 1 if messages[0].get("role") == "system" else 0
        folded = _sanitize(messages[start:anchor_index] + messages[anchor_index + 1 :])
        lines = [
            _RECOVERY_PROLOGUE,
            *(json.dumps(message) for message in folded),
            _RECOVERY_CONTINUATION,
        ]
        return [
            {"type": "text", "text": "\n".join(lines)},
            *_user_content(messages[anchor_index]),
        ]

    def _record_attempt(
        self,
        attempt: int,
        body: dict[str, Any],
        status: int | None,
        response_text: str | None,
        error: str | None,
        t_start: float,
        duration_s: float,
    ) -> None:
        if self._capture is None:
            return
        self._capture.record(
            attempt=attempt,
            endpoint="/interactions",
            request=body,
            status=status,
            response_text=response_text,
            error=error,
            t_start=t_start,
            duration_s=duration_s,
        )


def _translate_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    return {
        "type": "function",
        "name": function["name"],
        "description": function["description"],
        "parameters": function["parameters"],
    }


def _translate_suffix(
    messages: list[dict[str, Any]],
    call_names: dict[str, str],
) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            translated.extend(_user_content(message))
        elif role == "tool":
            call_id = str(message["tool_call_id"])
            name = call_names.get(call_id)
            if name is None:
                raise RuntimeError(
                    f"Interactions has no function name recorded for tool call {call_id!r}"
                )
            content = message.get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"Interactions tool result {call_id!r} content must be a string")
            translated.append(
                {"type": "function_result", "name": name, "call_id": call_id, "result": content}
            )
        elif role in {"assistant", "system"}:
            continue
        else:
            raise RuntimeError(f"unsupported chat message role {role!r}")
    if not translated:
        raise RuntimeError("wire='interactions' has no un-streamed user or tool message to send")
    return translated


def _user_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [_content_part(part) for part in content]
    raise RuntimeError(f"unsupported Interactions user content {content!r}")


def _content_part(part: Any) -> dict[str, Any]:
    if not isinstance(part, dict):
        raise RuntimeError(f"unsupported content part type {type(part).__name__!r}")
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": part["text"]}
    if part_type == "image_url":
        image_url = part.get("image_url")
        url = str(image_url.get("url", "")) if isinstance(image_url, dict) else ""
        header, separator, data = url.partition(",")
        if not separator or not header.startswith("data:") or not header.endswith(";base64"):
            raise RuntimeError(f"image content part is not a base64 data URI: {url[:60]!r}")
        mime_type = header[len("data:") : -len(";base64")]
        if not mime_type:
            raise RuntimeError(f"image content part has no data-URI MIME type: {url[:60]!r}")
        return {"type": "image", "mime_type": mime_type, "data": data}
    raise RuntimeError(f"unsupported content part type {part_type!r}")


def _parse_response(response: httpx.Response) -> tuple[AssistantMessage, str]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Interactions response has invalid JSON: {response.text[:500]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise _shape_error("response object", response.text)
    status = payload.get("status")
    if not isinstance(status, str):
        raise _shape_error("status", response.text)
    if status not in _SUCCESS_STATUSES:
        raise RuntimeError(f"Interactions response ended with status {status!r}")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise _shape_error("steps", response.text)
    interaction_id = payload.get("id")
    if not isinstance(interaction_id, str):
        raise _shape_error("id", response.text)

    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for step in steps:
        if not isinstance(step, dict):
            raise _shape_error("steps[]", response.text)
        step_type = step.get("type")
        if step_type == "function_call":
            call_id = step.get("id")
            name = step.get("name")
            arguments = step.get("arguments")
            if not isinstance(call_id, str):
                raise _shape_error("steps[].id", response.text)
            if not isinstance(name, str):
                raise _shape_error("steps[].name", response.text)
            if isinstance(arguments, dict):
                normalized_arguments = json.dumps(arguments)
            elif isinstance(arguments, str):
                normalized_arguments = arguments
            else:
                raise _shape_error("steps[].arguments", response.text)
            calls.append(ToolCall(id=call_id, name=name, arguments=normalized_arguments))
        elif step_type == "model_output":
            content = step.get("content")
            if not isinstance(content, list):
                raise _shape_error("steps[].content", response.text)
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if not isinstance(text, str):
                        raise _shape_error("steps[].content[].text", response.text)
                    text_parts.append(text)
        # Unknown step types (e.g. server-added thinking output once
        # thinking_level is set) are skipped, matching the Responses wire's
        # tolerance for OpenAI's reasoning items.

    usage = _normalized_usage(payload.get("usage"))
    return (
        AssistantMessage(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tuple(calls),
            usage=usage or None,
        ),
        interaction_id,
    )


def _shape_error(field: str, body: str) -> RuntimeError:
    return RuntimeError(f"Interactions response has missing or invalid {field}: {body[:500]!r}")


def _normalized_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    usage: dict[str, int] = {}
    for source, target in _USAGE_KEYS.items():
        counter = value.get(source)
        if isinstance(counter, int) and not isinstance(counter, bool):
            usage[target] = counter
    return usage
