"""Stateful Gemini Live wire client with bounded fresh-session recovery.

The policy's append-only chat-format list remains the conversation source of
truth. This boundary streams only new list entries, tracks the streamed prefix
by object identity, and folds that source of truth into a text prologue when a
socket must be replaced mid-trial.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as ws_connect

from inspect_robots_agent._llm import AssistantMessage, Provider, ToolCall

from ._capture import WireCapture

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

_CONNECTION_ERRORS = (WebSocketException, OSError, EOFError, TimeoutError)
_RECOVERY_PROLOGUE = (
    "Recovered session. Conversation so far, oldest first, one JSON object per line:"
)
_RECOVERY_CONTINUATION = "Continue the task from the latest observation below."
_DRAIN_CAP = 64


class _RecoverableError(Exception):
    """A transport or liveness failure that requires a fresh Live session."""


class GeminiLiveClient:
    """Stream append-only chat history over one recoverable Gemini Live session.

    The streamed cursor contains references, not copied values. A changed
    prefix therefore means a new trial even when two scenes produce
    byte-identical prompts.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        capture: WireCapture | None = None,
    ) -> None:
        self._provider = provider
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._capture = capture
        bare_model = provider.model.removeprefix("google/")
        self._model_resource = f"models/{bare_model}"
        # Kept as pending state so configuration tests can verify model
        # normalization without opening the real endpoint.
        self._setup: dict[str, Any] = {"model": self._model_resource}
        self._ws: ClientConnection | None = None
        self._streamed: list[dict[str, Any]] = []
        self._call_names: dict[str, str] = {}
        self._needs_recovery = False
        self._go_away = False

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """Return one assistant turn after streaming only the unseen suffix."""
        del reasoning_effort
        self._check_identity_prefix(messages)
        if self._go_away:
            self._abandon()
            self._needs_recovery = bool(self._streamed)
            self._go_away = False

        usage: dict[str, int] = {}
        last_error = "unknown error"
        for attempt in range(self._max_retries):
            sent: list[dict[str, Any]] = []
            received: list[dict[str, Any]] = []
            t_start = time.time() if self._capture is not None else 0.0
            try:
                if self._ws is None:
                    self._open(messages, tools, temperature, sent, received, usage)
                if self._needs_recovery:
                    self._send_recovery(messages, sent)
                    self._needs_recovery = False
                else:
                    self._send_suffix(messages, sent)
                result = self._read_exchange(received, usage)
            except _RecoverableError as exc:
                last_error = _redact_query(str(exc), self._provider.base_url)
                self._record_attempt(
                    attempt, sent, received, last_error, t_start, time.time() - t_start
                )
                self._abandon()
                # A first-call failure has no absorbed state to fold: retry as
                # a plain suffix send. A pending goAway died with the socket.
                self._needs_recovery = bool(self._streamed)
                self._go_away = False
            except Exception as exc:
                error = _redact_query(str(exc), self._provider.base_url)
                self._record_attempt(attempt, sent, received, error, t_start, time.time() - t_start)
                self._abandon()
                raise
            else:
                self._record_attempt(attempt, sent, received, None, t_start, time.time() - t_start)
                return AssistantMessage(
                    content=result.content,
                    tool_calls=result.tool_calls,
                    usage=usage or None,
                )
            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)

        raise RuntimeError(f"LLM request failed after {self._max_retries} attempts — {last_error}")

    def close(self) -> None:
        """Drop the Live socket without allowing close failures to escape."""
        self._abandon()
        self._needs_recovery = bool(self._streamed)
        self._go_away = False

    def _check_identity_prefix(self, messages: list[dict[str, Any]]) -> None:
        prefix_matches = len(messages) >= len(self._streamed) and all(
            messages[index] is streamed for index, streamed in enumerate(self._streamed)
        )
        if prefix_matches:
            return
        if messages and self._streamed and messages[0] is self._streamed[0]:
            raise RuntimeError(
                "Gemini Live received a rewritten conversation view without a trial reset"
            )
        self._abandon()
        self._streamed.clear()
        self._call_names.clear()
        self._needs_recovery = False
        self._go_away = False

    def _open(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None,
        sent: list[dict[str, Any]],
        received: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> None:
        url = self._provider.base_url
        if self._provider.api_key:
            url = f"{url}?key={quote_plus(self._provider.api_key)}"
        try:
            self._ws = ws_connect(url, max_size=None, open_timeout=self._timeout_s)
        except _CONNECTION_ERRORS as exc:
            raise _RecoverableError(_redact_query(str(exc), self._provider.base_url)) from exc

        setup: dict[str, Any] = {"model": self._model_resource}
        if messages and messages[0].get("role") == "system":
            setup["systemInstruction"] = {"parts": [{"text": str(messages[0].get("content", ""))}]}
        if tools:
            declarations: list[dict[str, Any]] = []
            for tool in tools:
                function = tool["function"]
                declarations.append(
                    {
                        "name": function["name"],
                        "description": function["description"],
                        "parameters": function["parameters"],
                    }
                )
            setup["tools"] = [{"functionDeclarations": declarations}]
        if temperature is not None:
            setup["generationConfig"] = {"temperature": temperature}
        self._setup = setup
        self._send_message({"setup": setup}, sent)

        for _ in range(_DRAIN_CAP):
            message = self._recv_message(received, usage)
            if "setupComplete" in message:
                return
            if "goAway" in message:
                self._go_away = True
        raise _RecoverableError("Gemini Live setup drain cap exhausted after 64 messages")

    def _send_suffix(
        self,
        messages: list[dict[str, Any]],
        sent: list[dict[str, Any]],
    ) -> None:
        suffix = messages[len(self._streamed) :]
        users: list[dict[str, Any]] = []
        tool_runs: list[list[dict[str, Any]]] = []
        current_tools: list[dict[str, Any]] = []
        for message in suffix:
            role = message.get("role")
            if role == "tool":
                current_tools.append(message)
                continue
            if current_tools:
                tool_runs.append(current_tools)
                current_tools = []
            if role == "user":
                users.append(message)
            elif role in {"assistant", "system"}:
                continue
            else:
                raise RuntimeError(f"unsupported chat message role {role!r}")
        if current_tools:
            tool_runs.append(current_tools)

        has_tools = bool(tool_runs)
        for index, message in enumerate(users):
            final_user_only = not has_tools and index == len(users) - 1
            self._send_message(
                _client_content(message, turn_complete=final_user_only),
                sent,
            )
        for run in tool_runs:
            self._send_message(self._tool_response(run), sent)
        if has_tools and users:
            self._send_message({"clientContent": {"turnComplete": True}}, sent)
        if not users and not has_tools:
            raise RuntimeError("Gemini Live has no un-streamed user or tool message to send")

        # A partial send cannot move the cursor: recovery must fold every
        # possibly absorbed message from the authoritative source list.
        self._streamed.extend(suffix)

    def _tool_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        responses: list[dict[str, Any]] = []
        for message in messages:
            call_id = str(message["tool_call_id"])
            name = self._call_names.get(call_id)
            if name is None:
                raise RuntimeError(
                    f"Gemini Live has no function name recorded for tool call {call_id!r}"
                )
            content = message.get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"Gemini Live tool result {call_id!r} content must be a string")
            responses.append(
                {
                    "id": call_id,
                    "name": name,
                    "response": {"output": content},
                }
            )
        return {"toolResponse": {"functionResponses": responses}}

    def _send_recovery(
        self,
        messages: list[dict[str, Any]],
        sent: list[dict[str, Any]],
    ) -> None:
        anchor_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        if anchor_index is None:
            raise RuntimeError("Gemini Live recovery requires at least one user message")

        # Deferred to avoid policy.py -> _gemini_live.py -> policy.py at import.
        from inspect_robots_agent.policy import _sanitize

        # The fresh session already carries a leading system message as its
        # systemInstruction; folding it again would duplicate the prompt.
        start = 1 if messages[0].get("role") == "system" else 0
        folded = _sanitize(messages[start:anchor_index] + messages[anchor_index + 1 :])
        lines = [
            _RECOVERY_PROLOGUE,
            *(json.dumps(message) for message in folded),
            _RECOVERY_CONTINUATION,
        ]
        prologue = {"role": "user", "content": "\n".join(lines)}
        self._send_message(_client_content(prologue, turn_complete=False), sent)
        self._send_message(_client_content(messages[anchor_index], turn_complete=True), sent)
        self._streamed = list(messages)
        self._call_names.clear()

    def _read_exchange(
        self,
        received: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> AssistantMessage:
        text_parts: list[str] = []
        for _ in range(_DRAIN_CAP):
            message = self._recv_message(received, usage)
            if "goAway" in message:
                self._go_away = True

            tool_call = message.get("toolCall")
            if isinstance(tool_call, dict):
                calls: list[ToolCall] = []
                for function_call in tool_call.get("functionCalls") or []:
                    call_id = str(function_call["id"])
                    name = str(function_call["name"])
                    self._call_names[call_id] = name
                    calls.append(
                        ToolCall(
                            id=call_id,
                            name=name,
                            arguments=json.dumps(function_call.get("args") or {}),
                        )
                    )
                if calls:
                    return AssistantMessage(content=None, tool_calls=tuple(calls))

            server_content = message.get("serverContent")
            if not isinstance(server_content, dict):
                continue
            model_turn = server_content.get("modelTurn")
            if isinstance(model_turn, dict):
                for part in model_turn.get("parts") or []:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
            if server_content.get("turnComplete"):
                if text_parts:
                    return AssistantMessage(content="".join(text_parts), tool_calls=())
                # The buffered toolResponse generation completes empty before
                # the final clientContent trigger produces the useful turn.
                text_parts.clear()
        raise _RecoverableError("Gemini Live drain cap exhausted after 64 messages")

    def _send_message(
        self,
        message: dict[str, Any],
        sent: list[dict[str, Any]],
    ) -> None:
        ws = self._ws
        if ws is None:
            raise _RecoverableError("Gemini Live socket is not connected")
        sent.append(message)
        try:
            ws.send(json.dumps(message))
        except _CONNECTION_ERRORS as exc:
            raise _RecoverableError(_redact_query(str(exc), self._provider.base_url)) from exc

    def _recv_message(
        self,
        received: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise _RecoverableError("Gemini Live socket is not connected")
        try:
            raw = ws.recv(timeout=self._timeout_s)
        except _CONNECTION_ERRORS as exc:
            message = "Gemini Live read timed out" if isinstance(exc, TimeoutError) else str(exc)
            raise _RecoverableError(_redact_query(message, self._provider.base_url)) from exc
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"Gemini Live returned invalid JSON: {str(raw)[:200]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini Live returned a non-object JSON message")
        received.append(parsed)
        _add_usage(parsed, usage)
        return parsed

    def _record_attempt(
        self,
        attempt: int,
        sent: list[dict[str, Any]],
        received: list[dict[str, Any]],
        error: str | None,
        t_start: float,
        duration_s: float,
    ) -> None:
        if self._capture is None:
            return
        self._capture.record(
            attempt=attempt,
            endpoint="bidi",
            request={"messages": sent},
            status=None,
            response_text=json.dumps({"messages": received}),
            error=error,
            t_start=t_start,
            duration_s=duration_s,
        )

    def _abandon(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception:
            return


def _client_content(message: dict[str, Any], *, turn_complete: bool) -> dict[str, Any]:
    """Translate one chat user message into a verified Live content turn."""
    content = message.get("content")
    if isinstance(content, str):
        parts = [{"text": content}]
    elif isinstance(content, list):
        parts = [_content_part(part) for part in content]
    else:
        raise RuntimeError(f"unsupported Gemini Live user content {content!r}")
    return {
        "clientContent": {
            "turns": [{"role": "user", "parts": parts}],
            "turnComplete": turn_complete,
        }
    }


def _content_part(part: Any) -> dict[str, Any]:
    """Translate text and data-URI image parts without accepting other shapes."""
    if not isinstance(part, dict):
        raise RuntimeError(f"unsupported content part type {type(part).__name__!r}")
    part_type = part.get("type")
    if part_type == "text":
        return {"text": part["text"]}
    if part_type == "image_url":
        url = str(part.get("image_url", {}).get("url", ""))
        header, separator, data = url.partition(",")
        if not separator or not header.startswith("data:") or not header.endswith(";base64"):
            raise RuntimeError(f"image content part is not a base64 data URI: {url[:60]!r}")
        mime_type = header[len("data:") : -len(";base64")]
        if not mime_type:
            raise RuntimeError(f"image content part has no data-URI MIME type: {url[:60]!r}")
        return {"inlineData": {"mimeType": mime_type, "data": data}}
    raise RuntimeError(f"unsupported content part type {part_type!r}")


def _add_usage(message: dict[str, Any], totals: dict[str, int]) -> None:
    """Add one Live envelope's billing counters to normalized exchange totals."""
    usage = message.get("usageMetadata")
    if not isinstance(usage, dict):
        return
    keys = {
        "promptTokenCount": "input_tokens",
        "candidatesTokenCount": "output_tokens",
        "totalTokenCount": "total_tokens",
    }
    for source, target in keys.items():
        value = usage.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            totals[target] = totals.get(target, 0) + value


def _redact_query(message: str, base_url: str) -> str:
    """Remove the Live credential query from transport diagnostics."""
    redacted = re.sub(r"\?key=[^\s'\"),]+", "", message)
    return re.sub(re.escape(base_url) + r"\?[^\s'\"),]+", base_url, redacted)
