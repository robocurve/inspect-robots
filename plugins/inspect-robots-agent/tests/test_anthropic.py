"""Native Anthropic Messages wire client and policy integration (plan 0026)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._anthropic import (
    _DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicClient,
)
from inspect_robots_agent._llm import Provider
from inspect_robots_agent._png import png_data_url

# -- fixtures --------------------------------------------------------------------


def _anthropic_response(*blocks: dict[str, Any], stop_reason: str = "tool_use") -> dict[str, Any]:
    return {"id": "msg_1", "type": "message", "content": list(blocks), "stop_reason": stop_reason}


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _thinking(thinking: str = "step one", signature: str = "sig-abc") -> dict[str, Any]:
    return {"type": "thinking", "thinking": thinking, "signature": signature}


def _tool_use(
    block_id: str = "toolu_1", name: str = "done", tool_input: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block_id,
        "name": name,
        "input": {"summary": "ok"} if tool_input is None else tool_input,
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _schema(name: str = "done") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} the task",
            "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}},
        },
    }


def _client(handler: Any, *, provider: Provider | None = None, **kwargs: Any) -> AnthropicClient:
    resolved = provider or Provider(
        base_url="http://llm.test/v1", api_key="sk-test", model="claude-opus-5"
    )
    kwargs.setdefault("max_output_tokens", _DEFAULT_MAX_OUTPUT_TOKENS)
    kwargs.setdefault("backoff_s", 0.0)
    return AnthropicClient(resolved, transport=httpx.MockTransport(handler), **kwargs)


def _capture(*responses: dict[str, Any], status: int = 200) -> tuple[list[httpx.Request], Any]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(status, json=payload)

    return seen, handler


_SYSTEM = {"role": "system", "content": "you drive a robot"}
_USER = {"role": "user", "content": "Goal: pick the cube"}


# -- request shape ---------------------------------------------------------------


def test_request_shape_and_headers() -> None:
    seen, handler = _capture(_anthropic_response(_text("hi"), stop_reason="end_turn"))
    client = _client(handler)

    client.complete([_SYSTEM, _USER], [_schema()], reasoning_effort="low")

    assert seen[0].url.path == "/v1/messages"
    body = json.loads(seen[0].content)
    assert body["model"] == "claude-opus-5"
    assert body["max_tokens"] == _DEFAULT_MAX_OUTPUT_TOKENS
    assert body["thinking"] == {"type": "adaptive"}
    assert body["system"] == "you drive a robot"
    assert body["output_config"] == {"effort": "low"}
    assert body["tools"] == [
        {
            "name": "done",
            "description": "done the task",
            "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}},
        }
    ]
    assert "speed" not in body
    assert "temperature" not in body
    assert seen[0].headers["x-api-key"] == "sk-test"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert "anthropic-beta" not in seen[0].headers


def test_optional_fields_omitted_when_unset() -> None:
    seen, handler = _capture(_anthropic_response(_text("hi"), stop_reason="end_turn"))

    _client(handler).complete([_USER], [])

    body = json.loads(seen[0].content)
    assert "tools" not in body
    assert "output_config" not in body
    assert "system" not in body


def test_temperature_forwarded_when_set() -> None:
    seen, handler = _capture(_anthropic_response(_text("hi"), stop_reason="end_turn"))

    _client(handler).complete([_USER], [], temperature=0.2)

    assert json.loads(seen[0].content)["temperature"] == 0.2


def test_fast_mode_sends_speed_and_beta_header() -> None:
    seen, handler = _capture(_anthropic_response(_text("hi"), stop_reason="end_turn"))

    _client(handler, speed="fast").complete([_USER], [])

    assert json.loads(seen[0].content)["speed"] == "fast"
    assert seen[0].headers["anthropic-beta"] == "fast-mode-2026-02-01"


def test_empty_api_key_omits_header() -> None:
    seen, handler = _capture(_anthropic_response(_text("hi"), stop_reason="end_turn"))
    provider = Provider(base_url="http://llm.test/v1", api_key="", model="m")

    _client(handler, provider=provider).complete([_USER], [])

    assert "x-api-key" not in seen[0].headers


# -- translation -----------------------------------------------------------------


def test_image_part_becomes_base64_source() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    url = png_data_url(image)
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]

    _client(handler).complete(history, [])

    parts = json.loads(seen[0].content)["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image"
    assert parts[1]["source"]["media_type"] == "image/png"
    assert parts[1]["source"]["data"] == url.removeprefix("data:image/png;base64,")
    assert not parts[1]["source"]["data"].startswith("data:")


def test_non_png_image_url_raises() -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))
    history = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        }
    ]

    with pytest.raises(RuntimeError, match="not a PNG data URL"):
        _client(handler).complete(history, [])


def test_unknown_content_part_raises() -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))
    history = [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]

    with pytest.raises(RuntimeError, match="unsupported content part type 'audio'"):
        _client(handler).complete(history, [])


def test_system_message_at_later_index_raises() -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))

    with pytest.raises(RuntimeError, match="system message at index 2"):
        _client(handler).complete([_SYSTEM, _USER, _SYSTEM], [])


def test_consecutive_tool_messages_merge_in_history_order() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    history = [
        _USER,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call("toolu_a", "done", "{}"),
                _tool_call("toolu_b", "done", "{}"),
                _tool_call("toolu_c", "done", "{}"),
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_b", "content": "ignored"},
        {"role": "tool", "tool_call_id": "toolu_c", "content": "ignored"},
        {"role": "tool", "tool_call_id": "toolu_a", "content": "executed"},
    ]

    _client(handler).complete(history, [])

    messages = json.loads(seen[0].content)["messages"]
    results = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    merged = results[-1]["content"]
    assert [b["tool_use_id"] for b in merged] == ["toolu_b", "toolu_c", "toolu_a"]
    assert all("is_error" not in block for block in merged)
    assert sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], list)) == 1


def test_interleaved_tool_runs_do_not_merge_across_an_observation() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    history = [
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("t1", "done", "{}")]},
        {"role": "tool", "tool_call_id": "t1", "content": "one"},
        _USER,
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("t2", "done", "{}")]},
        {"role": "tool", "tool_call_id": "t2", "content": "two"},
    ]

    _client(handler).complete(history, [])

    messages = json.loads(seen[0].content)["messages"]
    blocks = [
        m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert [[b["tool_use_id"] for b in group] for group in blocks] == [["t1"], ["t2"]]


def test_empty_assistant_turn_emits_no_message() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    history = [
        _USER,
        {"role": "assistant", "content": None},
        {"role": "user", "content": "Respond with exactly one tool call."},
    ]

    _client(handler).complete(history, [])

    messages = json.loads(seen[0].content)["messages"]
    assert [m["role"] for m in messages] == ["user", "user"]


def test_empty_string_content_with_tool_calls_emits_no_text_block() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    history = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("t1", "done", "{}")]},
    ]

    _client(handler).complete(history, [])

    blocks = json.loads(seen[0].content)["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["tool_use"]


def test_unparseable_tool_arguments_raise_naming_the_tool() -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))
    history = [
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("t1", "move", "{oops")]}
    ]

    with pytest.raises(RuntimeError, match="tool call 'move' has unparseable arguments"):
        _client(handler).complete(history, [])


def test_non_object_tool_arguments_raise_naming_the_tool() -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))
    history = [
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("t1", "move", "[1,2]")]}
    ]

    with pytest.raises(RuntimeError, match="tool call 'move' arguments must be a JSON object"):
        _client(handler).complete(history, [])


# -- thinking replay -------------------------------------------------------------


def test_thinking_block_is_replayed_verbatim_on_the_next_turn() -> None:
    first = _anthropic_response(_thinking(), _text("moving"), _tool_use("toolu_1"))
    second = _anthropic_response(_text("done"), stop_reason="end_turn")
    seen, handler = _capture(first, second)
    client = _client(handler)

    message = client.complete([_USER], [])
    history = [
        _USER,
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [_tool_call("toolu_1", "done", message.tool_calls[0].arguments)],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"},
    ]
    client.complete(history, [])

    replayed = json.loads(seen[1].content)["messages"][1]["content"]
    assert replayed[0] == _thinking()
    assert [b["type"] for b in replayed] == ["thinking", "text", "tool_use"]
    assert sum(1 for b in replayed if b["type"] == "text") == 1


def test_multi_tool_turn_replays_cached_blocks_once() -> None:
    first = _anthropic_response(_thinking(), _tool_use("toolu_a"), _tool_use("toolu_b"))
    second = _anthropic_response(_text("done"), stop_reason="end_turn")
    seen, handler = _capture(first, second)
    client = _client(handler)

    client.complete([_USER], [])
    history = [
        _USER,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call("toolu_a", "done", "{}"),
                _tool_call("toolu_b", "done", "{}"),
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_a", "content": "ok"},
        {"role": "tool", "tool_call_id": "toolu_b", "content": "ok"},
    ]
    client.complete(history, [])

    replayed = json.loads(seen[1].content)["messages"][1]["content"]
    assert [b["type"] for b in replayed] == ["thinking", "tool_use", "tool_use"]


def test_cache_miss_synthesizes_blocks() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    history = [
        {"role": "assistant", "content": "hi", "tool_calls": [_tool_call("foreign", "done", "{}")]},
    ]

    _client(handler).complete(history, [])

    blocks = json.loads(seen[0].content)["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]


def test_cache_prunes_against_history_including_tool_message_ids() -> None:
    first = _anthropic_response(_thinking(), _tool_use("toolu_1"))
    second = _anthropic_response(_text("done"), stop_reason="end_turn")
    _, handler = _capture(first, second)
    client = _client(handler)

    client.complete([_USER], [])
    assert "toolu_1" in client._raw_blocks_by_tool_use_id
    # A history that references the id only from the tool message must keep it.
    client.complete([_USER, {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"}], [])
    assert "toolu_1" in client._raw_blocks_by_tool_use_id
    # A fresh history (reset()) drops it.
    client.complete([_USER], [])
    assert client._raw_blocks_by_tool_use_id == {}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_terminal_responses_do_not_populate_the_cache(stop_reason: str) -> None:
    _, handler = _capture(_anthropic_response(_tool_use("toolu_1"), stop_reason=stop_reason))
    client = _client(handler)

    with pytest.raises(RuntimeError):
        client.complete([_USER], [])

    assert client._raw_blocks_by_tool_use_id == {}


# -- parsing ---------------------------------------------------------------------


def test_parses_text_and_tool_use() -> None:
    _, handler = _capture(
        _anthropic_response(_thinking(), _text("a"), _text("b"), _tool_use("toolu_1", "done"))
    )

    message = _client(handler).complete([_USER], [])

    assert message.content == "ab"
    assert [c.id for c in message.tool_calls] == ["toolu_1"]
    assert json.loads(message.tool_calls[0].arguments) == {"summary": "ok"}


def test_empty_text_normalizes_to_none() -> None:
    _, handler = _capture(_anthropic_response(_text(""), _tool_use("toolu_1")))

    message = _client(handler).complete([_USER], [])

    assert message.content is None


def test_refusal_raises_with_category() -> None:
    payload = _anthropic_response(stop_reason="refusal")
    payload["stop_details"] = {"type": "refusal", "category": "cyber"}
    seen, handler = _capture(payload)

    with pytest.raises(RuntimeError, match=r"refused the request.*cyber"):
        _client(handler).complete([_USER], [])

    assert len(seen) == 1


def test_refusal_with_null_stop_details_raises_cleanly() -> None:
    payload = _anthropic_response(stop_reason="refusal")
    payload["stop_details"] = None
    _, handler = _capture(payload)

    with pytest.raises(RuntimeError, match="refused the request"):
        _client(handler).complete([_USER], [])


def test_truncation_raises_and_never_yields_the_partial_tool_call() -> None:
    _, handler = _capture(_anthropic_response(_tool_use("toolu_1"), stop_reason="max_tokens"))

    with pytest.raises(RuntimeError, match=r"(?s)truncated.*fix: raise -P max_output_tokens="):
        _client(handler).complete([_USER], [])


@pytest.mark.parametrize("stop_reason", ["model_context_window_exceeded", "pause_turn"])
def test_unrecognized_stop_reason_raises_naming_it(stop_reason: str) -> None:
    _, handler = _capture(_anthropic_response(_text("partial"), stop_reason=stop_reason))

    with pytest.raises(RuntimeError, match=stop_reason):
        _client(handler).complete([_USER], [])


# -- retries and guided errors ---------------------------------------------------


def test_fast_mode_4xx_names_the_fix() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="speed: fast is not supported for this model")

    with pytest.raises(RuntimeError, match=r"fix: fast mode needs Claude Opus 5"):
        _client(handler, speed="fast").complete([_USER], [])


def test_unrelated_4xx_gets_no_fast_mode_guidance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="messages: at least one message is required")

    with pytest.raises(RuntimeError) as excinfo:
        _client(handler, speed="fast").complete([_USER], [])

    assert "fix:" not in str(excinfo.value)


def test_effort_4xx_names_the_accepted_values() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="output_config.effort: invalid value 'minimal'")

    with pytest.raises(RuntimeError, match=r"none and minimal are OpenAI-only values"):
        _client(handler).complete([_USER], [], reasoning_effort="minimal")


def test_temperature_guidance_only_when_temperature_was_sent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="temperature is not supported")

    # The word appears in the body, but this request never sent the parameter.
    with pytest.raises(RuntimeError) as excinfo:
        _client(handler).complete([_USER], [])

    assert "drop -P temperature=" not in str(excinfo.value)


def test_system_content_must_be_a_string() -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))
    history = [{"role": "system", "content": [{"type": "text", "text": "x"}]}, _USER]

    with pytest.raises(RuntimeError, match="system message content must be a string"):
        _client(handler).complete(history, [])


@pytest.mark.parametrize("role", ["developer", "function"])
def test_unsupported_role_raises(role: str) -> None:
    _, handler = _capture(_anthropic_response(_text("ok")))

    with pytest.raises(RuntimeError, match=f"unsupported message role {role!r}"):
        _client(handler).complete([{"role": role, "content": "x"}], [])


def test_assistant_role_is_not_rejected_by_the_role_guard() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))

    _client(handler).complete([_USER, {"role": "assistant", "content": "hi"}], [])

    assert [m["role"] for m in json.loads(seen[0].content)["messages"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.parametrize("status", [408, 409])
def test_408_and_409_are_retried(status: int) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(status, text="retry me")
        return httpx.Response(200, json=_anthropic_response(_text("ok"), stop_reason="end_turn"))

    assert _client(handler).complete([_USER], []).content == "ok"
    assert calls["n"] == 2


def test_timeout_scales_with_a_large_output_cap() -> None:
    _, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))

    small_cap = _client(handler, max_output_tokens=1000)
    large_cap = _client(handler, max_output_tokens=64000)
    explicit = _client(handler, max_output_tokens=64000, timeout_s=30.0)

    # Floored at the 120s the other clients use, then scaled above it.
    assert small_cap._http.timeout.read == 120.0
    assert large_cap._http.timeout.read == pytest.approx(512.0)
    assert explicit._http.timeout.read == 30.0
    # A big read budget must not stretch connect: an unroutable base_url
    # should fail fast, not hold the whole read timeout on every retry.
    assert large_cap._http.timeout.connect == 10.0
    # And the read budget itself is capped rather than unbounded.
    assert _client(handler, max_output_tokens=1_000_000)._http.timeout.read == 600.0


def test_temperature_4xx_names_the_fix() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="temperature: unsupported parameter")

    with pytest.raises(RuntimeError, match=r"fix: this model rejects temperature"):
        _client(handler).complete([_USER], [], temperature=0.5)


def test_exhausted_429_with_fast_mode_names_the_separate_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limit")

    with pytest.raises(RuntimeError, match=r"fix: fast mode has its own rate limit"):
        _client(handler, speed="fast").complete([_USER], [])


def test_exhausted_429_without_fast_mode_is_unguided() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limit")

    with pytest.raises(RuntimeError) as excinfo:
        _client(handler).complete([_USER], [])

    assert "fix:" not in str(excinfo.value)


@pytest.mark.parametrize("status", [408, 409])
def test_exhausted_retryable_non_429_gets_no_rate_limit_guidance(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="request timeout")

    with pytest.raises(RuntimeError) as excinfo:
        _client(handler, speed="fast").complete([_USER], [])

    assert "its own rate limit" not in str(excinfo.value)


@pytest.mark.parametrize("tail", ["transport", "server"])
def test_429_then_other_failure_drops_the_rate_limit_guidance(tail: str) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limit")
        if tail == "transport":
            raise httpx.ConnectError("boom")
        return httpx.Response(500, text="server error")

    with pytest.raises(RuntimeError) as excinfo:
        _client(handler, speed="fast").complete([_USER], [])

    assert "its own rate limit" not in str(excinfo.value)


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_anthropic_response(_text("ok"), stop_reason="end_turn"))

    assert _client(handler).complete([_USER], []).content == "ok"
    assert calls["n"] == 3


def test_client_4xx_fails_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, text="bad request")

    with pytest.raises(RuntimeError, match="rejected"):
        _client(handler).complete([_USER], [])

    assert calls["n"] == 1


def test_close_releases_the_pool() -> None:
    _, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    client = _client(handler)

    client.close()

    assert client._http.is_closed


# -- policy wiring ---------------------------------------------------------------


_ENV = {"ANTHROPIC_API_KEY": "sk-test"}


def _policy(**kwargs: Any) -> LLMAgentPolicy:
    kwargs.setdefault("model", "anthropic/claude-opus-5")
    kwargs.setdefault("env", dict(_ENV))
    return LLMAgentPolicy(**kwargs)


def test_policy_records_wire_speed_and_resolved_max_output_tokens() -> None:
    policy = _policy(wire="anthropic", speed="fast")

    assert policy.config.wire == "anthropic"
    assert policy.config.speed == "fast"
    assert policy.config.max_output_tokens == _DEFAULT_MAX_OUTPUT_TOKENS


def test_policy_passes_speed_and_cap_through_to_the_request() -> None:
    """Guards the policy -> client wiring, which config assertions cannot see.

    Config recording and request building are separate code paths: a dropped
    ``speed`` still records ``speed="fast"`` in the eval log while billing at
    standard rates, and no coverage metric catches it.
    """
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    policy = _policy(
        wire="anthropic",
        speed="fast",
        max_output_tokens=2000,
        transport=httpx.MockTransport(handler),
    )

    policy._client.complete([_USER], [])

    body = json.loads(seen[0].content)
    assert body["speed"] == "fast"
    assert body["max_tokens"] == 2000
    assert seen[0].headers["anthropic-beta"] == "fast-mode-2026-02-01"


def test_policy_passes_the_default_cap_when_unset() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    policy = _policy(wire="anthropic", transport=httpx.MockTransport(handler))

    policy._client.complete([_USER], [])

    body = json.loads(seen[0].content)
    assert body["max_tokens"] == _DEFAULT_MAX_OUTPUT_TOKENS
    assert "speed" not in body
    assert "anthropic-beta" not in seen[0].headers


def test_policy_sends_the_prefix_stripped_model_id() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    policy = _policy(wire="anthropic", transport=httpx.MockTransport(handler))

    policy._client.complete([_USER], [])

    assert json.loads(seen[0].content)["model"] == "claude-opus-5"


def test_chat_wire_records_no_max_output_tokens() -> None:
    policy = LLMAgentPolicy(model="anthropic/claude-opus-5", env=dict(_ENV))

    assert policy.config.max_output_tokens is None
    assert policy.config.speed is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wire": "anthropic", "speed": "turbo"}, "speed must be one of"),
        ({"wire": "anthropic", "max_output_tokens": 0}, "must be an int >= 1"),
        ({"wire": "anthropic", "max_output_tokens": 1e5}, "must be an int >= 1"),
        ({"wire": "anthropic", "max_output_tokens": True}, "must be an int >= 1"),
        ({"wire": "antropic"}, "wire must be one of"),
    ],
)
def test_invalid_configurations_raise(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _policy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"wire": "chat", "speed": "fast"}, {"wire": "chat", "max_output_tokens": 2000}],
)
def test_wire_gated_params_raise_config_error_so_the_cli_renders_them(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError, match=r"only supported on wire='anthropic'"):
        _policy(**kwargs)


def test_misspelled_wire_reports_the_wire_not_the_speed() -> None:
    # Ordering guard: wire is validated before the params gated on it.
    with pytest.raises(ValueError, match="wire must be one of"):
        _policy(wire="antropic", speed="fast")


def test_openrouter_fallback_is_refused_with_guidance() -> None:
    with pytest.raises(ConfigError, match=r"fix: set \$ANTHROPIC_API_KEY"):
        LLMAgentPolicy(
            model="anthropic/claude-opus-5",
            wire="anthropic",
            env={"OPENROUTER_API_KEY": "sk-or"},
        )


def test_bare_model_id_is_told_to_add_the_prefix_not_to_set_the_key() -> None:
    # The key is already set; the actual mistake is the missing prefix.
    with pytest.raises(ConfigError, match=r"-P model=anthropic/claude-opus-5"):
        LLMAgentPolicy(
            model="claude-opus-5",
            wire="anthropic",
            env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_explicit_base_url_suppresses_the_openrouter_guard() -> None:
    policy = LLMAgentPolicy(
        model="claude-opus-5",
        wire="anthropic",
        base_url="http://gateway.test/v1",
        env={"OPENROUTER_API_KEY": "sk-or", "ANTHROPIC_API_KEY": "sk-ant"},
    )

    assert policy.config.base_url == "http://gateway.test/v1"


def test_non_anthropic_prefix_is_not_told_to_set_the_anthropic_key() -> None:
    with pytest.raises(ConfigError, match=r"fix: use an anthropic/ model id"):
        LLMAgentPolicy(
            model="meta-llama/llama-3",
            wire="anthropic",
            env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_variant_suffix_is_told_to_drop_it_not_to_set_the_key() -> None:
    # ':free' routes to OpenRouter whatever keys are set, so the key advice
    # would point at something the user already did right.
    with pytest.raises(
        ConfigError,
        match=r"fix: drop the OpenRouter variant suffix \(-P model=anthropic/claude-opus-5\)",
    ):
        LLMAgentPolicy(
            model="anthropic/claude-opus-5:free",
            wire="anthropic",
            env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_bare_variant_id_is_fixed_in_one_step() -> None:
    # Both the prefix and the suffix are wrong; advice that fixes only one
    # earns a second refusal on the retry.
    with pytest.raises(
        ConfigError,
        match=(
            r"fix: use -P model=anthropic/claude-opus-5 "
            r"\(the :variant suffix routes to OpenRouter\)"
        ),
    ):
        LLMAgentPolicy(
            model="claude-opus-5:free",
            wire="anthropic",
            env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_variant_strip_keeps_fine_tune_colons() -> None:
    # Only the last segment is the variant; rpartition, not partition, or the
    # advice would truncate the fine-tune id to 'anthropic/ft'.
    with pytest.raises(ConfigError, match=r"-P model=anthropic/ft:gpt-4o-mini\b"):
        LLMAgentPolicy(
            model="ft:gpt-4o-mini:free",
            wire="anthropic",
            env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_empty_api_key_env_does_not_send_the_openrouter_key_to_a_gateway() -> None:
    # '-P api_key_env=' parses to '', which resolve_provider treats as unset
    # and falls back to $OPENROUTER_API_KEY. An `is None` test here would hand
    # a third-party gateway the OpenRouter secret.
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    policy = LLMAgentPolicy(
        model="claude-opus-5",
        wire="anthropic",
        base_url="https://gw.example/v1",
        api_key_env="",
        transport=httpx.MockTransport(handler),
        env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
    )

    policy._client.complete([_USER], [])

    assert seen[0].headers["x-api-key"] == "sk-ant"


@pytest.mark.parametrize("model", ["anthropic/", ":free", "anthropic/:free"])
def test_ids_with_nothing_usable_left_get_a_whole_command(model: str) -> None:
    # Each of these strips to an empty id, where echoing the remainder would
    # send the user in a circle (or name $ANTHROPIC_API_KEY, already set).
    with pytest.raises(ConfigError, match=r"fix: pass a full model id"):
        LLMAgentPolicy(
            model=model,
            wire="anthropic",
            env={"ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_empty_base_url_still_hits_the_guard() -> None:
    # '-P base_url=' parses to '', which resolve_provider ignores. An `is None`
    # test here would wave it through and POST /v1/messages to OpenRouter.
    with pytest.raises(ConfigError, match=r"resolved to OpenRouter"):
        LLMAgentPolicy(
            model="anthropic/claude-opus-5",
            wire="anthropic",
            base_url="",
            env={"OPENROUTER_API_KEY": "sk-or"},
        )


def test_model_from_the_environment_resolves_like_the_argument() -> None:
    with pytest.raises(ConfigError, match=r"fix: prefix the model id"):
        LLMAgentPolicy(
            wire="anthropic",
            env={
                "INSPECT_ROBOTS_MODEL": "claude-opus-5",
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENROUTER_API_KEY": "sk-or",
            },
        )


def test_another_direct_provider_is_refused_not_sent_to_its_endpoint() -> None:
    # With $OPENAI_API_KEY set this resolves straight to api.openai.com, which
    # would have taken a /v1/messages POST and 404ed at the first LLM call.
    # The message names the id the user typed, not the prefix-stripped one.
    with pytest.raises(
        ConfigError, match=r"'openai/gpt-5\.6' resolved to https://api\.openai\.com/v1"
    ):
        LLMAgentPolicy(
            model="openai/gpt-5.6",
            wire="anthropic",
            env={"OPENAI_API_KEY": "sk-oai", "OPENROUTER_API_KEY": "sk-or"},
        )


def test_direct_provider_guidance_uses_the_requested_id_not_the_stripped_one() -> None:
    # resolve_provider strips 'groq/', so branching on the resolved id would
    # read it as bare and suggest the nonsense 'anthropic/llama-3'.
    with pytest.raises(ConfigError, match=r"fix: use an anthropic/ model id"):
        LLMAgentPolicy(model="groq/llama-3", wire="anthropic", env={"GROQ_API_KEY": "sk-groq"})


def test_gateway_defaults_the_key_env_to_anthropic() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    policy = LLMAgentPolicy(
        model="claude-opus-5",
        wire="anthropic",
        base_url="http://gateway.test/v1",
        transport=httpx.MockTransport(handler),
        env={"OPENROUTER_API_KEY": "sk-or", "ANTHROPIC_API_KEY": "sk-ant"},
    )

    policy._client.complete([_USER], [])

    assert seen[0].headers["x-api-key"] == "sk-ant"
    # The config records what the user passed, not the substitution.
    assert policy.config.api_key_env is None


def test_act_drives_a_multi_turn_trial_and_replays_thinking() -> None:
    """The full messages array across turns: nudge path, merge, and replay."""
    requests: list[httpx.Request] = []
    # Turn 1: no tool call, so act() nudges. Turn 2: a real move. Turn 3 (next
    # act()) starts from tool results followed by a fresh observation.
    payloads = [
        # Thinking only: no text, no tool call, so the parsed turn has
        # content None and translates to no message at all.
        _anthropic_response(_thinking(), stop_reason="end_turn"),
        _anthropic_response(
            _thinking("now I move"),
            _tool_use("toolu_turn2", "done", {"summary": "ok"}),
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = payloads[min(len(requests) - 1, len(payloads) - 1)]
        return httpx.Response(200, json=payload)

    policy = LLMAgentPolicy(
        model="anthropic/claude-opus-5",
        wire="anthropic",
        transport=httpx.MockTransport(handler),
        env=dict(_ENV),
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    policy.act(Observation())

    first = json.loads(requests[0].content)
    assert first["system"].startswith("You are controlling a real robot")
    assert all(m["role"] != "system" for m in first["messages"])

    # Turn 2 carries the dropped-assistant-then-nudge shape: the text-only turn
    # emits no assistant message, so two user messages sit adjacent.
    second = json.loads(requests[1].content)
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "user", "user"]
    assert second["messages"][-1]["content"] == "Respond with exactly one tool call."

    # Next act(): tool results merge into one user turn, then the observation.
    policy.act(Observation())
    third = json.loads(requests[2].content)
    assistant_turns = [m for m in third["messages"] if m["role"] == "assistant"]
    replayed = assistant_turns[-1]["content"]
    assert replayed[0]["type"] == "thinking"
    assert replayed[0]["thinking"] == "now I move"
    results = [
        m
        for m in third["messages"]
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and m["content"][0].get("type") == "tool_result"
    ]
    assert len(results) == 1


def test_explicit_api_key_env_wins_over_the_default() -> None:
    seen, handler = _capture(_anthropic_response(_text("ok"), stop_reason="end_turn"))
    policy = LLMAgentPolicy(
        model="claude-opus-5",
        wire="anthropic",
        base_url="http://gateway.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        transport=httpx.MockTransport(handler),
        env={"OPENROUTER_API_KEY": "sk-or", "ANTHROPIC_API_KEY": "sk-ant"},
    )

    policy._client.complete([_USER], [])

    assert seen[0].headers["x-api-key"] == "sk-or"
