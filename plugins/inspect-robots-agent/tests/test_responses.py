"""OpenAI Responses wire client and policy integration."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._llm import Provider
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent.policy import AgentPolicyConfig


def _response(*output: dict[str, Any]) -> dict[str, Any]:
    return {"id": "resp_1", "status": "completed", "output": list(output)}


def _message(text: str, *, item_id: str = "msg_1") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call(
    call_id: str = "call_1",
    name: str = "done",
    arguments: str = '{"summary":"ok"}',
    *,
    item_id: str = "fc_1",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _client(handler: Any, **kwargs: Any) -> ResponsesClient:
    provider = Provider(base_url="http://llm.test/v1", api_key="sk-test", model="m")
    return ResponsesClient(provider, transport=httpx.MockTransport(handler), **kwargs)


def test_translates_history_tools_and_request_options() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_response(_message("done")))

    client = _client(handler)
    image_url = "data:image/png;base64,cG5n"
    client.complete(
        messages=[
            {"role": "system", "content": "control the robot"},
            {"role": "user", "content": "reach the cube"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Current observation."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
            {
                "role": "assistant",
                "content": "I will move.",
                "tool_calls": [
                    _tool_call("call_move", "move_by", '{"deltas":{"dx":0.1}}'),
                    _tool_call("call_extra", "give_up", '{"reason":"extra"}'),
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_extra",
                "content": "ignored: one tool call per turn",
            },
            {"role": "tool", "tool_call_id": "call_move", "content": "moved"},
            {"role": "assistant", "content": "plain assistant history"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "move_by",
                    "description": "Move by a delta.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        temperature=0.2,
        reasoning_effort="medium",
    )

    (request,) = seen
    assert request.url == httpx.URL("http://llm.test/v1/responses")
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {
        "model": "m",
        "input": [
            {"role": "system", "content": "control the robot"},
            {"role": "user", "content": "reach the cube"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Current observation."},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
            {"role": "assistant", "content": "I will move."},
            {
                "type": "function_call",
                "call_id": "call_move",
                "name": "move_by",
                "arguments": '{"deltas":{"dx":0.1}}',
            },
            {
                "type": "function_call",
                "call_id": "call_extra",
                "name": "give_up",
                "arguments": '{"reason":"extra"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_extra",
                "output": "ignored: one tool call per turn",
            },
            {
                "type": "function_call_output",
                "call_id": "call_move",
                "output": "moved",
            },
            {"role": "assistant", "content": "plain assistant history"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "move_by",
                "description": "Move by a delta.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "temperature": 0.2,
        "reasoning": {"effort": "medium"},
    }
    history_messages = [item for item in body["input"] if "role" in item]
    assert all("type" not in item for item in history_messages)


def test_optional_request_fields_are_omitted_when_unset() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler)
    client.complete(messages=[], tools=[])

    assert bodies[0]["tools"] == []
    assert "temperature" not in bodies[0]
    assert "reasoning" not in bodies[0]


@pytest.mark.parametrize("content", [None, ""])
def test_empty_assistant_turn_emits_no_input_item(content: str | None) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler)
    client.complete(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Respond with exactly one tool call."},
        ],
        tools=[],
    )

    assert bodies[0]["input"] == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "Respond with exactly one tool call."},
    ]


def test_cache_miss_synthesizes_tool_call_only_turn_without_null_message() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler)
    client.complete(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("external", "done", '{"summary":"ok"}')],
            },
            {"role": "tool", "tool_call_id": "external", "content": "done: ok"},
        ],
        tools=[],
    )

    assert bodies[0]["input"] == [
        {
            "type": "function_call",
            "call_id": "external",
            "name": "done",
            "arguments": '{"summary":"ok"}',
        },
        {
            "type": "function_call_output",
            "call_id": "external",
            "output": "done: ok",
        },
    ]
    assert not any("content" in item and item["content"] is None for item in bodies[0]["input"])


def test_replays_all_raw_items_once_before_function_output() -> None:
    reasoning = {
        "id": "rs_1",
        "type": "reasoning",
        "encrypted_content": "encrypted-turn-one",
        "summary": [],
    }
    message = _message("I will move once.")
    call = _call("call_move", "move_by", '{"deltas":{"dx":0.1}}')
    requests: list[dict[str, Any]] = []
    responses = [_response(reasoning, message, call), _response()]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    client = _client(handler)
    first = client.complete(messages=[{"role": "user", "content": "move"}], tools=[])
    client.complete(
        messages=[
            {"role": "user", "content": "move"},
            first.raw(),
            {"role": "tool", "tool_call_id": "call_move", "content": "moved"},
        ],
        tools=[],
    )

    replay = requests[1]["input"]
    assert replay[1:4] == [reasoning, message, call]
    assert replay[4] == {
        "type": "function_call_output",
        "call_id": "call_move",
        "output": "moved",
    }
    assert replay.count(reasoning) == 1
    assert replay.count(message) == 1
    assert replay.count(call) == 1
    assert json.dumps(replay).count("I will move once.") == 1


def test_two_function_calls_replay_cached_output_once_per_assistant_turn() -> None:
    reasoning = {"id": "rs_1", "type": "reasoning", "encrypted_content": "encrypted"}
    first_call = _call("call_1", "move_by", "{}", item_id="fc_1")
    second_call = _call("call_2", "give_up", "{}", item_id="fc_2")
    requests: list[dict[str, Any]] = []
    responses = [_response(reasoning, first_call, second_call), _response()]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    client = _client(handler)
    first = client.complete(messages=[{"role": "user", "content": "move"}], tools=[])
    client.complete(
        messages=[
            {"role": "user", "content": "move"},
            first.raw(),
            {"role": "tool", "tool_call_id": "call_2", "content": "ignored"},
            {"role": "tool", "tool_call_id": "call_1", "content": "moved"},
        ],
        tools=[],
    )

    replay = requests[1]["input"]
    for cached_item in (reasoning, first_call, second_call):
        assert replay.count(cached_item) == 1
    # No synthesized duplicates alongside the cached items: exactly the two
    # cached function_call items, however they are keyed.
    assert sum(item.get("type") == "function_call" for item in replay) == 2


def test_cache_is_pruned_when_submitted_history_drops_call_ids() -> None:
    cached_call = _call("old_call")
    requests: list[dict[str, Any]] = []
    responses = [_response(cached_call), _response(), _response()]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    client = _client(handler)
    client.complete(messages=[{"role": "user", "content": "first"}], tools=[])
    client.complete(messages=[{"role": "user", "content": "fresh history"}], tools=[])
    client.complete(
        messages=[
            {
                "role": "assistant",
                "content": "synthesized",
                "tool_calls": [_tool_call("old_call", "done", '{"summary":"ok"}')],
            }
        ],
        tools=[],
    )

    assert requests[2]["input"] == [
        {"role": "assistant", "content": "synthesized"},
        {
            "type": "function_call",
            "call_id": "old_call",
            "name": "done",
            "arguments": '{"summary":"ok"}',
        },
    ]


@pytest.mark.parametrize(
    ("output", "content", "calls"),
    [
        ([_message("hello")], "hello", []),
        ([_call("call_1", "done", "{}")], None, [("call_1", "done", "{}")]),
        (
            [_message("moving"), _call("call_2", "move_by", '{"deltas":{}}')],
            "moving",
            [("call_2", "move_by", '{"deltas":{}}')],
        ),
        (
            [_message("one", item_id="msg_1"), _message("two", item_id="msg_2")],
            "onetwo",
            [],
        ),
    ],
    ids=["text", "tool", "text_and_tool", "multiple_messages"],
)
def test_parses_response_output(
    output: list[dict[str, Any]],
    content: str | None,
    calls: list[tuple[str, str, str]],
) -> None:
    client = _client(lambda request: httpx.Response(200, json=_response(*output)))

    result = client.complete(messages=[], tools=[])

    assert result.content == content
    assert [(call.id, call.name, call.arguments) for call in result.tool_calls] == calls


def test_parser_concatenates_only_output_text_parts() -> None:
    item = _message("first")
    item["content"].extend(
        [
            {"type": "refusal", "refusal": "no"},
            {"type": "output_text", "text": " second", "annotations": []},
        ]
    )
    client = _client(lambda request: httpx.Response(200, json=_response(item)))

    result = client.complete(messages=[], tools=[])

    assert result.content == "first second"


def test_failed_status_raises_once_and_does_not_populate_cache() -> None:
    failed_call = _call("failed_call")
    requests: list[dict[str, Any]] = []
    responses = [
        {
            "id": "resp_failed",
            "status": "failed",
            "error": {"message": "reasoning token limit reached"},
            "output": [failed_call],
        },
        _response(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    client = _client(handler, max_retries=3, backoff_s=0.0)
    with pytest.raises(RuntimeError, match="reasoning token limit reached"):
        client.complete(messages=[], tools=[])
    assert len(requests) == 1

    client.complete(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("failed_call", "done", "{}")],
            }
        ],
        tools=[],
    )
    assert requests[1]["input"] == [
        {
            "type": "function_call",
            "call_id": "failed_call",
            "name": "done",
            "arguments": "{}",
        }
    ]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_transient_http_errors_retry_then_succeed(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(status_code, text="temporary")
        return httpx.Response(200, json=_response(_message("ok")))

    client = _client(handler, backoff_s=0.0)

    assert client.complete(messages=[], tools=[]).content == "ok"
    assert calls == 3


def test_transport_errors_retry_then_succeed() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json=_response())

    client = _client(handler, backoff_s=0.0)

    client.complete(messages=[], tools=[])
    assert calls == 2


def test_retries_exhausted_and_client_errors_fail_fast() -> None:
    server_calls = 0

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal server_calls
        server_calls += 1
        return httpx.Response(503, text="down")

    client = _client(unavailable, backoff_s=0.0, max_retries=2)
    with pytest.raises(RuntimeError, match="503"):
        client.complete(messages=[], tools=[])
    assert server_calls == 2

    client_calls = 0

    def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal client_calls
        client_calls += 1
        return httpx.Response(400, json={"error": {"message": "bad input"}})

    client = _client(rejected, backoff_s=0.0)
    with pytest.raises(RuntimeError, match="bad input"):
        client.complete(messages=[], tools=[])
    assert client_calls == 1


def test_close_closes_underlying_http_client() -> None:
    client = _client(lambda request: httpx.Response(200, json=_response()))

    client.close()

    assert client._http.is_closed


def test_policy_uses_responses_wire_through_act_and_records_config() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response(_call()))

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        wire="responses",
        transport=httpx.MockTransport(handler),
        env={},
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    policy.act(Observation())

    assert requests[0].url.path == "/v1/responses"
    body = json.loads(requests[0].content)
    assert body["reasoning"] == {"effort": "low"}
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.wire == "responses"


def test_policy_rejects_invalid_wire_and_defaults_config_to_chat() -> None:
    with pytest.raises(ValueError, match="wire must be one of"):
        LLMAgentPolicy(
            model="test/model",
            base_url="http://llm.test/v1",
            wire="messages",
            env={},
        )

    policy = LLMAgentPolicy(model="test/model", base_url="http://llm.test/v1", env={})
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.wire == "chat"
