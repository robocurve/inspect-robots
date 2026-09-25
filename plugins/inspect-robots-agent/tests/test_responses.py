"""OpenAI Responses wire client and policy integration."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import pytest

from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._capture import WireCapture
from inspect_robots_agent._llm import Provider
from inspect_robots_agent._responses import (
    ResponsesClient,
    _translate_content_parts,
    _translate_tools,
)
from inspect_robots_agent.policy import AgentPolicyConfig, _evicted_view


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


def _client(handler: Any, *, model: str = "m", **kwargs: Any) -> ResponsesClient:
    provider = Provider(base_url="http://llm.test/v1", api_key="sk-test", model=model)
    return ResponsesClient(provider, transport=httpx.MockTransport(handler), **kwargs)


def _wire_rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "wire/run-1/scene-e0/calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def test_unknown_content_part_is_ignored() -> None:
    assert _translate_content_parts([{"type": "vendor_extension", "value": "x"}]) == []


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra", "openai/gpt-6-astra"])
def test_matches_instruction_and_elision_anchors_without_changing_history(
    role: str, model: str
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    messages: list[dict[str, Any]] = [
        {"role": role, "content": "control the robot"},
        {"role": "user", "content": [{"type": "text", "text": "older elided observation"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "previous state"},
                {"type": "text", "text": "[1 camera frame(s) elided]"},
            ],
            "cache_anchor": True,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "current state"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
            ],
        },
    ]
    original = deepcopy(messages)
    client = _client(handler, model=model)
    client.complete(messages=messages, tools=[])
    client.complete(messages=messages, tools=[])

    body = bodies[0]
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert body["input"][0] == {
        "role": role,
        "content": [
            {
                "type": "input_text",
                "text": "control the robot",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ],
    }
    assert body["input"][2]["content"] == [
        {"type": "input_text", "text": "previous state"},
        {
            "type": "input_text",
            "text": "[1 camera frame(s) elided]",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
    ]
    assert body["input"][3] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "current state"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,cG5n",
            },
            {
                "type": "input_text",
                "text": "",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
        ],
    }
    assert json.dumps(body).count('"prompt_cache_breakpoint"') == 3
    assert "cache_anchor" not in json.dumps(body)
    assert messages == original
    assert bodies[1] == body


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-5.5", "o3", "custom-model", "gpt-60"])
def test_older_and_unknown_models_keep_existing_caching_behavior(model: str) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    _client(handler, model=model).complete(
        messages=[
            {"role": "system", "content": "control the robot"},
            {"role": "user", "content": "elided history", "cache_anchor": True},
        ],
        tools=[],
    )
    assert "prompt_cache_options" not in bodies[0]
    assert bodies[0]["input"] == [
        {"role": "system", "content": "control the robot"},
        {"role": "user", "content": "elided history"},
    ]


def _marked_items(body: dict[str, Any]) -> list[int]:
    return [
        index
        for index, item in enumerate(body["input"])
        if isinstance(blocks := item.get("content", item.get("output")), list)
        and any("prompt_cache_breakpoint" in block for block in blocks)
    ]


def test_append_only_reuse_survives_nudges_without_marker_accumulation() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra")
    messages: list[dict[str, Any]] = [{"role": "system", "content": "instructions"}]
    for turn in range(25):
        messages.append({"role": "user", "content": f"observation or nudge {turn}"})
        client.complete(messages=messages, tools=[])
    assert _marked_items(bodies[0]) == [0, 1]
    assert _marked_items(bodies[1]) == [0, 1, 2]
    assert _marked_items(bodies[2]) == [0, 2, 3]
    assert _marked_items(bodies[-1]) == [0, 24, 25]
    assert all(isinstance(item["content"], list) for item in bodies[-1]["input"])
    assert "prompt_cache_breakpoint" not in json.dumps(messages)


def test_eviction_retains_only_unchanged_prefixes_and_moves_the_anchor() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra")
    messages: list[dict[str, Any]] = [{"role": "system", "content": "instructions"}]
    for turn in range(4):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"state {turn}"},
                    {"type": "text", "text": "camera 'top':"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{turn}"}},
                ],
            }
        )
        client.complete(messages=_evicted_view(messages, 1, mark_anchor=True), tools=[])
    assert [_marked_items(body) for body in bodies] == [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 2, 3, 4],
    ]
    assert bodies[-1]["input"][-1]["content"][-2]["type"] == "input_image"
    assert bodies[-1]["input"][-1]["content"][-1]["text"] == ""
    # Changing an early prefix invalidates later entries, even if their own blocks match.
    messages[1]["content"][0]["text"] = "corrected state"
    client.complete(messages=_evicted_view(messages, 1, mark_anchor=True), tools=[])
    assert _marked_items(bodies[-1]) == [0, 3, 4]
    assert all(len(_marked_items(body)) <= 4 for body in bodies)


def test_full_image_history_preserves_images_and_reuses_previous_image_endpoint() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra")
    messages: list[dict[str, Any]] = [{"role": "system", "content": "instructions"}]
    urls = [f"data:image/png;base64,{turn}" for turn in range(4)]
    for url in urls:
        messages.append(
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
        )
        client.complete(messages=messages, tools=[])
    assert [_marked_items(body) for body in bodies] == [[0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 4]]
    assert [item["content"][0]["image_url"] for item in bodies[-1]["input"][1:]] == urls
    assert "prompt_cache_breakpoint" not in json.dumps(messages)


@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-4.1"])
def test_image_endpoints_keep_empty_text_anchor_after_marker_moves(role: str, model: str) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    content = (
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}]
        if role == "user"
        else [{"type": "input_image", "image_url": "data:image/png;base64,cG5n"}]
    )
    messages: list[dict[str, Any]] = [{"role": role, "content": content}]
    if role == "tool":
        messages[0]["tool_call_id"] = "capture"
    original = deepcopy(messages)
    client = _client(handler, model=model)
    for turn in range(3):
        client.complete(messages=messages, tools=[])
        messages.append({"role": "user", "content": f"next observation {turn}"})
    field = "content" if role == "user" else "output"
    first = bodies[0]["input"][0][field]
    if model == "gpt-6-astra":
        assert first == [
            {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
            {"type": "input_text", "text": "", "prompt_cache_breakpoint": {"mode": "explicit"}},
        ]
        assert bodies[1]["input"][0][field] == first
        assert bodies[2]["input"][0][field] == [first[0], {"type": "input_text", "text": ""}]
        assert [_marked_items(body) for body in bodies] == [[0], [0, 1], [1, 2]]
    else:
        assert first == [{"type": "input_image", "image_url": "data:image/png;base64,cG5n"}]
    assert messages[:1] == original


@pytest.mark.parametrize("retry", [False, True])
def test_tool_tail_normalization_keeps_previous_output_eligible_for_reuse(retry: bool) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if retry and len(bodies) == 1:
            return httpx.Response(429, text="retry")
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra", backoff_s=0)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "instructions"},
        {"role": "tool", "tool_call_id": "one", "content": "first result"},
        {"role": "tool", "tool_call_id": "two", "content": "second result"},
    ]
    original = deepcopy(messages)
    client.complete(messages=messages, tools=[])
    if retry:
        assert bodies[0] == bodies[1]
        bodies.pop(0)
    messages.append({"role": "user", "content": "try another call"})
    client.complete(messages=messages, tools=[])
    assert _marked_items(bodies[0]) == [0, 2]
    assert _marked_items(bodies[1]) == [0, 2, 3]
    assert bodies[1]["input"][1]["output"] == [{"type": "input_text", "text": "first result"}]
    assert bodies[0]["input"][2] == bodies[1]["input"][2]
    assert messages[:3] == original


@pytest.mark.parametrize("changed", ["tools", "effort", "temperature"])
def test_cache_settings_changes_invalidate_prior_endpoints(changed: str) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "A"},
    ]
    client.complete(messages=messages, tools=[], reasoning_effort="low")
    messages.append({"role": "user", "content": "B"})
    tools = [{"function": {"name": "move", "description": "move", "parameters": {}}}]
    client.complete(
        messages=messages,
        tools=tools if changed == "tools" else [],
        reasoning_effort="high" if changed == "effort" else "low",
        temperature=0.2 if changed == "temperature" else None,
    )
    assert _marked_items(bodies[-1]) == [0, 2]


@pytest.mark.parametrize("failure", ["http", "failed", "incomplete"])
def test_unsuccessful_requests_do_not_add_prefix_candidates(failure: str) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 2:
            if failure == "http":
                return httpx.Response(400, text="rejected")
            return httpx.Response(200, json={"status": failure, "output": []})
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "A"},
    ]
    client.complete(messages=messages, tools=[])
    messages.append({"role": "user", "content": "B"})
    if failure == "incomplete":
        client.complete(messages=messages, tools=[])
    else:
        with pytest.raises(RuntimeError):
            client.complete(messages=messages, tools=[])
    messages.append({"role": "user", "content": "C"})
    client.complete(messages=messages, tools=[])
    assert _marked_items(bodies[-1]) == [0, 1, 3]


def test_cache_retry_body_is_stable_and_reset_drops_old_candidates() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(429, text="retry")
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra", backoff_s=0)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "A"},
    ]
    client.complete(messages=messages, tools=[])
    assert bodies[0] == bodies[1]
    client._reset_cache_tracking()
    messages.append({"role": "user", "content": "B"})
    client.complete(messages=messages, tools=[])
    assert _marked_items(bodies[-1]) == [0, 2]


def test_unsupported_assistant_tail_does_not_get_a_marker() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    _client(handler, model="gpt-6-astra").complete(
        messages=[{"role": "assistant", "content": "unchanged assistant text"}], tools=[]
    )
    assert _marked_items(bodies[0]) == []
    assert bodies[0]["input"] == [{"role": "assistant", "content": "unchanged assistant text"}]


def test_policy_reset_clears_candidates_even_for_identical_scene_content() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    policy = LLMAgentPolicy(
        model="gpt-6-astra",
        base_url="http://llm.test/v1",
        wire="responses",
        transport=httpx.MockTransport(handler),
        env={},
    )
    policy.bind(CubePickEmbodiment().info)
    scene = Scene(id="same", instruction="same goal")
    policy.reset(scene)
    policy._messages.append({"role": "user", "content": "A"})
    policy._client.complete(messages=policy._messages, tools=[])
    assert _marked_items(bodies[-1]) == [0, 2]
    policy.reset(scene)
    policy._messages.extend([{"role": "user", "content": "A"}, {"role": "user", "content": "B"}])
    policy._client.complete(messages=policy._messages, tools=[])
    assert _marked_items(bodies[-1]) == [0, 3]


@pytest.mark.parametrize(
    "messages, expected",
    [
        ([], []),
        ([{"role": "system", "content": "instructions"}], [0]),
        ([{"role": "user", "content": "elided", "cache_anchor": True}], [0]),
        ([{"role": "user", "content": []}], []),
    ],
)
def test_coincident_or_absent_cache_targets_are_not_duplicated(
    messages: list[dict[str, Any]], expected: list[int]
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _client(handler, model="gpt-6-astra")
    for _ in range(2):
        client.complete(messages=messages, tools=[])
    assert [_marked_items(body) for body in bodies] == [expected, expected]


def test_capture_history_keeps_function_outputs_in_call_order_before_images() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    _client(handler).complete(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("move", "move_joints", '{"targets":{"joint":0.2}}'),
                    _tool_call("pic", "take_pic", '{"note":"inspect"}'),
                ],
            },
            {"role": "tool", "tool_call_id": "move", "content": "executing move"},
            {
                "role": "tool",
                "tool_call_id": "pic",
                "content": "captured 1 frame(s): 'top'",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "camera 'top' (step 1):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,cG5n"},
                    },
                ],
            },
        ],
        tools=[],
    )

    assert bodies[0]["input"] == [
        {
            "type": "function_call",
            "call_id": "move",
            "name": "move_joints",
            "arguments": '{"targets":{"joint":0.2}}',
        },
        {
            "type": "function_call",
            "call_id": "pic",
            "name": "take_pic",
            "arguments": '{"note":"inspect"}',
        },
        {"type": "function_call_output", "call_id": "move", "output": "executing move"},
        {
            "type": "function_call_output",
            "call_id": "pic",
            "output": "captured 1 frame(s): 'top'",
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "camera 'top' (step 1):"},
                {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
            ],
        },
    ]


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
    assert "service_tier" not in bodies[0]


@pytest.mark.parametrize("service_tier", ["auto", "default", "flex", "priority", "fast"])
def test_service_tier_is_sent_on_every_retry_and_captured(
    service_tier: str, tmp_path: Path
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(503, json={"error": {"message": "retry"}})
        return httpx.Response(200, json={**_response(), "service_tier": "default"})

    capture = WireCapture()
    capture.begin_trial(log_dir=str(tmp_path), run_id="run-1", trial_id="scene-e0")
    client = _client(handler, service_tier=service_tier, capture=capture, backoff_s=0)
    try:
        client.complete(messages=[], tools=[], reasoning_effort="medium")
    finally:
        client.close()
        capture.end_trial()

    assert len(bodies) == 2
    assert all(body["service_tier"] == service_tier for body in bodies)
    assert all(body["reasoning"] == {"effort": "medium"} for body in bodies)
    rows = _wire_rows(tmp_path)
    assert len(rows) == 2
    assert all(row["request"]["service_tier"] == service_tier for row in rows)
    assert rows[-1]["response"]["service_tier"] == "default"


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


@pytest.mark.parametrize("model", ["m", "gpt-6-astra"])
def test_replays_all_raw_items_once_before_function_output(model: str) -> None:
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

    client = _client(handler, model=model)
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
    expected_output: Any = "moved"
    if model == "gpt-6-astra":
        expected_output = [
            {
                "type": "input_text",
                "text": "moved",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
    assert replay[4] == {
        "type": "function_call_output",
        "call_id": "call_move",
        "output": expected_output,
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


def test_capture_precedes_failed_payload_raise(tmp_path: Path) -> None:
    payload = {
        "id": "resp_failed",
        "status": "failed",
        "error": {"message": "reasoning token limit reached"},
        "output": [],
    }
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    client = _client(
        lambda request: httpx.Response(200, json=payload),
        capture=capture,
    )

    with pytest.raises(RuntimeError, match="reasoning token limit reached"):
        client.complete(messages=[], tools=[])

    (row,) = _wire_rows(tmp_path)
    assert row["status"] == 200
    assert row["response"] == payload


def test_capture_records_responses_transport_error(tmp_path: Path) -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("responses offline", request=request)

    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    client = _client(offline, max_retries=1, capture=capture)

    with pytest.raises(RuntimeError, match="responses offline"):
        client.complete(messages=[], tools=[])

    (row,) = _wire_rows(tmp_path)
    assert row["status"] is None
    assert row["response"] is None
    assert row["error"] == "responses offline"


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


def test_retry_after_header_overrides_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("inspect_robots_agent._responses.time.sleep", sleeps.append)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        return httpx.Response(200, json=_response(_message("ok")))

    _client(handler, backoff_s=1.0).complete(messages=[], tools=[])

    assert sleeps == [7.0]


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


@pytest.mark.parametrize("service_tier", [None, "auto", "default", "flex", "priority", "fast"])
def test_policy_uses_responses_wire_through_act_and_records_config(
    service_tier: str | None,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response(_call()))

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        wire="responses",
        service_tier=service_tier,
        transport=httpx.MockTransport(handler),
        env={},
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    policy.act(Observation())

    assert requests[0].url.path == "/v1/responses"
    body = json.loads(requests[0].content)
    assert "reasoning" not in body
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.wire == "responses"
    assert policy.config.effort is None
    assert asdict(policy.config)["service_tier"] == service_tier
    if service_tier is None:
        assert "service_tier" not in body
    else:
        assert body["service_tier"] == service_tier


@pytest.mark.parametrize("service_tier", ["", "turbo", "FAST", " fast", 42, True, ["fast"]])
def test_policy_rejects_invalid_service_tier(service_tier: Any) -> None:
    with pytest.raises(ConfigError, match="service_tier"):
        LLMAgentPolicy(
            model="test/model",
            base_url="http://llm.test/v1",
            wire="responses",
            service_tier=service_tier,
            env={},
        )


@pytest.mark.parametrize("wire", ["chat", "messages", "anthropic", "gemini-live", "interactions"])
def test_policy_rejects_service_tier_on_other_wires(wire: str) -> None:
    with pytest.raises(ConfigError, match="service_tier is only supported on wire='responses'"):
        LLMAgentPolicy(
            model="test/model",
            base_url="http://llm.test/v1",
            wire=wire,
            service_tier="fast",
            env={},
        )


def test_policy_rejects_invalid_wire_and_defaults_config_to_chat() -> None:
    with pytest.raises(ConfigError, match="wire must be one of"):
        LLMAgentPolicy(
            model="test/model",
            base_url="http://llm.test/v1",
            wire="response",
            env={},
        )

    policy = LLMAgentPolicy(model="test/model", base_url="http://llm.test/v1", env={})
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.wire == "chat"


def test_translate_tools_handles_missing_description_and_parameters() -> None:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "simple_tool",
            },
        }
    ]
    translated = _translate_tools(tools)
    assert translated == [
        {
            "type": "function",
            "name": "simple_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


@pytest.mark.parametrize(
    ("wire", "base_url"),
    [
        ("chat", "http://llm.test/v1"),
        ("responses", "http://llm.test/v1"),
        ("messages", "http://llm.test/v1"),
        ("gemini-live", "ws://llm.test/v1beta"),
        ("interactions", "http://llm.test/v1beta"),
    ],
)
def test_policy_forwards_retry_configuration_to_every_wire(
    wire: str,
    base_url: str,
) -> None:
    policy = LLMAgentPolicy(
        model="m",
        base_url=base_url,
        wire=wire,
        max_retries=8,
        backoff_s=2.5,
        wire_capture=False,
        env={},
    )

    assert policy._client._max_retries == 8
    assert policy._client._backoff_s == 2.5
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.max_retries == 8
    assert policy.config.backoff_s == 2.5


def test_policy_keeps_existing_positional_parameter_order() -> None:
    policy = LLMAgentPolicy(
        "m",
        "http://llm.test/v1",
        None,
        "chat",
        False,
        None,
        None,
        100,
        0.25,
        max_retries=8,
        backoff_s=2.5,
        env={},
    )

    assert policy._temperature == 0.25
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.temperature == 0.25


def test_policy_config_keeps_existing_positional_field_order() -> None:
    config = AgentPolicyConfig(
        1,
        None,
        0.25,
        "m",
        None,
        None,
        "chat",
        True,
        None,
        "flex",
        None,
        100,
        "none",
        0.1,
        False,
        "always",
        "render",
        2,
        None,
        None,
        None,
    )

    assert config.effort == "none"
    assert config.max_speed_frac == 0.1
    assert config.pre_check is None
    assert config.service_tier == "flex"
    assert config.max_retries == 3
    assert config.backoff_s == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": 0},
        {"max_retries": True},
        {"backoff_s": -1.0},
        {"backoff_s": float("inf")},
        {"backoff_s": float("nan")},
        {"backoff_s": True},
    ],
)
def test_policy_rejects_invalid_retry_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match=r"max_retries|backoff_s"):
        LLMAgentPolicy(
            model="m",
            base_url="http://llm.test/v1",
            wire_capture=False,
            env={},
            **kwargs,
        )
