"""Gemini Live v1beta wire behavior and policy integration (plan 0039)."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from _stub_bidi_server import StubBidiServer, completed_text, tool_call
from websockets.sync.server import ServerConnection

import inspect_robots_agent._gemini_live as live_module
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots_agent import GeminiLiveClient, LLMAgentPolicy
from inspect_robots_agent._capture import WireCapture
from inspect_robots_agent._llm import Provider
from inspect_robots_agent.policy import _GEMINI_LIVE_BASE, AgentPolicyConfig

_PNG_BYTES = b"\x89PNG\r\n\x1a\ngemini-live-test"
_PNG_PAYLOAD = base64.b64encode(_PNG_BYTES).decode("ascii")


@pytest.fixture
def server_factory() -> Iterator[Callable[..., StubBidiServer]]:
    servers: list[StubBidiServer] = []

    def make(*args: Any, **kwargs: Any) -> StubBidiServer:
        server = StubBidiServer(*args, **kwargs)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


def _client(
    server: StubBidiServer,
    *,
    model: str = "gemini-robotics-er-2-streaming-preview",
    api_key: str = "",
    capture: WireCapture | None = None,
    max_retries: int = 3,
) -> GeminiLiveClient:
    return GeminiLiveClient(
        Provider(base_url=server.url, api_key=api_key, model=model),
        timeout_s=1.0,
        backoff_s=0.0,
        max_retries=max_retries,
        capture=capture,
    )


def _messages(*users: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "drive safely"},
        {"role": "user", "content": "Goal: inspect"},
        *users,
    ]


def _observation(text: str = "observation") -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _image_observation(text: str, payload: str = _PNG_PAYLOAD) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "text", "text": "camera 'top':"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
    }


def _schema(name: str = "done") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} the task",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
            },
        },
    }


def _tool_message(call_id: str, output: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _content_messages(server: StubBidiServer, connection: int = 0) -> list[dict[str, Any]]:
    return [
        message
        for message in server.connections[connection]
        if "clientContent" in message or "toolResponse" in message
    ]


def _wire_rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "wire/run-1/scene-e0/calls.jsonl"
    return [cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines()]


def test_setup_round_trip_and_first_call_suffix_shape(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call()]])
    client = _client(server, model="google/gemini-robotics-er-2-streaming-preview")
    messages = _messages(_observation())

    client.complete(messages, [_schema()], temperature=0.2)

    setup = server.connections[0][0]["setup"]
    assert setup == {
        "model": "models/gemini-robotics-er-2-streaming-preview",
        "systemInstruction": {"parts": [{"text": "drive safely"}]},
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "done",
                        "description": "done the task",
                        "parameters": {
                            "type": "object",
                            "properties": {"summary": {"type": "string"}},
                        },
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    assert "toolConfig" not in setup
    content = _content_messages(server)
    assert len(content) == 2
    assert content[0]["clientContent"]["turnComplete"] is False
    assert content[1]["clientContent"]["turnComplete"] is True
    assert content[0]["clientContent"]["turns"][0]["parts"] == [{"text": "Goal: inspect"}]
    assert "turns" in content[1]["clientContent"]
    assert all("toolResponse" not in message for message in content)


def test_default_endpoint_and_pending_setup_are_socketless() -> None:
    policy = LLMAgentPolicy(
        model="google/gemini-robotics-er-2-streaming-preview",
        wire="gemini-live",
        wire_capture=False,
        env={"GEMINI_API_KEY": "secret"},
    )

    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.base_url == _GEMINI_LIVE_BASE
    assert isinstance(policy._client, GeminiLiveClient)
    assert policy._client._setup == {"model": "models/gemini-robotics-er-2-streaming-preview"}
    assert "secret" not in json.dumps(policy.config.__dict__)


def test_normal_step_orders_observation_before_tool_response_and_empty_trigger(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    empty = {"serverContent": {"turnComplete": True}}
    server = server_factory([[tool_call("fc_1")], [empty], [tool_call("fc_2")]])
    client = _client(server)
    history = _messages(_observation("old"))
    first = client.complete(history, [])
    history.extend(
        [
            first.raw(),
            _tool_message("fc_1", "motion played"),
            _observation("fresh"),
        ]
    )

    second = client.complete(history, [])

    assert second.tool_calls[0].id == "fc_2"
    tail = _content_messages(server)[-3:]
    assert tail[0]["clientContent"]["turnComplete"] is False
    assert tail[0]["clientContent"]["turns"][0]["parts"] == [{"text": "fresh"}]
    assert tail[1] == {
        "toolResponse": {
            "functionResponses": [
                {
                    "id": "fc_1",
                    "name": "done",
                    "response": {"output": "motion played"},
                }
            ]
        }
    }
    assert tail[2] == {"clientContent": {"turnComplete": True}}


def test_nudge_is_user_only_without_tool_response(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([completed_text("describe"), [tool_call("fc_2")]])
    client = _client(server)
    history = _messages(_observation())
    first = client.complete(history, [])
    history.extend(
        [
            first.raw(),
            {"role": "user", "content": "Respond with exactly one tool call."},
        ]
    )

    client.complete(history, [])

    tail = _content_messages(server)[-1]
    assert tail["clientContent"]["turnComplete"] is True
    assert tail["clientContent"]["turns"][0]["parts"] == [
        {"text": "Respond with exactly one tool call."}
    ]
    assert "toolResponse" not in tail


def test_tool_only_retry_has_no_trailing_turn_complete(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call("fc_1")], [tool_call("fc_2")]])
    client = _client(server)
    history = _messages(_observation())
    first = client.complete(history, [])
    history.extend([first.raw(), _tool_message("fc_1", "pre-check rejected")])

    client.complete(history, [])

    assert _content_messages(server)[-1] == {
        "toolResponse": {
            "functionResponses": [
                {
                    "id": "fc_1",
                    "name": "done",
                    "response": {"output": "pre-check rejected"},
                }
            ]
        }
    }


def test_on_demand_tool_and_image_message_use_three_part_sequence(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    empty = {"serverContent": {"turnComplete": True}}
    server = server_factory([[tool_call("pic", "take_pic")], [empty], [tool_call("move")]])
    client = _client(server)
    history = _messages(_observation())
    first = client.complete(history, [])
    history.extend(
        [
            first.raw(),
            _tool_message("pic", "captured 1 frame(s): 'top'"),
            _image_observation("camera delivery"),
        ]
    )

    second = client.complete(history, [])

    assert second.tool_calls[0].id == "move"
    tail = _content_messages(server)[-3:]
    assert tail[0]["clientContent"]["turnComplete"] is False
    assert "inlineData" in tail[0]["clientContent"]["turns"][0]["parts"][-1]
    assert "toolResponse" in tail[1]
    assert tail[2] == {"clientContent": {"turnComplete": True}}


def test_multiple_function_calls_aggregate_ordered_function_responses(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    two_calls = {
        "toolCall": {
            "functionCalls": [
                {"name": "move_joints", "args": {"x": 1}, "id": "move"},
                {"name": "take_pic", "args": {"camera": "top"}, "id": "pic"},
            ]
        }
    }
    server = server_factory([[two_calls], [tool_call("next")]])
    client = _client(server)
    history = _messages(_observation())

    first = client.complete(history, [])

    assert [call.id for call in first.tool_calls] == ["move", "pic"]
    assert [json.loads(call.arguments) for call in first.tool_calls] == [
        {"x": 1},
        {"camera": "top"},
    ]
    history.extend(
        [
            first.raw(),
            _tool_message("move", "executed"),
            _tool_message("pic", "queued"),
        ]
    )
    client.complete(history, [])
    response = _content_messages(server)[-1]["toolResponse"]["functionResponses"]
    assert response == [
        {
            "id": "move",
            "name": "move_joints",
            "response": {"output": "executed"},
        },
        {"id": "pic", "name": "take_pic", "response": {"output": "queued"}},
    ]
    assert sum("toolResponse" in message for message in _content_messages(server)) == 1


def test_assistant_turns_are_never_resent_as_client_content(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([completed_text("thinking"), [tool_call()]])
    client = _client(server)
    history = _messages(_observation())
    first = client.complete(history, [])
    history.extend([first.raw(), {"role": "user", "content": "nudge"}])

    client.complete(history, [])

    turns = [
        turn
        for message in _content_messages(server)
        for turn in message.get("clientContent", {}).get("turns", [])
    ]
    assert all(turn["role"] == "user" for turn in turns)
    assert "thinking" not in json.dumps(turns)


def test_png_data_uri_translates_to_inline_data(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call()]])

    _client(server).complete(_messages(_image_observation("look")), [])

    parts = _content_messages(server)[-1]["clientContent"]["turns"][0]["parts"]
    assert parts[-1] == {"inlineData": {"mimeType": "image/png", "data": _PNG_PAYLOAD}}


def test_unknown_user_part_fails_loudly(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory()
    history = _messages({"role": "user", "content": [{"type": "audio", "data": "x"}]})

    with pytest.raises(RuntimeError, match="unsupported content part type 'audio'"):
        _client(server, max_retries=1).complete(history, [])


def test_policy_text_turn_takes_existing_nudge_path_end_to_end(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory(
        [completed_text("I should use a tool"), [tool_call("done", "done", {"summary": "ok"})]]
    )
    policy = LLMAgentPolicy(
        model="test/model",
        base_url=server.url,
        wire="gemini-live",
        wire_capture=False,
        env={},
    )
    embodiment = CubePickEmbodiment()
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="inspect")
    policy.reset(scene)

    chunk = policy.act(embodiment.reset(scene, seed=0))

    assert chunk.actions[0].meta["request_stop"] is True
    assert chunk.actions[0].meta["stop_reason"] == "done"
    assert any(
        message.get("clientContent", {}).get("turns", [{}])[0].get("parts", [{}])[0].get("text")
        == "Respond with exactly one tool call."
        for message in _content_messages(server)
        if message.get("clientContent", {}).get("turns")
    )


def test_drain_resumption_and_empty_turn_before_tool_call(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory(
        [
            [
                {"sessionResumptionUpdate": {"newHandle": "discard-me"}},
                {"serverContent": {"turnComplete": True}},
                tool_call(),
            ]
        ]
    )

    message = _client(server).complete(_messages(_observation()), [])

    assert message.tool_calls[0].id == "fc_1"


def test_drain_cap_exhaustion_recovers_then_raises(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    noise = [{"sessionResumptionUpdate": {"newHandle": str(index)}} for index in range(64)]
    server = server_factory([noise, noise, noise])

    with pytest.raises(RuntimeError, match=r"failed after 3 attempts.*drain cap exhausted"):
        _client(server).complete(_messages(_observation()), [])

    assert len(server.connections) == 3


def test_usage_metadata_is_summed_and_normalized(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory(
        [
            [
                {
                    "sessionResumptionUpdate": {},
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 8,
                    },
                },
                {
                    **tool_call(),
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 11,
                        "totalTokenCount": 18,
                    },
                },
            ]
        ]
    )

    result = _client(server).complete(_messages(_observation()), [])

    assert result.usage == {"input_tokens": 10, "output_tokens": 16, "total_tokens": 26}


def test_usage_from_failed_attempt_survives_recovery(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        if connection == 0 and message.get("clientContent", {}).get("turnComplete") is True:
            server.send(
                ws,
                {
                    "sessionResumptionUpdate": {},
                    "usageMetadata": {"promptTokenCount": 9},
                },
            )
            ws.close()
            return True
        return False

    server = server_factory(
        [[{**tool_call(), "usageMetadata": {"promptTokenCount": 4}}]],
        hook=hook,
    )

    result = _client(server).complete(_messages(_observation()), [])

    assert result.usage == {"input_tokens": 13}


def test_mid_step_drop_recovers_with_sanitized_jsonl_and_one_anchor_image(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    old_payload = base64.b64encode(b"old-image").decode("ascii")

    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        content = message.get("clientContent")
        if connection == 0 and isinstance(content, dict) and content.get("turns"):
            text = json.dumps(content["turns"])
            if "newest observation" in text and content.get("turnComplete") is False:
                ws.close()
                return True
        return False

    server = server_factory([[tool_call("old")], [tool_call("recovered")]], hook=hook)
    client = _client(server)
    history = _messages(_image_observation("older observation", old_payload))
    first = client.complete(history, [])
    history.extend(
        [
            first.raw(),
            _tool_message("old", "played"),
            _image_observation("newest observation"),
        ]
    )

    result = client.complete(history, [])

    assert result.tool_calls[0].id == "recovered"
    recovery = server.connections[1]
    assert all("toolResponse" not in message for message in recovery)
    assert all(
        turn.get("role") != "model"
        for message in recovery
        for turn in message.get("clientContent", {}).get("turns", [])
    )
    wire = json.dumps(recovery)
    assert wire.count(_PNG_PAYLOAD) == 1
    assert old_payload not in wire
    assert "[image omitted: streamed camera frame]" in wire
    content = _content_messages(server, 1)
    prologue = content[0]["clientContent"]["turns"][0]["parts"][0]["text"]
    lines = prologue.splitlines()
    assert lines[0] == live_module._RECOVERY_PROLOGUE
    assert lines[-1] == live_module._RECOVERY_CONTINUATION
    assert all(isinstance(json.loads(line), dict) for line in lines[1:-1])
    # The fresh session re-sends the system prompt as systemInstruction;
    # folding it into the prologue too would duplicate it.
    assert all(json.loads(line).get("role") != "system" for line in lines[1:-1])
    anchor = content[1]["clientContent"]
    assert anchor["turnComplete"] is True
    assert "newest observation" in json.dumps(anchor)


def test_go_away_finishes_exchange_then_recovers_at_next_boundary(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory(
        [
            [{"goAway": {"timeLeft": "5s"}}, tool_call("old")],
            [tool_call("new")],
        ]
    )
    client = _client(server)
    history = _messages(_observation("old observation"))
    first = client.complete(history, [])
    history.extend([first.raw(), _tool_message("old", "played"), _observation("new observation")])

    result = client.complete(history, [])

    assert result.tool_calls[0].id == "new"
    assert len(server.connections) == 2
    assert all("toolResponse" not in message for message in server.connections[1])


def test_go_away_then_drop_recovers_once_without_stale_flag(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        if connection == 0 and message.get("clientContent", {}).get("turnComplete") is True:
            server.send(ws, {"goAway": {"timeLeft": "1s"}})
            ws.close()
            return True
        return False

    server = server_factory([[tool_call("recovered")], [tool_call("next")]], hook=hook)
    client = _client(server)
    history = _messages(_observation("first observation"))

    first = client.complete(history, [])

    assert first.tool_calls[0].id == "recovered"
    history.extend(
        [first.raw(), _tool_message("recovered", "played"), _observation("second observation")]
    )

    second = client.complete(history, [])

    # The goAway died with its socket: the recovered session must survive the
    # next boundary instead of being torn down by the stale flag.
    assert second.tool_calls[0].id == "next"
    assert len(server.connections) == 2


def test_first_call_transient_failure_retries_without_recovery_prologue(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        del server
        if connection == 0 and "setup" in message:
            ws.close()
            return True
        return False

    server = server_factory([[tool_call("fc_1")]], hook=hook)

    result = _client(server).complete(_messages(_observation()), [])

    assert result.tool_calls[0].id == "fc_1"
    assert len(server.connections) == 2
    retry_wire = json.dumps(server.connections[1])
    assert live_module._RECOVERY_PROLOGUE not in retry_wire
    tail = _content_messages(server, 1)[-1]["clientContent"]
    assert tail["turnComplete"] is True


def test_persistent_drop_exhausts_retries(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        del server, connection
        if message.get("clientContent", {}).get("turnComplete") is True:
            ws.close()
            return True
        return False

    server = server_factory(hook=hook)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _client(server).complete(_messages(_observation()), [])

    assert len(server.connections) == 3


def test_on_demand_frameless_anchor_recovers(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        del server
        content = message.get("clientContent")
        if (
            connection == 0
            and isinstance(content, dict)
            and content.get("turnComplete") is False
            and "frameless anchor" in json.dumps(content)
        ):
            ws.close()
            return True
        return False

    server = server_factory([[tool_call("pic", "take_pic")], [tool_call("new")]], hook=hook)
    client = _client(server)
    history = _messages(_observation())
    first = client.complete(history, [])
    history.extend(
        [first.raw(), _tool_message("pic", "no frame"), _observation("frameless anchor")]
    )

    result = client.complete(history, [])

    assert result.tool_calls[0].id == "new"
    anchor = _content_messages(server, 1)[1]
    assert "inlineData" not in json.dumps(anchor)
    assert anchor["clientContent"]["turnComplete"] is True


def test_byte_identical_new_trial_reopens_by_identity(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call("first")], [tool_call("second")]])
    client = _client(server)
    first_history = _messages(_observation("same"))
    second_history = json.loads(json.dumps(first_history))

    client.complete(first_history, [])
    client.complete(second_history, [])

    assert len(server.connections) == 2
    assert server.connections[1][0]["setup"]["model"].startswith("models/")


def test_rewritten_view_with_same_system_identity_raises_guard(
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call()]])
    client = _client(server)
    history = _messages(_observation())
    client.complete(history, [])
    view = [history[0], *json.loads(json.dumps(history[1:]))]

    with pytest.raises(RuntimeError, match="rewritten conversation view"):
        client.complete(view, [])


def test_policy_trial_end_closes_and_next_trial_reopens_lazily(
    tmp_path: Path,
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call("first")], [tool_call("second")]])
    policy = LLMAgentPolicy(
        model="test/model",
        base_url=server.url,
        wire="gemini-live",
        wire_capture=False,
        env={},
    )
    assert isinstance(policy._client, GeminiLiveClient)
    policy._client.complete(_messages(_observation("same")), [])
    policy.on_trial_end(
        TrialRecord(scene_id="s0", epoch=0, seed=0, status="success"),
        str(tmp_path),
        "run",
    )
    server.wait_for_closed(1)
    assert len(server.connections) == 1

    policy._client.complete(_messages(_observation("same")), [])

    assert len(server.connections) == 2


def test_trial_end_is_clean_when_server_is_already_gone(
    tmp_path: Path,
    server_factory: Callable[..., StubBidiServer],
) -> None:
    server = server_factory([[tool_call()]])
    policy = LLMAgentPolicy(
        model="test/model",
        base_url=server.url,
        wire="gemini-live",
        wire_capture=False,
        env={},
    )
    assert isinstance(policy._client, GeminiLiveClient)
    policy._client.complete(_messages(_observation()), [])
    server.stop()

    policy.on_trial_end(
        TrialRecord(scene_id="s0", epoch=0, seed=0, status="success"),
        str(tmp_path),
        "run",
    )


def test_per_wire_default_resolution_and_explicit_none() -> None:
    live = LLMAgentPolicy(
        model="google/gemini-robotics-er-2-streaming-preview",
        wire="gemini-live",
        wire_capture=False,
        env={"GEMINI_API_KEY": "g"},
    )
    explicit_none = LLMAgentPolicy(
        model="m",
        base_url="ws://stub.test",
        wire="gemini-live",
        effort=None,
        image_horizon=None,
        wire_capture=False,
        env={},
    )
    chat = LLMAgentPolicy(model="m", base_url="http://stub.test", wire_capture=False, env={})
    responses = LLMAgentPolicy(
        model="m",
        base_url="http://stub.test",
        wire="responses",
        wire_capture=False,
        env={},
    )
    anthropic = LLMAgentPolicy(
        model="m",
        base_url="http://stub.test",
        wire="anthropic",
        wire_capture=False,
        env={},
    )

    assert isinstance(live.config, AgentPolicyConfig)
    assert isinstance(explicit_none.config, AgentPolicyConfig)
    assert isinstance(chat.config, AgentPolicyConfig)
    assert isinstance(responses.config, AgentPolicyConfig)
    assert isinstance(anthropic.config, AgentPolicyConfig)
    assert (live.config.effort, live.config.image_horizon) == (None, None)
    assert (explicit_none.config.effort, explicit_none.config.image_horizon) == (
        None,
        None,
    )
    assert (chat.config.effort, chat.config.image_horizon) == ("low", 2)
    assert (responses.config.effort, responses.config.image_horizon) == ("low", 2)
    assert (anthropic.config.effort, anthropic.config.image_horizon) == ("low", 2)


def test_live_rejects_explicit_effort_and_image_horizon_with_guidance() -> None:
    common: dict[str, Any] = {
        "model": "m",
        "base_url": "ws://stub.test",
        "wire": "gemini-live",
        "wire_capture": False,
        "env": {},
    }
    with pytest.raises(ConfigError) as effort_error:
        LLMAgentPolicy(**common, effort="low")
    assert str(effort_error.value) == (
        "effort is not supported on wire='gemini-live'.\nfix: drop -P effort="
    )
    with pytest.raises(ConfigError) as horizon_error:
        LLMAgentPolicy(**common, image_horizon=2)
    assert "Live API's own context-window compression is the equivalent mechanism" in str(
        horizon_error.value
    )
    assert "already-streamed frames cannot be evicted" in str(horizon_error.value)


def test_existing_invalid_effort_and_horizon_messages_are_preserved() -> None:
    common: dict[str, Any] = {
        "model": "m",
        "base_url": "ws://stub.test",
        "wire": "gemini-live",
        "env": {},
    }
    with pytest.raises(ConfigError) as effort_error:
        LLMAgentPolicy(**common, effort="default")
    assert str(effort_error.value) == (
        "effort must be one of ['high', 'low', 'max', 'medium', 'minimal', 'none', "
        "'xhigh'], or None to omit the field, got 'default'"
    )
    with pytest.raises(ConfigError) as horizon_error:
        LLMAgentPolicy(**common, image_horizon=0)
    assert str(horizon_error.value) == (
        "image_horizon must be an int >= 1, or None to send full image history.\n"
        "fix: pass -P image_horizon=N or -P image_horizon=none"
    )
    with pytest.raises(ConfigError, match="image_horizon must be an int"):
        LLMAgentPolicy(**common, image_horizon=True)


def test_wrong_provider_openrouter_and_http_base_url_are_guided_without_key_leak() -> None:
    expected_fix = "fix: use -P model=google/... and set $GEMINI_API_KEY"
    with pytest.raises(ConfigError) as foreign:
        LLMAgentPolicy(
            model="openai/gpt-test",
            wire="gemini-live",
            env={"OPENAI_API_KEY": "openai-secret"},
        )
    assert expected_fix in str(foreign.value)

    with pytest.raises(ConfigError) as router:
        LLMAgentPolicy(
            model="google/gemini-test",
            wire="gemini-live",
            env={"OPENROUTER_API_KEY": "router-secret"},
        )
    assert expected_fix in str(router.value)
    assert "router-secret" not in str(router.value)

    with pytest.raises(ConfigError) as http_error:
        LLMAgentPolicy(
            model="google/gemini-test",
            base_url="http://stub.test",
            wire="gemini-live",
            env={},
        )
    assert "requires a ws:// or wss:// base_url" in str(http_error.value)
    assert "fix:" in str(http_error.value)


def test_capture_rows_attempts_responses_blobs_and_key_redaction(
    tmp_path: Path,
    server_factory: Callable[..., StubBidiServer],
) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")

    def hook(
        server: StubBidiServer,
        ws: ServerConnection,
        connection: int,
        message: dict[str, Any],
    ) -> bool:
        if connection == 0 and message.get("clientContent", {}).get("turnComplete") is True:
            server.send(ws, {"sessionResumptionUpdate": {"padding": "x" * 3000}})
            ws.close()
            return True
        return False

    long_batch = [
        {"sessionResumptionUpdate": {"padding": str(index) + "y" * 200}} for index in range(20)
    ] + [tool_call("recovered")]
    server = server_factory([long_batch], hook=hook)
    client = _client(server, api_key="capture-secret", capture=capture)

    client.complete(_messages(_image_observation("anchor")), [])
    capture.end_trial()

    rows = _wire_rows(tmp_path)
    assert len(rows) == 2
    assert [row["attempt"] for row in rows] == [0, 1]
    assert [row["call"] for row in rows] == [0, 0]
    assert all(row["endpoint"] == "bidi" for row in rows)
    assert "setup" in rows[0]["request"]["messages"][0]
    assert "setup" in rows[1]["request"]["messages"][0]
    assert len(rows[1]["response"]["messages"]) == 22
    assert len(rows[0]["response"]["messages"][1]["sessionResumptionUpdate"]["padding"]) == 3000
    serialized = json.dumps(rows)
    assert "capture-secret" not in serialized
    assert _PNG_PAYLOAD not in serialized
    digest = hashlib.sha256(_PNG_BYTES).hexdigest()
    assert f"$blob:{digest}" in serialized
    assert (tmp_path / f"wire/run-1/blobs/{digest}.png").read_bytes() == _PNG_BYTES


def test_transport_error_url_is_redacted_before_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "wss://live.test/BidiGenerateContent"
    secret = "never-print-this"

    def fail_connect(url: str, **kwargs: Any) -> None:
        del kwargs
        raise OSError(f"cannot connect to {url}")

    monkeypatch.setattr(live_module, "ws_connect", fail_connect)
    client = GeminiLiveClient(
        Provider(base_url=endpoint, api_key=secret, model="google/model"),
        max_retries=1,
        backoff_s=0.0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.complete(_messages(_observation()), [])

    message = str(exc_info.value)
    assert endpoint in message
    assert "?key=" not in message
    assert secret not in message
