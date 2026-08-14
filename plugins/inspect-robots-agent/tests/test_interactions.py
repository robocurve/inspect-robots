"""Gemini Interactions wire behavior and policy integration (plan 0068)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import inspect_robots_agent._gemini_live as live_module
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots_agent import InteractionsClient, LLMAgentPolicy
from inspect_robots_agent._capture import WireCapture
from inspect_robots_agent._llm import Provider
from inspect_robots_agent.policy import _INTERACTIONS_BASE, AgentPolicyConfig

_PNG_BYTES = b"\x89PNG\r\n\x1a\ninteractions-test"
_PNG_PAYLOAD = base64.b64encode(_PNG_BYTES).decode("ascii")
_BLOB_RE = re.compile(r"\$blob:([0-9a-f]{64})")


def _response(
    interaction_id: str,
    *steps: dict[str, Any],
    status: str = "completed",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": interaction_id,
        "status": status,
        "steps": list(steps),
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _call(
    call_id: str = "call_1",
    name: str = "done",
    arguments: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": call_id,
        "name": name,
        "arguments": {} if arguments is None else arguments,
    }


def _text(*parts: str) -> dict[str, Any]:
    return {
        "type": "model_output",
        "content": [{"type": "text", "text": part} for part in parts],
    }


def _schema(name: str = "done") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} the task",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _messages(*extra: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "drive safely"},
        {"role": "user", "content": "Goal: inspect"},
        *extra,
    ]


def _image_message(text: str, payload: str = _PNG_PAYLOAD) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
    }


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "sk-test",
    max_retries: int = 3,
    capture: WireCapture | None = None,
) -> InteractionsClient:
    return InteractionsClient(
        Provider(
            base_url="http://llm.test/v1beta",
            api_key=api_key,
            model="google/gemini-3.7-flash",
        ),
        max_retries=max_retries,
        backoff_s=0.0,
        transport=httpx.MockTransport(handler),
        capture=capture,
    )


def _body(request: httpx.Request) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(request.content))


def _wire_rows(path: Path) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines()]


def _inline_blobs(value: Any, blob_dir: Path) -> Any:
    if isinstance(value, str):

        def replace_blob(match: re.Match[str]) -> str:
            payload = (blob_dir / f"{match.group(1)}.png").read_bytes()
            return base64.b64encode(payload).decode("ascii")

        return _BLOB_RE.sub(replace_blob, value)
    if isinstance(value, list):
        return [_inline_blobs(item, blob_dir) for item in value]
    if isinstance(value, dict):
        return {key: _inline_blobs(item, blob_dir) for key, item in value.items()}
    return value


def test_first_call_translates_body_tools_system_image_and_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_response("i1", _text("ok")))

    client = _client(handler)
    client.complete(_messages(_image_message("Current observation.")), [_schema()])

    (request,) = seen
    assert request.url == httpx.URL("http://llm.test/v1beta/interactions")
    assert request.headers["x-goog-api-key"] == "sk-test"
    body = _body(request)
    assert body == {
        "model": "gemini-3.7-flash",
        "input": [
            {"type": "text", "text": "Goal: inspect"},
            {"type": "text", "text": "Current observation."},
            {"type": "image", "mime_type": "image/png", "data": _PNG_PAYLOAD},
        ],
        "store": True,
        "tools": [
            {
                "type": "function",
                "name": "done",
                "description": "done the task",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "system_instruction": "drive safely",
    }
    assert "previous_interaction_id" not in body


def test_chained_call_sends_only_tool_and_user_delta_with_scoped_fields() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json=_response(
                    "i1",
                    _call("tool-call-id", "move_by", {"deltas": {"dx": 0.1}}),
                    status="requires_action",
                ),
            )
        return httpx.Response(200, json=_response("i2", _text("done")))

    client = _client(handler)
    history = _messages({"role": "user", "content": "old observation"})
    first = client.complete(history, [_schema("move_by")])
    history.extend(
        [
            first.raw(),
            {"role": "tool", "tool_call_id": "tool-call-id", "content": "moved"},
            {"role": "user", "content": "new observation"},
        ]
    )

    client.complete(history, [_schema("move_by")])

    second = bodies[1]
    assert second["previous_interaction_id"] == "i1"
    assert second["input"] == [
        {
            "type": "function_result",
            "call_id": "tool-call-id",
            "result": "moved",
        },
        {"type": "text", "text": "new observation"},
    ]
    assert second["tools"] == bodies[0]["tools"]
    assert second["system_instruction"] == "drive safely"


def test_function_calls_normalize_object_and_string_arguments() -> None:
    payload = _response(
        "i1",
        _call("one", "move_by", {"deltas": {"dx": 0.1}}),
        _call("two", "take_pic", '{"camera":"top"}'),
        status="requires_action",
    )
    result = _client(lambda request: httpx.Response(200, json=payload)).complete(_messages(), [])

    assert [call.id for call in result.tool_calls] == ["one", "two"]
    assert [call.name for call in result.tool_calls] == ["move_by", "take_pic"]
    assert [json.loads(call.arguments) for call in result.tool_calls] == [
        {"deltas": {"dx": 0.1}},
        {"camera": "top"},
    ]


def test_text_blocks_concatenate_and_empty_steps_return_empty_message() -> None:
    responses = iter(
        [
            _response("i1", _text("one", " two")),
            _response("i2"),
        ]
    )
    client = _client(lambda request: httpx.Response(200, json=next(responses)))

    first = client.complete(_messages(), [])
    second = client.complete(json.loads(json.dumps(_messages())), [])

    assert first.content == "one two"
    assert first.tool_calls == ()
    assert second.content is None
    assert second.tool_calls == ()


def test_usage_counters_are_normalized_and_non_int_values_skipped() -> None:
    usage = {
        "total_input_tokens": 10,
        "total_output_tokens": 4,
        "total_cached_tokens": 3,
        "total_thought_tokens": 2,
        "total_tokens": 16,
        "ignored": 99,
    }
    result = _client(
        lambda request: httpx.Response(200, json=_response("i1", usage=usage))
    ).complete(_messages(), [])
    assert result.usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 3,
        "thought_tokens": 2,
        "total_tokens": 16,
    }

    result = _client(
        lambda request: httpx.Response(
            200,
            json=_response(
                "i2",
                usage={
                    "total_input_tokens": "10",
                    "total_output_tokens": True,
                    "total_tokens": None,
                },
            ),
        )
    ).complete(_messages(), [])
    assert result.usage is None


def test_transient_retry_success_and_exhaustion() -> None:
    calls = 0

    def succeeds(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="temporary")
        return httpx.Response(200, json=_response("i1", _text("ok")))

    assert _client(succeeds).complete(_messages(), []).content == "ok"
    assert calls == 2

    failed_calls = 0

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal failed_calls
        failed_calls += 1
        return httpx.Response(503, text="down")

    with pytest.raises(RuntimeError, match=r"failed after 3 attempts.*503"):
        _client(unavailable).complete(_messages(), [])
    assert failed_calls == 3


def test_plain_and_thinking_level_400_errors_fail_fast_with_guidance() -> None:
    calls = 0

    def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad tool schema")

    with pytest.raises(RuntimeError, match="bad tool schema"):
        _client(rejected).complete(_messages(), [])
    assert calls == 1

    client = _client(lambda request: httpx.Response(400, text="unsupported thinking_level minimal"))
    with pytest.raises(RuntimeError) as exc_info:
        client.complete(_messages(), [])
    assert str(exc_info.value).endswith(
        "fix: supported thinking_level values vary by model; "
        "gemini-3.7-flash accepts low|medium|high"
    )


def test_chain_loss_folds_sanitized_history_then_chains_to_new_head() -> None:
    bodies: list[dict[str, Any]] = []
    old_payload = base64.b64encode(b"old image").decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        index = len(bodies)
        if index == 1:
            return httpx.Response(200, json=_response("i1", _call("move", "move_by")))
        if index == 2:
            return httpx.Response(404, text="interaction expired")
        if index == 3:
            return httpx.Response(200, json=_response("folded", _text("recovered")))
        return httpx.Response(200, json=_response("i3", _text("next")))

    client = _client(handler, max_retries=3)
    history = _messages(_image_message("older observation", old_payload))
    first = client.complete(history, [])
    history.extend(
        [
            first.raw(),
            {"role": "tool", "tool_call_id": "move", "content": "played"},
            _image_message("newest observation"),
        ]
    )

    client.complete(history, [])

    assert bodies[1]["previous_interaction_id"] == "i1"
    fold = bodies[2]
    assert "previous_interaction_id" not in fold
    assert len(fold["input"]) == 3
    prologue = fold["input"][0]["text"]
    lines = prologue.splitlines()
    assert lines[0] == live_module._RECOVERY_PROLOGUE
    assert lines[-1] == live_module._RECOVERY_CONTINUATION
    assert all(isinstance(json.loads(line), dict) for line in lines[1:-1])
    assert all(json.loads(line).get("role") != "system" for line in lines[1:-1])
    assert "[image omitted: streamed camera frame]" in prologue
    assert old_payload not in json.dumps(fold)
    assert fold["input"][1:] == [
        {"type": "text", "text": "newest observation"},
        {"type": "image", "mime_type": "image/png", "data": _PNG_PAYLOAD},
    ]

    history.extend(
        [
            {"role": "assistant", "content": "recovered"},
            {"role": "user", "content": "one more observation"},
        ]
    )
    client.complete(history, [])
    assert bodies[3]["previous_interaction_id"] == "folded"
    assert bodies[3]["input"] == [{"type": "text", "text": "one more observation"}]


def test_fold_persists_across_transient_failure_and_400_chain_loss() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        index = len(bodies)
        if index == 1:
            return httpx.Response(200, json=_response("i1", _text("first")))
        if index == 2:
            return httpx.Response(400, text="unknown previous_interaction_id")
        if index == 3:
            return httpx.Response(500, text="fold temporarily unavailable")
        return httpx.Response(200, json=_response("folded", _text("ok")))

    client = _client(handler, max_retries=4)
    history = _messages()
    first = client.complete(history, [])
    history.extend([first.raw(), {"role": "user", "content": "new observation"}])

    client.complete(history, [])

    assert "previous_interaction_id" in bodies[1]
    assert "previous_interaction_id" not in bodies[2]
    assert "previous_interaction_id" not in bodies[3]
    assert bodies[2]["input"] == bodies[3]["input"]


def test_final_attempt_chain_loss_leaves_fold_for_next_complete() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        if len(bodies) == 1:
            return httpx.Response(200, json=_response("i1", _text("first")))
        if len(bodies) == 2:
            return httpx.Response(404, text="expired")
        return httpx.Response(200, json=_response("folded", _text("recovered")))

    client = _client(handler, max_retries=1)
    history = _messages()
    first = client.complete(history, [])
    history.extend([first.raw(), {"role": "user", "content": "new observation"}])

    with pytest.raises(RuntimeError, match="404"):
        client.complete(history, [])
    client.complete(history, [])

    assert bodies[1]["previous_interaction_id"] == "i1"
    assert "previous_interaction_id" not in bodies[2]
    assert live_module._RECOVERY_PROLOGUE in bodies[2]["input"][0]["text"]


def test_unchained_404_fails_immediately_without_fold() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return httpx.Response(404, text="model path missing")

    with pytest.raises(RuntimeError, match="model path missing"):
        _client(handler).complete(_messages(), [])
    assert len(bodies) == 1
    assert live_module._RECOVERY_PROLOGUE not in json.dumps(bodies)


def test_empty_delta_guard_rejects_assistant_only_suffix() -> None:
    client = _client(lambda request: httpx.Response(200, json=_response("i1", _text("plain text"))))
    history = _messages()
    first = client.complete(history, [])
    history.append(first.raw())

    with pytest.raises(RuntimeError, match="no un-streamed user or tool message"):
        client.complete(history, [])


def test_rewritten_view_guard_and_identity_based_fresh_trial_reset() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return httpx.Response(200, json=_response(f"i{len(bodies)}", _text("ok")))

    client = _client(handler)
    history = _messages({"role": "user", "content": "observation"})
    client.complete(history, [])
    rewritten = [history[0], *json.loads(json.dumps(history[1:]))]

    with pytest.raises(RuntimeError, match="rewritten conversation view"):
        client.complete(rewritten, [])

    fresh = json.loads(json.dumps(history))
    client.complete(fresh, [])
    assert len(bodies) == 2
    assert "previous_interaction_id" not in bodies[1]
    assert bodies[1]["input"] == bodies[0]["input"]


def test_failed_status_raises_and_incomplete_returns_partial_text() -> None:
    failed = _client(lambda request: httpx.Response(200, json=_response("i1", status="failed")))
    with pytest.raises(RuntimeError, match="status 'failed'"):
        failed.complete(_messages(), [])

    incomplete = _client(
        lambda request: httpx.Response(
            200,
            json=_response("i2", _text("partial"), status="incomplete"),
        )
    ).complete(_messages(), [])
    assert incomplete.content == "partial"


def test_missing_steps_raise_and_unknown_step_types_are_skipped() -> None:
    client = _client(lambda request: httpx.Response(200, json={"id": "i1", "status": "completed"}))
    with pytest.raises(RuntimeError, match="steps"):
        client.complete(_messages(), [])

    tolerant = _client(
        lambda request: httpx.Response(
            200,
            json=_response(
                "i2",
                {"type": "thinking", "content": [{"type": "text", "text": "pondering"}]},
                {"type": "model_output", "content": [{"type": "text", "text": "done"}]},
            ),
        )
    )
    message = tolerant.complete(_messages(), [])
    assert message.content == "done"
    assert message.tool_calls == ()


def test_capture_records_interactions_endpoint_and_flat_image_blob(tmp_path: Path) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    client = _client(
        lambda request: httpx.Response(200, json=_response("i1", _text("ok"))),
        capture=capture,
    )

    client.complete(_messages(_image_message("look")), [])
    capture.end_trial()

    (row,) = _wire_rows(tmp_path / "wire/run-1/scene-e0/calls.jsonl")
    assert row["endpoint"] == "/interactions"
    image = next(part for part in row["request"]["input"] if part["type"] == "image")
    digest = hashlib.sha256(_PNG_BYTES).hexdigest()
    assert image == {
        "type": "image",
        "mime_type": "image/png",
        "data": f"$blob:{digest}",
    }
    assert (tmp_path / f"wire/run-1/blobs/{digest}.png").read_bytes() == _PNG_BYTES


def test_generation_config_placement_keyless_header_omission_and_close() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert "x-goog-api-key" not in request.headers
        bodies.append(_body(request))
        return httpx.Response(200, json=_response(f"i{len(bodies)}"))

    client = _client(handler, api_key="")
    client.complete(_messages(), [], temperature=0.2, reasoning_effort="medium")
    client.complete(json.loads(json.dumps(_messages())), [])

    assert bodies[0]["generation_config"] == {
        "temperature": 0.2,
        "thinking_level": "medium",
    }
    assert "temperature" not in bodies[0]
    assert "thinking_level" not in bodies[0]
    assert "generation_config" not in bodies[1]
    client.close()
    assert client._http.is_closed


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_policy_accepts_interactions_effort_levels(effort: str) -> None:
    policy = LLMAgentPolicy(
        model="m",
        base_url="http://llm.test/v1beta",
        wire="interactions",
        effort=effort,
        wire_capture=False,
        env={},
    )
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.effort == effort
    assert policy.config.image_horizon is None


@pytest.mark.parametrize("effort", ["none", "xhigh", "max", 0.5, None])
def test_policy_rejects_interactions_unsupported_effort(
    effort: str | float | None,
) -> None:
    with pytest.raises(ConfigError) as exc_info:
        LLMAgentPolicy(
            model="m",
            base_url="http://llm.test/v1beta",
            wire="interactions",
            effort=effort,
            wire_capture=False,
            env={},
        )
    assert str(exc_info.value).endswith(
        "fix: pass -P effort=minimal|low|medium|high (maps to thinking_level), or drop -P effort="
    )


def test_policy_interactions_horizon_and_ws_base_url_guards() -> None:
    with pytest.raises(ConfigError) as horizon_error:
        LLMAgentPolicy(
            model="m",
            base_url="http://llm.test/v1beta",
            wire="interactions",
            image_horizon=2,
            env={},
        )
    assert "frames already absorbed by the chain cannot be evicted client-side" in str(
        horizon_error.value
    )

    with pytest.raises(ConfigError) as base_error:
        LLMAgentPolicy(
            model="m",
            base_url="wss://llm.test/interactions",
            wire="interactions",
            env={},
        )
    assert str(base_error.value).endswith(
        "fix: wire='interactions' is HTTP; drop -P base_url= or pass the Live wire "
        "a websocket endpoint via -P wire=gemini-live"
    )


def test_policy_interactions_resolution_key_default_and_client_selection() -> None:
    with pytest.raises(ConfigError) as provider_error:
        LLMAgentPolicy(
            model="openai/gpt-test",
            wire="interactions",
            env={"OPENAI_API_KEY": "openai-secret"},
        )
    assert str(provider_error.value) == (
        "wire='interactions' needs Google's direct provider.\n"
        "fix: use -P model=google/... and set $GEMINI_API_KEY"
    )

    explicit = LLMAgentPolicy(
        model="google/gemini-test",
        base_url="https://proxy.test/v1beta",
        wire="interactions",
        wire_capture=False,
        env={"GEMINI_API_KEY": "gemini-secret", "OPENROUTER_API_KEY": "wrong-secret"},
    )
    assert isinstance(explicit._client, InteractionsClient)
    assert explicit._client._provider.api_key == "gemini-secret"

    direct = LLMAgentPolicy(
        model="google/gemini-3.7-flash",
        wire="interactions",
        wire_capture=False,
        env={"GEMINI_API_KEY": "gemini-secret"},
    )
    assert isinstance(direct._client, InteractionsClient)
    assert isinstance(direct.config, AgentPolicyConfig)
    assert direct.config.base_url == _INTERACTIONS_BASE
    assert direct.config.model == "gemini-3.7-flash"
    assert direct.config.effort is None


def test_policy_act_chains_delta_and_capture_round_trips_blobs(tmp_path: Path) -> None:
    raw_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw_requests.append(_body(request))
        if len(raw_requests) == 1:
            return httpx.Response(
                200,
                json=_response(
                    "i1",
                    _call(
                        "move-call",
                        "move_by",
                        {
                            "deltas": {"dx": 0.02, "dy": 0.02},
                            "note": "The cube is ahead, so I will move toward it.",
                        },
                    ),
                    status="requires_action",
                ),
            )
        return httpx.Response(
            200,
            json=_response(
                "i2",
                _call(
                    "done-call",
                    "done",
                    {"summary": "finished", "hindsight": "none"},
                ),
                status="requires_action",
            ),
        )

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1beta",
        wire="interactions",
        depth="off",
        transport=httpx.MockTransport(handler),
        env={},
    )
    embodiment = CubePickEmbodiment()
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="reach the cube")
    policy.reset(scene)
    policy.on_trial_start("s0", 0, str(tmp_path), "run-1")

    first_chunk = policy.act(embodiment.reset(scene, seed=0))
    next_observation = embodiment.step(first_chunk.actions[-1]).observation
    stop_chunk = policy.act(next_observation)

    assert stop_chunk.actions[0].meta["request_stop"] is True
    second = raw_requests[1]
    assert second["previous_interaction_id"] == "i1"
    assert second["input"][0]["type"] == "function_result"
    assert second["input"][0]["call_id"] == "move-call"
    assert any(part.get("type") == "image" for part in second["input"])
    assert all(part.get("type") != "function_call" for part in second["input"])
    assert "The cube is ahead" not in json.dumps(second["input"])

    record = TrialRecord(scene_id="s0", epoch=0, seed=0, status="success")
    policy.on_trial_end(record, str(tmp_path), "run-1")
    pointer = record.metadata["wire_capture"]
    assert isinstance(pointer, str)
    capture_path = tmp_path / pointer
    rows = _wire_rows(capture_path)
    assert [row["endpoint"] for row in rows] == ["/interactions", "/interactions"]
    blob_dir = capture_path.parent.parent / "blobs"
    assert [_inline_blobs(row["request"], blob_dir) for row in rows] == raw_requests


def test_policy_text_turn_takes_existing_nudge_path_end_to_end() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json=_response(
                    "i1",
                    _text("I should use a tool"),
                    status="incomplete",
                ),
            )
        return httpx.Response(
            200,
            json=_response(
                "i2",
                _call(
                    "done-call",
                    "done",
                    {"summary": "ok", "hindsight": "none"},
                ),
                status="requires_action",
            ),
        )

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1beta",
        wire="interactions",
        wire_capture=False,
        transport=httpx.MockTransport(handler),
        env={},
    )
    embodiment = CubePickEmbodiment()
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="inspect")
    policy.reset(scene)

    chunk = policy.act(embodiment.reset(scene, seed=0))

    assert chunk.actions[0].meta["request_stop"] is True
    assert bodies[1]["previous_interaction_id"] == "i1"
    assert bodies[1]["input"] == [{"type": "text", "text": "Respond with exactly one tool call."}]
    assert "I should use a tool" not in json.dumps(bodies[1]["input"])
