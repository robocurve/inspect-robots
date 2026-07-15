"""Provider resolution ladder + OpenAI-compatible chat client (plan 0008 §4a)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inspect_robots.errors import ConfigError
from inspect_robots_agent._llm import ChatClient, resolve_provider

# --- provider resolution ladder ----------------------------------------------


def test_explicit_base_url_wins_over_everything() -> None:
    env = {"OPENROUTER_API_KEY": "or-key", "ANTHROPIC_API_KEY": "ant-key"}
    p = resolve_provider(
        model="anthropic/claude-fable-5",
        base_url="http://localhost:8000/v1",
        api_key_env=None,
        env=env,
    )
    assert p.base_url == "http://localhost:8000/v1"
    assert p.api_key == "or-key"  # default api_key_env is OPENROUTER_API_KEY
    assert p.model == "anthropic/claude-fable-5"  # custom endpoints get the raw string


def test_explicit_base_url_with_custom_key_env() -> None:
    env = {"MY_KEY": "sk-mine"}
    p = resolve_provider(
        model="local/foo", base_url="http://box:1234/v1", api_key_env="MY_KEY", env=env
    )
    assert p.api_key == "sk-mine"


def test_explicit_base_url_without_key_allows_keyless_endpoints() -> None:
    p = resolve_provider(model="m", base_url="http://localhost:8000/v1", api_key_env=None, env={})
    assert p.api_key == ""


def test_anthropic_model_with_anthropic_key() -> None:
    p = resolve_provider(
        model="anthropic/claude-fable-5",
        base_url=None,
        api_key_env=None,
        env={"ANTHROPIC_API_KEY": "ant-key", "OPENROUTER_API_KEY": "or-key"},
    )
    assert "api.anthropic.com" in p.base_url
    assert p.api_key == "ant-key"
    assert p.model == "claude-fable-5"  # provider prefix stripped for the compat endpoint


def test_openai_model_with_openai_key() -> None:
    p = resolve_provider(
        model="openai/gpt-5.2", base_url=None, api_key_env=None, env={"OPENAI_API_KEY": "oai"}
    )
    assert "api.openai.com" in p.base_url
    assert p.api_key == "oai"
    assert p.model == "gpt-5.2"


def test_openrouter_is_the_universal_fallback() -> None:
    p = resolve_provider(
        model="anthropic/claude-fable-5",
        base_url=None,
        api_key_env=None,
        env={"OPENROUTER_API_KEY": "or-key"},  # no ANTHROPIC_API_KEY
    )
    assert "openrouter.ai" in p.base_url
    assert p.model == "anthropic/claude-fable-5"  # OpenRouter takes the full string


@pytest.mark.parametrize(
    ("model", "key_env", "host", "bare"),
    [
        (
            "google/gemini-3.1-flash-lite",
            "GEMINI_API_KEY",
            "googleapis.com",
            "gemini-3.1-flash-lite",
        ),
        ("x-ai/grok-4.3", "XAI_API_KEY", "api.x.ai", "grok-4.3"),
        ("xai/grok-4.3", "XAI_API_KEY", "api.x.ai", "grok-4.3"),
        # Groq serves ids that themselves contain a slash; only the first
        # segment is the routing prefix.
        (
            "groq/meta-llama/llama-4-scout-17b-16e-instruct",
            "GROQ_API_KEY",
            "api.groq.com",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        ("mistralai/mistral-medium-3", "MISTRAL_API_KEY", "api.mistral.ai", "mistral-medium-3"),
        ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY", "api.deepseek.com", "deepseek-chat"),
    ],
)
def test_direct_provider_table(model: str, key_env: str, host: str, bare: str) -> None:
    p = resolve_provider(
        model=model,
        base_url=None,
        api_key_env=None,
        env={key_env: "sk-provider", "OPENROUTER_API_KEY": "or-key"},
    )
    assert host in p.base_url
    assert p.api_key == "sk-provider"  # provider key preferred over OpenRouter
    assert p.model == bare


def test_openrouter_variant_suffix_is_never_claimed_directly() -> None:
    # ":nitro"/":free" mean something to OpenRouter only; the direct endpoint
    # would 404 on them even with the provider key set.
    p = resolve_provider(
        model="google/gemini-3.5-flash:nitro",
        base_url=None,
        api_key_env=None,
        env={"GEMINI_API_KEY": "gk", "OPENROUTER_API_KEY": "or-key"},
    )
    assert "openrouter.ai" in p.base_url
    assert p.model == "google/gemini-3.5-flash:nitro"


def test_variant_suffix_without_openrouter_key_is_a_guided_error() -> None:
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        resolve_provider(
            model="google/gemini-3.5-flash:nitro",
            base_url=None,
            api_key_env=None,
            env={"GEMINI_API_KEY": "gk"},
        )


def test_bare_prefix_without_model_id_is_not_claimed() -> None:
    # "anthropic" alone must not resolve to an empty model id on the compat
    # endpoint; it falls through the ladder like any unroutable string.
    p = resolve_provider(
        model="anthropic",
        base_url=None,
        api_key_env=None,
        env={"ANTHROPIC_API_KEY": "ant-key", "OPENROUTER_API_KEY": "or-key"},
    )
    assert "openrouter.ai" in p.base_url


def test_guided_error_names_the_new_provider_keys() -> None:
    with pytest.raises(ConfigError) as excinfo:
        resolve_provider(model="google/gemini-3.5-flash", base_url=None, api_key_env=None, env={})
    message = str(excinfo.value)
    assert "GEMINI_API_KEY" in message
    assert "google/*" in message


def test_missing_model_is_a_guided_error() -> None:
    with pytest.raises(ConfigError, match="INSPECT_ROBOTS_MODEL"):
        resolve_provider(model=None, base_url=None, api_key_env=None, env={})


def test_no_matching_key_is_a_guided_error() -> None:
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY") as excinfo:
        resolve_provider(model="openai/gpt-5.2", base_url=None, api_key_env=None, env={})
    assert "OPENAI_API_KEY" in str(excinfo.value)


# --- chat client ---------------------------------------------------------------


def _tool_call_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "moving now",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "move_joints",
                                "arguments": '{"targets": {"j0": 0.5}, "duration_s": 1.0}',
                            },
                        }
                    ],
                }
            }
        ]
    }


def _client(handler: Any, **kwargs: Any) -> ChatClient:
    provider = resolve_provider(
        model="m", base_url="http://llm.test/v1", api_key_env="K", env={"K": "sk-test"}
    )
    return ChatClient(provider, transport=httpx.MockTransport(handler), **kwargs)


def test_complete_sends_wire_format_and_parses_tool_calls() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_tool_call_response())

    client = _client(handler)
    msg = client.complete(
        messages=[{"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "move_joints"}}],
        temperature=0.2,
    )
    (request,) = seen
    assert request.url == httpx.URL("http://llm.test/v1/chat/completions")
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "m"
    assert body["tools"][0]["function"]["name"] == "move_joints"
    assert body["temperature"] == 0.2
    assert msg.content == "moving now"
    (call,) = msg.tool_calls
    assert call.id == "call_1"
    assert call.name == "move_joints"
    assert json.loads(call.arguments)["duration_s"] == 1.0


def test_reasoning_effort_is_sent_when_set_and_omitted_when_none() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_call_response())

    client = _client(handler)
    client.complete(messages=[], tools=[], reasoning_effort="low")
    client.complete(messages=[], tools=[])
    assert bodies[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in bodies[1]


def test_keyless_provider_sends_no_authorization_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=_tool_call_response())

    provider = resolve_provider(model="m", base_url="http://llm.test/v1", api_key_env=None, env={})
    client = ChatClient(provider, transport=httpx.MockTransport(handler))
    client.complete(messages=[], tools=[])


def test_transient_errors_retry_then_succeed() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_tool_call_response())

    client = _client(handler, backoff_s=0.0)
    msg = client.complete(messages=[], tools=[])
    assert calls["n"] == 3
    assert msg.tool_calls


def test_retries_exhausted_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = _client(handler, backoff_s=0.0, max_retries=2)
    with pytest.raises(RuntimeError, match="503"):
        client.complete(messages=[], tools=[])


def test_client_error_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad tool schema"}})

    client = _client(handler, backoff_s=0.0)
    with pytest.raises(RuntimeError, match="bad tool schema"):
        client.complete(messages=[], tools=[])
    assert calls["n"] == 1  # 4xx is our bug, not weather; retrying can't help
