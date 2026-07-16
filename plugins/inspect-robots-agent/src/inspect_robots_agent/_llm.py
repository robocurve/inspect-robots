"""OpenAI-compatible chat client + provider resolution (plan 0008 §4a).

No provider SDKs: one ``httpx`` client speaking the chat-completions wire
format covers OpenRouter, OpenAI, local vLLM/Ollama, and Anthropic's
OpenAI-compat endpoint — the same "speak the protocol, don't import the
package" doctrine as the xpolicylab plugin.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from inspect_robots.errors import ConfigError

ENV_MODEL = "INSPECT_ROBOTS_MODEL"

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_KEY = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class _DirectProvider:
    """A provider with a stable OpenAI-compatible endpoint and conventional key name."""

    base_url: str
    key_env: str


# OpenRouter-style prefixes that resolve to the provider's own endpoint when its
# key is present. The prefix is stripped: these endpoints want the bare model id.
_DIRECT_PROVIDERS: dict[str, _DirectProvider] = {
    "anthropic": _DirectProvider("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "openai": _DirectProvider("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "google": _DirectProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"
    ),
    "x-ai": _DirectProvider("https://api.x.ai/v1", "XAI_API_KEY"),
    "xai": _DirectProvider("https://api.x.ai/v1", "XAI_API_KEY"),
    "groq": _DirectProvider("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistralai": _DirectProvider("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": _DirectProvider("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}


# OpenRouter routing-variant suffixes. The variant only means something to
# OpenRouter, so ids carrying one are never claimed by a direct provider. A
# closed set, not "any colon": OpenAI/Mistral fine-tune ids legitimately
# contain colons (ft:gpt-4o-mini:org:suffix:id) and must keep routing direct.
_OPENROUTER_VARIANTS = frozenset({"free", "nitro", "floor", "extended", "online", "thinking"})


def _has_openrouter_variant(model_id: str) -> bool:
    """True when the id ends in a known OpenRouter ``:variant`` suffix."""
    _, sep, suffix = model_id.rpartition(":")
    return bool(sep) and suffix in _OPENROUTER_VARIANTS


def _provider_key_hints() -> str:
    """One ``$KEY for prefix/*`` hint per distinct key, for the guided error."""
    seen: dict[str, str] = {}
    for prefix, direct in _DIRECT_PROVIDERS.items():
        seen.setdefault(direct.key_env, prefix)
    return ", ".join(f"${key} for {prefix}/*" for key, prefix in seen.items())


@dataclass(frozen=True)
class Provider:
    """A resolved OpenAI-compatible endpoint: where, with which key, which model."""

    base_url: str
    api_key: str
    model: str


def resolve_provider(
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    env: Mapping[str, str],
) -> Provider:
    """The key/base-url ladder (plan 0008 §4a); first match wins.

    1. Explicit ``base_url`` — any OpenAI-compatible endpoint; the key comes
       from ``api_key_env`` (default ``OPENROUTER_API_KEY``), and a missing
       key is allowed (local vLLM/Ollama endpoints are typically keyless).
    2. A known ``prefix/*`` model + that provider's key (``_DIRECT_PROVIDERS``:
       anthropic, openai, google, x-ai/xai, groq, mistralai, deepseek) — the
       provider's own endpoint, prefix stripped from the model id. Ids ending
       in a known OpenRouter variant suffix (``_OPENROUTER_VARIANTS``, e.g.
       ``:free``, ``:nitro``) are never claimed here — the variant only means
       something to OpenRouter. Other colons (fine-tune ids) pass through.
    3. ``OPENROUTER_API_KEY`` — OpenRouter, which takes the full
       ``provider/model`` string.

    Anything else is a guided [`ConfigError`][inspect_robots.errors.ConfigError]
    naming the fixes — never a traceback at the user.
    """
    if not model:
        raise ConfigError(
            "no model configured for the agent policy.\n"
            f"fix: pass -P model=provider/model (e.g. anthropic/claude-fable-5) "
            f"or set ${ENV_MODEL}"
        )
    if base_url:
        key_env = api_key_env or _OPENROUTER_KEY
        return Provider(base_url=base_url.rstrip("/"), api_key=env.get(key_env, ""), model=model)
    provider_prefix, _, bare_model = model.partition("/")
    direct = _DIRECT_PROVIDERS.get(provider_prefix)
    if (
        direct is not None
        and bare_model  # a bare "anthropic" must not resolve to an empty model id
        and not _has_openrouter_variant(bare_model)
        and (key := env.get(direct.key_env))
    ):
        return Provider(base_url=direct.base_url, api_key=key, model=bare_model)
    if key := env.get(_OPENROUTER_KEY):
        return Provider(base_url=_OPENROUTER_BASE, api_key=key, model=model)
    raise ConfigError(
        f"no API key found for model {model!r}.\n"
        f"fix: set ${_OPENROUTER_KEY} (works for any model), or the provider's "
        f"key ({_provider_key_hints()}), "
        "or pass -P base_url=... (+ -P api_key_env=NAME) for a custom endpoint"
    )


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for; ``arguments`` is raw JSON text."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantMessage:
    """The parsed ``choices[0].message`` of a chat completion."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]

    def raw(self) -> dict[str, Any]:
        """The wire-format dict to append back onto the conversation."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        return message


class ChatClient:
    """Blocking chat-completions client with bounded retry on transient failures.

    Retries 429/5xx and transport errors with exponential backoff; a 4xx is
    our request's fault and fails immediately. Persistent failure raises
    ``RuntimeError``, which the rollout wraps as ``PolicyError``.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._provider = provider
        self._max_retries = max_retries
        self._backoff_s = backoff_s
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
        reasoning_effort: str | None = None,
    ) -> AssistantMessage:
        body: dict[str, Any] = {"model": self._provider.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        last_error = "unknown error"
        for attempt in range(self._max_retries):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    return _parse_message(response.json())
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in (429,) and response.status_code < 500:
                    # A 4xx is our request's fault; retrying cannot help.
                    raise RuntimeError(f"LLM request rejected — {last_error}")
            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)
        raise RuntimeError(f"LLM request failed after {self._max_retries} attempts — {last_error}")

    def close(self) -> None:
        self._http.close()


def _parse_message(payload: dict[str, Any]) -> AssistantMessage:
    message = payload["choices"][0]["message"]
    calls = tuple(
        ToolCall(
            id=str(c["id"]),
            name=str(c["function"]["name"]),
            arguments=str(c["function"]["arguments"]),
        )
        for c in message.get("tool_calls") or []
    )
    content = message.get("content")
    return AssistantMessage(content=content if content is None else str(content), tool_calls=calls)
