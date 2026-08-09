"""Wire-neutral provider resolution and the Chat Completions client.

No provider SDKs: small ``httpx`` clients speak the selected protocol while
this module resolves direct providers, OpenRouter, and explicit endpoints.
Chat Completions still covers OpenRouter, OpenAI, local vLLM/Ollama, and
provider compatibility endpoints (plan 0008 §4a).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from inspect_robots.errors import ConfigError

from ._capture import WireCapture

ENV_MODEL = "INSPECT_ROBOTS_MODEL"

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_KEY = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class _DirectProvider:
    """A provider with a stable endpoint, native wire, and conventional key name."""

    base_url: str
    key_env: str
    wire: str = "chat"
    keep_prefix: bool = False


# OpenRouter-style prefixes that resolve to the provider's own endpoint when its
# key is present. Prefixes are stripped unless an entry requests the full id.
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
    "thinkingmachines": _DirectProvider(
        "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1",
        "TINKER_API_KEY",
        wire="messages",
        keep_prefix=True,
    ),
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


def _provider_key_hints(native_wires: frozenset[str]) -> str:
    """Name keys only for direct providers whose native wire the caller supports."""
    seen: dict[str, str] = {}
    for prefix, direct in _DIRECT_PROVIDERS.items():
        if direct.wire not in native_wires:
            continue
        seen.setdefault(direct.key_env, prefix)
    return ", ".join(f"${key} for {prefix}/*" for key, prefix in seen.items())


def _direct_claim(
    model: str | None,
    env: Mapping[str, str],
    *,
    native_wires: frozenset[str] = frozenset({"chat"}),
) -> tuple[str, _DirectProvider] | None:
    """Return the claimable prefix and entry, or ``None`` when the ladder must continue."""
    if not model:
        return None
    provider_prefix, _, bare_model = model.partition("/")
    direct = _DIRECT_PROVIDERS.get(provider_prefix)
    if (
        direct is not None
        and direct.wire in native_wires
        and bare_model
        and not _has_openrouter_variant(bare_model)
        and env.get(direct.key_env)
    ):
        return provider_prefix, direct
    return None


@dataclass(frozen=True)
class Provider:
    """A resolved request target and any direct provider's native-wire opinion.

    ``wire`` is ``"chat"`` for explicit endpoints and OpenRouter, and records
    a direct table entry's native wire when that entry claims the model.
    """

    base_url: str
    api_key: str
    model: str
    wire: str = "chat"


def resolve_provider(
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    env: Mapping[str, str],
    *,
    native_wires: frozenset[str] = frozenset({"chat"}),
) -> Provider:
    """The key/base-url ladder (plan 0008 §4a); first match wins.

    1. Explicit ``base_url``: any caller-compatible endpoint; the key comes
       from ``api_key_env`` (default ``OPENROUTER_API_KEY``), and a missing
       key is allowed (local vLLM/Ollama endpoints are typically keyless).
    2. A known ``prefix/*`` model + that provider's key (``_DIRECT_PROVIDERS``:
       anthropic, openai, google, x-ai/xai, groq, mistralai, deepseek,
       thinkingmachines): the provider's own endpoint, with the prefix
       stripped unless the entry requires the full id. ``native_wires`` names
       the protocols the caller can speak; entries on other wires do not claim
       and the ladder continues. Ids ending in a known OpenRouter variant
       suffix (``_OPENROUTER_VARIANTS``, e.g. ``:free``, ``:nitro``) are never
       claimed here. Other colons (fine-tune and checkpoint ids) pass through.
    3. ``OPENROUTER_API_KEY``: OpenRouter, which takes the full
       ``provider/model`` string.

    Anything else is a guided [`ConfigError`][inspect_robots.errors.ConfigError]
    naming the fixes, never a traceback at the user.
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
    claim = _direct_claim(model, env, native_wires=native_wires)
    if claim is not None:
        _, direct = claim
        _, _, bare_model = model.partition("/")
        return Provider(
            base_url=direct.base_url,
            api_key=env[direct.key_env],
            model=model if direct.keep_prefix else bare_model,
            wire=direct.wire,
        )
    if key := env.get(_OPENROUTER_KEY):
        return Provider(base_url=_OPENROUTER_BASE, api_key=key, model=model)
    raise ConfigError(
        f"no API key found for model {model!r}.\n"
        f"fix: set ${_OPENROUTER_KEY} (works for any model), or the provider's "
        f"key ({_provider_key_hints(native_wires)}), "
        "or pass -P base_url=... (+ -P api_key_env=NAME) for a custom endpoint"
    )


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for; ``arguments`` is raw JSON text."""

    id: str
    name: str
    arguments: str
    extra_content: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssistantMessage:
    """One parsed assistant turn shared by the supported API wire clients."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: dict[str, int] | None = None

    def raw(self) -> dict[str, Any]:
        """The wire-format dict to append back onto the conversation."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                    **({"extra_content": c.extra_content} if c.extra_content is not None else {}),
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
        capture: WireCapture | None = None,
    ):
        self._provider = provider
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._capture = capture
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
        body: dict[str, Any] = {"model": self._provider.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        last_error = "unknown error"
        for attempt in range(self._max_retries):
            t_start = time.time() if self._capture is not None else 0.0
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if self._capture is not None:
                    self._capture.record(
                        attempt=attempt,
                        endpoint="/chat/completions",
                        request=body,
                        status=None,
                        response_text=None,
                        error=str(exc),
                        t_start=t_start,
                        duration_s=time.time() - t_start,
                    )
                last_error = str(exc)
            else:
                if self._capture is not None:
                    self._capture.record(
                        attempt=attempt,
                        endpoint="/chat/completions",
                        request=body,
                        status=response.status_code,
                        response_text=response.text,
                        error=None,
                        t_start=t_start,
                        duration_s=time.time() - t_start,
                    )
                if response.status_code == 200:
                    return _parse_message(response.json())
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in (429,) and response.status_code < 500:
                    # A 4xx is our request's fault; retrying cannot help.
                    guidance = ""
                    if "reasoning_effort" in response.text and "/v1/responses" in response.text:
                        guidance = (
                            "\nfix: pass -P wire=responses (needs a direct OpenAI endpoint, e.g. "
                            "$OPENAI_API_KEY set), or -P effort=none"
                        )
                    raise RuntimeError(f"LLM request rejected — {last_error}{guidance}")
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
            extra_content=c.get("extra_content"),
        )
        for c in message.get("tool_calls") or []
    )
    content = message.get("content")
    return AssistantMessage(content=content if content is None else str(content), tool_calls=calls)
