"""Dependency-free OpenAI-compatible chat wire shared by summarize and the VLM grader.

Blocking POSTs to ``<base_url>/chat/completions`` over stdlib ``urllib``,
with an injectable transport seam (``http_post``) so callers and tests never
touch the network. The token cap is sent as ``max_tokens``; when a 400
response body names ``max_completion_tokens`` (OpenAI reasoning models
reject ``max_tokens`` and suggest that substitute), the identical request
is retried once with the cap under that key. A caller-supplied reasoning
effort rides on both sends as ``reasoning_effort`` and is omitted from the
body entirely when unset. Errors are raised as guided
``ConfigError``s whose prefix and fix hint the caller labels for its own
command surface.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, cast

from inspect_robots.errors import ConfigError

HttpPost = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]

_RESPONSE_EXCERPT_LIMIT = 500

_TOKEN_CAP = 8192


class _ChatHTTPError(ConfigError):
    """The endpoint answered with a non-2xx status, kept in ``status``.

    Still a ``ConfigError``, so existing callers behave as before; callers that
    must tell a rejected request (4xx) from an outage (5xx) read ``status``.
    """

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class _ChatTransportError(ConfigError):
    """The request never got an HTTP answer (DNS, refused, timeout, dropped)."""


def _response_excerpt(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return (text or "(empty response body)")[:_RESPONSE_EXCERPT_LIMIT]


def _check_url(url: str) -> None:
    """Reject a URL that can never work, so it reads as configuration, not an outage.

    urllib only discovers a bad scheme, a missing host, a bad port or embedded
    whitespace while opening the connection, where it looks like a transport
    failure. Checking first keeps a typo such as ``htps://`` from being
    mistaken for a network blip (plan 0085).
    """
    problem = None
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:  # e.g. an unbalanced IPv6 bracket
        parts = None
        problem = str(exc)
    if parts is None:
        pass
    elif parts.scheme not in ("http", "https"):
        problem = "the scheme must be http or https"
    elif not parts.hostname:
        problem = "no host given"
    elif any(ch.isspace() or ord(ch) < 32 for ch in url):
        problem = "it contains whitespace or control characters"
    else:
        try:
            parts.port  # noqa: B018 - raises ValueError for a bad port
        except ValueError as exc:
            problem = str(exc)
    if problem is not None:
        raise ConfigError(
            f"chat request failed: invalid URL {url!r}: {problem}.\n"
            "fix: check the base URL (e.g. https://api.example.com/v1)"
        )


def _urllib_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
    """Send one blocking HTTP POST and preserve HTTP error bodies for guided failures."""
    _check_url(url)
    request = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except (OSError, http.client.HTTPException):
            # Keep the status: a 4xx with an unreadable body is still a
            # rejected request, not a transport failure.
            body = b""
        return exc.code, body
    except urllib.error.URLError as exc:
        raise _ChatTransportError(
            f"chat request failed: {exc.reason}.\n"
            "fix: check the base URL and network connectivity, then retry"
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        # A read timeout or dropped connection arrives here rather than as a
        # URLError; it is still a transport failure, not a request problem.
        raise _ChatTransportError(
            f"chat request failed: {type(exc).__name__}: {exc}.\n"
            "fix: check the base URL and network connectivity, then retry"
        ) from exc


def _post_chat(
    post: HttpPost,
    url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    token_param: str,
    effort: str | float | None,
) -> tuple[int, bytes]:
    """Send one chat-completions request with the token cap keyed under ``token_param``.

    ``effort`` arrives already normalized by the calling surface and is sent
    verbatim as ``reasoning_effort``; ``None`` and ``""`` leave the body
    without the key.
    """
    payload: dict[str, Any] = {"model": model, "messages": messages, token_param: _TOKEN_CAP}
    if effort is not None and effort != "":
        payload["reasoning_effort"] = effort
    body = json.dumps(payload).encode("utf-8")
    return post(url, headers, body)


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    what: str = "summary",
    fix_hint: str = "check --base-url, --model, and the configured API key",
    http_post: HttpPost | None = None,
    effort: str | float | None = None,
) -> str:
    """Return one OpenAI-compatible chat completion or raise a guided configuration error.

    ``what`` labels error prefixes for the calling command and ``fix_hint``
    names that command's remedy flags in the non-2xx failure message.
    ``effort`` is sent verbatim as ``reasoning_effort``; ``None`` and ``""``
    omit the key so the provider default applies.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    post = _urllib_post if http_post is None else http_post
    status, response_body = _post_chat(post, url, headers, model, messages, "max_tokens", effort)
    if status == 400 and "max_completion_tokens" in response_body.decode("utf-8", errors="replace"):
        status, response_body = _post_chat(
            post, url, headers, model, messages, "max_completion_tokens", effort
        )
    if not 200 <= status < 300:
        raise _ChatHTTPError(
            f"{what} request failed with HTTP {status}: {_response_excerpt(response_body)}\n"
            f"fix: {fix_hint}",
            status,
        )

    try:
        payload = cast(dict[str, Any], json.loads(response_body))
        choices = cast(list[Any], payload["choices"])
        message = cast(dict[str, Any], choices[0]["message"])
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
        raise ConfigError(
            f"{what} endpoint returned a malformed reply: {_response_excerpt(response_body)}\n"
            "fix: use an OpenAI-compatible /chat/completions endpoint"
        ) from None
    return content
