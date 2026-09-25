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
command surface. Ordinary response-body read failures are normalized too;
interrupts propagate unchanged.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from http.client import IncompleteRead
from typing import Any, cast

from inspect_robots.errors import ConfigError

HttpPost = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]

_RESPONSE_EXCERPT_LIMIT = 500

_TOKEN_CAP = 8192


class _HTTPErrorBodyReadError(ConfigError):
    """Keep an HTTP error status and body-read cause together for the wire caller."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code} error response body read failed: {detail}")


def _response_excerpt(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return (text or "(empty response body)")[:_RESPONSE_EXCERPT_LIMIT]


def _body_read_failure_detail(exc: IncompleteRead | OSError) -> str:
    """Return safe guidance describing an ordinary response-body read failure."""
    if isinstance(exc, TimeoutError):
        return "the response body read timed out"
    if isinstance(exc, IncompleteRead):
        return "the response body ended in an incomplete transfer"
    return "the connection was interrupted while reading the response body"


def _read_response_body(read: Callable[[], bytes]) -> bytes:
    """Normalize ordinary network failures raised while consuming a response body."""
    try:
        return read()
    except (IncompleteRead, OSError) as exc:
        detail = _body_read_failure_detail(exc)
        raise ConfigError(
            f"chat request failed: {detail}.\n"
            "fix: check the server response and network connectivity, then retry"
        ) from exc


def _urllib_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
    """Send a blocking POST, preserving complete HTTP error bodies and guiding read failures."""
    request = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            return int(response.status), _read_response_body(response.read)
    except urllib.error.HTTPError as exc:
        with exc:
            try:
                body = exc.read()
            except (IncompleteRead, OSError) as read_error:
                raise _HTTPErrorBodyReadError(
                    exc.code, _body_read_failure_detail(read_error)
                ) from read_error
            return exc.code, body
    except urllib.error.URLError as exc:
        raise ConfigError(
            f"chat request failed: {exc.reason}.\n"
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
    body_read_error: _HTTPErrorBodyReadError | None = None
    try:
        status, response_body = _post_chat(
            post, url, headers, model, messages, "max_tokens", effort
        )
    except _HTTPErrorBodyReadError as exc:
        status, response_body, body_read_error = exc.status_code, b"", exc
    if (
        body_read_error is None
        and status == 400
        and "max_completion_tokens" in response_body.decode("utf-8", errors="replace")
    ):
        try:
            status, response_body = _post_chat(
                post, url, headers, model, messages, "max_completion_tokens", effort
            )
        except _HTTPErrorBodyReadError as exc:
            status, response_body, body_read_error = exc.status_code, b"", exc
    if not 200 <= status < 300:
        excerpt = (
            f"(error response body unavailable: {body_read_error.detail})"
            if body_read_error is not None
            else _response_excerpt(response_body)
        )
        error = ConfigError(f"{what} request failed with HTTP {status}: {excerpt}\nfix: {fix_hint}")
        if body_read_error is not None:
            raise error from body_read_error
        raise error

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
