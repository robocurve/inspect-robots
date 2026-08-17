"""Dependency-free OpenAI-compatible chat wire shared by summarize and the VLM grader.

One blocking POST to ``<base_url>/chat/completions`` over stdlib ``urllib``,
with an injectable transport seam (``http_post``) so callers and tests never
touch the network. Errors are raised as guided ``ConfigError``s whose prefix
and fix hint the caller labels for its own command surface.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, cast

from inspect_robots.errors import ConfigError

HttpPost = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]

_RESPONSE_EXCERPT_LIMIT = 500


def _response_excerpt(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return (text or "(empty response body)")[:_RESPONSE_EXCERPT_LIMIT]


def _urllib_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
    """Send one blocking HTTP POST and preserve HTTP error bodies for guided failures."""
    request = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise ConfigError(
            f"chat request failed: {exc.reason}.\n"
            "fix: check the base URL and network connectivity, then retry"
        ) from exc


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    what: str = "summary",
    fix_hint: str = "check --base-url, --model, and the configured API key",
    http_post: HttpPost | None = None,
) -> str:
    """Return one OpenAI-compatible chat completion or raise a guided configuration error.

    ``what`` labels error prefixes for the calling command and ``fix_hint``
    names that command's remedy flags in the non-2xx failure message.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 8192}).encode("utf-8")
    post = _urllib_post if http_post is None else http_post
    status, response_body = post(url, headers, body)
    if not 200 <= status < 300:
        raise ConfigError(
            f"{what} request failed with HTTP {status}: {_response_excerpt(response_body)}\n"
            f"fix: {fix_hint}"
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
