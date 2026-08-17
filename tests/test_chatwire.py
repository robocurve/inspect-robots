"""The shared chat wire labels its guided errors for the calling command."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from email.message import Message

import pytest

from inspect_robots._chatwire import HttpPost, _urllib_post, chat_completion
from inspect_robots.errors import ConfigError


def _post(status: int, body: bytes) -> HttpPost:
    def post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        return status, body

    return post


def test_chat_completion_parses_a_reply() -> None:
    reply = b'{"choices":[{"message":{"content":"hello"}}]}'
    post = _post(200, reply)
    assert chat_completion("https://x.test/v1", "k", "m", [], http_post=post) == "hello"


def test_non_2xx_uses_the_callers_labels() -> None:
    with pytest.raises(ConfigError, match=r"grading request failed with HTTP 500.*\nfix: use -G"):
        chat_completion(
            "https://x.test/v1",
            "k",
            "m",
            [],
            what="grading",
            fix_hint="use -G",
            http_post=_post(500, b"boom"),
        )


def test_non_2xx_defaults_keep_summarize_wording() -> None:
    with pytest.raises(
        ConfigError,
        match=r"summary request failed with HTTP 404.*\nfix: check --base-url, --model",
    ):
        chat_completion("https://x.test/v1", "k", "m", [], http_post=_post(404, b"missing"))


def test_malformed_reply_uses_the_what_prefix() -> None:
    with pytest.raises(ConfigError, match=r"grading endpoint returned a malformed reply"):
        chat_completion(
            "https://x.test/v1", "k", "m", [], what="grading", http_post=_post(200, b"not json")
        )


def test_urllib_post_preserves_http_error_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.HTTPError(
            request.full_url, 429, "rate limited", Message(), io.BytesIO(b"slow down")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _urllib_post("https://x.test", {}, b"{}") == (429, b"slow down")


def test_urllib_post_translates_url_errors_to_neutral_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ConfigError, match=r"chat request failed: offline.\nfix: check the base"):
        _urllib_post("https://x.test", {}, b"{}")
