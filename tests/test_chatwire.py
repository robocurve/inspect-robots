"""The shared chat wire labels its guided errors for the calling command."""

from __future__ import annotations

import pytest

from inspect_robots._chatwire import HttpPost, chat_completion
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
