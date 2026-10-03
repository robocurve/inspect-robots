"""The shared chat wire labels its guided errors for the calling command."""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from inspect_robots._chatwire import (
    HttpPost,
    _ChatHTTPError,
    _ChatTransportError,
    _urllib_post,
    chat_completion,
)
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


_Call = tuple[str, dict[str, str], bytes]

_OPENAI_MAX_TOKENS_400 = json.dumps(
    {
        "error": {
            "message": (
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            ),
            "type": "invalid_request_error",
            "param": "max_tokens",
            "code": "unsupported_parameter",
        }
    }
).encode("utf-8")

_REPLY_200 = b'{"choices":[{"message":{"content":"hello"}}]}'


def _scripted_post(responses: list[tuple[int, bytes]]) -> tuple[HttpPost, list[_Call]]:
    """A recording transport that serves ``responses`` in order and raises past the script."""
    calls: list[_Call] = []

    def post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        calls.append((url, headers, body_bytes))
        if len(calls) > len(responses):
            raise AssertionError(f"post called {len(calls)} times, scripted {len(responses)}")
        return responses[len(calls) - 1]

    return post, calls


def test_retries_with_max_completion_tokens_when_a_400_names_it() -> None:
    post, calls = _scripted_post([(400, _OPENAI_MAX_TOKENS_400), (200, _REPLY_200)])
    messages = [{"role": "user", "content": "hi"}]

    assert chat_completion("https://x.test/v1", "k", "m", messages, http_post=post) == "hello"

    assert len(calls) == 2
    first, second = calls
    assert first[0] == second[0]
    assert first[1] == second[1]
    first_body = json.loads(first[2])
    second_body = json.loads(second[2])
    assert first_body["max_tokens"] == 8192
    assert second_body["max_completion_tokens"] == 8192
    assert "max_tokens" not in second_body
    assert second_body["model"] == first_body["model"] == "m"
    assert second_body["messages"] == first_body["messages"] == messages


def test_retry_failure_surfaces_the_retry_response() -> None:
    retry_400 = b'{"error": {"message": "max_completion_tokens still rejected"}}'
    post, calls = _scripted_post([(400, _OPENAI_MAX_TOKENS_400), (400, retry_400)])

    with pytest.raises(ConfigError) as excinfo:
        chat_completion("https://x.test/v1", "k", "m", [], http_post=post)

    assert len(calls) == 2
    message = str(excinfo.value)
    assert message.startswith("summary request failed with HTTP 400")
    assert "still rejected" in message
    assert "Unsupported parameter" not in message


def test_a_400_without_the_marker_raises_after_a_single_post() -> None:
    post, calls = _scripted_post([(400, b'{"error": {"message": "bad request"}}')])

    with pytest.raises(ConfigError, match=r"summary request failed with HTTP 400"):
        chat_completion("https://x.test/v1", "k", "m", [], http_post=post)

    assert len(calls) == 1


def test_a_non_400_with_the_marker_raises_after_a_single_post() -> None:
    post, calls = _scripted_post([(500, b"try max_completion_tokens later")])

    with pytest.raises(ConfigError, match=r"summary request failed with HTTP 500"):
        chat_completion("https://x.test/v1", "k", "m", [], http_post=post)

    assert len(calls) == 1


def test_a_2xx_with_the_marker_does_not_retry() -> None:
    reply = b'{"choices":[{"message":{"content":"use max_completion_tokens next time"}}]}'
    post, calls = _scripted_post([(200, reply)])

    result = chat_completion("https://x.test/v1", "k", "m", [], http_post=post)

    assert result == "use max_completion_tokens next time"
    assert len(calls) == 1


def test_the_marker_check_reads_the_full_body() -> None:
    long_400 = b'{"error": "' + b"x" * 600 + b' use max_completion_tokens"}'
    post, calls = _scripted_post([(400, long_400), (200, _REPLY_200)])

    assert chat_completion("https://x.test/v1", "k", "m", [], http_post=post) == "hello"

    assert len(calls) == 2


def test_effort_is_sent_as_reasoning_effort() -> None:
    post, calls = _scripted_post([(200, _REPLY_200)])

    chat_completion("https://x.test/v1", "k", "m", [], effort="high", http_post=post)

    assert json.loads(calls[0][2])["reasoning_effort"] == "high"


def test_the_default_call_sends_no_reasoning_effort() -> None:
    post, calls = _scripted_post([(200, _REPLY_200)])

    chat_completion("https://x.test/v1", "k", "m", [], http_post=post)

    assert "reasoning_effort" not in json.loads(calls[0][2])


@pytest.mark.parametrize("absent", [None, ""])
def test_none_and_empty_effort_omit_the_key(absent: str | None) -> None:
    post, calls = _scripted_post([(200, _REPLY_200)])

    chat_completion("https://x.test/v1", "k", "m", [], effort=absent, http_post=post)

    assert "reasoning_effort" not in json.loads(calls[0][2])


def test_the_retry_carries_effort_on_both_sends() -> None:
    post, calls = _scripted_post([(400, _OPENAI_MAX_TOKENS_400), (200, _REPLY_200)])

    chat_completion("https://x.test/v1", "k", "m", [], effort="high", http_post=post)

    assert len(calls) == 2
    assert json.loads(calls[0][2])["reasoning_effort"] == "high"
    assert json.loads(calls[1][2])["reasoning_effort"] == "high"


@pytest.mark.parametrize("falsy_or_numeric", [0, 0.5])
def test_non_string_effort_is_serialized_verbatim(falsy_or_numeric: float) -> None:
    post, calls = _scripted_post([(200, _REPLY_200)])

    chat_completion("https://x.test/v1", "k", "m", [], effort=falsy_or_numeric, http_post=post)

    assert json.loads(calls[0][2])["reasoning_effort"] == falsy_or_numeric


class _BodyRaises:
    """A urlopen response whose body read fails mid-stream."""

    status = 200

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __enter__(self) -> _BodyRaises:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("timed out"), http.client.RemoteDisconnected("closed")],
)
def test_urllib_post_wraps_read_timeouts_and_dropped_connections(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """Failures that are not URLError are still transport errors, not raw exceptions."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        if isinstance(exc, TimeoutError):
            return _BodyRaises(exc)
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(_ChatTransportError, match=r"^chat request failed: "):
        _urllib_post("https://x.test", {}, b"{}")


def test_unreadable_http_error_body_keeps_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx whose body cannot be read is still a rejected request with its status."""

    class _Broken(io.BytesIO):
        def read(self, *args: object) -> bytes:
            raise http.client.IncompleteRead(b"")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.HTTPError(request.full_url, 404, "missing", Message(), _Broken())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _urllib_post("https://x.test", {}, b"{}") == (404, b"")
    with pytest.raises(_ChatHTTPError, match=r"HTTP 404: \(empty response body\)") as info:
        chat_completion("https://x.test/v1", "k", "m", [])
    assert info.value.status == 404


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("api.example/v1", "the scheme must be http or https"),
        ("htps://api.example/v1", "the scheme must be http or https"),
        ("ftp://x.test/v1", "the scheme must be http or https"),
        ("https:///v1", "no host given"),
        ("http://x .test/v1", "whitespace or control characters"),
        ("http://x.test/v 1", "whitespace or control characters"),
        ("http://x.test:abc/v1", "Port could not be cast"),
        ("http://[::1/v1", "Invalid IPv6 URL"),
        ("http://user:pass@x.test/v1", "credentials in the URL are not supported"),
        ("http://x.test:0/v1", "port 0 is not valid"),
    ],
)
def test_malformed_url_is_a_plain_config_error_not_a_transport_error(
    url: str, problem: str
) -> None:
    """A URL that can never work is configuration: it must not look like an outage."""
    with pytest.raises(ConfigError, match="invalid URL") as info:
        _urllib_post(url, {}, b"{}")
    assert type(info.value) is ConfigError
    assert problem in str(info.value)


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("http://127.0.0.1:1/v1/chat/completions", "sk-ok\n"),
        ("http://a..b/v1/chat/completions", "sk-ok"),
    ],
)
def test_requests_that_can_never_be_sent_are_plain_config_errors(url: str, key: str) -> None:
    """A newline in the key or an unencodable host fails before connecting, as config."""
    with pytest.raises(ConfigError, match=r"invalid request") as info:
        _urllib_post(url, {"Authorization": f"Bearer {key}"}, b"{}")
    assert type(info.value) is ConfigError


@pytest.mark.parametrize(
    "url",
    ["HTTPS://api.example.com/v1", "http://[::1]:8080/v1?api-version=1", "https://bücher.de/v1"],
)
def test_valid_urls_pass_the_check(url: str) -> None:
    from inspect_robots._chatwire import _check_url

    _check_url(url)
