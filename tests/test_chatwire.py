"""The shared chat wire labels its guided errors for the calling command."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from email.message import Message
from http.client import IncompleteRead
from typing import cast

import pytest

from inspect_robots._chatwire import (
    HttpPost,
    _HTTPErrorBodyReadError,
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
    body = io.BytesIO(b"slow down")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.HTTPError(request.full_url, 429, "rate limited", Message(), body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _urllib_post("https://x.test", {}, b"{}") == (429, b"slow down")
    assert body.closed


def test_urllib_post_reads_and_closes_a_success_response(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_body = b"complete response"

    class Response:
        status = 200

        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def read(self) -> bytes:
            return expected_body

    response = Response()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    assert _urllib_post("https://x.test", {}, b"{}") == (200, expected_body)
    assert response.closed


def test_urllib_post_translates_url_errors_to_neutral_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ConfigError, match=r"chat request failed: offline.\nfix: check the base"):
        _urllib_post("https://x.test", {}, b"{}")


@pytest.mark.parametrize(
    ("error", "description"),
    [
        pytest.param(TimeoutError("read timed out"), "timed out", id="timeout"),
        pytest.param(IncompleteRead(b"partial", 4), "incomplete transfer", id="incomplete-read"),
        pytest.param(
            ConnectionResetError("peer reset connection"), "connection was interrupted", id="reset"
        ),
    ],
)
def test_urllib_post_normalizes_response_body_read_failures(
    monkeypatch: pytest.MonkeyPatch, error: Exception, description: str
) -> None:
    class FailingResponse:
        status = 200

        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> FailingResponse:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def read(self) -> bytes:
            raise error

    response = FailingResponse()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(
        ConfigError,
        match=rf"chat request failed:.*{description}.*\nfix:",
    ) as exc_info:
        _urllib_post("https://x.test", {}, b"{}")

    assert exc_info.value.__cause__ is error
    assert response.closed


def test_urllib_post_normalizes_http_error_body_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = TimeoutError("error body read timed out")

    class FailingBody:
        closed = False

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            raise error

        def close(self) -> None:
            self.closed = True

    body = FailingBody()

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "server error",
            Message(),
            body,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(_HTTPErrorBodyReadError) as exc_info:
        _urllib_post("https://x.test", {}, b"{}")

    assert exc_info.value.status_code == 500
    assert "body read timed out" in exc_info.value.detail
    assert exc_info.value.__cause__ is error
    assert body.closed


def test_chat_completion_preserves_http_400_and_skips_retry_when_body_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = TimeoutError("error body read timed out")

    class FailingBody:
        closed = False

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            raise error

        def close(self) -> None:
            self.closed = True

    body = FailingBody()

    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        requests.append(request)
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            Message(),
            body,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(
        ConfigError,
        match=r"grading request failed with HTTP 400:.*body unavailable",
    ) as exc_info:
        chat_completion("https://x.test/v1", "k", "m", [], what="grading", effort="none")

    assert "grading request failed with HTTP 400:" in str(exc_info.value)
    assert "error response body unavailable" in str(exc_info.value)
    assert "body read timed out" in str(exc_info.value)
    assert "\nfix:" in str(exc_info.value)
    read_error = exc_info.value.__cause__
    assert isinstance(read_error, _HTTPErrorBodyReadError)
    assert read_error.status_code == 400
    assert read_error.__cause__ is error
    assert body.closed
    assert len(requests) == 1


def test_chat_completion_retains_read_failure_from_max_completion_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = TimeoutError("retry error body read timed out")

    class FailingBody:
        closed = False

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            raise error

        def close(self) -> None:
            self.closed = True

    retry_body = FailingBody()
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "bad request",
                Message(),
                io.BytesIO(_OPENAI_MAX_TOKENS_400),
            )
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "service unavailable",
            Message(),
            retry_body,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(
        ConfigError,
        match=r"grading request failed with HTTP 503:.*body unavailable.*\nfix:",
    ) as exc_info:
        chat_completion("https://x.test/v1", "k", "m", [], what="grading", effort="none")

    assert len(requests) == 2
    first_payload = json.loads(cast(bytes, requests[0].data))
    retry_payload = json.loads(cast(bytes, requests[1].data))
    assert first_payload["max_tokens"] == 8192
    assert retry_payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in retry_payload
    assert first_payload["reasoning_effort"] == "none"
    assert retry_payload["reasoning_effort"] == "none"
    read_error = exc_info.value.__cause__
    assert isinstance(read_error, _HTTPErrorBodyReadError)
    assert read_error.status_code == 503
    assert read_error.__cause__ is error
    assert retry_body.closed


def test_urllib_post_does_not_convert_keyboard_interrupt_during_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedResponse:
        status = 200

        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> InterruptedResponse:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def read(self) -> bytes:
            raise KeyboardInterrupt

    response = InterruptedResponse()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(KeyboardInterrupt):
        _urllib_post("https://x.test", {}, b"{}")

    assert response.closed


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
