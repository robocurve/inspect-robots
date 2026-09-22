import io
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any, ClassVar

import pytest

from inspect_robots_jev import _decisions
from inspect_robots_jev._decisions import ChoiceAnswer, DecisionsClient, DecisionsError

OK: dict[str, Any] = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "move": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.7, "b": 0.3},
            "confidence": 0.4,
        }
    },
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


class FakePost:
    def __init__(self, responses: list[tuple[int, dict[str, str], Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((url, headers, json.loads(body)))
        status, hdrs, payload = self.responses.pop(0)
        return status, hdrs, json.dumps(payload).encode()


def test_openrouter_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = FakePost([(200, {}, OK)])
    client = DecisionsClient(http_post=post)
    ans = client.choose(state={"x": 1}, instructions="pick", criteria={"a": "A", "b": "B"})
    url, headers, body = post.calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer k"
    assert body == {
        "model": "typesafe/jev-1.13",
        "state": {"x": 1},
        "questions": {
            "move": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}
        },
    }
    assert ans == ChoiceAnswer(
        choice="a",
        probabilities={"a": 0.7, "b": 0.3},
        confidence=0.4,
        model="typesafe/jev-1.13-20260917",
        usage={"input_tokens": 10, "output_tokens": 2},
        raw=OK,
    )


def test_typesafe_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    post = FakePost([(200, {}, OK)])
    client = DecisionsClient(api="typesafe", http_post=post)
    client.choose(state="s", instructions="i", criteria={"a": "A"})
    url, _headers, body = post.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert body["model"] == "jev-latest"
    assert client.api == "typesafe"


def test_unknown_api() -> None:
    with pytest.raises(ValueError, match="api"):
        DecisionsClient(api="nope", api_key="k")


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(DecisionsError, match="OPENROUTER_API_KEY"):
        DecisionsClient(http_post=FakePost([]))


def test_retry_then_success() -> None:
    slept: list[float] = []
    post = FakePost(
        [
            (429, {"retry-after": "2"}, {"error": "slow"}),
            (503, {}, {"error": "down"}),
            (200, {}, OK),
        ]
    )
    client = DecisionsClient(api_key="k", http_post=post, sleep=slept.append)
    assert client.choose(state="s", instructions="i", criteria={"a": "A", "b": "B"}).choice == "a"
    assert slept == [2.0, 3.0]


def test_gives_up_after_max_attempts() -> None:
    post = FakePost([(500, {}, {"error": "x"})] * 2)
    client = DecisionsClient(api_key="k", http_post=post, sleep=lambda s: None, max_attempts=2)
    with pytest.raises(DecisionsError, match="500"):
        client.choose(state="s", instructions="i", criteria={"a": "A"})
    assert len(post.calls) == 2


def test_client_error_no_retry() -> None:
    post = FakePost([(400, {}, {"error": {"message": "bad"}})])
    with pytest.raises(DecisionsError, match="400"):
        DecisionsClient(api_key="k", http_post=post).choose(
            state="s", instructions="i", criteria={"a": "A"}
        )
    assert len(post.calls) == 1


def test_choice_not_in_criteria() -> None:
    bad = {**OK, "answers": {"move": {**OK["answers"]["move"], "choice": "zzz"}}}
    with pytest.raises(DecisionsError, match="zzz"):
        DecisionsClient(api_key="k", http_post=FakePost([(200, {}, bad)])).choose(
            state="s", instructions="i", criteria={"a": "A"}
        )


def test_malformed_body() -> None:
    post = FakePost([(200, {}, {"answers": {}})])
    with pytest.raises(DecisionsError, match="move"):
        DecisionsClient(api_key="k", http_post=post).choose(
            state="s", instructions="i", criteria={"a": "A"}
        )


def test_missing_optional_fields_default() -> None:
    sparse = {"answers": {"move": {"type": "choice", "choice": "a"}}}
    ans = DecisionsClient(api_key="k", http_post=FakePost([(200, {}, sparse)])).choose(
        state="s", instructions="i", criteria={"a": "A"}
    )
    assert (ans.probabilities, ans.confidence, ans.model, ans.usage) == (
        {},
        0.0,
        "typesafe/jev-1.13",
        {},
    )


def test_default_http_post_is_urllib(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status = 200
        headers: ClassVar[dict[str, str]] = {"Content-Type": "application/json"}

        def read(self) -> bytes:
            return json.dumps(OK).encode()

        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

    seen: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> Resp:
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, hdrs, body = _decisions.urllib_post("https://x.test/", {"A": "b"}, b"{}", 1.0)
    assert status == 200 and json.loads(body) == OK and hdrs["content-type"] == "application/json"
    assert seen == {"url": "https://x.test/", "method": "POST"}


def test_default_http_post_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req: Any, timeout: float) -> Any:
        hdrs = Message()
        hdrs["Retry-After"] = "1"
        raise urllib.error.HTTPError("https://x.test/", 429, "slow", hdrs, io.BytesIO(b'{"e":1}'))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, hdrs, body = _decisions.urllib_post("https://x.test/", {}, b"{}", 1.0)
    assert status == 429 and hdrs["retry-after"] == "1" and body == b'{"e":1}'


def test_default_post_used_when_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, dict[str, str], bytes]:
        called.append(url)
        return 200, {}, json.dumps(OK).encode()

    monkeypatch.setattr(_decisions, "urllib_post", fake_post)
    DecisionsClient(api_key="k").choose(state="s", instructions="i", criteria={"a": "A"})
    assert called == ["https://openrouter.ai/api/alpha/decisions"]


def test_retry_after_http_date_and_cap() -> None:
    from inspect_robots_jev._decisions import _retry_delay

    assert _retry_delay("Wed, 21 Oct 2026 07:28:00 GMT", 2) == 3.0
    assert _retry_delay("3600", 1) == 30.0
    assert _retry_delay("-5", 1) == 0.0
    assert _retry_delay(None, 30) == 30.0


def test_connection_errors_become_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req: Any, timeout: float) -> Any:
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, hdrs, body = _decisions.urllib_post("https://x.test/", {}, b"{}", 1.0)
    assert status == 599 and hdrs == {} and b"connection reset" in body


def test_non_json_and_non_object_bodies_raise_decisions_error() -> None:
    class RawPost:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __call__(
            self, url: str, headers: dict[str, str], body: bytes, timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            return 200, {}, self.payload

    with pytest.raises(DecisionsError, match="non-JSON"):
        DecisionsClient(api_key="k", http_post=RawPost(b"<html>")).choose(
            state="s", instructions="i", criteria={"a": "A"}
        )
    with pytest.raises(DecisionsError, match="non-object"):
        DecisionsClient(api_key="k", http_post=RawPost(b"[1, 2]")).choose(
            state="s", instructions="i", criteria={"a": "A"}
        )


def test_synthetic_599_is_retried_through_choose() -> None:
    post = FakePost([(599, {}, {"error": "connection error"}), (200, {}, OK)])
    client = DecisionsClient(api_key="k", http_post=post, sleep=lambda s: None)
    assert client.choose(state="s", instructions="i", criteria={"a": "A", "b": "B"}).choice == "a"
    assert len(post.calls) == 2
