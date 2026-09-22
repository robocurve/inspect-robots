"""Minimal client for Jev's Decisions API (OpenRouter alpha or TypeSafe direct).

One request is one ``state`` plus one Choice question. The HTTP edge is an
injectable ``http_post`` so tests never touch the network. Retries follow the
providers' guidance: 429 and 5xx retry, honouring ``retry-after`` when given.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

HttpPost = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]

_BACKENDS: dict[str, tuple[str, str, str]] = {
    "openrouter": (
        "https://openrouter.ai/api/alpha/decisions",
        "OPENROUTER_API_KEY",
        "typesafe/jev-1.13",
    ),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-latest"),
}


#: Never sleep longer than this between attempts, whatever ``retry-after`` says.
_MAX_RETRY_DELAY_S = 30.0


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait: a numeric ``retry-after`` (capped), else a linear backoff."""
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), _MAX_RETRY_DELAY_S)
        except ValueError:  # HTTP-date form: fall back to backoff
            pass
    return min(1.5 * attempt, _MAX_RETRY_DELAY_S)


class DecisionsError(RuntimeError):
    """The Decisions API refused, failed, or returned an unusable answer."""


@dataclass(frozen=True)
class ChoiceAnswer:
    """One Choice answer: the pick, every option's probability, and confidence."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    model: str
    usage: Mapping[str, Any]
    raw: Mapping[str, Any]


def _lower(headers: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def urllib_post(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, dict[str, str], bytes]:
    """Default ``HttpPost`` over stdlib urllib; HTTP errors become status tuples."""
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), _lower(resp.headers), bytes(resp.read())
    except urllib.error.HTTPError as err:
        return int(err.code), _lower(err.headers), bytes(err.read())
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        # Connection reset, DNS failure, socket timeout: report as a synthetic
        # 5xx so the client's retry loop handles it like a provider outage.
        return 599, {}, f"connection error: {err}".encode()


class DecisionsClient:
    """Ask Jev one Choice question about one state."""

    def __init__(
        self,
        *,
        api: str = "openrouter",
        model: str | None = None,
        api_key: str | None = None,
        http_post: HttpPost | None = None,
        timeout_s: float = 30.0,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if api not in _BACKENDS:
            raise ValueError(f"api must be one of {sorted(_BACKENDS)}, got {api!r}")
        url, key_env, default_model = _BACKENDS[api]
        key = api_key if api_key is not None else os.environ.get(key_env)
        if not key:
            raise DecisionsError(f"no API key: pass api_key= or set {key_env}")
        self.api = api
        self.model = model or default_model
        self._url = url
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._post = http_post if http_post is not None else urllib_post
        self._timeout = timeout_s
        self._max_attempts = max_attempts
        self._sleep = sleep

    def choose(
        self,
        *,
        state: Any,
        instructions: str,
        criteria: Mapping[str, str],
        question_id: str = "move",
    ) -> ChoiceAnswer:
        """POST one Choice question; return the validated answer."""
        question = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
        body = json.dumps(
            {"model": self.model, "state": state, "questions": {question_id: question}}
        ).encode()
        attempt = 0
        while True:
            attempt += 1
            status, headers, payload = self._post(self._url, self._headers, body, self._timeout)
            if (status == 429 or status >= 500) and attempt < self._max_attempts:
                self._sleep(_retry_delay(headers.get("retry-after"), attempt))
                continue
            break
        if status != 200:
            raise DecisionsError(f"decisions API returned HTTP {status}: {payload[:300]!r}")
        try:
            raw = json.loads(payload)
        except ValueError as err:
            raise DecisionsError(
                f"decisions API returned non-JSON body: {payload[:200]!r}"
            ) from err
        if not isinstance(raw, dict):
            raise DecisionsError(f"decisions API returned a non-object body: {payload[:200]!r}")
        answer = raw.get("answers", {}).get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise DecisionsError(f"no choice answer for question {question_id!r}: {raw!r}"[:500])
        choice = str(answer.get("choice"))
        if choice not in criteria:
            raise DecisionsError(f"choice {choice!r} is not one of the offered options")
        return ChoiceAnswer(
            choice=choice,
            probabilities={str(k): float(v) for k, v in answer.get("probabilities", {}).items()},
            confidence=float(answer.get("confidence", 0.0)),
            model=str(raw.get("model", self.model)),
            usage=dict(raw.get("usage", {})),
            raw=raw,
        )
