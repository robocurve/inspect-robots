# 0082: Retry server-side `status: "failed"` payloads on the responses wire

Closes #448. Branch: `fix/responses-failed-retry`.

## Problem

`ResponsesClient.complete()` (`plugins/inspect-robots-agent/src/inspect_robots_agent/_responses.py:75-115`)
retries transport errors, HTTP 429 and HTTP 5xx up to `max_retries` with
exponential backoff, and fails fast on any other 4xx. A 200 response is handed to
`_parse_response()`, which raises `RuntimeError("LLM response failed — <message>")`
the moment `payload["status"] == "failed"`. Plan 0022 chose that deliberately:
"failed status fails fast, no retry (one request on the wire, mirroring the 4xx
path): the deliberate, message-bearing terminal state of a request the server
accepted, unlike a 5xx where the server never answered."

OpenRouter's Responses endpoint breaks the premise. When the upstream provider
drops the generation stream, OpenRouter answers HTTP 200 with

```json
{"status": "failed",
 "error": {"code": "server_error", "message": "stream closed with reason: error"},
 "error_type": "server",
 "output": [{"type": "reasoning", "status": "completed", "...": "..."}]}
```

That is a 5xx carried inside a 200. On the chat wire the same hiccup arrives as
an HTTP 5xx and is retried; on the responses wire it ends the trial on attempt 0
(observed on a live rig, 12th call of a DeepSeek run, 2026-09-14). The
`reasoning token limit reached` case that motivated 0022 is deterministic for
the same input and must keep failing fast.

## Design

One change in `ResponsesClient.complete()`, in the `status_code == 200` branch:

```python
if response.status_code == 200:
    payload = response.json()
    failure = _transient_failure(payload)
    if failure is None:
        message, output = _parse_response(payload)
        ...cache population unchanged...
        return message
    last_error = failure
else:
    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
    if response.status_code not in (429,) and response.status_code < 500:
        raise RuntimeError(f"LLM request rejected — {last_error}")
# existing: sleep(backoff * 2**attempt) if attempts remain
```

`_transient_failure(payload: dict[str, Any]) -> str | None` (module-private,
next to `_parse_response`, with a docstring stating this contract) classifies
a 200 payload. It returns a non-None string exactly when the payload is
`status == "failed"` **and** the error is server-side:

- `error.code` in `{"server_error", "rate_limit_exceeded"}` (the two values of
  the OpenAI Responses `ResponseError.code` enum matching the conditions the
  HTTP path already retries, 5xx and 429), or
- `payload.get("error_type") == "server"` (an OpenRouter extension field
  observed on the wire 2026-09-14, not in OpenRouter's docs; only the literal
  `"server"` matches, anything else falls through to fail-fast).

Field access is defensive, mirroring `_parse_response`: `error =
payload.get("error")`; `code = error.get("code") if isinstance(error, dict)
else None`; same for `message`. `error: null` is plausible on the wire.

Return value for a transient failure: `f"response failed ({code or
'server'}): {message or 'unknown error'}"`, never None even when `message` is
absent (so `{"status": "failed", "error_type": "server"}` with no `error`
object is still retried). `None` means, and only means, "not a retryable
failure; let `_parse_response` handle it". The caller sets `last_error =
failure`, so the exhausted-retries text reads
`LLM request failed after 3 attempts — response failed (server_error): stream
closed with reason: error` and names the code that triggered the retries.

Only `_transient_failure` reads those fields. `_parse_response` keeps its
fail-fast raise for every other failed payload (the plan 0022 invariant is
retained: one request on the wire, cache never populated, `error.message` in
the text) and is otherwise untouched.

Invariants restated at the point of modification:

- A retried failed payload never populates `_raw_items_by_call_id`; the cache
  write stays inside the `failure is None` branch.
- Wire capture records every attempt exactly as today: `self._capture.record`
  already ran before the status check, so the failed payload is captured with
  `status=200` and the full body on every attempt.
- Retries exhausted raise the existing
  `RuntimeError(f"LLM request failed after {n} attempts — {last_error}")`, so
  the final text contains both "failed after N attempts" and the original
  "LLM response failed — stream closed ..." message.
- No change to `ChatClient`, `_llm.py`, policy wiring, or config surface. The
  retry count and backoff remain the constructor defaults (3, 1.0 s); exposing
  them is #441.

Bump `plugins/inspect-robots-agent/pyproject.toml` version 0.26.0 to 0.26.1 so
the fix publishes with the next core release, then run `uv lock` and commit
`uv.lock` with it: the lockfile pins the workspace member's version and CI's
`uv sync --locked` rejects drift (the 0.25.0 to 0.26.0 bump, commit 80b1915,
touched `uv.lock` for the same reason).

## Tests (`plugins/inspect-robots-agent/tests/test_responses.py`)

Existing tests are not modified. In particular
`test_failed_status_raises_once_and_does_not_populate_cache` (error has no
`code`, no `error_type`) must keep passing unchanged: it is the fail-fast case.

New:

- `test_server_error_failed_payload_retries_then_succeeds` (behaviour: content
  and cache): handler returns the OpenRouter payload above, with
  `_call("failed_call")` in `output`, on calls 1 and 2, and
  `_response(_message("ok"))` (no function_call of its own) on call 3;
  `backoff_s=0.0`. Assert content `"ok"`, `len(requests) == 3`, and
  `"failed_call" not in client._raw_items_by_call_id` right after that
  `complete()` (tests already reach into private state, see
  `test_close_closes_underlying_http_client`).
- `test_failed_payload_retry_variants` (the classification matrix,
  parametrised over the error shape; the `server_error` arm intentionally
  overlaps the test above): `{"error": {"code": "rate_limit_exceeded"}}`,
  `{"error_type": "server"}` with no `error` object at all, and
  `{"error": {"code": "server_error", "message": "boom"}}` each retry (handler
  fails once, then succeeds; two requests); `{"error": {"code":
  "invalid_prompt", "message": "bad"}}`, `{"error": {"message": "reasoning
  token limit reached"}}` and `{"error": None}` each fail fast with exactly one
  request and `pytest.raises(RuntimeError, match="LLM response failed")`.
- `test_server_error_failed_payload_exhausts_retries`: always-failed handler,
  `max_retries=2`, `backoff_s=0.0`;
  `pytest.raises(RuntimeError, match=r"failed after 2 attempts.*stream closed")`;
  two requests.
- `test_capture_records_each_failed_attempt` (extends the capture invariant):
  with `WireCapture`, two failed attempts then success produce three rows, all
  with `call == 0`, `attempt == 0, 1, 2` in order, the first two with
  `status == 200` and `response == payload`.

Test code is annotated per house style even though the plugin CI job
type-checks only `src/` (ci.yml lines 357-360), not `tests/`.

Gates (plugin job in `ci.yml`): `ruff check plugins/inspect-robots-agent`,
`ruff format --check plugins/inspect-robots-agent`, `mypy --config-file
plugins/inspect-robots-agent/pyproject.toml
plugins/inspect-robots-agent/src/inspect_robots_agent`, `pytest
plugins/inspect-robots-agent/tests -q`. Plus `uv sync --locked` must succeed
after the lockfile update.

## Docs

`plugins/inspect-robots-agent/README.md`, a short paragraph directly after the
`-P wire=` table (the table ends around line 90): one sentence that on the
`responses`
wire a `status: "failed"` body whose error is server-side (`server_error`,
`rate_limit_exceeded`, or OpenRouter's `error_type: "server"`) is retried like
an HTTP 5xx, while other failed bodies (for example a reasoning token limit)
still fail fast with the message. No em dash in the sentence, including when
quoting the error text outside a code span (CLAUDE.md writing style).

Plan 0022, inside the "Decision: failed status fails fast" paragraph, append
"(narrowed by plan 0082: server-side failed bodies retry like a 5xx)". No
other change to 0022.
