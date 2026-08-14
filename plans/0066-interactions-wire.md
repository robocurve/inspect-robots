# 0066 — wire=interactions: Gemini Interactions API as a stateful HTTP wire

- **Status:** draft (R0, not yet critiqued)
- **Issue:** #378

## Problem

The agent plugin speaks four wires: `chat`, `responses`, `messages`, and
`gemini-live`. Google's Interactions API is now GA and is the recommended
surface for the latest Gemini models; notably `gemini-3.7-flash` does **not**
support the Live API, so the only way to drive it today is the stateless
OpenAI-compat `chat` wire, which re-uploads the (image-bearing) request view
on every step and leans on `image_horizon` eviction to stay affordable.

The Interactions API offers server-side conversation state: each call can
chain to the previous one with `previous_interaction_id`, so the request
carries only the new observation and function results. Google's docs state
stateful chaining "allows the system to more easily utilize implicit caching
for the conversation history, which improves performance and reduces costs."
That is the same property the `gemini-live` wire buys via a websocket, but
for GA HTTP models, with per-request retry semantics instead of a fragile
long-lived socket.

## Contract (from Google's API reference, ai.google.dev/api/interactions-api)

- `POST https://generativelanguage.googleapis.com/v1beta/interactions`,
  API-key auth via the `x-goog-api-key` header (the Gemini native-REST
  convention; the Live wire's `?key=` query form is websocket-specific).
- Request: `model` (bare id, e.g. `gemini-3.7-flash`), `input` (string, or
  array of content blocks / steps), `previous_interaction_id` (optional),
  `tools` (function tools: `{"type": "function", "name", "description",
  "parameters"}` — flat, *not* nested under `"function"` like Chat
  Completions), `system_instruction` (string), `generation_config`
  (`thinking_level` ∈ minimal/low/medium/high, `temperature`,
  `max_output_tokens`, `tool_choice`), `store` (bool, default true),
  `stream` (bool; we do not stream).
- Input content blocks: `{"type": "text", "text": ...}` and
  `{"type": "image", "mime_type": "image/png", "data": "<base64>"}`.
  Function results are steps: `{"type": "function_result", "call_id": ...,
  "result": ...}`.
- Response: `id`, `status` (`completed`, `requires_action`, `failed`,
  `incomplete`, `budget_exceeded`, …), `steps` array containing
  `{"type": "function_call", "id", "name", "arguments": {…}}` steps and
  `{"type": "model_output", "content": [{"type": "text", "text": …}, …]}`
  steps, and `usage` (`total_input_tokens`, `total_output_tokens`,
  `total_cached_tokens`, `total_thought_tokens`, `total_tokens`).
- Statefulness: `tools`, `system_instruction`, and `generation_config` are
  interaction-scoped (never inherited across the chain), so every request
  re-sends them; only the conversation history lives server-side.

### Contract risks (unverified against the live endpoint)

No `GEMINI_API_KEY` was available on the authoring machine, so the shapes
above come from the API reference, not a probe. The implementation therefore:

1. parses defensively — a response missing `steps`, or a step of an unknown
   type, raises `RuntimeError` naming the field and quoting the first 500
   characters of the body, never a `KeyError` traceback;
2. accepts `arguments` as either a JSON object (documented) or a string
   (Chat-Completions habit), normalizing to the JSON-text `ToolCall.arguments`
   contract either way;
3. ships with `tests/manual_interactions_smoke.py` (excluded from pytest
   collection by its non-`test_` name), a ~40-line script that exercises one
   basic call, one chained call, and one function round-trip against the real
   endpoint when `GEMINI_API_KEY` is set. First run on a keyed machine is a
   release gate for announcing the wire.

## Design

### New module: `plugins/inspect-robots-agent/src/inspect_robots_agent/_interactions.py`

`InteractionsClient`, same `complete(messages, tools, temperature,
reasoning_effort) -> AssistantMessage` contract as the other clients, httpx
only. Constructor `(provider, *, timeout_s=120.0, max_retries=3,
backoff_s=1.0, transport=None, capture=None)` — `transport` for
`httpx.MockTransport` tests, mirroring `ChatClient`.

State, mirroring `GeminiLiveClient`'s identity-prefix discipline:

- `_streamed: list[dict]` — references (not copies) of the chat-format
  messages already absorbed by the server chain. A changed prefix means a new
  trial: reset `_streamed`, `_last_interaction_id`, and `_call_ids`. The
  rewritten-view guard is identical to the Live wire's: same first message
  object but non-prefix history raises "rewritten conversation view without a
  trial reset". This is also why `image_horizon` is rejected on this wire
  (below): `_evicted_view` rewrites history and would trip this guard by
  design.
- `_last_interaction_id: str | None` — the chain head.
- Bare model: `provider.model.removeprefix("google/")`, as in the Live wire.

Request construction per `complete()` call:

- Suffix = `messages[len(_streamed):]`. Translate, in order:
  - `system` (only ever message 0, only on an unchained request) → the
    `system_instruction` field, not an input block;
  - `user` → content blocks (string content → one text block; list content →
    text parts verbatim, `image_url` data-URL parts decoded into
    `{"type": "image", "mime_type": <from the data URL>, "data": <base64>}`;
    any other part type raises, as in the Live translator);
  - `tool` → `{"type": "function_result", "call_id": <tool_call_id>,
    "result": <content>}` (content must be `str`, else raise);
  - `assistant` → skipped (the server already has it when chained; on an
    unchained fold it arrives inside the sanitized prologue).
- Body: `model`, `input` (the translated block/step array), `store: true`,
  `tools` (translated from Chat-Completions shape to the flat Interactions
  shape, every request), `system_instruction` (every request, from
  `messages[0]` when it is a system message), `generation_config` with
  `temperature` and/or `thinking_level` when set, and
  `previous_interaction_id` when `_last_interaction_id` is set.
- `reasoning_effort` arrives already validated by the policy (below) as one
  of minimal/low/medium/high and goes out as
  `generation_config.thinking_level`.

Response handling:

- 200 → parse JSON. `status` of `completed` or `requires_action` → walk
  `steps`: collect every `function_call` step into a `ToolCall` (arguments
  normalized to JSON text), concatenate every `model_output` step's text
  blocks into `content`. Any other terminal `status` raises `RuntimeError`
  naming it. On success: `_last_interaction_id = response["id"]`,
  `_streamed.extend(suffix)` — the cursor advances only on a parsed 200, so
  a failed attempt resends the same delta (one POST is atomic server-side;
  there is no partial-absorb case like the Live socket's).
- 429/5xx/transport error → bounded retry with exponential backoff
  (`backoff_s * 2**attempt`), as in `ChatClient`.
- Chain loss — HTTP 404, or a 400 whose body mentions
  `previous_interaction`: rebuild as an *unchained* fold (below) and retry
  within the same attempt loop.
- Other 4xx → immediate `RuntimeError` with the truncated body (the
  request's fault; retrying cannot help). When the body mentions
  `thinking_level`, append a fix line naming the levels this model accepts
  may differ (e.g. 3.7-flash serves low/medium/high).

Fold recovery (chain loss only), mirroring `_send_recovery`:

- Anchor = last `user` message. Input =
  `[text: _RECOVERY_PROLOGUE, *json-lines of _sanitize(history minus system
  and anchor), text: _RECOVERY_CONTINUATION, *anchor's translated blocks]`,
  no `previous_interaction_id`. Reuse the Live wire's prologue strings
  (import them) so transcripts read identically. `_sanitize` is imported
  deferred inside the function, exactly as `_gemini_live.py` does, to avoid
  the import cycle. On success the cursor covers the full message list.

Usage mapping (per call, summed by the policy):
`total_input_tokens → input_tokens`, `total_output_tokens → output_tokens`,
`total_cached_tokens → cache_read_input_tokens`,
`total_thought_tokens → thought_tokens`, `total_tokens → total_tokens`.
Non-int values are skipped, as in the Live wire's `_add_usage`.

Capture: one `capture.record(...)` per attempt with
`endpoint="/interactions"`, the exact request body, status, and response
text — the same shape `ChatClient` records.

`close()`: closes the httpx client. No per-trial teardown hook is needed —
the identity-prefix check already resets chain state on the next trial's
first call, and unlike the Live socket there is no half-dead transport to
drop, so `on_trial_end` stays untouched.

### `policy.py` changes

- `_WIRE_FORMATS` gains `"interactions"`.
- `_INTERACTIONS_BASE = "https://generativelanguage.googleapis.com/v1beta"`.
- Construction guards (same order block as the existing wire checks):
  - `effort`: on `wire="interactions"`, only minimal/low/medium/high pass;
    `none`, `xhigh`, `max`, and fractions raise `ConfigError` with
    "fix: pass -P effort=minimal|low|medium|high (maps to thinking_level),
    or drop -P effort=". Unset stays unset (provider default).
  - `image_horizon`: explicit value raises, default resolves to `None` —
    same shape as the `gemini-live` guard, with the fix text naming
    server-side history as the reason (frames already absorbed by the chain
    cannot be evicted client-side).
  - `base_url`: when given, must start with `http://` or `https://`
    (a ws:// URL gets "fix: wire='interactions' is HTTP; drop -P base_url=
    or pass the Live wire a websocket endpoint via -P wire=gemini-live").
  - `api_key_env` default when `base_url` is set without one:
    `GEMINI_API_KEY` (same clause as the `gemini-live` branch).
- Resolution: like `gemini-live` — without an explicit `base_url`,
  resolution must land on `_GOOGLE_BASE`, else `ConfigError`
  ("wire='interactions' needs Google's direct provider.\nfix: use
  -P model=google/... and set $GEMINI_API_KEY"); then the provider is
  re-based onto `_INTERACTIONS_BASE`. The `resolve_provider` failure remap
  the Live wire does is extended to cover this wire.
- Client selection: `wire == "interactions"` →
  `InteractionsClient(provider, transport=transport, capture=self._capture)`.
- `speed` / `max_output_tokens` stay messages-only (no scope creep; the
  Interactions `generation_config.max_output_tokens` can come later).
- `AgentPolicyConfig` needs no new fields; `wire` records the choice.

### `_capture.py` change

`_replace_blobs` gains the Interactions image shape: a dict with
`"type": "image"` and a *string* `"data"` value (no `"source"` sub-dict)
gets `data` replaced by the blob sentinel. The existing Anthropic branch
(`"type": "image"` with `source.type == "base64"`) is untouched; the new
branch fires only when `"source"` is absent and `"data"` is a `str`, so the
two cannot collide. Module docstring's supported-shapes sentence gains
"Interactions".

### Docs

- Plugin README: new row in the `-P wire=` table
  (`interactions` | `/interactions` | Google's stateful HTTP API: server-side
  history for GA Gemini models, e.g. `gemini-3.7-flash`); a short section
  after the Gemini Robotics ER 2 one with a full example command; the
  effort section gains the thinking_level mapping sentence; the
  image_horizon table row's "unset on `gemini-live`" note extends to this
  wire.
- `__init__.py` module docstring sentence listing wires gains
  `interactions`.
- Plugin `pyproject.toml`: minor version bump (new feature), per the
  release convention for `plugins/*`.

## Tests (`plugins/inspect-robots-agent/tests/test_interactions.py` + policy tests)

Client tests run against `httpx.MockTransport` handlers that assert on the
request body and script responses:

1. First call: body carries `model` (prefix stripped), `store: true`, flat
   `tools`, `system_instruction` from the system message, translated
   text+image input blocks, no `previous_interaction_id`; `x-goog-api-key`
   header set from the provider key.
2. Chained call: after a 200 with `id: "i1"`, the next `complete()` sends
   `previous_interaction_id: "i1"` and only the delta (the new tool result
   as a `function_result` step with the recorded `call_id`, plus the new
   user observation), while still re-sending `tools` and
   `system_instruction`.
3. Function-call parsing: a `requires_action` response with two
   `function_call` steps yields two `ToolCall`s with JSON-text arguments;
   object and string `arguments` both normalize.
4. Text parsing: `model_output` text blocks concatenate; empty steps yield
   `content=None` and no calls.
5. Usage mapping: the five counters land under the normalized names;
   missing/non-int counters are skipped.
6. Retry: one 500 then a 200 succeeds; three 500s raise after
   `max_retries`; a plain 400 raises immediately with the body excerpt.
7. Chain-loss fold: a 404 on a chained call triggers an unchained retry
   whose input starts with the recovery prologue and ends with the anchor
   user message's blocks; a subsequent call chains to the fold's new id.
8. Rewritten-view guard and fresh-trial reset: same two behaviors the Live
   client pins.
9. Terminal failure statuses (`failed`) raise with the status named.
10. Capture: a recorded row has `endpoint="/interactions"` and the image
    `data` replaced by a `$blob:` sentinel (also covers the `_capture.py`
    branch).
11. Policy construction: effort accept/reject matrix, image_horizon
    rejection, ws base_url rejection, non-Google resolution rejection with
    the guided message, `api_key_env` default with explicit base_url, and
    client-type selection.

Gates: plugin pytest, `ruff check`, `ruff format --check`, strict mypy —
same bars the plugin already passes; no core `src/inspect_robots` changes,
so the 100% core coverage gate is unaffected.

## Out of scope

- Streaming (`stream: true`), `background`, response_format, and agent ids.
- Deleting stored interactions at trial end (Google owns retention; a
  follow-up could DELETE the chain head on `on_trial_end` if retention
  becomes a concern).
- Making `interactions` the default wire for `google/*` (chat remains the
  default; this wire is explicit opt-in, like `gemini-live`).
- `max_output_tokens` support outside `wire=messages`.
