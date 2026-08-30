# 0039 — `-P wire=gemini-live`: Gemini Robotics ER 2 over the Live API

Issue: #252. PR: #227. Revised after critiques R1 (13 findings) and R2
(10 findings); all resolved below.

## Problem

Google's guidance for Gemini Robotics ER 2 is the Live API: "Gemini Robotics
ER 2 integrates into the Gemini Live API, using a bidirectional streaming
endpoint optimized for latency-sensitive tasks" (launch post, confirmed by its
author). The Live-served model is a distinct id,
`gemini-robotics-er-2-streaming-preview`, which supports only
`bidiGenerateContent`: it cannot be reached by any HTTP wire the agent plugin
speaks today. Conversely the chat-wire id (`gemini-robotics-er-2-preview`) is
rejected on the socket, so this is not a transport preference; it is a
different serving surface.

On the chat wire every step re-uploads the whole multi-image transcript. A
real-rig rollout is one motion per LLM call, so per-call overhead dominates
wall-clock rig time. A Live session holds conversation state server-side:
after setup, each step sends only the new observation and tool result.

All protocol facts below were verified against the live socket on 2026-08-01
(spike transcript in issue #252). Where the public docs and the socket
disagreed, the socket won.

## Design

One new wire, `gemini-live`, in `inspect-robots-agent`. Raw
`BidiGenerateContent` JSON over WSS, no google SDK — the same
speak-the-protocol doctrine as the other wires, and the same `websockets>=12`
sync-client dependency and stub-server test pattern the xpolicylab plugin
already established in this repo.

### Verified protocol facts (socket, v1beta, 2026-08-01)

- Endpoint: `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<KEY>`.
- Setup accepts `model` (sent as `models/<bare-id>`), `systemInstruction`,
  `tools`, `generationConfig`. `toolConfig` is **rejected at the protocol
  level** ("Unknown name \"toolConfig\" at 'setup'"): there is no
  config-forced tool choice on this surface. Prompt-level forcing produced
  well-formed motion tool calls (with `note` args) in 4/4 spike sessions; the
  policy's existing no-tool-call nudge/retry path is the backstop.
- Camera frames work as `inlineData` parts inside `clientContent` turns —
  verified for both `image/jpeg` and `image/png` (the policy emits PNG).
- Lockstep sequencing (the eval-semantics requirement that the model acts on
  the *fresh* observation): sending `toolResponse` alone resumes generation
  immediately and the model emits its next action **without** new frames.
  Buffering the new user content first (`turnComplete: false`), then sending
  the `toolResponse`, yields an **empty** resumed turn instead of a
  stale-state action; a final empty `clientContent {"turnComplete": true}`
  then fires generation on the buffered content.
- `functionResponses` entries carrying `id`, `name`, and an object-valued
  `response` are accepted (the spike always sent all three; the client always
  will too).
- `sessionResumptionUpdate` messages arrive unprompted and must be drained;
  `GoAway` carries `timeLeft` before server-side termination.
- Server envelopes the client and stub must both speak, as parsed
  successfully against the real socket by the spike:
  `{"setupComplete": {}}`;
  `{"serverContent": {"modelTurn": {"parts": [{"text": "..."}]}}}` and
  `{"serverContent": {"turnComplete": true}}` (the empty resumed turn is
  exactly this, with no `modelTurn`);
  `{"toolCall": {"functionCalls": [{"name": "move_joints", "args": {...},
  "id": "fc_15275099834869453056"}]}}` — note `args` is a JSON object, not
  a string, hence the re-serialization when building `ToolCall.arguments`.

### `wire` param and per-wire defaults

`_WIRE_FORMATS` grows to `{"chat", "responses", "anthropic", "gemini-live"}`.

`effort` and `image_horizon` currently have non-None constructor defaults
(`"low"`, `2`), so "reject when explicitly set" cannot be expressed with the
`speed`/`max_output_tokens` pattern (those default to `None`). Both switch to
an out-of-band sentinel that no CLI or config value can produce (in-band
sentinels like `"default"`/`0` would legalize inputs that are rejected today
— `tests/test_policy_eviction.py` pins the `image_horizon=0` rejection):

```python
class _Unset:
    """Marker type for constructor defaults resolved per wire."""

_UNSET: Final = _Unset()
```

- `effort: str | None | _Unset = _UNSET` — resolves to `"low"` on
  `chat`/`responses`/`anthropic`, `None` on `gemini-live` (bidi
  `generationConfig` has no effort/thinking field on v1beta). An explicit
  effort level with `wire=gemini-live` is a guided `ConfigError` (fix: drop
  `-P effort=`); explicit `None` ("omit the field") equals this wire's
  resolved value and passes.
- `image_horizon: int | None | _Unset = _UNSET` — resolves to `2` on the
  HTTP wires, `None` on `gemini-live`. An explicit `int >= 1` with
  `wire=gemini-live` is a guided `ConfigError` (we cannot evict what we
  already streamed; the Live API's own context-window compression is the
  equivalent mechanism, and the fix line says so); explicit `None` ("full
  history") passes.

Validation order: the `wire` membership check runs before sentinel resolution
and the per-wire rejections (today the `_EFFORT_LEVELS` check at
policy.py:318 precedes the wire check at :328 — it moves after; its
user-facing message is unchanged since `_UNSET` is not user-expressible).
All existing rejections (`image_horizon=0`, `effort="default"`, booleans)
behave exactly as today — including on `gemini-live`, where the generic
`image_horizon=0` message suggests `-P image_horizon=N` and the wire then
rejects `N >= 1`; the two-bounce error path is accepted (both messages are
guided, and special-casing one invalid input per wire isn't worth it).
`AgentPolicyConfig` keeps its `str | None` / `int | None` field types and
records the *resolved* values. A bare `-P wire=gemini-live` run constructs
successfully (regression test).

Other params: `temperature` passes through as `generationConfig.temperature`.
`speed`/`max_output_tokens` stay anthropic-only (unchanged checks).
`wire_capture`, `images=always|on_demand`, `depth`, `replan_interval`,
`max_llm_calls`, and `transcript_echo` work unchanged — they live in the
policy loop, which this wire does not touch.

### Provider resolution and endpoint derivation

`resolve_provider` is untouched, but `gemini-live` adds a wrong-provider
guard in `__init__` (the analog of the anthropic-wire guard at
policy.py:393-440): without an explicit `base_url`, resolution must land on
the google direct provider — i.e. a `google/*` model with `$GEMINI_API_KEY`
set. Otherwise (OpenRouter fallback, other prefixes) a guided `ConfigError`
(fix: use `-P model=google/...` and set `$GEMINI_API_KEY`), which also
guarantees an OpenRouter secret is never placed in a Google `?key=` query.

- Default endpoint: the wss URL above; the key is appended as `?key=` at
  connect time only, and only when non-empty.
- An explicit `base_url` starting with `ws://` or `wss://` wins (this is how
  tests reach the stub). With a ws(s) `base_url`, `api_key_env` defaults to
  `GEMINI_API_KEY` (not the OpenRouter default `resolve_provider` would
  pick); a missing key is allowed (stubs are keyless). `wire=gemini-live`
  with an `http(s)://` `base_url` is a guided `ConfigError`.
- Setup `model` is always `models/<bare-id>`, where the bare id strips a
  leading `google/` when present — this normalizes both paths (the direct
  provider already strips; the explicit-base_url path of `_llm.py:111-113`
  does not). Asserted in the setup round-trip test.
- `AgentPolicyConfig.base_url` records the derived ws(s) endpoint **without**
  the key. The key must never appear in config, capture rows, transcripts, or
  exception text: connect/send/recv errors are wrapped and re-raised with the
  query string stripped before they can carry the URL upward.

### `GeminiLiveClient` (new module `_gemini_live.py`)

Implements the same duck interface as the other three clients —
`complete(messages, tools, temperature, reasoning_effort) -> AssistantMessage`
and `close()` — so `policy.py` changes are confined to validation, client
selection, and one lifecycle hook (below). The client is stateful where the
others are stateless; the policy's append-only chat-format message list is
the source of truth, and the client keeps **a list of references to the
message dicts it has streamed** (the cursor). Synchronous
`websockets.sync.client`, matching the blocking policy loop.

**Invariant:** on this wire the resolved `image_horizon` is always `None`,
so `complete()` receives `self._messages` itself — never an `_evicted_view`
copy — and the identity cursor is sound. Guard: if the prefix check fails
while `messages[0] is streamed[0]`, that is a view-not-reset bug (a real
`reset()` replaces the system message too) and the client raises
`RuntimeError` instead of silently re-sessioning every call.

Session/cursor rules, in order, on each `complete(messages, ...)`:

1. **Prefix check by identity.** If any streamed reference is not `is`-equal
   to the message at its position (or the list is shorter), this is a new
   trial (`reset()` rebuilt the list — value equality is unsafe: re-running
   the same scene rebuilds a byte-identical prefix): close the socket and
   start fresh. Object identity is O(1) per message and immune to that
   collision.
2. **Open lazily.** If no session is open: setup from the leading `system`
   message (`systemInstruction`) plus `tools` as `functionDeclarations` and
   `generationConfig` (temperature when set); wait for `setupComplete`. On a
   *reconnect* (mid-trial recovery), the send is the recovery send defined
   below, not the normal suffix send.
3. **Send the un-streamed suffix.** Message classes:
   - `assistant` messages generated by the currently open session are
     advanced past without sending — the server already holds them as its
     own turns; re-sending would duplicate them in server context. (The
     suffix after a normal step is `[assistant, tool, user]`: the assistant
     entry is the `raw()` echo of what this session just produced.)
   - `user` messages → `clientContent` turns; plain-string content becomes
     a single text part, text parts pass through, `image_url` data-URI
     parts become `inlineData` (mime from the URI header), and any other
     part type raises `RuntimeError` (only text and `image_url` occur in
     this plugin; silently dropping or forwarding unverified constructs is
     worse than failing loudly).
   - `tool` messages: **all consecutive un-streamed tool messages aggregate
     into one `toolResponse`**, `functionResponses` entries in list order,
     every issued call id answered (the loop appends one tool message per
     call, so `[assistant, tool, tool, user]` is a real normal-step suffix —
     ignored-second-call results and motion+queued-`take_pic` pairs). Each
     entry carries the original `toolCall` `id`, the `name` from the
     client's id→name map (recorded when the `toolCall` message was
     received — chat-format tool messages carry no name), and
     `response={"output": <the string content>}`.

   The generation trigger depends on the suffix class:

   | suffix contains | send order | trigger |
   |---|---|---|
   | user only (first call: `[system, user(goal), user(obs)]`; nudge) | `clientContent` turns | `turnComplete: true` set on the **final content turn** itself |
   | tool + user (normal step; on-demand `take_pic`) | user turns with `turnComplete: false`, then `toolResponse` | trailing **empty** `clientContent {"turnComplete": true}` |
   | tool only (in-step tool-error retry) | `toolResponse` | the `toolResponse` itself — **no** turn-complete (one would fire a second, stray generation and desync the next read) |

   Advance the cursor only after the full send completes.
4. **Read** with a 120 s per-message timeout (the `timeout_s` the HTTP wires
   use) and a drain cap of 64 messages per exchange, draining
   `sessionResumptionUpdate` and empty/interim turns, until either a
   `toolCall` (→ `AssistantMessage` with the calls, `args` re-serialized as
   JSON text; multiple `functionCalls` become multiple `ToolCall`s; ids and
   names recorded in the id→name map) or a completed text turn
   (→ `AssistantMessage(content=text, tool_calls=())`, which lands in the
   policy's existing nudge path). Timeout or cap exhaustion → recovery
   (below). `usageMetadata` is summed across **all** messages in the
   exchange and normalized (`promptTokenCount` → `input_tokens`,
   `candidatesTokenCount` → `output_tokens`, `totalTokenCount` →
   `total_tokens`). This is billing truth: a normal step comprises two
   generations (the empty resumed turn plus the triggered one), so
   `input_tokens` reads higher than single-generation wires; the README
   note says so. Messages received on an attempt that later fails mid-read
   (recovery follows) still count — they were billed. Pinned by a stub test
   with `usageMetadata` on two messages of one exchange.

The suffix shapes in the table are exhaustive for the current policy loop
(`act()` appends `user(observation)` before its first `complete()`; tool
results are appended as they execute; the nudge appends a bare user turn;
on-demand `take_pic` appends `[tool(result), user(images)]`;
`_forced_give_up` constructs its chunk without calling `complete()`).
The stub tests pin each shape.

### Recovery (always a fresh session; text prologue; no handles)

v1 does **not** use `sessionResumption` handles: after a mid-step transport
failure the client cannot know how much of the send the server absorbed, so
resuming server state risks double-delivering an observation or answering an
already-answered call. Handles are drained and discarded. Recovery never
depends on Google keeping state.

Triggers: socket close, connect/send/recv transport error, read timeout,
drain-cap exhaustion. On `GoAway`: finish the current exchange normally (the
server keeps it alive for `timeLeft`), mark the session, and reconnect at
the next `complete()` boundary before sending.

The recovery send, on the fresh session (after setup):

- **Everything except the newest user content is folded into one text
  prologue** — streamed history *and* any un-streamed assistant/tool
  messages alike. Model-role `clientContent` turns and `toolResponse`s for
  call ids the new session never issued are unverified constructs on this
  surface; the prologue uses only verified ones, and it also preserves the
  dead session's assistant turns that rule 3 would otherwise skip.
- Prologue format (concrete): a single `user` turn whose text is
  `"Recovered session. Conversation so far, oldest first, one JSON object
  per line:"` followed by the `_sanitize`-rendered messages (`_sanitize`
  lives in `policy.py`, which imports the client module — use a deferred
  function-level import inside the recovery path to avoid the circular
  top-level import) (the exact
  JSONL rendering `on_trial_end` already writes; image parts become
  `_sanitize`'s literal `"[image omitted: streamed camera frame]"` text
  stubs, with camera names surviving in the adjacent label parts) for every
  message being folded, then `"Continue the task from the latest
  observation below."`.
- The anchor is the **newest `user` message, images or not** (under
  `images=on_demand` observation turns usually carry no frames, and a
  camera-less embodiment never has any): it is excluded from the prologue
  and sent as a normal `clientContent` turn, with its images intact when it
  has them — exactly one copy (in the mid-step case the pending observation
  and the anchor are the same message, so nothing is double-sent). Older
  image-bearing messages (e.g. stale `take_pic` deliveries) always fold
  into the prologue as stubs — stale frames are never re-sent.
- The final content turn carries `turnComplete: true`. A recovery send
  never contains a `toolResponse`.
- Afterwards the cursor marks the entire list streamed, and the id→name map
  is cleared (no live call ids can span sessions).

Bounded by the same retry discipline as `ChatClient` (3 attempts,
exponential backoff); persistent failure raises `RuntimeError`, which the
rollout wraps as `PolicyError`. This is also the session-lifetime answer: a
`max_steps=1200` rollout that outlives any single Live session survives on
recoveries.

### Lifecycle

`policy.on_trial_end` closes the live session (an `isinstance` check, the
`mark_anchor` precedent): trials never share sessions anyway (identity rule),
this stops a session idling through scoring until the server `GoAway`s it,
and nothing else would ever close the final trial's socket (`eval()` closes
only the embodiment). Closing is **best-effort**: `GeminiLiveClient.close()`
swallows all transport/close exceptions — the sessions most likely to be
closed here are half-dead (post-`GoAway`, server already gone), and an
exception escaping `on_trial_end` would flip a successful trial to `"error"`
(eval.py:428-442). Stub test: server drops the connection, `on_trial_end`
still returns cleanly.

### Capture and transcript

One `WireCapture.record` row per socket exchange **attempt** (recovery
attempts get their own rows, matching the one-row-per-attempt contract):
`endpoint="bidi"`, `request={"messages": [...]}` (a dict, as `record`'s
signature requires) containing the client messages sent in that attempt —
including `setup` when the attempt opened a session — and
`response_text=json.dumps({"messages": [...]})` for the received messages.
The object wrapper matters: `_capture._response_value` stores parsed JSON
only when it is a dict and truncates anything else to 2000 chars, which
would silently lose multi-message exchanges. `_replace_blobs` gains an
`inlineData` branch (Gemini parts have no `type` key) so frames dedupe into
content-addressed blobs like the HTTP wires.

The client-side message list remains the authoritative transcript exactly as
on the HTTP wires — deliberate: with server-held state and context
compression, our side is the only complete record, and `on_trial_end`
already persists it.

## Tests (`tests/test_gemini_live.py` + `tests/_stub_bidi_server.py`)

Stub `BidiGenerateContent` server on `websockets.sync.server` (xpolicylab's
`_stub_server.py` pattern: free port, thread, steerable failure modes,
recorded inbound messages exposed for order assertions), speaking exact
v1beta JSON:

- setup round-trip over the stub (explicit ws base_url, including with a
  `google/`-prefixed model): `models/<bare-id>`, systemInstruction from the
  system message, functionDeclarations from toolset schemas, temperature
  when set; asserts `toolConfig` absent. The no-base_url `google/*` path is
  asserted **socketlessly** — `AgentPolicyConfig.base_url` records the
  derived wss endpoint and the client's pending setup payload carries
  `models/<bare-id>` — since the default endpoint is the real Google URL
  and the client has no HTTP-style `transport=` seam (the socket opens
  lazily inside `complete()`).
- suffix-shape sequencing, one test per table row, with stub-side order
  assertions: first-call (`[system, user, user]` → content turns with
  `turnComplete: true` on the final content turn, no empty third message,
  no toolResponse); normal step (obs turn `turnComplete: false` **before**
  the `toolResponse`, empty turn-complete last); nudge (no toolResponse);
  in-step tool-error retry (toolResponse only, **no** trailing
  turn-complete); on-demand `take_pic` (`[tool, user]` → three-part
  sequence).
- `functionResponses` shape: id echoed, `name` recovered from the earlier
  `toolCall`, `response == {"output": <string>}`; two `functionCalls` in
  one `toolCall` message → two `ToolCall`s, and their two tool-result
  messages aggregate into **one** `toolResponse` with entries in list
  order.
- assistant turns never re-sent mid-session (stub asserts no model-role
  content between setup and trial end).
- image translation: PNG data-URI part → `inlineData` with `image/png`.
- text-only completed turn → nudge path end-to-end (next exchange returns a
  tool call).
- drain behavior: interleaved `sessionResumptionUpdate` and empty turns
  before the `toolCall`; drain-cap exhaustion → recovery, then
  `RuntimeError` after retries.
- usage: `usageMetadata` on two messages of one exchange → summed,
  normalized keys.
- recovery: server drops between `clientContent` and `toolResponse`
  (mid-step) and separately sends `GoAway`; fresh session gets a text
  prologue (stub asserts: no model-role turns, no toolResponse, exactly one
  copy of the newest observation's images, older images as `_sanitize`'s
  `"[image omitted: streamed camera frame]"` stubs, prologue lines parse as
  JSONL) and the step completes; a server that keeps dropping exhausts
  retries into `RuntimeError`; an `images=on_demand` recovery (anchor user
  message carries no frames) completes with a frameless anchor turn.
- new-trial detection: `reset()` to a byte-identical scene still produces a
  fresh session (identity check, not value equality); view-not-reset guard
  raises `RuntimeError`.
- lifecycle: `on_trial_end` closes the socket (stub observes the close);
  next trial reopens lazily; with the server already gone, `on_trial_end`
  still returns cleanly (best-effort close).
- validation: bare `wire=gemini-live` with `google/*` + key constructs and
  resolves `effort=None`/`image_horizon=None`; explicit `effort=low` and
  explicit `image_horizon=2` → guided `ConfigError`s; explicit
  `effort=none`/`image_horizon=none` pass; sentinel resolution on the HTTP
  wires still lands on `"low"`/`2`, and `image_horizon=0`/`effort=default`
  are still rejected with today's messages (regressions); non-google model
  without ws base_url, OpenRouter-fallback (no `GEMINI_API_KEY`,
  `OPENROUTER_API_KEY` set — message must not leak the key), and `http://`
  base_url → guided `ConfigError`s.
- capture: one row per attempt including the recovery attempt; setup in the
  opening row; a long multi-message response survives un-truncated (object
  wrapper); `inlineData` frames deduped to blobs; no key material in rows.
- key redaction: transport error whose exception text contains the URL
  surfaces without the `?key=` query.

Existing pinned tests updated: `test_package.py` version pin → `0.20.0` and
`__all__` gains `GeminiLiveClient` (also exported from `__init__.py`, the
`AnthropicClient`/`ResponsesClient` precedent). `test_llm.py` is untouched
(provider resolution unchanged). Note the agent plugin's CI coverage is
report-only (`--cov-fail-under=0`, ci.yml:327-331) — the exhaustive test
list above is the real gate, not a coverage number.

## Docs

`plugins/inspect-robots-agent/README.md`: add the `gemini-live` row to the
wire table ("Google's Live API — required for the `-streaming-` robotics
model ids"); replace the interim ER 2 paragraph from the docs commit with a
short "Gemini Robotics ER 2" subsection covering both ids —
`gemini-robotics-er-2-preview` on the default chat wire,
`gemini-robotics-er-2-streaming-preview` via `-P wire=gemini-live` — the
one-liner example, and the per-wire notes (no `effort`, no `image_horizon`,
Live context compression instead; usage counts include two generations per
step). The paragraph's "Interactions API" fallback note is superseded and
dropped. CHANGELOG entry under the plugin.

## Version

`inspect-robots-agent` 0.19.1 → 0.20.0 (new wire, new dependency
`websockets>=12`; the `_UNSET` sentinel change is behavior-preserving for
every expressible input). Core is untouched — no core version bump.
`uv lock` run and committed for the plugin dependency addition (CI installs
`--locked`).
