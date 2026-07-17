# 0021 — `-P wire=responses`: dial reasoning effort on OpenAI models

Issue: #131. Status: accepted (3 critique rounds).

## Problem

The agent plugin speaks only the Chat Completions wire format. OpenAI rejects
`tools` + `reasoning_effort` together on `/v1/chat/completions` for its recent
models (observed on `gpt-5.6-sol`; the restriction dates back to gpt-5.4):

```
Function tools with reasoning_effort are not supported for gpt-5.6-sol in
/v1/chat/completions. To use function tools, use /v1/responses or set
reasoning_effort to 'none'.
```

Since the agent policy is tool calls or nothing, the only working
configuration for OpenAI models today is `-P effort=none` — reasoning off. An
eval of an OpenAI reasoning model cannot be run at all. (First hit as a failed
YAM fork-task eval, 2026-07-15.)

## Design

One new policy param, one new client class, zero changes to the conversation
loop. The chat-completions message format stays the single source of truth for
`_messages`, the transcript, sanitization, and echo; the Responses client
translates at the wire boundary only. Same no-SDK doctrine: httpx, raw JSON.

### `wire` param

`LLMAgentPolicy(wire="chat")`, values `"chat" | "responses"`; anything else is
a `ValueError` at construction (mirrors the `effort` validation). Recorded in
`AgentPolicyConfig` so it lands in the eval log. The CLI forwards
`-P wire=responses` for free.

Default stays `"chat"`: it works for OpenRouter, vLLM, Ollama, and the
Anthropic/Gemini compat endpoints, none of which serve `/responses`. No
auto-selection by provider — implicit switching would change wire behavior
under users' feet based on which env key happens to be set.

`base_url` composes: `wire=responses` posts to `{base_url}/responses`, so an
Azure or proxy endpoint that serves the Responses API works unchanged.

### `ResponsesClient` (new module `_responses.py`)

Same constructor (including `transport=` injection) and
`complete(messages, tools, temperature, reasoning_effort)
-> AssistantMessage` and `close()` surface as `ChatClient`, same bounded
retry policy (429/5xx/transport retried with exponential backoff, other 4xx
fail fast), so `policy.py`'s changes are confined to `__init__` (the `wire`
param, its `ValueError` validation, client selection) and the config
dataclass.
`Provider`, `ToolCall`, `AssistantMessage`, and the retry loop shape are
shared vocabulary from `_llm.py`; the retry loop itself is small enough to
keep duplicated rather than extracting a base class.

Request body: `model`, `input` (translated items), `tools` (translated),
`store: false`, `include: ["reasoning.encrypted_content"]`, and when set
`temperature` and `reasoning: {"effort": ...}` (the Responses spelling of
`reasoning_effort`). Current OpenAI returns encrypted reasoning content by
default when `store: false`, so the explicit `include` is belt-and-braces
for older/Azure/proxy deployments, not load-bearing against openai.com —
keep it, but it is not the thing to "fix" if reasoning replay breaks.

One effort caveat: `_EFFORT_LEVELS` includes `"max"`, which `/responses`
does not accept (none/minimal/low/medium/high/xhigh). Construction-time
validation stays wire-agnostic; `wire=responses` + `effort=max` fails fast
with OpenAI's own 400, which names the valid values. Accepted, not guarded.

Stateless by design (`store: false`): no server-side conversation state to
lose, works on ZDR orgs, every request can be legally rebuilt from the
transcript alone (byte-identical replay additionally needs the in-memory
raw-item cache; a fresh client synthesizes a legal, if different, request),
and tests are plain request/response pairs against `httpx.MockTransport`.
`previous_response_id` chaining was rejected: it leaves conversation state on
OpenAI's servers, breaks on ZDR orgs, and makes retries after an ambiguous
transport failure double-append.

### Translation: chat messages → Responses input items

| Chat form | Responses input item |
|---|---|
| `{"role": "system"/"user"/"assistant", "content": "<str>"}` | same-role **untyped** message (`{"role": ..., "content": "<str>"}`, no `"type"` key) — the untyped EasyInputMessage form is load-bearing: the typed `{"type": "message"}` item rejects plain-string assistant content (assistant parts must be `output_text`), so a "uniformity" refactor adding `type` would 400 |
| user content part `{"type": "text", ...}` | `{"type": "input_text", "text": ...}` |
| user content part `{"type": "image_url", "image_url": {"url": u}}` | `{"type": "input_image", "image_url": u}` |
| assistant msg `tool_calls[i]` | `{"type": "function_call", "call_id": id, "name": ..., "arguments": ...}`, after the assistant text item if any — but a raw-item cache hit (below) replaces **every** synthesized item for that assistant message, text and tool calls alike, with the cached item list verbatim in cached order; emitting both would show the model its own text twice per turn |
| `{"role": "tool", "tool_call_id": id, "content": c}` | `{"type": "function_call_output", "call_id": id, "output": c}` |
| assistant msg, no tool calls, content `None` or `""` | **no input item** — the `incomplete`-status retry path appends exactly this message (`raw()` of an empty turn) before the nudge, and consecutive user messages are legal input; emitting `content: null` would 400 and turn a recoverable no-tool-call turn fatal |

`ToolCall.id` carries the Responses `call_id` (that is what
`function_call_output` must reference), so the policy's existing
`tool_call_id` bookkeeping works untouched.

Tool schemas flatten from the chat nesting to the Responses shape:
`{"type": "function", "function": {name, description, parameters}}` →
`{"type": "function", "name": ..., "description": ..., "parameters": ...,
"strict": false}`. `strict: false` is explicit and non-negotiable: unlike
Chat Completions (non-strict unless asked), Responses auto-normalizes tool
schemas into strict mode "when possible", and the move tool's
`targets`/`deltas` parameter is a free-form object with no `properties`
(dimension names live in the description) — exactly the shape strict
normalization would mangle into an empty closed object, making every move
call unrepresentable. Explicit `strict: false` preserves chat-wire parity.
`Toolset.schemas()` stays chat-shaped; the client owns the translation both
ways.

### The reasoning-item constraint

OpenAI reasoning models require that when a `function_call` item is resent,
the `reasoning` item that preceded it is resent too; a bare replay of
chat-format history 400s with "function_call was provided without its
required reasoning item". With `store: false` the reasoning item's content
comes back encrypted (`reasoning.encrypted_content`) and is replayed
verbatim.

So the client keeps a raw-item cache: after each successfully *parsed*
response (never a failed or raising one), the response's verbatim `output`
item list is stored keyed by the `call_id` of each `function_call` item in
it. During translation, an assistant message whose first `tool_call.id` hits
the cache emits the cached raw items in place of all synthesized items for
that message (this preserves item `id`s and any interleaved message items
exactly as the API produced them). A miss falls back to the synthesized
translation — correct for histories this client instance did not produce,
and for non-reasoning models, where no reasoning item is required.

Cache lifetime: entries whose `call_id` no longer appears in the submitted
history are pruned on each `complete()` call, against the submitted history
**before** the new response is cached (prune-after-store would evict every
fresh entry and make the cache always miss) — call ids are harvested from
both places history carries them, assistant `tool_calls[i].id` and tool
message `tool_call_id`, so a future history-trimming feature cannot silently
break replay. A `reset()` (fresh `_messages`) empties the cache without the
client needing a reset hook.

Accepted loss: text-only assistant turns (the no-tool-call retry path) cache
nothing, so their reasoning items are dropped from replay. That is legal —
only `function_call` items require their preceding reasoning item — and
costs at most some reasoning reuse on the next turn.

### Response parsing → `AssistantMessage`

From `response["output"]`: concatenate the `text` of every
`output_text` content part of `message` items (None if there are none), and
one `ToolCall(id=item["call_id"], name, arguments)` per `function_call` item,
in output order. `reasoning` items influence nothing here — they only matter
for the cache. An `incomplete` status with no usable output falls into the
existing no-tool-call retry path in `act()`.

A body with `status: "failed"` (HTTP 200, top-level `error` object) raises
`RuntimeError` carrying `error.message` — parsing it as an empty
`AssistantMessage` would burn three "Respond with exactly one tool call."
nudges and then die with a generic no-tool-call error while the real cause
sits discarded in the body. Decision: failed status fails fast, no retry
(one request on the wire, mirroring the 4xx path) — the deliberate,
message-bearing terminal state of a request the server accepted, unlike a
5xx where the server never answered. A failed response never populates the
raw-item cache.

### Guided error for the motivating failure

When `ChatClient` (wire=chat) gets the 4xx above — detected by
`"reasoning_effort"` and `"/v1/responses"` both appearing in the error body —
the raised `RuntimeError` appends:
`fix: pass -P wire=responses (needs a direct OpenAI endpoint, e.g.
$OPENAI_API_KEY set), or -P effort=none`.
The endpoint caveat matters: the same error is reachable via OpenRouter
routing to an OpenAI model, where `wire=responses` alone would dead-end at a
nonexistent `openrouter.ai/api/v1/responses`. House style: the error names
the fix, never just the failure.

## Tests (`tests/test_responses.py` + small additions)

All against `httpx.MockTransport`, mirroring `test_llm.py` conventions.

- Translation goldens: each row of the table above, including multi-part
  observation content with an image data URL, and multi-tool-call turns with
  their ignored-extra `function_call_output`s.
- Request body: `store: false`, `include`, `reasoning.effort` present only
  when effort set, `temperature` only when set, tools flattened with
  `strict: false`, history messages emitted in the untyped (no `"type"` key)
  form, and the request path is `{base_url}/responses`.
- Reasoning-item replay: turn 1 returns `[reasoning, message, function_call]`;
  the turn-2 request must contain all three verbatim, before the
  `function_call_output`, with the message text appearing exactly once (no
  extra synthesized assistant message for that turn). A second replay case
  returns two `function_call` items on turn 1 and asserts each cached item
  appears exactly once in the turn-2 request (cache hit is per assistant
  message, keyed on the first call id — not per tool call). Plus cache-prune
  on a fresh history, and cache-miss synthesis — including the tool-call-only turn
  (`content: None`), which must emit no null-content message item, only the
  `function_call`.
- Empty assistant turn (`content: None`/`""`, no tool calls — the
  `incomplete` retry path) translates to no input item.
- Parsing: text-only, tool-call-only, text+tool-call, `output_text` spread
  across multiple message items; `status: "failed"` raises with
  `error.message` in the text, exactly one request on the wire, cache left
  unpopulated.
- Retry/fail-fast parity with `ChatClient`, plus `close()`.
- Policy wiring: `wire="responses"` end-to-end through `LLMAgentPolicy.act()`
  (mock embodiment), invalid `wire` value raises, `wire` recorded in config.
- Chat-side: the guided error suffix on the motivating 4xx body.

## Docs

Plugin README: a "Reasoning effort on OpenAI models" section — the error, why
it happens (Chat Completions restriction, not an inspect-robots bug), and the
`-P wire=responses -P effort=medium` fix. One line in the wire-format
paragraph noting the default and when to switch.

## Version

`inspect-robots-agent` 0.9.0 → 0.10.0 (new feature, backward compatible;
default behavior unchanged). Core untouched.
