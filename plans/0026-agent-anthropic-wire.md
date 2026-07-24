# 0026 — `-P wire=anthropic`: native Messages API unlocks fast mode

Issue: #165. Status: draft.

## Problem

The agent plugin reaches Anthropic models only through the OpenAI-compat shim
at `https://api.anthropic.com/v1/chat/completions` (`_llm.py` `_DIRECT_PROVIDERS`).
The compat shim cannot express the Claude-native request surface, so the knob
that matters most for this plugin is unreachable:

**Fast mode** runs Claude Opus 5 and Opus 4.8 at up to 2.5x higher output
tokens/sec. It needs three things simultaneously: native `POST /v1/messages`,
the `anthropic-beta: fast-mode-2026-02-01` header, and `speed: "fast"` as a
top-level body field. `ChatClient` posts to `/chat/completions`, sets only
`Authorization: Bearer`, and builds a closed body (`model`, `messages`,
`tools`, `temperature`, `reasoning_effort`) — none of the three is reachable,
and there is no `extra_body`/`extra_headers` escape hatch to smuggle them in.

This is not a cosmetic gap for a robotics eval. The arm stands still while the
model thinks, which is exactly why `effort` already defaults to `low`
(`policy.py:143-146`). Fast mode is the same trade on the serving side, and it
is the one lever that buys latency without trading reasoning depth.

The native wire also unlocks `output_config.effort` in its own spelling
(`low` through `max`, where the compat shim's `reasoning_effort` mapping is
undocumented) and correct `stop_reason: "refusal"` surfacing, which the compat
shim flattens into an ordinary empty completion.

## Design

Third wire, one new client class, zero changes to the conversation loop. This
is deliberately the same shape as plan 0022 (`wire=responses`): the
chat-completions message format stays the single source of truth for
`_messages`, the transcript, sanitization, and echo; the new client translates
at the wire boundary only. Same no-SDK doctrine: httpx, raw JSON.

### `wire` param gains `"anthropic"`

`_WIRE_FORMATS` becomes `{"chat", "responses", "anthropic"}`. Default stays
`"chat"`: it works for OpenRouter, vLLM, Ollama, and the Anthropic/Gemini
compat endpoints. No auto-selection when the model id starts with
`anthropic/` — implicit switching would change wire behavior, billing surface,
and error shape under users' feet based on which env key happens to be set,
and the compat path remains the right default for OpenRouter-routed Claude.

`base_url` composes: `wire=anthropic` posts to `{base_url}/messages`, so a
proxy or gateway that serves the Messages API works unchanged.

### `speed` param (the point of the exercise)

`LLMAgentPolicy(speed=None)`, values `None | "fast"`. Anything else is a
`ValueError` at construction, mirroring the `effort` validation. `-P speed=fast`
forwards for free. Recorded in `AgentPolicyConfig` so the eval log distinguishes
a fast-mode run from a standard one — they are the same model at different
prices ($10/$50 vs $5/$25 per MTok), so a cost reconciliation that cannot see
this field is wrong by 2x.

`speed="fast"` with `wire != "anthropic"` is a construction-time `ValueError`,
not a silent no-op. The compat shim accepts unknown body fields by ignoring
them, so a silently-dropped `speed` would bill at standard rates while the
user believes fast mode is on. This is the failure mode most worth failing
loudly on.

No model-id gating. Fast mode is Opus 5 and Opus 4.8 today, but the supported
set moves and a hardcoded allowlist would reject a model the API accepts.
Unsupported models get the API's own 400, wrapped by the guided error below.

### `max_tokens` param (forced by the wire)

`max_tokens` is **required** on `/v1/messages` and has no default — unlike
both OpenAI wires, where omitting it is legal. `LLMAgentPolicy(max_tokens=4096)`,
validated `>= 1`, recorded in the config, sent on every `wire=anthropic`
request and ignored by the other two wires.

4096 is chosen against this plugin's output shape, not the model's ceiling: a
turn is one tool call plus a one-or-two-sentence `note`. It also has to clear
adaptive thinking, which is on by default on Opus 5 and bills against the same
`max_tokens` ceiling — a tight limit would truncate mid-thought and burn a
turn on the no-tool-call retry path.

### `AnthropicClient` (new module `_anthropic.py`)

Same constructor (including `transport=` injection), same
`complete(messages, tools, temperature, reasoning_effort) -> AssistantMessage`
and `close()` surface as `ChatClient` and `ResponsesClient`, same bounded retry
policy (429/5xx/transport retried with exponential backoff, other 4xx fail
fast). `policy.py`'s changes stay confined to `__init__` (the new param
validation and client selection) and the config dataclass.

`Provider`, `ToolCall`, and `AssistantMessage` are shared vocabulary from
`_llm.py`. As in 0022, the retry loop is small enough to keep duplicated rather
than extracting a base class across three clients.

Headers: `x-api-key` (not `Authorization: Bearer` — the native API's own
scheme), `anthropic-version: 2023-06-01`, and `anthropic-beta:
fast-mode-2026-02-01` only when `speed="fast"`. The beta header is what makes
`POST /v1/messages` the beta endpoint over raw HTTP; there is no separate URL.

Request body: `model`, `max_tokens`, `messages` (translated), `system`
(hoisted, see below), and when set `tools` (flattened), `temperature`,
`output_config: {"effort": ...}`, `speed`.

### Effort mapping, and why `thinking` is never disabled

`_EFFORT_LEVELS` is `{none, minimal, low, medium, high, xhigh, max}`; the
Messages API accepts `low | medium | high | xhigh | max`. The two extras are
rejected at construction for `wire=anthropic` with a guided `ValueError`
naming the accepted set.

This departs from 0022, which let `effort=max` on `wire=responses` fail with
OpenAI's own 400 and kept validation wire-agnostic. The difference is the
failure mode. `max` on Responses is a hard 400: loud, self-describing,
unmissable. The natural native translation of `effort="none"` is
`thinking: {"type": "disabled"}`, which the API **accepts** — and on Opus 5
disabled thinking has a documented failure mode where the model writes a tool
call into its visible text instead of emitting a `tool_use` block. The turn
succeeds, the call never runs, no error is raised. For a policy that is tool
calls or nothing, that is three silent no-tool-call turns and then a generic
`RuntimeError` pointing nowhere near the cause. A guard is justified where the
alternative is silent degradation, not where it is a 400.

`effort=None` (the Python `None`, distinct from the wire string `"none"` —
see the README's existing note) omits `output_config` entirely and lets the
model default apply. `thinking` is never sent in any configuration: adaptive
is the Opus 5 default and the right one here.

### Translation: chat messages → Messages API

| Chat form | Messages API |
| --- | --- |
| `{"role": "system", "content": "<str>"}` | hoisted to the top-level `system` string, not a message. Only `_messages[0]` is ever a system message in this loop; a system message anywhere else raises `RuntimeError` rather than being silently dropped |
| `{"role": "user"/"assistant", "content": "<str>"}` | same role, `content` kept as the plain string |
| user content part `{"type": "text", ...}` | `{"type": "text", "text": ...}` unchanged |
| user content part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,B"}}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "B"}}` — the `data:` prefix is stripped; `_png.png_data_url` always emits PNG, so the media type is not parsed out of the URL, but a URL that does not match the expected prefix raises rather than shipping a malformed block |
| assistant msg `content` (non-empty) | `{"type": "text", "text": ...}` block first |
| assistant msg `tool_calls[i]` | `{"type": "tool_use", "id": id, "name": ..., "input": json.loads(arguments)}`, after the text block if any. Non-object or unparseable `arguments` raises `RuntimeError` naming the tool — a `tool_use` block with a non-object `input` is a 400 whose message does not identify which call was malformed |
| `{"role": "tool", "tool_call_id": id, "content": c}` | `{"type": "tool_result", "tool_use_id": id, "content": c}` inside a **user** message; **consecutive tool messages merge into one user message** (see below) |
| assistant msg, no tool calls, content `None` or `""` | **no message** — same accepted loss as 0022. The `incomplete`/nudge retry path appends exactly this, and an assistant message with an empty `content` array is a 400 |

The merge rule is load-bearing, not tidiness. `act()` appends one `role: "tool"`
message per ignored extra tool call and then one for the executed call
(`policy.py:269-280`), so a multi-tool-call turn produces several consecutive
tool messages. The Messages API requires every `tool_result` answering one
assistant turn to sit in a single user message; emitting one user message per
result makes the second one an unanswered-`tool_use` 400. Runs of consecutive
`role: "tool"` messages therefore collapse into one user message with one
`tool_result` block each, in order.

Tool schemas flatten from the chat nesting:
`{"type": "function", "function": {name, description, parameters}}` →
`{"name": ..., "description": ..., "input_schema": ...}`. `Toolset.schemas()`
stays chat-shaped; the client owns the translation both ways. No strict-mode
concern here — the Messages API does not auto-normalize schemas the way the
Responses API does, so the move tool's free-form `targets`/`deltas` object
survives unchanged.

### Response parsing → `AssistantMessage`

From `response["content"]`: concatenate the `text` of every `text` block (None
if there are none), and one `ToolCall(id=block["id"], name=block["name"],
arguments=json.dumps(block["input"]))` per `tool_use` block, in order.
`thinking` blocks are ignored — the plugin does not replay them, and with
`display` left at its `"omitted"` default they carry no text anyway.

Re-serializing `input` back to JSON text keeps `ToolCall.arguments` a string
across all three wires, so `_tools.py` validation and the transcript format
stay identical and the echo path is untouched.

`stop_reason: "refusal"` raises `RuntimeError` carrying
`stop_details.category` when present. This is an HTTP 200 whose `content` is
empty or partial; parsing it as an empty `AssistantMessage` would burn three
"Respond with exactly one tool call." nudges and then die with a generic
no-tool-call error while the real cause sits discarded in the body. Same
fail-fast-no-retry decision as 0022's `status: "failed"`: the server
deliberately answered. `stop_details` is informational and may be `null` even
on a refusal, so the branch keys on `stop_reason` alone and treats the
category as optional.

### Guided errors

House style: the error names the fix, never just the failure.

**OpenRouter fallback.** `resolve_provider`'s ladder falls through to
OpenRouter whenever `ANTHROPIC_API_KEY` is unset, and
`openrouter.ai/api/v1/messages` does not exist. Construction raises
`ConfigError` when `wire="anthropic"` resolved to the OpenRouter base URL,
naming the two fixes (set `$ANTHROPIC_API_KEY`, or pass `-P base_url=` for a
gateway that serves the Messages API). Only the OpenRouter base is rejected;
an explicit `base_url` is always honored, since proxies serving `/v1/messages`
are a legitimate deployment.

**Fast mode unsupported.** A 4xx whose body mentions `speed` or `fast` while
`speed="fast"` appends: `fix: fast mode needs Claude Opus 5 or Opus 4.8 on the
Claude API (not Bedrock/Vertex/Foundry), or drop -P speed=fast`.

**Fast-mode rate limit.** Fast mode draws on a rate-limit pool separate from
standard Opus, so a fast-mode run can 429 while standard capacity is idle.
After retries are exhausted on a 429 with `speed="fast"`, the terminal error
appends: `fix: fast mode has its own rate limit; retry later or drop
-P speed=fast to fall back to standard speed`.

No automatic fallback to standard speed on 429. Silently downgrading changes
the billing rate and the latency profile mid-eval, and it invalidates the
prompt cache (switching `speed` is a cache-invalidating change), so the
"recovery" costs a full cache re-write and produces an eval log whose `speed`
field no longer describes what ran. The user chose fast mode; the fix belongs
in their hands.

## Tests (`tests/test_anthropic.py` + small additions)

All against `httpx.MockTransport`, mirroring `test_llm.py` and
`test_responses.py` conventions.

- Translation goldens: each row of the table above, including multi-part
  observation content with a PNG data URL (asserting the `data:` prefix is
  stripped and `media_type` is set), and a malformed image URL raising.
- The merge rule: a turn with three tool calls produces exactly one user
  message carrying three `tool_result` blocks in order, not three messages.
  Plus the interleaved case (tool messages, then a user observation) to show
  only *consecutive* runs merge.
- System hoisting: `_messages[0]` becomes top-level `system` and no message
  carries `role: "system"`; a system message at a later index raises.
- Request body: `max_tokens` always present, `system` present, tools flattened
  to `input_schema`, `output_config.effort` present only when effort is set,
  `temperature` only when set, `speed` only when set, and the request path is
  `{base_url}/messages`.
- Headers: `x-api-key` and `anthropic-version` on every request; the
  `anthropic-beta: fast-mode-2026-02-01` header present with `speed="fast"`
  and **absent** without it.
- Parsing: text-only, tool-use-only, text + tool-use, text spread across
  multiple blocks, `thinking` blocks ignored, `input` round-tripping to JSON
  text. `stop_reason: "refusal"` raises with the category in the message and
  puts exactly one request on the wire; a refusal with `stop_details: null`
  raises without an AttributeError.
- Assistant turn with `content: None` and no tool calls emits no message.
- Guided errors: the OpenRouter-fallback `ConfigError` (and that an explicit
  `base_url` suppresses it), the fast-mode 4xx guidance, and the fast-mode 429
  guidance after retries are exhausted.
- Retry/fail-fast parity with `ChatClient`, plus `close()`.
- Policy wiring: `wire="anthropic"` end-to-end through `LLMAgentPolicy.act()`
  (mock embodiment); `speed="fast"` with `wire="chat"` raises; invalid `speed`
  raises; `effort="none"`/`"minimal"` with `wire="anthropic"` raises;
  `max_tokens=0` raises; `wire`, `speed`, and `max_tokens` recorded in config.

## Docs

- `plugins/inspect-robots-agent/README.md`: add `speed` and `max_tokens` to
  the knobs list (line 101), extend the wire-format paragraph (line 48), and
  add a short "Fast mode on Claude" section next to the existing "Reasoning
  effort on OpenAI models" section, including the price delta and the
  Claude-API-only caveat. Repo writing style applies: no em dashes in prose.
- No `docs/` guide change. The quickstart and cookbook mention `-P model=` and
  `-P effort=`; wire selection stays plugin-README territory, as it did for
  `wire=responses`.

## Non-goals

- Prompt caching (`cache_control` breakpoints on the system block and tool
  schemas). The stable prefix here is large and reused every turn, so this is
  the obvious follow-up, but it is a cost optimization with its own placement
  and invalidation design, not part of unlocking fast mode.
- Adaptive-thinking display (`thinking: {"display": "summarized"}`) and
  surfacing thinking summaries into the transcript.
- Task budgets (`output_config.task_budget`), extended context, and the
  Batches API.
- `AnthropicBedrock`/Vertex/Foundry variants. Fast mode is Claude API only, so
  they would inherit the wire without its motivating feature.
