# 0026 — `-P wire=anthropic`: native Messages API unlocks fast mode

Issue: #165. Status: draft (revised after critique rounds 1 and 2).

## Problem

The agent plugin reaches Anthropic models only through the OpenAI-compat shim
at `https://api.anthropic.com/v1/chat/completions` (`_llm.py:37`). The compat
shim cannot express the Claude-native request surface, so the knob that matters
most for this plugin is unreachable:

Fast mode runs Claude Opus 5 and Opus 4.8 at up to 2.5x higher output
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

Scope note: fast mode is Claude API only. Not Bedrock, Vertex, Foundry, or
Claude Platform on AWS, and not usable with the Batch API or Priority Tier.
Opus 4.7 fast mode was removed and now errors, so this wire does not resurrect
it.

## Design

Third wire, one new client class, zero changes to the conversation loop. Same
shape as plan 0022 (`wire=responses`): the chat-completions message format
stays the single source of truth for `_messages`, the transcript, sanitization,
and echo; the new client translates at the wire boundary only. Same no-SDK
doctrine: httpx, raw JSON.

The one place this is not a thin mirror of 0022 is opaque-block replay (see
"The thinking-block constraint"). 0022 needed a raw-item cache for OpenAI
reasoning items; this wire needs the same mechanism for Claude thinking blocks.

### What this plan deliberately does not touch

Two earlier drafts proposed a per-wire `_ACCEPTED_EFFORTS` table, partly to
close 0022's known `wire=responses` + `effort=max` hole in the same PR. It is
cut, for three reasons that only surfaced under review:

- Its stated motive was that users would paste `-P effort=none` into a Claude
  run. False: the CLI reads a bare `none` as Python `None` (`README.md:118`),
  which omits the field and is legal on every wire. Only the quoted
  `-P effort="'none'"` form would trip a guard, and that form exists precisely
  because a user wanted the literal wire value.
- Keying accepted efforts on the **wire** treats a protocol as a vendor. Both
  wires advertise gateway support via `-P base_url=`, and a table would reject
  configurations the gateway's model accepts.
- It is the only part of the PR that changes behavior for an existing wire,
  which is the argument for a separate PR, not for bundling.

So `effort` validation stays exactly as it is today (`policy.py:128-132`), and
out-of-set values fail with the API's own 400 — the same disposition 0022 chose
for `effort=max`. `effort="none"`/`"minimal"` on this wire will 400; that is
consistent, loud, and self-describing.

Also cut for scope, all listed under Non-goals: `retry-after`-aware backoff,
exposing response `usage`, and adding a field to `Provider`.

### `wire` param gains `"anthropic"`

`_WIRE_FORMATS` becomes `{"chat", "responses", "anthropic"}`. Default stays
`"chat"`: it works for OpenRouter, vLLM, Ollama, and the Anthropic/Gemini
compat endpoints. No auto-selection when the model id starts with
`anthropic/` — implicit switching would change wire behavior, billing surface,
and error shape under users' feet based on which env key happens to be set.

`base_url` composes: `wire=anthropic` posts to `{base_url}/messages`, so a
proxy or gateway serving the Messages API works unchanged.

**Model id form.** This wire uses the same `anthropic/<id>` prefix as every
other model string in the plugin; `resolve_provider` strips it and sends the
bare id (`_llm.py:112-120`). A bare `-P model=claude-opus-5` partitions to
prefix `claude-opus-5` with an empty remainder, misses the direct-provider
table, and dead-ends at "no API key found" (`_llm.py:123`) even with
`ANTHROPIC_API_KEY` set. Stated here and in the README because a user who knows
the native API will reach for the bare form.

### `speed` param (the point of the exercise)

`LLMAgentPolicy(speed=None)`, values `None | "fast"`. Anything else is a
`ValueError` at construction, mirroring the `effort` validation. `-P speed=fast`
forwards for free.

`speed="fast"` with `wire != "anthropic"` is a construction-time `ValueError`,
not a silent no-op. The compat shim accepts unknown body fields by ignoring
them, so a silently-dropped `speed` would bill at standard rates while the user
believes fast mode is on.

No model-id gating. Fast mode's supported set moves (4.7 was removed, 5 was
added) and a hardcoded allowlist would reject a model the API accepts.
Unsupported models get the API's own 400, wrapped by the guided error below.

### `max_output_tokens` param (forced by the wire)

`max_tokens` is required on `/v1/messages` and has no server default, unlike
both OpenAI wires where omitting it is legal.

Named `max_output_tokens`, not `max_tokens`: next to `max_llm_calls` (a
per-trial budget) and `max_speed_frac`, a bare `max_tokens` reads as a
trial-wide token budget. It is a per-response cap, and the name matches the
Responses spelling so forwarding it on the other wires later is an extension
rather than a rename.

`LLMAgentPolicy(max_output_tokens: int | None = None)`. Resolution happens in
`policy.__init__`, not in the client:

```python
_DEFAULT_MAX_OUTPUT_TOKENS = 16000   # in _anthropic.py, imported by policy.py

resolved = (
    (max_output_tokens if max_output_tokens is not None else _DEFAULT_MAX_OUTPUT_TOKENS)
    if wire == "anthropic"
    else None
)
```

The `wire` gate is load-bearing: an ungated resolution would record `16000` in
the config for a `wire=chat` run, where nothing capped the output.

`AnthropicClient` therefore takes `max_output_tokens: int` (no default, no
second source of truth for a number that lands in an eval log). The resolved
value is what `AgentPolicyConfig` records.

### `__init__` validation order

The new checks are order-dependent, so spell the sequence out rather than
leaving it to be rediscovered. Today `resolve_provider` runs at
`policy.py:120`, before the `wire` check at `policy.py:133-134`; that has to
change, because the `api_key_env` substitution must precede resolution and the
OpenRouter check must follow it.

1. `max_speed_frac`, `max_llm_calls`, `effort` (unchanged)
2. `wire` membership in `_WIRE_FORMATS` — **first** among the new checks, so
   `-P wire=antropic -P speed=fast` reports the misspelled wire rather than the
   speed error
3. `speed` value, and `speed`/`max_output_tokens` against `wire`
4. resolve the effective `api_key_env` (see the guided-errors section)
5. `resolve_provider(...)`
6. the OpenRouter-fallback `ConfigError`
7. resolve `max_output_tokens`, build the client

Step 6 needs `from inspect_robots.errors import ConfigError` in `policy.py`,
which has no such import today (only `_llm.py:18` imports it).

Validation: `>= 1` when set, checked on every wire. `max_output_tokens` set
with `wire != "anthropic"` is a construction-time `ValueError`, mirroring
`speed` — a user passing `-P max_output_tokens=2000 -P wire=chat` believes a
cap applies, and none would. The config records `None` on the other two wires.

Why 16000: the visible output of a turn is one tool call plus a
one-or-two-sentence `note`, on the order of 100-200 tokens. Everything above
that is headroom for adaptive thinking, which bills against this same ceiling.
There is no published number for adaptive spend, so the default follows the
repo-wide guidance for non-streaming requests rather than the visible-output
estimate, and truncation is a hard error (below) so a too-small value is loud
rather than silent.

**Known limit, documented not guarded:** `effort` of `xhigh` or `max` wants
`max_tokens` at 64K or above, and this client has no streaming path and a
120s timeout, so those settings will either truncate (hard error, naming
`-P max_output_tokens=`) or time out. The README says to keep effort at `high`
or below on this wire until streaming lands. Guarding it in code would mean
another wire-keyed allowlist, which is what this plan just cut.

### `AnthropicClient` (new module `_anthropic.py`)

```python
AnthropicClient(
    provider: Provider,
    *,
    max_output_tokens: int,
    speed: str | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    backoff_s: float = 1.0,
    transport: httpx.BaseTransport | None = None,
)
```

`speed` and `max_output_tokens` are constructor-injected. `complete()` keeps
the exact existing signature —
`complete(messages, tools, temperature, reasoning_effort) -> AssistantMessage`
— so `act()` (`policy.py:268-273`) is untouched and the
`ChatClient | ResponsesClient | AnthropicClient` union at `policy.py:136` stays
structurally compatible under mypy strict (the call site uses keyword
arguments, so parameter *names* are load-bearing, not just types). `close()` is
unchanged. `policy.py`'s changes stay confined to `__init__` and the config
dataclass.

`Provider`, `ToolCall`, and `AssistantMessage` are shared vocabulary from
`_llm.py`, imported as they are today. No change to `Provider`.

Headers: `x-api-key` (the native API's own scheme, not `Authorization:
Bearer`), `anthropic-version: 2023-06-01`, and `anthropic-beta:
fast-mode-2026-02-01` only when `speed="fast"`. The beta header is what makes
`POST /v1/messages` the beta endpoint over raw HTTP; there is no separate URL.
An empty `provider.api_key` omits `x-api-key` entirely, matching `ChatClient`
(`_llm.py:182-184`) so keyless local proxies work.

Request body: `model`, `max_tokens`, `thinking`, `messages` (translated), and
when applicable `system` (hoisted), `tools` (flattened), `output_config:
{"effort": ...}`, `temperature`, `speed`.

**`thinking: {"type": "adaptive"}` is always sent.** Omitting it is only
equivalent to adaptive on Opus 5; on Opus 4.8 and 4.7 an omitted `thinking`
runs with thinking **off**. Since fast mode covers Opus 5 and 4.8, omitting it
would silently disable thinking on half the target set and make the replay
cache below dead code there. The cost of sending it explicitly: pre-4.6 models
(Sonnet 4.5, Haiku 4.5) do not support `{"type": "adaptive"}`, so this wire
does not reach them. Per-model support is readable at
`GET /v1/models/<id>` → `capabilities.thinking.types.adaptive.supported`; the
exact failure shape has not been observed, so the README says "not supported"
rather than naming a status code. That is the right trade for a wire whose
reason to exist is an Opus-5-and-4.8 feature; the compat wire still serves
those models.

`temperature` is forwarded when set, not rejected. Opus 5, 4.8, and 4.7 reject
it with a 400, but Opus 4.6 and Sonnet 4.6 accept it over this same wire, so a
construction-time guard would forbid legal requests — the same anti-allowlist
reasoning applied to `speed` above. A guided error wraps the 400 instead.

### The thinking-block constraint

A typical assistant turn returns `[thinking, text?, tool_use]`. The hard
constraint is on **tool-use continuation**: when an assistant turn containing
`tool_use` is replayed, the thinking block that preceded it must be replayed
unchanged, or the request can fail on block ordering or signature validation.
`AssistantMessage` (`_llm.py:140-159`) carries only `content` and `tool_calls`,
so it is structurally lossy for opaque blocks.

This is 0022's problem with different block names, and it gets 0022's
mechanism. `AnthropicClient` keeps
`_raw_blocks_by_tool_use_id: dict[str, list[dict[str, Any]]]`:

- After each successfully parsed response (never a raising or terminal one),
  the response's verbatim `content` array is stored keyed by the `id` of each
  `tool_use` block in it.
- During translation, an assistant message whose first `tool_call.id` hits the
  cache emits `{"role": "assistant", "content": <cached blocks verbatim>}` in
  place of all synthesized blocks for that message. Emitting both would show
  the model its own text twice.
- A miss falls back to synthesis: correct for histories this client instance
  did not produce, and for non-thinking models.
- Entries whose `tool_use_id` no longer appears in the submitted history are
  pruned at the top of `complete()`, **before** the new response is cached
  (prune-after-store would evict every fresh entry and always miss). Ids are
  harvested from both places history carries them: assistant `tool_calls[i].id`
  and tool-message `tool_call_id`.
- `reset()` produces a fresh `_messages`, so the prune empties the cache
  without the client needing a reset hook. The client outlives a trial; the
  cache does not.

Accepted loss, inherited from 0022: a text-only assistant turn (the nudge retry
path, `policy.py:283-290`) carries no `tool_use`, so it caches nothing and its
thinking block is dropped from replay. That is safe because the constraint
attaches to tool-use continuation. Note such a turn is not terminal:
`_MAX_CONSECUTIVE_FAILURES = 3` (`policy.py:38`) means it stays in the history
for up to two further `complete()` calls.

### Translation: chat messages → Messages API

Unless a row says otherwise, a violation raises `RuntimeError` naming the
offending element.

| Chat form | Messages API |
| --- | --- |
| `{"role": "system", "content": "<str>"}` at index 0 | hoisted to the top-level `system` string, not a message. Omitted entirely when `_messages[0]` is not a system message (`act()` is reachable without `reset()`; the only guard is `bind()`) |
| a system message at index >= 1 | raises |
| `{"role": "user"/"assistant", "content": "<str>"}` | same role, `content` kept as the plain string |
| user content part `{"type": "text", ...}` | `{"type": "text", "text": ...}` unchanged |
| user content part `{"type": "image_url", "image_url": {"url": u}}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B}}` where `B` is `u` with the exact prefix `data:image/png;base64,` removed. `_png.png_data_url` always emits that prefix; any other `u` raises |
| any other content part type | raises. `_responses.py:151-159` silently drops unknown parts; this wire does not, because a dropped camera frame is an invisible capability loss mid-trial |
| assistant msg `content` (non-empty) | `{"type": "text", "text": ...}` block first |
| assistant msg `tool_calls[i]` | `{"type": "tool_use", "id": id, "name": ..., "input": json.loads(arguments)}` after the text block. Unparseable JSON and parseable-but-non-object both raise: a `tool_use` with a non-object `input` 400s with a message that does not identify the offending call |
| assistant msg with a cache hit on `tool_calls[0].id` | the cached raw block list verbatim, replacing every synthesized block |
| `{"role": "tool", "tool_call_id": id, "content": c}` | `{"type": "tool_result", "tool_use_id": id, "content": c}` inside a **user** message; consecutive tool messages merge into one (below) |
| assistant msg, no tool calls, content `None` or `""` | no message. The nudge path appends exactly this, and an assistant message with an empty content array is a 400 |

`tool_result.is_error` is deliberately never set. `act()` feeds tool failures
back as ordinary content (`policy.py:306`) and the model reads them fine on the
other two wires; adding the flag here would make this wire's transcript
semantics diverge from the shared history for no demonstrated gain.

Merging consecutive tool messages: `act()` appends one `role: "tool"` message
per ignored extra tool call (`policy.py:292-303`) and then one for the executed
call (`policy.py:304-307`). Splitting `tool_result` blocks across several user
messages does not 400 (the API coalesces consecutive same-role messages), but
it does silently train the model to stop making parallel calls, so the run
collapses into one user message with one `tool_result` block each.

Block order within the merged message is history order, which for a multi-call
turn is `[extra_1..extra_n, executed]` while the assistant's `tool_use` order
was `[executed, extra_1..extra_n]`. Blocks match by `tool_use_id`, not
position, so this is legal; it is called out because the mismatch looks like a
bug to a reader diffing the two arrays.

Two consecutive-user-message sequences fall out of the loop, both legal under
same-role coalescing and both easy to miss, so both get goldens:

- **Dropped assistant turn then nudge.** An assistant turn with no tool calls
  and empty content emits no message, so the observation user message is
  immediately followed by the nudge user message.
- **Tool results then next observation.** After a successful call `act()`
  returns; the next `act()` appends an observation, so a user message of
  `tool_result` blocks is immediately followed by one of text and images.

Tool schemas flatten from the chat nesting:
`{"type": "function", "function": {name, description, parameters}}` →
`{"name": ..., "description": ..., "input_schema": ...}`. `Toolset.schemas()`
stays chat-shaped; the client owns translation both ways. No strict-mode
concern: the Messages API does not auto-normalize schemas the way the Responses
API does, so the move tool's free-form `targets`/`deltas` object survives.
`tools=[]` omits the `tools` key, matching `ChatClient` (`_llm.py:200-201`).
Not `ResponsesClient`, which sends `tools: []` unconditionally
(`_responses.py:58-64`).

### Response parsing → `AssistantMessage`

From `response["content"]`: concatenate the `text` of every `text` block, and
emit one `ToolCall(id=block["id"], name=block["name"],
arguments=json.dumps(block["input"]))` per `tool_use` block, in order.
`thinking` blocks contribute nothing here; they matter only to the cache.

Empty concatenated text normalizes to `None`, never `""`. A `""` would
round-trip through `raw()` (`_llm.py:149`) into the next request as an empty
`text` block, which is a 400.

Re-serializing `input` to JSON text keeps `ToolCall.arguments` a string across
all three wires, so `_tools.py` validation, the transcript format, and the echo
path are untouched.

**Terminal stop reasons.** `stop_reason` outside
`{end_turn, tool_use, stop_sequence}` raises `RuntimeError` and is never
retried, matching 0022's `status: "failed"` decision — the server deliberately
answered. Handled as a default-deny rather than an enumeration of two, because
the enumeration would miss `model_context_window_exceeded`, which is live here:
`max_llm_calls` defaults to 100 and every turn appends fresh multi-camera PNGs.

Two get specific guidance:

- `refusal` carries `stop_details.category` when present. `stop_details` is
  informational and may be `null`, so the message is built with
  `(payload.get("stop_details") or {}).get("category")` — no attribute access
  on a possibly-`None` dict.
- `max_tokens` carries `fix: raise -P max_output_tokens=`. This is the one
  silent-degradation risk the PR newly creates: a truncated turn is an HTTP 200
  with partial content, and without the guard it is either no `tool_use` (three
  nudge turns, then a generic `RuntimeError` at `policy.py:286` with the cause
  discarded) or a partial `tool_use` whose `input` still parses and reaches
  `toolset.execute()` (`policy.py:304`) as robot motion. Raising before parsing
  content is what prevents the second case.

Anything else outside the permissive set gets a generic message naming the
`stop_reason` verbatim. No terminal response populates the cache.

`pause_turn` is deliberately in the deny set. It arises only from the
server-side sampling loop hitting its iteration limit (web search, web fetch,
code execution), and this wire declares client-side custom tools only, so it is
unreachable today. It is also the one denied reason that is *resumable* rather
than fatal, so a future PR adding a server tool must revisit this branch
instead of inheriting it.

### Guided errors

House style: the error names the fix, never just the failure.

**OpenRouter fallback.** `resolve_provider` falls through to OpenRouter
whenever `ANTHROPIC_API_KEY` is unset, and `openrouter.ai/api/v1/messages` does
not exist. `policy.py` compares `provider.base_url` against
`_llm._OPENROUTER_BASE`, imported alongside `ENV_MODEL` (`policy.py:33` already
imports from that private module, so this adds no new coupling), and raises
`ConfigError` when `wire="anthropic"` matched it, naming both fixes: set
`$ANTHROPIC_API_KEY`, or pass `-P base_url=` for a gateway serving the Messages
API.

Deliberately narrow: `-P model=openai/gpt-5 -P wire=anthropic` with
`$OPENAI_API_KEY` set resolves to `api.openai.com/v1` and dead-ends at a 404.
Enumerating every wrong-endpoint pairing is not worth it, and the 404 names the
URL.

**Key resolution with an explicit `base_url`.** `resolve_provider` consults
`api_key_env` only inside its `if base_url:` branch (`_llm.py:109-111`),
defaulting to `OPENROUTER_API_KEY`, so
`-P base_url=https://gateway/v1 -P wire=anthropic` would send an OpenRouter key
as `x-api-key` and 401 with no hint. For `wire="anthropic"` **with a
`base_url`**, `policy.py` defaults `api_key_env` to `ANTHROPIC_API_KEY` before
calling `resolve_provider`; an explicit `-P api_key_env=` still wins. With no
`base_url` the parameter is dead, so nothing changes and
`AgentPolicyConfig.api_key_env` records the user's original value rather than
the substituted one.

**Fast mode unsupported.** A 4xx while `speed="fast"` whose body contains both
`"speed"` and `"fast"` (case-insensitive, matched against the full
`response.text`, as `_llm.py:220` does) appends: `fix: fast mode needs Claude
Opus 5 or Opus 4.8 on the Claude API (not Bedrock, Vertex, Foundry, or Claude
Platform on AWS), or drop -P speed=fast`. The two-token AND mirrors 0022's
matcher shape; a bare `fast` substring is too weak.

**Sampling parameter rejected.** A 4xx whose body mentions `temperature`
appends: `fix: this model rejects temperature (Opus 5, 4.8, 4.7, Sonnet 5, and
Fable 5 all do; Opus 4.6 and Sonnet 4.6 accept it); drop -P temperature=`.
Model-agnostic phrasing on purpose: an Opus-only list reads as inapplicable to
a Sonnet 5 user who just hit the 400.

**Fast-mode rate limit.** Fast mode draws on a rate-limit pool separate from
standard Opus, so a fast-mode run can 429 while standard capacity sits idle.
The retry loop tracks `last_status: int | None` alongside `last_error`, **reset
to `None` in the `httpx.TransportError` branch** so a 429-then-transport-error
sequence does not emit rate-limit guidance for a connection failure. When
retries are exhausted with `last_status == 429` and `speed="fast"`, the
terminal error appends: `fix: fast mode has its own rate limit; retry later or
drop -P speed=fast to fall back to standard speed`.

No automatic fallback to standard speed on 429: silently downgrading changes
the billing rate and the latency profile mid-eval and produces an eval log
whose `speed` field no longer describes what ran.

### What the eval log records

`AgentPolicyConfig` gains `speed: str | None` and `max_output_tokens: int |
None` (the resolved value on this wire, `None` elsewhere); the existing `wire`
field carries the new value. The config is frozen with all-defaulted fields
(`policy.py:75-91`), so appending is safe, and `_html.py:555` renders
`sorted(policy_config.items())`, so new keys render with no viewer change.

`speed` records intent, not what was served. The response's `usage.speed`
reports which speed actually ran, and a fast-mode request served at standard
speed is a 2x billing difference the log cannot currently distinguish. Left as
a documented gap rather than half-solved with an unused public attribute; see
Non-goals.

## Version and changelog

Plugin `0.12.0` → `0.13.0` (`plugins/inspect-robots-agent/pyproject.toml:7`).
`AnthropicClient` joins `__all__` for parity with `ResponsesClient`
(`__init__.py:34,44`), inserted after `"AgentPolicyConfig"` to keep the list
sorted.

`tests/test_package.py` breaks on both the version pin (line 9) and the exact
`__all__` pin (lines 11-23); that pin exists because a stale hardcoded version
shipped through two releases. No other existing test changes: `effort`
validation is untouched, so `test_policy_e2e.py:879` and
`test_responses.py:534` keep passing.

`CHANGELOG.md` gains an entry under `## [Unreleased]` → `### Added`, matching
the `wire=responses` line already at `CHANGELOG.md:55`.

Also update `__init__.py:1-20`, whose package docstring still describes the
plugin as speaking only OpenAI-compatible APIs.

## Tests (`tests/test_anthropic.py` + small additions)

All against `httpx.MockTransport`, mirroring `test_llm.py` and
`test_responses.py`. Plugin coverage is report-only (`ci.yml:326-331` runs with
`--cov-fail-under=0`) and plugin tests are **not** type-checked (root
`[tool.mypy] files` covers the core's `tests/`, not this one), so this list is
the gate rather than a safety net under one.

New fixtures, since nothing existing emits Anthropic-shaped payloads
(`test_policy_e2e.py`'s `_Script` and response builders are chat-shaped):
`_anthropic_response(*blocks, stop_reason="tool_use")` building a `content`
array, and `_AnthropicScript` mirroring `_Script`
(`test_policy_e2e.py:130-140`) for multi-turn policy tests. A `_client(...)`
helper that does what `test_llm.py:225` does plus accepts a `Provider`
override, as `test_llm.py:273` constructs by hand, so the empty-api-key case is
constructible.

Translation:

- Each table row, including multi-part observation content with a PNG data URL
  (asserting prefix stripping and `media_type`), a malformed image URL raising,
  and an unknown content part type raising.
- System hoisting: index-0 system message becomes top-level `system` with no
  `role: "system"` message on the wire; a history with no system message omits
  the `system` key; a system message at index >= 1 raises.
- Merge rule: three tool calls produce exactly one user message with three
  `tool_result` blocks **in history order**, none of them carrying `is_error`;
  the interleaved case (only consecutive runs merge); and a history *ending* in
  a tool run, the shape on the wire every time `act()` retries after a tool
  error (`policy.py:305-313`).
- Both consecutive-user-message sequences, asserted on the full `messages`
  array across a multi-turn trial via `_AnthropicScript`, not per-message.
- `json.loads(arguments)` unparseable and parseable-but-non-object as separate
  cases.
- Assistant turn with `content: None` and no tool calls emits no message;
  assistant turn with `content: ""` **and** tool calls emits no empty text
  block.

Thinking replay:

- Turn 1 returns `[thinking, text, tool_use]`; the turn-2 request contains all
  three verbatim, text appearing exactly once.
- Two `tool_use` blocks in one turn: each cached block appears exactly once in
  turn 2 (the hit is per assistant message, keyed on the first call id).
- Cache prune on a fresh history, including the tool-message `tool_call_id`
  harvest path; and cache-miss synthesis.
- No terminal response (refusal, max_tokens) populates the cache.

Request shape:

- `max_tokens` present with the resolved value (asserting `16000` by default);
  `thinking: {"type": "adaptive"}` always present; `tools=[]` omits `tools`;
  `output_config.effort` only when effort is set; `temperature` only when set;
  `speed` only when set; path is `{base_url}/messages`; model id is the
  prefix-stripped bare form.
- Headers: `x-api-key` and `anthropic-version` on every request;
  `anthropic-beta: fast-mode-2026-02-01` present with `speed="fast"` and absent
  without it; empty api_key omits `x-api-key`.

Parsing and terminal states:

- Text-only, tool-use-only, text + tool-use, text across multiple blocks, a
  zero-length text block normalizing to `None`, `thinking` ignored, `input`
  round-tripping to JSON text.
- `refusal` raises with the category, one request on the wire; `refusal` with
  `stop_details: null` raises cleanly.
- `max_tokens` raises naming `-P max_output_tokens=`, and a truncated response
  containing a partial `tool_use` raises rather than returning that call.
- An unrecognized `stop_reason` raises naming it verbatim, with
  `model_context_window_exceeded` and `pause_turn` as the two named cases (the
  first is the live one motivating default-deny; the second is the resumable
  reason a future server-tool PR must revisit).

Errors and retries:

- OpenRouter-fallback `ConfigError`; an explicit `base_url` suppresses it.
- `api_key_env` defaulting to `ANTHROPIC_API_KEY` **with `base_url` set**
  (asserted on the outgoing `x-api-key` header), an explicit `-P api_key_env=`
  overriding it, and `AgentPolicyConfig.api_key_env` recording the user's value.
- Fast-mode 4xx guidance, and the negative case: a 4xx unrelated to speed while
  `speed="fast"` gets no guidance.
- `temperature` 4xx guidance.
- 429 terminal message with `speed="fast"` (guided) and without it (unguided);
  429-then-transport-error emitting no rate-limit guidance; and 429-then-5xx
  likewise, which catches a `last_status` set only on 429 rather than on every
  response.
- Retry/fail-fast parity with `ChatClient`, plus `close()`.

Policy wiring:

- `wire="anthropic"` end-to-end through `LLMAgentPolicy.act()`.
- Raises: `speed="fast"` with `wire="chat"`; invalid `speed`;
  `max_output_tokens` set with `wire="chat"`; `max_output_tokens=0`; invalid
  `wire` still raises the existing guided `ValueError`.
- `wire`, `speed`, and the resolved `max_output_tokens` recorded in config,
  including `None` on a `wire=chat` run.

## Docs

`plugins/inspect-robots-agent/README.md`:

- Add `speed` and `max_output_tokens` to the knobs list (lines 100-102).
- Replace the two-sentence wire paragraph (lines 48-50) with a small wire
  table, matching the provider table at lines 38-46.
- Update the provider table row at line 40, which says
  `Anthropic (OpenAI-compat)` and after this PR serves both wires.
- Add a "Fast mode on Claude" section beside "Reasoning effort on OpenAI
  models" (starts line 124): the `anthropic/<id>` model form, the
  Claude-API-only caveat, supported models, the keep-effort-at-`high`-or-below
  note, and cost described as roughly double the standard price (it is double
  on both input and output) with a link to Anthropic's pricing page. No
  figures: no README here carries a price, and a stale one would ship to PyPI
  on every release.

Repo writing style applies (CLAUDE.md "Writing style"): no em dashes in prose,
bold only for `**term:**` lead-ins, no decorative emoji.

No `docs/` guide change. Wire selection stays plugin-README territory, as it
did for `wire=responses`.

## Retry-loop duplication

Three clients each carry a near-identical retry loop. 0022 justified the
duplication at two as "small enough"; that premise is already false, since
`_llm.py:207-228` and `_responses.py:70-88` differ in their guided-error block,
and this client adds three guidance branches plus `last_status`.

Keep it duplicated anyway, for a better reason: extraction needs at least two
injection points (4xx guidance, terminal message), so a base class buys
template-method indirection rather than deduplication. The only genuinely
shared logic is "is this status retryable" plus the backoff sleep, three lines.
Revisit at a fourth wire, and do not inherit the "small enough" claim, which
will not survive it.

## Non-goals

- Streaming, and therefore `effort` of `xhigh`/`max` on this wire.
- Prompt caching (`cache_control` on the system block and tool schemas). The
  stable prefix is large and reused every turn, so this is the obvious
  follow-up, with its own placement and invalidation design.
- Surfacing response `usage`, including served `speed`, into the trial record.
  Worth doing, but a public attribute whose only consumer is one assertion is
  worse than a documented gap.
- `retry-after`-aware backoff. Correct, but it needs a parse rule (delta-seconds
  vs HTTP-date), a status scope, and a sleep-duration assertion pattern this
  suite does not have.
- Adding a field to `Provider`. The wire is expressible with the three it
  already has, and provenance does not belong on a value object.
- Server-side refusal `fallbacks`. This wire's target set carries elevated
  safety classifiers, so the question will come up, but the answer is the same
  one that rules out 429 auto-downgrade: silently serving from a different
  model produces an eval log that no longer describes what ran.
- Per-wire `effort` validation, and with it 0022's `effort=max` hole. Separate
  concern, separate PR.
- Adaptive-thinking display (`thinking: {"display": "summarized"}`) and thinking
  summaries in the transcript.
- Task budgets (`output_config.task_budget`) and the Batches API.
- Bedrock/Vertex/Foundry variants: fast mode is Claude API only, so they would
  inherit the wire without its motivating feature.
