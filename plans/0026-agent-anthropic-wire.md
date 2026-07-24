# 0026 — `-P wire=anthropic`: native Messages API unlocks fast mode

Issue: #165. Status: draft (revised after critique round 1).

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

The native wire also unlocks `output_config.effort` in its own spelling
(`low` through `max`) and correct `stop_reason: "refusal"` surfacing, which the
compat shim flattens into an ordinary empty completion.

Scope note: fast mode is Claude API only. Not Bedrock, Vertex, Foundry, or
Claude Platform on AWS, and not usable with the Batch API or Priority Tier.
Opus 4.7 fast mode was removed and now errors, so this wire does not resurrect
it.

## Design

Third wire, one new client class, zero changes to the conversation loop. This
is deliberately the same shape as plan 0022 (`wire=responses`): the
chat-completions message format stays the single source of truth for
`_messages`, the transcript, sanitization, and echo; the new client translates
at the wire boundary only. Same no-SDK doctrine: httpx, raw JSON.

The one place this plan is *not* a thin mirror of 0022 is opaque-block replay
(see "The thinking-block constraint"). 0022 needed a raw-item cache for
OpenAI reasoning items; this wire needs the same mechanism for Claude thinking
blocks, for the same underlying reason.

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
forwards for free.

`speed="fast"` with `wire != "anthropic"` is a construction-time `ValueError`,
not a silent no-op. The compat shim accepts unknown body fields by ignoring
them, so a silently-dropped `speed` would bill at standard rates while the
user believes fast mode is on.

No model-id gating. Fast mode's supported set moves (4.7 was removed, 5 was
added) and a hardcoded allowlist would reject a model the API accepts.
Unsupported models get the API's own 400, wrapped by the guided error below.

### `max_output_tokens` param (forced by the wire)

`max_tokens` is **required** on `/v1/messages` and has no server default,
unlike both OpenAI wires where omitting it is legal.

Named `max_output_tokens`, not `max_tokens`: sitting next to `max_llm_calls`
(a per-trial budget) and `max_speed_frac`, a bare `max_tokens` reads as a
trial-wide token budget. It is a per-response cap. The name also matches the
Responses spelling, so forwarding it on the other two wires later is an
extension rather than a rename.

`LLMAgentPolicy(max_output_tokens: int | None = None)`, validated `>= 1` when
set. `None` means "use the wire's default": `AnthropicClient` applies
`_DEFAULT_MAX_OUTPUT_TOKENS = 16000` internally, and the other two wires ignore
the field entirely.

Recorded in `AgentPolicyConfig` as the **effective** value — the resolved
number on `wire=anthropic`, and `None` on the other two wires, where nothing
constrained the output. Logging `16000` for a `wire=chat` run would assert a
limit that did not exist.

Why 16000 and not 4096: the visible output of a turn is one tool call plus a
one-or-two-sentence `note`, on the order of 100-200 tokens. Everything above
that is headroom for adaptive thinking, which is on by default on Opus 5 and
bills against this same ceiling. There is no published number for how much
adaptive thinking will spend, and this wire permits effort up to `max`, so the
default is set by the repo-wide guidance for non-streaming requests (~16000)
rather than by the visible-output estimate. Truncation is then a hard error
rather than a silent degradation (below), so a too-small value is loud.

### `AnthropicClient` (new module `_anthropic.py`)

```python
AnthropicClient(
    provider: Provider,
    *,
    speed: str | None = None,
    max_output_tokens: int | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    backoff_s: float = 1.0,
    transport: httpx.BaseTransport | None = None,
)
```

`speed` and `max_output_tokens` are **constructor**-injected. `complete()`
keeps the exact existing signature —
`complete(messages, tools, temperature, reasoning_effort) -> AssistantMessage`
— so `act()` (`policy.py:268-273`) is untouched and the
`ChatClient | ResponsesClient | AnthropicClient` union at `policy.py:136`
stays structurally compatible under mypy strict. `close()` is unchanged.
`policy.py`'s changes stay confined to `__init__` and the config dataclass.

`Provider`, `ToolCall`, and `AssistantMessage` are shared vocabulary from
`_llm.py`.

Headers: `x-api-key` (not `Authorization: Bearer` — the native API's own
scheme), `anthropic-version: 2023-06-01`, and `anthropic-beta:
fast-mode-2026-02-01` only when `speed="fast"`. The beta header is what makes
`POST /v1/messages` the beta endpoint over raw HTTP; there is no separate URL.
An empty `provider.api_key` omits the `x-api-key` header entirely, matching
`ChatClient` (`_llm.py:182-184`) so keyless local proxies still work.

Request body: `model`, `max_tokens`, `messages` (translated), and when
applicable `system` (hoisted), `tools` (flattened), `output_config:
{"effort": ...}`, `speed`.

`temperature` is **never sent**. Claude Opus 5, Opus 4.8, and Opus 4.7 reject
`temperature`, `top_p`, and `top_k` with a 400, and those are exactly the
models this wire exists to reach. `temperature is not None` with
`wire="anthropic"` is a construction-time `ValueError` naming the affected
models, matching the `speed`-on-wrong-wire precedent. The policy still accepts
`-P temperature=` for the other two wires.

### Effort: a per-wire accepted set

Replace the wire-agnostic check at `policy.py:128-134` with a table:

```python
_ACCEPTED_EFFORTS = {
    "chat": _EFFORT_LEVELS,
    "responses": _EFFORT_LEVELS - {"max"},
    "anthropic": _EFFORT_LEVELS - {"none", "minimal"},
}
```

Four lines, no special case, and it closes 0022's own known-ugly hole in the
same PR: 0022 §"One effort caveat" documented that `wire=responses` +
`effort=max` fails with OpenAI's 400 and accepted it rather than guarding.
Leaving two inconsistent validation policies side by side is worse than fixing
both.

The honest justification for guarding at all — the earlier draft's reasoning
was wrong and is retracted. Since this client never sends `thinking`, an
out-of-set effort would produce a plain 400, not the silent tool-call-as-text
degradation that disabled thinking causes on Opus 5. So the "guard silent
degradation, not 400s" principle does not apply here. What does apply: a 400
surfaces only after `bind()` and the first observation, so it burns a trial and
costs a round trip, and `-P effort=none` is the documented GPT-5.x workaround
sitting in this plugin's own README (line 117) that users will paste into a
Claude run. Failing at construction is worth four lines.

`effort=None` (the Python `None`, distinct from the wire string `"none"`)
omits `output_config` entirely. `thinking` is never sent in any configuration:
adaptive is the Opus 5 default and the right one here.

### The thinking-block constraint

Adaptive thinking is on by default on Opus 5, so a typical assistant turn
returns `[thinking, text?, tool_use]`. Thinking blocks must be echoed back
**unchanged** when the conversation continues on the same model; replaying a
history with them stripped triggers ordering and signature 400s from the
second turn onward. `AssistantMessage` (`_llm.py:140-159`) carries only
`content` and `tool_calls`, so it is structurally lossy for opaque blocks.

This is the same problem 0022 solved for OpenAI reasoning items, and it gets
the same mechanism. `AnthropicClient` keeps
`_raw_blocks_by_tool_use_id: dict[str, list[dict[str, Any]]]`:

- After each successfully *parsed* response (never a failed or raising one),
  the response's verbatim `content` array is stored keyed by the `id` of each
  `tool_use` block in it.
- During translation, an assistant message whose first `tool_call.id` hits the
  cache emits the cached raw blocks in place of **all** synthesized blocks for
  that message, text and tool calls alike. Emitting both would show the model
  its own text twice.
- A miss falls back to synthesis — correct for histories this client instance
  did not produce, and for non-thinking models.
- Entries whose `tool_use_id` no longer appears in the submitted history are
  pruned at the top of `complete()`, **before** the new response is cached
  (prune-after-store would evict every fresh entry and make the cache always
  miss). Ids are harvested from both places history carries them: assistant
  `tool_calls[i].id` and tool-message `tool_call_id`.
- A `reset()` (fresh `_messages`) empties the cache without the client needing
  a reset hook.

Accepted loss, inherited from 0022: text-only assistant turns (the nudge retry
path) cache nothing, so their thinking blocks are dropped from replay. Those
turns carry no `tool_use`, which is what the pairing constraint attaches to.

### Translation: chat messages → Messages API

| Chat form | Messages API |
| --- | --- |
| `{"role": "system", "content": "<str>"}` at index 0 | hoisted to the top-level `system` string, not a message. Omitted entirely when `_messages[0]` is not a system message (`act()` is reachable without `reset()`; the only guard is `bind()`) |
| a system message at index >= 1 | `RuntimeError` rather than a silent drop |
| `{"role": "user"/"assistant", "content": "<str>"}` | same role, `content` kept as the plain string |
| user content part `{"type": "text", ...}` | `{"type": "text", "text": ...}` unchanged |
| user content part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,B"}}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "B"}}`. `_png.png_data_url` always emits PNG, so the media type is fixed rather than parsed; a URL not matching the expected prefix raises rather than shipping a malformed block |
| assistant msg `content` (non-empty) | `{"type": "text", "text": ...}` block first |
| assistant msg `tool_calls[i]` | `{"type": "tool_use", "id": id, "name": ..., "input": json.loads(arguments)}` after the text block. Unparseable JSON and parseable-but-non-object both raise `RuntimeError` naming the tool: a `tool_use` with a non-object `input` 400s with a message that does not identify the offending call |
| assistant msg with a cache hit on `tool_calls[0].id` | the cached raw block list verbatim, replacing every synthesized block for that message |
| `{"role": "tool", "tool_call_id": id, "content": c}` | `{"type": "tool_result", "tool_use_id": id, "content": c}` inside a **user** message; consecutive tool messages merge into one (below) |
| assistant msg, no tool calls, content `None` or `""` | no message. The nudge retry path appends exactly this, and an assistant message with an empty content array is a 400 |

Merging consecutive tool messages: `act()` appends one `role: "tool"` message
per ignored extra tool call (`policy.py:292-303`) and then one for the executed
call (`policy.py:304-307`), so a multi-tool-call turn produces a run of
consecutive tool messages. Splitting `tool_result` blocks across several user
messages does not 400 (the API coalesces consecutive same-role messages), but
it does silently train the model to stop making parallel calls, so the run
collapses into one user message carrying one `tool_result` block each.

Block order within the merged message is **history order**, which for a
multi-call turn is `[extra_1..extra_n, executed]` while the assistant's
`tool_use` order was `[executed, extra_1..extra_n]`. Blocks are matched by
`tool_use_id`, not position, so this is legal; it is called out here only
because the mismatch looks like a bug to a reader diffing the two arrays.

Two consecutive-user-message sequences fall out of the loop and are legal
under same-role coalescing, but are easy to miss and both get goldens:

- **Dropped assistant turn then nudge.** An assistant turn with no tool calls
  and empty content emits no message, so the wire sees the observation user
  message immediately followed by the nudge user message.
- **Tool results then next observation.** After a successful call `act()`
  returns; the next `act()` appends an observation, so a user message of
  `tool_result` blocks is immediately followed by a user message of text and
  images.

Tool schemas flatten from the chat nesting:
`{"type": "function", "function": {name, description, parameters}}` →
`{"name": ..., "description": ..., "input_schema": ...}`. `Toolset.schemas()`
stays chat-shaped; the client owns the translation both ways. No strict-mode
concern: the Messages API does not auto-normalize schemas the way the Responses
API does, so the move tool's free-form `targets`/`deltas` object survives.
`tools=[]` omits the `tools` key entirely, matching `ResponsesClient`.

### Response parsing → `AssistantMessage`

From `response["content"]`: concatenate the `text` of every `text` block, and
emit one `ToolCall(id=block["id"], name=block["name"],
arguments=json.dumps(block["input"]))` per `tool_use` block, in order.
`thinking` blocks contribute nothing here — they matter only to the cache.

Empty concatenated text normalizes to `None`, never `""`. A `""` would
round-trip through `raw()` (`_llm.py:149`) into the next request as an empty
`text` block, which is a 400.

Re-serializing `input` back to JSON text keeps `ToolCall.arguments` a string
across all three wires, so `_tools.py` validation, the transcript format, and
the echo path are untouched.

Two terminal stop reasons raise `RuntimeError` and are never retried, matching
0022's `status: "failed"` decision — the server deliberately answered:

- `stop_reason: "refusal"`, carrying `stop_details.category` when present.
  `stop_details` is informational and may be `null` even on a refusal, so the
  branch keys on `stop_reason` alone.
- `stop_reason: "max_tokens"`, carrying `fix: raise -P max_output_tokens=`.
  A truncated turn is an HTTP 200 with partial content, and it is the one
  silent-degradation risk this PR newly creates: either there is no `tool_use`
  block (three nudge turns, then a generic `RuntimeError` at `policy.py:286`
  with the cause discarded), or there is a partial `tool_use` whose `input`
  still re-serializes to valid JSON and gets executed as a motion by
  `toolset.execute()` (`policy.py:304`). Truncated responses never yield tool
  calls.

Neither terminal response populates the cache.

### Guided errors

House style: the error names the fix, never just the failure.

**OpenRouter fallback.** `resolve_provider`'s ladder falls through to
OpenRouter whenever `ANTHROPIC_API_KEY` is unset, and
`openrouter.ai/api/v1/messages` does not exist. `Provider` gains a public
`from_openrouter_fallback: bool` field set by `resolve_provider`, so `policy.py`
does not reach into `_llm`'s privates to compare base URLs. Construction raises
`ConfigError` when `wire="anthropic"` and that flag is set, naming the two
fixes (set `$ANTHROPIC_API_KEY`, or pass `-P base_url=` for a gateway serving
the Messages API).

This guard covers the fallback case only, deliberately. `-P model=openai/gpt-5
-P wire=anthropic` with `$OPENAI_API_KEY` set resolves to `api.openai.com/v1`
and dead-ends at a 404 — enumerating every wrong-endpoint combination is not
worth it, and the 404 names the URL.

**Key resolution with an explicit `base_url`.** `resolve_provider` defaults
`api_key_env` to `OPENROUTER_API_KEY` (`_llm.py:109-111`), so
`-P base_url=https://gateway/v1 -P wire=anthropic` would send an OpenRouter key
as `x-api-key` and 401 with no hint. For `wire="anthropic"`, `policy.py`
defaults `api_key_env` to `ANTHROPIC_API_KEY` before calling
`resolve_provider`. An explicit `-P api_key_env=` still wins.

**Fast mode unsupported.** A 4xx while `speed="fast"` whose body contains both
`"speed"` and `"fast"` appends: `fix: fast mode needs Claude Opus 5 or Opus 4.8
on the Claude API (not Bedrock/Vertex/Foundry), or drop -P speed=fast`. The
two-token AND mirrors 0022's matcher shape (`_llm.py:220`); a bare `fast`
substring is too weak a signal.

**Fast-mode rate limit.** Fast mode draws on a rate-limit pool separate from
standard Opus, so a fast-mode run can 429 while standard capacity is idle. The
retry loop tracks `last_status: int | None` alongside `last_error`; when
retries are exhausted with `last_status == 429` and `speed="fast"`, the
terminal error appends: `fix: fast mode has its own rate limit; retry later or
drop -P speed=fast to fall back to standard speed`. One extra variable, and it
saves a user staring at a 429 while their standard-speed quota sits unused.

**429 backoff honors `retry-after`.** When the response carries a parseable
`retry-after`, sleep that instead of `backoff_s * 2**attempt`, capped at
`timeout_s`. Inherited fixed backoff ignores the server's own answer.

No automatic fallback to standard speed on 429. Silently downgrading changes
the billing rate and the latency profile mid-eval and produces an eval log
whose `speed` field no longer describes what ran.

### What the eval log records

`AgentPolicyConfig` gains `speed: str | None`, `max_output_tokens: int | None`
(effective value, see above), and the existing `wire` field carries the new
value. The config is frozen with all-defaulted fields (`policy.py:75-91`), so
appending is safe; `_html.py:555` renders `sorted(policy_config.items())`, so
new keys render with no viewer change.

Requested `speed` records intent, not what was served. The response's
`usage.speed` reports which speed actually ran, and a fast-mode request that
silently served at standard speed is a 2x billing difference the log should
not hide. `AnthropicClient` exposes the last response's `usage` (including
`speed`, `input_tokens`, `output_tokens`) as a plain attribute; wiring it into
the trial record is left to a follow-up, but the fast-mode test asserts
`usage.speed` is surfaced.

## Version

Plugin `0.12.0` → `0.13.0` (`plugins/inspect-robots-agent/pyproject.toml:7`).
`AnthropicClient` joins the package's `__all__` for parity with
`ResponsesClient` (`__init__.py:34,44`). Both changes break
`tests/test_package.py` (the version pin at line 9 and the exact `__all__` pin
at lines 11-23), which is updated in the same PR — that pin exists because a
stale hardcoded version shipped twice.

## Tests (`tests/test_anthropic.py` + small additions)

All against `httpx.MockTransport`, mirroring `test_llm.py` and
`test_responses.py` conventions. Plugin coverage is report-only
(`ci.yml:326-331` runs with `--cov-fail-under=0`), so this list is the gate,
not a safety net under one.

Translation:

- Each row of the table above, including multi-part observation content with a
  PNG data URL (asserting the `data:` prefix is stripped and `media_type` is
  set), and a malformed image URL raising.
- System hoisting: index-0 system message becomes top-level `system` with no
  `role: "system"` message on the wire; a history with **no** system message
  omits the `system` key; a system message at index >= 1 raises.
- The merge rule: three tool calls produce exactly one user message with three
  `tool_result` blocks; the interleaved case (only consecutive runs merge); and
  a history **ending** in a tool run, which is the shape on the wire every time
  `act()` retries after a tool error (`policy.py:305-313`).
- Both consecutive-user-message sequences, asserted on the full `messages`
  array across a multi-turn trial, not per-message.
- `json.loads(arguments)` unparseable and parseable-but-non-object as two
  separate cases.
- Assistant turn with `content: None` and no tool calls emits no message;
  assistant turn with `content: ""` **and** tool calls emits no empty text
  block.

Thinking replay:

- Turn 1 returns `[thinking, text, tool_use]`; the turn-2 request contains all
  three verbatim, with the text appearing exactly once.
- Two `tool_use` blocks in one turn: each cached block appears exactly once in
  turn 2 (the hit is per assistant message, keyed on the first call id).
- Cache prune on a fresh history, and cache-miss synthesis.
- Neither a `refusal` nor a `max_tokens` response populates the cache.

Request shape:

- `max_tokens` always present; `tools=[]` omits the `tools` key;
  `output_config.effort` only when effort is set; `speed` only when set;
  `temperature` never present; path is `{base_url}/messages`.
- Headers: `x-api-key` and `anthropic-version` on every request;
  `anthropic-beta: fast-mode-2026-02-01` present with `speed="fast"` and absent
  without it; empty api_key omits `x-api-key`.

Parsing and terminal states:

- Text-only, tool-use-only, text + tool-use, text across multiple blocks,
  `thinking` blocks ignored, `input` round-tripping to JSON text.
- `refusal` raises with the category, one request on the wire; `refusal` with
  `stop_details: null` raises without an AttributeError.
- `max_tokens` raises naming `-P max_output_tokens=`, and a truncated response
  containing a partial `tool_use` raises rather than returning that call.
- `usage.speed` surfaced on a fast-mode response.

Errors and retries:

- OpenRouter-fallback `ConfigError`; an explicit `base_url` suppresses it.
- `api_key_env` defaulting to `ANTHROPIC_API_KEY` for this wire, and an
  explicit `-P api_key_env=` overriding it.
- Fast-mode 4xx guidance; and the negative case — a 4xx unrelated to speed
  while `speed="fast"` gets no guidance.
- 429 terminal message with `speed="fast"` (guided) and without it (unguided).
- `retry-after` honored when present, fixed backoff when absent.
- Retry/fail-fast parity with `ChatClient`, plus `close()`.

Policy wiring:

- `wire="anthropic"` end-to-end through `LLMAgentPolicy.act()` (mock
  embodiment).
- Raises: `speed="fast"` with `wire="chat"`; invalid `speed`;
  `effort="none"`/`"minimal"` with `wire="anthropic"`; `effort="max"` with
  `wire="responses"` (the newly closed 0022 hole); `temperature` set with
  `wire="anthropic"`; `max_output_tokens=0`.
- `wire`, `speed`, and the effective `max_output_tokens` recorded in the
  config, including `None` on a `wire=chat` run.

## Docs

`plugins/inspect-robots-agent/README.md`:

- Add `speed` and `max_output_tokens` to the knobs list (line 101).
- Replace the two-sentence wire paragraph (lines 48-50) with a small wire
  table, matching the provider table already at lines 38-46.
- Add a "Fast mode on Claude" section next to the existing "Reasoning effort on
  OpenAI models" section, with the Claude-API-only caveat and the supported
  models. Describe the cost as roughly double the standard output price and
  link Anthropic's pricing page rather than printing figures: no README here
  carries a price, and a stale one would ship to PyPI on every release.

Repo writing style applies to all of it (CLAUDE.md "Writing style"): no em
dashes in prose, bold only for `**term:**` lead-ins, no decorative emoji.

No `docs/` guide change. Wire selection stays plugin-README territory, as it
did for `wire=responses`.

## Retry-loop duplication

Three clients will each carry a near-identical retry loop. 0022 justified the
duplication at two as "small enough"; that premise is already false, since
`_llm.py:207-228` and `_responses.py:70-88` differ in their guided-error block,
and this client adds two more guidance branches plus `last_status`.

Keep it duplicated anyway, for a better reason: extraction would need at least
two injection points (4xx guidance, terminal message), so a base class buys
template-method indirection rather than deduplication. The only genuinely
shared logic is "is this status retryable" plus the backoff sleep, which is
three lines. Revisit at a fourth wire, and do not inherit the "small enough"
claim, which will not survive it.

## Non-goals

- Prompt caching (`cache_control` breakpoints on the system block and tool
  schemas). The stable prefix here is large and reused every turn, so this is
  the obvious follow-up, but it is a cost optimization with its own placement
  and invalidation design.
- Wiring `usage` into the trial record beyond exposing it on the client.
- Adaptive-thinking display (`thinking: {"display": "summarized"}`) and
  surfacing thinking summaries into the transcript.
- Task budgets (`output_config.task_budget`) and the Batches API.
- Bedrock/Vertex/Foundry variants: fast mode is Claude API only, so they would
  inherit the wire without its motivating feature.
