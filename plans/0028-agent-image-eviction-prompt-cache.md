# Agent Image Eviction + Prompt Caching Implementation Plan

> **For agentic workers:** Implement task-by-task; each task carries its own
> test cycle. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound the agent policy's per-request payload (fixes the HTTP 413 that
kills long episodes) and cut its quadratic token cost via Anthropic prompt
caching, with per-trial usage accounting to verify both.

**Architecture:** `LLMAgentPolicy` keeps its full chat history canonical but
sends the wire client a per-call *view* in which camera frames older than the
last `image_horizon` image-bearing messages are replaced by deterministic text
stubs — each message is rewritten exactly once when it ages out, so the prompt
prefix stays byte-stable and cacheable. The Anthropic wire client adds
`cache_control` breakpoints (system prompt, newest-stubbed anchor message,
final message) and surfaces the response `usage` block, which the policy
aggregates per trial into `record.metadata["llm_usage"]`.

**Tech Stack:** Python 3.10–3.13, httpx (no provider SDKs — repo rule),
pytest + mock transports (existing patterns in
`plugins/inspect-robots-agent/tests/`), ruff, strict mypy.

## Global Constraints

- Repo gates: `ruff check .`, `ruff format --check .`, strict `mypy` over src
  **and** tests, `pytest --cov` (plugin has its own coverage scope; keep it at
  its current gate — add tests for every new branch).
- Ruff D1: docstring on every public module/class/function — state the
  contract, not the symbol name.
- No provider SDKs in `plugins/inspect-robots-agent` — hand-built request
  bodies over httpx only.
- Every construction-time validation failure raises
  `inspect_robots.errors.ConfigError` with a `fix:` line (repo issue #168
  convention).
- Chat-format history stays canonical (plan 0026); wire clients translate.
- The stored history (`self._messages`), `transcript()`, `transcript_delta()`,
  and the frames side-car must be unaffected by eviction — eviction is
  view-only.
- Deterministic bytes everywhere in the outgoing view: no timestamps, no
  counters that change retroactively, no dict-ordering hazards.
- Context: issue #188; failed run `adhoc_3539bb48` (2026-07-27): 86 calls,
  264 accumulated frames ≈ 36 MB request → HTTP 413 `request_too_large`.

---

## File Structure

| File | Change |
|---|---|
| `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py` | `image_horizon` config + validation; `_evicted_view()`; call-site wiring; usage aggregation; `record.metadata["llm_usage"]` |
| `plugins/inspect-robots-agent/src/inspect_robots_agent/_anthropic.py` | `cache_control` on system / anchor / final blocks; parse `usage`; copy-on-write for replay-cache blocks |
| `plugins/inspect-robots-agent/src/inspect_robots_agent/_llm.py` | `AssistantMessage.usage` field; `ChatClient.complete` ignores new kwarg |
| `plugins/inspect-robots-agent/src/inspect_robots_agent/_responses.py` | `ResponsesClient.complete` ignores new kwarg |
| `plugins/inspect-robots-agent/tests/test_policy_eviction.py` | New: view-builder unit tests + eviction e2e + prefix-stability test |
| `plugins/inspect-robots-agent/tests/test_anthropic.py` | Extend: cache_control placement, usage parsing, replay-block immutability |
| `plugins/inspect-robots-agent/tests/test_policy_e2e.py` | Extend: usage metadata lands in `TrialRecord.metadata` |
| `plugins/inspect-robots-agent/README.md` | Document `-P image_horizon=`, caching behavior, `llm_usage` metadata |
| `CHANGELOG.md` | Entry under Unreleased |

---

### Task 1: `_evicted_view()` + `image_horizon` config (policy.py)

**Files:**
- Modify: `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py`
- Test: `plugins/inspect-robots-agent/tests/test_policy_eviction.py` (new)

**Interfaces:**
- Produces:
  ```python
  def _evicted_view(
      messages: list[dict[str, Any]],
      horizon: int,
      *,
      mark_anchor: bool = False,
  ) -> list[dict[str, Any]]
  ```
  Returns a new list. Messages whose `content` list contains `image_url`
  parts are "image-bearing". The last `horizon` image-bearing messages pass
  through untouched (same object, not copied). Every older image-bearing
  message is replaced by a shallow-copied dict whose content list has all
  camera parts (each `image_url` part *and* the label `text` part immediately
  preceding it) removed and one stub appended:
  `{"type": "text", "text": f"[{n} camera frame(s) elided]"}` where `n` is
  the count of removed `image_url` parts. When `mark_anchor=True` and at
  least one message was stubbed, the *newest* stubbed message dict also gets
  `"cache_anchor": True` (top-level key on the message dict, consumed by the
  Anthropic wire in Task 2).
- Produces: `LLMAgentPolicy(image_horizon: int | None = 2, ...)` and
  `AgentPolicyConfig.image_horizon: int | None = 2`. `None` disables eviction
  (send full history — pre-change behavior).

**Notes for the implementer:**
- Default is `2`: with 3 cameras that keeps ≤ 6 frames in flight, bounding
  request bodies at well under 1 MB versus the 32 MB ceiling.
- Validation mirrors `max_output_tokens`: reject `bool`, non-`int`, and
  `< 1` with `ConfigError("image_horizon must be an int >= 1, or None to "
  "send full image history")`. The CLI forwards `-P image_horizon=2` through
  the same coercion path as `max_output_tokens`.
- Untouched messages must be the *same objects* (identity), not copies —
  the prefix-stability test asserts this, and it is what makes the
  serialized prefix byte-stable across calls.
- The stub text contains no step number and no timestamp — determinism.
- Image-bearing counting covers both attachment paths: `images="always"`
  observation messages and the `on_demand` immediate-frames user messages
  appended after `take_pic` (both are `role: "user"` with `image_url` parts;
  the rule is shape-based, not mode-based).
- Call site in `act()`: replace the direct `self._client.complete(self._messages, ...)`
  with:
  ```python
  outgoing = self._messages
  if self._image_horizon is not None:
      outgoing = _evicted_view(
          self._messages,
          self._image_horizon,
          mark_anchor=isinstance(self._client, AnthropicClient),
      )
  message = self._client.complete(
      outgoing,
      toolset.schemas(),
      temperature=self._temperature,
      reasoning_effort=self._effort,
  )
  ```
  The view is rebuilt on every iteration of the inner `while True` loop (a
  nudge retry appends messages), not once per `act()`.

- [ ] **Step 1: Write failing unit tests** in
  `tests/test_policy_eviction.py` covering: (a) fewer image-bearing messages
  than horizon → same list contents, zero copies; (b) one aged-out
  observation message → label+image parts removed, stub text
  `"[3 camera frame(s) elided]"` appended after the state text part, original
  `self._messages` object unmodified; (c) kept messages are identical objects
  (`is`); (d) `mark_anchor=True` sets `cache_anchor` only on the newest
  stubbed message; (e) on_demand-style image-only message stubs to a single
  text part; (f) non-list content (plain string user messages) passes
  through untouched.
- [ ] **Step 2: Run tests, verify they fail** (`_evicted_view` undefined).
- [ ] **Step 3: Implement `_evicted_view` and the config plumbing**
  (`__init__` param + validation + `AgentPolicyConfig` field + `act()` call
  site).
- [ ] **Step 4: Run the new tests and the full plugin suite** —
  `uv run pytest plugins/inspect-robots-agent/tests/ -q`; all green.
- [ ] **Step 5: Write the prefix-stability test**: drive a policy through 4
  `act()` cycles with a mock transport (existing e2e fixtures), capture each
  outgoing request body, and assert byte-identical shared prefix: request
  k+1's serialized `messages[:boundary]` equals request k's, where
  `boundary` is everything up to request k's final two messages.
- [ ] **Step 6: Run, verify green, commit**
  `feat(agent): bound outgoing image history with image_horizon view eviction`

### Task 2: `cache_control` breakpoints (Anthropic wire)

**Files:**
- Modify: `plugins/inspect-robots-agent/src/inspect_robots_agent/_anthropic.py`
- Test: `plugins/inspect-robots-agent/tests/test_anthropic.py`

**Interfaces:**
- Consumes: `"cache_anchor": True` marker from Task 1.
- Produces: request bodies where
  1. `body["system"]` is a block list:
     `[{"type": "text", "text": <system>, "cache_control": {"type": "ephemeral"}}]`
     (unchanged hoisting logic; only the final shape changes),
  2. the translated turn for a marker-carrying message has
     `cache_control: {"type": "ephemeral"}` on its **last** content block,
  3. the last content block of the **final** translated message likewise.

**Notes for the implementer:**
- Max 4 breakpoints per request; this design uses exactly 3.
- Why the anchor: the eviction boundary is the only point where the prefix
  changes between calls. A breakpoint there chains to the previous call's
  anchor (~one observation cycle ≈ ~10 blocks back, inside Anthropic's
  20-block lookback), so the whole stable prefix reads from cache at 0.1×
  while only the ~`horizon`-message tail re-prefills.
- `_translate_messages` never forwards unknown keys (it builds fresh dicts),
  so consuming `message.get("cache_anchor")` needs no popping and nothing
  leaks onto the wire.
- **Copy-on-write:** blocks fetched from `self._raw_blocks_by_tool_use_id`
  (the thinking-replay cache) are stored verbatim for replay. If the final
  message is an assistant replay turn, adding `cache_control` in place would
  poison the stored copy and grow a second `cache_control` next call. Apply
  breakpoints via `{**block, "cache_control": {...}}` on a copied list, never
  by mutating.
- Anchor and final can coincide (rare; anchor is never the last message in
  practice because an observation follows every eviction) — if they do, one
  block carries one `cache_control`, not two; guard with an identity check.
- No breakpoint on tools: the system breakpoint caches tools+system together
  (tools render before system).
- Other wires: `ChatClient.complete` and `ResponsesClient.complete` gain no
  behavior — OpenAI-compatible gateways cache prefixes automatically.

- [ ] **Step 1: Write failing tests** extending `test_anthropic.py` (reuse
  its mock-transport request-capture pattern): (a) system hoisting now emits
  the block-list shape with `cache_control`; (b) a message carrying
  `cache_anchor: True` yields a translated turn whose last block has
  `cache_control` and the marker key itself does not appear anywhere in the
  serialized body; (c) the final translated message's last block has
  `cache_control`; (d) a replayed assistant turn as final message: the dict
  stored in `_raw_blocks_by_tool_use_id` is unmodified after `complete()`
  (no `cache_control` key), and the *next* request contains exactly one
  `cache_control` on that turn; (e) exactly 3 `cache_control` occurrences in
  a representative request.
- [ ] **Step 2: Run, verify failures.**
- [ ] **Step 3: Implement** in `complete()` + `_translate_messages`.
- [ ] **Step 4: Full plugin suite green.**
- [ ] **Step 5: Commit**
  `feat(agent): anthropic prompt-cache breakpoints on system/anchor/final`

### Task 3: Usage accounting

**Files:**
- Modify: `plugins/inspect-robots-agent/src/inspect_robots_agent/_llm.py`
  (`AssistantMessage`), `_anthropic.py` (`_parse_response`), `policy.py`
  (aggregation + metadata), `_responses.py` (kwarg parity only if Task 1's
  call-site change requires none — see note)
- Test: `plugins/inspect-robots-agent/tests/test_anthropic.py`,
  `plugins/inspect-robots-agent/tests/test_policy_e2e.py`

**Interfaces:**
- Produces: `AssistantMessage.usage: dict[str, int] | None = None` — new
  optional field, **excluded from `raw()`** (raw() round-trips into history;
  usage must not).
- Produces: `record.metadata["llm_usage"]` per trial, e.g.
  `{"llm_calls": 86, "input_tokens": 1690000, "output_tokens": 8100,
  "cache_creation_input_tokens": ..., "cache_read_input_tokens": ...}` —
  keys summed from each response's `usage` (only int-valued keys), plus
  `llm_calls` counted locally. Written in `on_trial_end` alongside the
  existing `record.metadata["transcript"]`; omitted when no calls were made.
- Note: only the Anthropic wire populates `usage` in this change; the chat
  and responses wires leave it `None` and the aggregate then contains only
  `llm_calls`. (Their payloads carry usage too — deliberately out of scope.)

**Notes for the implementer:**
- Reset accumulation in `reset()` (per-trial), same as `_calls_used`.
- With `transcript_echo`, emit one line per call:
  `[agent] -- usage: in=<input_tokens> cache_read=<cache_read_input_tokens> out=<output_tokens>`
  (keys absent from the response render as 0).
- The acceptance signal for Task 2 lives here: a live run should show
  `cache_read_input_tokens > 0` from call 2 onward.

- [ ] **Step 1: Write failing tests**: (a) `_parse_response` carries the
  payload's `usage` dict through; absent usage → `None`; (b) `raw()` output
  contains no `usage` key; (c) e2e: after a 2-cycle mock run,
  `record.metadata["llm_usage"]["llm_calls"] == 2` and token keys equal the
  mock's sums; (d) `reset()` zeroes the aggregate.
- [ ] **Step 2: Run, verify failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full plugin suite + `mypy` + `ruff` green.**
- [ ] **Step 5: Commit** `feat(agent): per-trial LLM usage accounting in trial metadata`

### Task 4: Docs

**Files:**
- Modify: `plugins/inspect-robots-agent/README.md`, `CHANGELOG.md`

- [ ] **Step 1:** README: add `image_horizon` to the `-P` options table
  (default 2, `None`/unset-via-`-P image_horizon=` semantics, why: request
  bodies otherwise grow ~420 KB per observation and 413 at ~85 observations
  with 3 cameras); a short "Prompt caching" paragraph under the anthropic
  wire section (automatic, 3 breakpoints, verify via `llm_usage`); document
  `record.metadata["llm_usage"]`.
- [ ] **Step 2:** CHANGELOG entry under Unreleased referencing issue #188.
- [ ] **Step 3:** Commit `docs(agent): document image_horizon, prompt caching, llm_usage`

---

## Self-Review Notes

- Spec coverage: 413 fix (Task 1), cost fix (Task 2), verification signal
  (Task 3), docs (Task 4). Issue #188 maps 1:1.
- Type consistency: `_evicted_view` returns `list[dict[str, Any]]` and is
  consumed as such at the `act()` call site; `cache_anchor` is a message-dict
  key produced in Task 1 and consumed in Task 2; `AssistantMessage.usage` is
  produced in Task 3's `_parse_response` and consumed in Task 3's policy
  aggregation.
- Deliberate scope cuts: no usage capture on chat/responses wires; no 1h TTL
  (calls are ~15 s apart, 5 min default suffices); no seconds-based horizon
  (plan 0026-seconds-based-horizons is unrelated).
