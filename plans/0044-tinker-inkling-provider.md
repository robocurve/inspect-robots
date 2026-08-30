# 0044 — Inkling on Tinker: `thinkingmachines/` provider and `wire=messages`

Issue: #278. Revised after critiques R1 (8 findings: ENV_MODEL wire
inference, the :451 fix ladder, capx blast radius, the hint's explicitness
signal, and four smaller), R2 (7 findings: key-hint leak into
default-`native_wires` errors, :451 branch conditions, conflict-guard text
vs the OpenRouter-routed config, inference no-op pinning, guard ordering vs
gemini-live, hint rationale, `speed` probe), R3 (7 findings:
`resolve_provider` docstring inventory, falsified README prose inventory,
and five polish items), and R4 (6 findings: ladder-test key
preconditions, two more README lines, and four inventory nits); all
resolved below.

## Problem

Tinker (Thinking Machines) now serves Inkling and Inkling-Small
(`thinkingmachines/Inkling`, `thinkingmachines/Inkling-Small`: hybrid
reasoners, vision, 64K context) via serverless inference. The agent plugin
already drives them end to end — verified live on 2026-08-03, both models
solved CubePick 1/1 through the Messages wire — but the working configuration
is three flags of boilerplate, and the obvious wrong configuration fails in
the worst possible way.

### Verified facts (live probes, 2026-08-03)

All probes used the plugin's exact request shapes (`_anthropic.py` bodies,
`_png.png_data_url` frames).

- **Anthropic-compatible endpoint**
  (`https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1`,
  auth via `x-api-key`): tool use (`tool_use` blocks, `stop_reason=tool_use`,
  valid JSON args), base64 PNG image inputs (solid-green sanity check answered
  "Green"), `thinking: {"type": "adaptive"}` accepted, thinking blocks
  returned and their verbatim multi-turn replay accepted across a 14-turn
  episode, system-block `cache_control` accepted (real cache reads served,
  despite docs saying caching is unsupported), `max_tokens` accepted at the
  plugin default 16000 and beyond (200000 probed).
- **Effort**: `output_config.effort` accepts exactly `low`, `medium`, `high`,
  `xhigh`, `max` — the same five values `_anthropic.py` sends. `none` and
  `minimal` are a 422 whose body contains "effort", so the existing
  `_rejection_guidance` branch already names the fix. Effort genuinely
  modulates thinking volume (141 thinking chars at `low` vs 426 at `max` on
  the same prompt).
- **Model ids**: the endpoint expects the **full** id
  (`thinkingmachines/Inkling`). Base-model names work on both compatible
  APIs; `tinker://` checkpoint paths also work (out of scope here).
- **`speed: "fast"` + the fast-mode beta header are silently ignored**
  (probed 2026-08-03: HTTP 200, normal-speed response, no error) — so a
  misdirected `-P speed=fast` neither fails nor helps on Tinker.
- **OpenAI-compatible endpoint** (`.../oai/api/v1`): no image support, and
  `tools` are **silently ignored** — HTTP 200 with the motion described in
  prose, no `tool_calls`. On `wire=chat` the policy burns
  `_MAX_CONSECUTIVE_FAILURES` (3) turns and dies at `policy.py:778` with the
  unguided `LLM produced no tool call in 3 consecutive turns`.
- **Usage quirk**: Tinker reports `input_tokens: 0`; input shows up as
  `cache_creation_input_tokens`/`cache_read_input_tokens`. Cosmetic
  (transcript `in=0`, EvalLog input-token stats read zero); document, don't
  code around.

### The two UX defects

1. `wire=anthropic` names a vendor, but the value selects a protocol — the
   Messages API — which now has multiple servers (Anthropic, Tinker,
   gateways). The other wire names are endpoint-shaped (`chat` →
   /chat/completions, `responses` → /responses); this one should be too.
2. Supporting a provider should not require users to carry
   `-P wire= -P base_url= -P api_key_env=` on every invocation. Every other
   supported provider is one model prefix plus one env var.

## Design

Agent plugin only; no core changes. Four parts.

### D1: `wire=messages`, with `anthropic` as a permanent alias

- `_WIRE_FORMATS` becomes `{"chat", "responses", "messages", "gemini-live"}`.
  A new `_WIRE_ALIASES = {"anthropic": "messages"}` is applied **before**
  membership validation, so `-P wire=anthropic` keeps working forever and an
  unknown value's `ConfigError` lists the canonical names.
- Every wire comparison against `"anthropic"` in `policy.py` (lines 369
  (negated guard), 432, 451, 513, 518 today) switches to `"messages"`. Error and fix strings that spell
  `wire=anthropic` / `-P wire=anthropic` (all in `policy.py` and
  `__init__.py`; `_anthropic.py` has none) say `wire=messages`.
- `AgentPolicyConfig.wire` records the **canonical** value: a run launched
  with `-P wire=anthropic` logs `wire: "messages"`. Old EvalLogs are
  immutable and stay as written; nothing reads the stored wire back for
  control flow. The `AgentPolicyConfig.wire` docstring names the alias.
- Prose that teaches the old name is part of this rename, not incidental:
  the package docstring (`__init__.py:6`, rendered into the generated API
  docs) and the `AgentPolicyConfig.max_output_tokens` field comment
  ("Effective per-response cap on ``wire=anthropic``", policy.py:216).
  The `_EFFORT_LEVELS` comment block (policy.py:72-77) is touched for a
  different reason: its "xhigh/max need a cap this client cannot stream"
  claim becomes Anthropic-endpoint-specific once Tinker shares the wire
  (Tinker returned 200 for xhigh/max at a 200000 cap, no streaming).

### D2: provider-native wires and the `thinkingmachines/` entry

`_llm.py` changes:

- `_DirectProvider` gains two fields with defaults that leave the existing
  eight entries (seven providers; `x-ai`/`xai` are one) untouched:
  `wire: str = "chat"` and `keep_prefix: bool = False`. The table comment
  "The prefix is stripped: these endpoints want the bare model id" is
  updated for `keep_prefix`, and so is the `_ANTHROPIC_BASE` comment
  ("The only endpoint that serves /v1/messages without an explicit
  base_url", policy.py:60), which the Tinker entry falsifies.
- `resolve_provider`'s **docstring** is public-API contract rendered into
  the generated API docs and is falsified twice: step 2's provider
  enumeration + "prefix stripped from the model id" (keep-prefix entries
  keep it), and the new `native_wires` kwarg needs its contract stated
  (which wires the caller can speak; entries with other native wires do
  not claim and the ladder falls through). Two more public docstrings the
  entry falsifies join the inventory: `Provider`'s ("A resolved
  OpenAI-compatible endpoint", `_llm.py:75` — no longer necessarily
  OpenAI-compatible, and it grows the `wire` field) and the `_llm.py`
  module docstring (lines 1-7, which enumerates chat-only coverage).
- The `speed`/`max_output_tokens` guard comment at policy.py:370-372
  ("Both fail loudly rather than silently") is scoped to what it still
  guarantees: with `messages` inferred for Tinker, `-P speed=fast` passes
  construction and Tinker ignores it server-side (probed); the comment and
  the README caveat must tell the same story.
- New entry:

  ```python
  "thinkingmachines": _DirectProvider(
      "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1",
      "TINKER_API_KEY",
      wire="messages",
      keep_prefix=True,
  ),
  ```

- The claim condition in `resolve_provider` (prefix in table + bare model
  non-empty + no OpenRouter `:variant` suffix + key env set) is extracted
  into a single helper, `_direct_claim(model, env, native_wires) -> tuple[str, _DirectProvider] | None`,
  used by both `resolve_provider` and the new wire inference below — one
  predicate, no drift. On a successful claim, the resolved model id is the
  full id when `keep_prefix` is set, else the bare id as today.
- **`resolve_provider` stays back-compatible for its other consumers.**
  `resolve_provider` and `Provider` are public plugin API, and
  inspect-robots-capx calls them with clients that speak only
  OpenAI-compatible wires; silently rerouting its `thinkingmachines/*`
  resolution from OpenRouter to a Messages-only base URL would trade a
  working config for an unguided hard failure. So `resolve_provider` (and
  `_direct_claim`) gain a keyword `native_wires: frozenset[str] = frozenset({"chat"})`:
  a table entry claims the model only when its wire is in the set,
  otherwise the ladder falls through to OpenRouter exactly as today. The
  agent policy passes `frozenset({"chat", "messages"})`; capx and any
  out-of-tree caller keep the default and see zero routing change. No capx
  edits, no capx release.
- `_provider_key_hints()` takes the same `native_wires` and only names
  entries claimable under it. Otherwise resolve_provider's no-key error
  under the default set would tell a capx user with `thinkingmachines/*`
  to `set $TINKER_API_KEY` — advice that cannot work there, since the
  default set declines the claim and the same error would recur.
- `Provider` gains `wire: str = "chat"`, recording the claimed entry's
  native wire. The explicit-`base_url` and OpenRouter paths leave it at
  `"chat"` (meaning: no provider opinion), so nothing downstream changes for
  them.

`policy.py` constructor changes (order per plan 0026: wire is resolved and
validated before the wire-only params, key-env defaulting before resolution,
endpoint checks after):

- The `wire` parameter becomes `str | _Unset = _UNSET` (the module's existing
  marker type; `agent_policy()` and the CLI pass strings through unchanged).
  Resolution:
  - explicit value → alias-normalize, validate against `_WIRE_FORMATS`;
  - `_UNSET` → `_direct_claim(requested_model, environ, ...)`'s native wire
    when the claim succeeds and no `base_url` was passed, else `"chat"`. An
    explicit `base_url` bypasses the provider table today and keeps doing
    so — no implied wire.
- **Inference operates on the same model string resolution uses.** The
  model can arrive via `$INSPECT_ROBOTS_MODEL`
  (`requested_model = model or environ.get(ENV_MODEL)`, policy.py:436,
  currently computed *after* wire validation). The `environ` and
  `requested_model` computations are hoisted above wire resolution;
  otherwise an env-supplied `thinkingmachines/Inkling` would infer
  `wire="chat"` while `resolve_provider` claims Tinker's Messages base URL,
  recreating exactly the silent chat-against-Messages failure this plan
  exists to kill. Hoisting is safe: neither computation depends on
  anything the moved-past checks establish.
- `AgentPolicyConfig.wire`'s dataclass default stays `"chat"` (the effective
  default for every model that doesn't claim a native-wire provider).
- **Conflict guard** (construction-time, immediately after provider
  resolution and **before** the endpoint checks at policy.py:451 and :499,
  so it wins over the gemini-live "needs Google's direct Live API provider"
  error for claiming models; the :499 guard stays reachable for
  non-claiming ones): when the claim succeeds, the entry's wire is not
  `"chat"`, and an explicit wire disagrees, raise a guided `ConfigError`:

  ```
  wire='chat' cannot drive thinkingmachines/* — the provider's direct
  endpoint serves only the Messages API.
  fix: drop -P wire= (thinkingmachines/* defaults to wire=messages), or
  pass -P base_url=... (+ -P api_key_env=NAME) to route this wire through
  a gateway such as OpenRouter deliberately
  ```

  The gateway half of the fix line appears only for `chat`/`responses`;
  `gemini-live` gets just the drop-`-P wire=` half (no gateway serves the
  Live protocol, and policy.py:414 requires a ws:// base_url there anyway).

  The text names the endpoint mismatch, not Tinker's OAI limitations: with
  both `TINKER_API_KEY` and `OPENROUTER_API_KEY` set, this exact config
  routed through OpenRouter and *worked* until now, so the message must not
  describe a failure the blocked config never had, and the fix line must
  include the gateway escape (`-P base_url=https://openrouter.ai/api/v1
  -P api_key_env=OPENROUTER_API_KEY`). The explicit-`base_url` escape hatch
  is untouched: users who want Tinker's OAI endpoint on `wire=chat` can
  still point at it and own the outcome (see D3).
- The Messages-endpoint check at policy.py:451 ("only Anthropic's own
  endpoint serves /v1/messages") learns about provider-native wires in both
  directions:
  - successful claim → skip the check (`provider.wire == "messages"`);
  - failed claim on a native-messages prefix → the fix-string ladder
    generalizes rather than gaining a bolt-on branch. The existing branches
    were built to never name a fix the user already has right, and the new
    prefix inherits that care:
    - "Messages-capable prefixes" is a defined set, not vibes:
      `{"anthropic"} ∪ {prefixes whose table entry has wire == "messages"}`.
      It cannot be derived from `entry.wire` alone because the `anthropic`
      entry's wire deliberately stays `"chat"` (its Messages service comes
      from `wire=messages` selecting the endpoint, not from a claim);
    - the empty-id branch (policy.py:464, today `removeprefix("anthropic/")`
      only) strips any Messages-capable prefix, so bare `thinkingmachines/`
      gets the full-command fix, not key advice for an unusable id;
    - the foreign-prefix branch (policy.py:469-474) treats Messages-capable
      prefixes as domestic — its comment "A foreign prefix can never
      resolve to the Messages API" is false once a second such prefix
      exists;
    - the `:variant` branch keeps precedence: `thinkingmachines/Inkling:free`
      with `TINKER_API_KEY` set gets "drop the OpenRouter variant suffix" —
      post-resolution, the ladder never names a key the user already set
      (the ladder only runs when `resolve_provider` succeeded, i.e. the
      OpenRouter fallback fired; without any usable key the error is
      `resolve_provider`'s own no-key raise, which legitimately names keys);
    - only then, a usable un-suffixed `thinkingmachines/*` id with the key
      env actually unset → `fix: set $TINKER_API_KEY` (spelled from the
      entry's `key_env`, not hardcoded).

  The `api_key_env` defaulting at line 432 (`wire=messages` + explicit
  `base_url` → `ANTHROPIC_API_KEY`) is rename-only.

Routing notes for the changelog (`### Changed`), stated as behavior changes,
not precedent: with both `TINKER_API_KEY` and `OPENROUTER_API_KEY` set,
`thinkingmachines/*` with `wire` unset previously fell through to OpenRouter
on `wire=chat` and now resolves direct to Tinker on `wire=messages`; and an
**explicit** `-P wire=chat` with that key pair, which previously worked via
OpenRouter, is now the construction-time `ConfigError` above (the gateway
escape re-enables it deliberately).

### D3: guided hint on the silent-tool-drop failure

The `policy.py:778` raise (`LLM produced no tool call in N consecutive
turns`) appends, only when `wire == "chat"` **and** an explicit `base_url`
was configured:

```
note: some OpenAI-compatible endpoints accept `tools` but silently ignore
them (Tinker's OpenAI-compatible API is one). If the provider serves the
Messages API, retry with -P wire=messages and its Messages base_url.
```

Not on first-party chat providers, and not on the `policy.py:931` raise,
which is a different failure (tool calls present but invalid). The gating
is about cost, not diagnosis: on the explicit-`base_url` path (gateways,
local vLLM/Ollama) a silent tool drop is *possible* and the hedged "some
endpoints" wording earns its place; on first-party endpoints it is not, so
the note would be pure misdirection. A local-model streak that is really
model behavior still reads the note and loses nothing.

The "explicit `base_url`" bit is not derivable at the raise site today:
`AgentPolicyConfig.base_url` stores the *resolved* `provider.base_url`
(policy.py:549), which is always non-None. The constructor stashes the
explicitness as a private attribute (`self._explicit_base_url = bool(base_url)`);
the config schema is unchanged.

### D4: documentation

Public-facing text follows the repo writing-style rules (no em dashes in
prose, no mid-sentence bold, headers use colons).

- **Plugin README**: new "Inkling on Tinker" section — the two-flag
  invocation, effort semantics (plugin default `low`; Inkling's own default
  is high per Tinker's thinking-effort cookbook page, so pass `-P effort=`
  deliberately; `none`/`minimal` rejected by the endpoint — the default
  claim is doc-sourced, not probe-sourced, and is attributed as such), the
  zero-input-tokens usage quirk, the beta
  caveat, a note that `-P speed=fast` is Claude-on-Claude-API only and is
  **silently ignored** by Tinker (probed: HTTP 200, normal speed — and with
  messages inferred it no longer trips the construction-time wire check, so
  nothing else will flag it), and a pointer that
  `:peft:262144` extended-context ids are untested. Wires table row renamed
  to `messages` with the `anthropic` alias noted; provider/key table gains
  the `thinkingmachines/*` / `TINKER_API_KEY` row and the `anthropic/*`
  row's "native with `-P wire=anthropic`" (README.md:40) takes the rename;
  the "Fast mode on Claude" section's examples switch to `-P wire=messages`.
  Existing prose the feature falsifies is part of the edit, not collateral:
  the key-ladder step "the provider's own endpoint, prefix stripped from
  the model id" (README.md:29) learns about keep-prefix entries; the
  configuration-knobs paragraph "`speed` and `max_output_tokens` apply to
  `-P wire=anthropic` only" (README.md:245) takes the rename; and the
  fast-mode section's opening "`-P wire=anthropic` drives Claude through
  the native Messages API" (README.md:341), its "Only Anthropic's own
  endpoint serves /v1/messages, so anything that resolves elsewhere is
  refused up front", and "The model id keeps the `anthropic/` prefix on
  this wire" (README.md:351-353) are rewritten for the second
  Messages-native provider.
- **capx README** (repo copy only, no capx code change or release): one
  qualifier line where it says routing "follow[s] the inspect-robots-agent
  plugin" (capx README.md:58-59) — capx keeps chat-only routing, so
  `thinkingmachines/*` resolves via OpenRouter there by design. Publishes
  whenever capx next bumps.
- **Root README**: the fast-mode example at line 142 switches to
  `-P wire=messages`.
- **docs/guide/logging-and-rerun.md:166**: "on the Anthropic wire" becomes
  "on the Messages wire" (the one prose use of the old name under `docs/`;
  released-version CHANGELOG history stays as written).
- **CHANGELOG**: agent-plugin entries split by class — the new provider,
  alias, and guards under `### Added`; the D2 routing note (direct-to-Tinker
  precedence over OpenRouter when both keys are set) under `### Changed`.

### Version

`plugins/inspect-robots-agent/pyproject.toml` `0.20.0` → `0.21.0` (new
provider, new accepted wire value; fully backward compatible). Publishes via
the existing `publish-agent` job on the next core release.

## Tests

All in `plugins/inspect-robots-agent/tests/`, following the existing
MockTransport / stub patterns; no live-API tests (beta service, secret key).

- `test_llm.py`: `thinkingmachines/Inkling` + `TINKER_API_KEY` claims direct
  (with `native_wires` including `"messages"`) with the full id preserved
  and `Provider.wire == "messages"`; the **default** `native_wires` skips
  the entry and routes to OpenRouter (the capx-compat contract); without the
  key it falls through to OpenRouter as before; explicit `base_url` bypasses
  the entry; `:peft:262144` suffix (a colon that is not an OpenRouter
  variant) still claims direct. `test_guided_error_names_the_new_provider_keys`
  (test_llm.py:183) updates for `$TINKER_API_KEY` in `_provider_key_hints()`
  **under the agent's wire set**, plus the inverse: the default
  `native_wires` hint string omits `$TINKER_API_KEY`.
- `test_policy_e2e.py` / construction tests:
  - `wire` unset + `thinkingmachines/*` + key → `AnthropicClient` selected,
    Tinker base URL, config records `wire="messages"`.
  - `wire` unset + model supplied only via `$INSPECT_ROBOTS_MODEL`
    (`thinkingmachines/*`) + key → same outcome as the explicit-model case.
  - `wire` unset + `anthropic/claude-*` + `ANTHROPIC_API_KEY` →
    `ChatClient` on the compat endpoint, config records `wire="chat"` (pins
    that no existing entry grows a native wire; the whole back-compat story
    rests on this).
  - `wire=messages` + `thinkingmachines/*` + `OPENROUTER_API_KEY` but no
    `TINKER_API_KEY` → `ConfigError` whose fix says `set $TINKER_API_KEY`.
  - `wire=messages` + `thinkingmachines/Inkling:free` with **both**
    `TINKER_API_KEY` and `OPENROUTER_API_KEY` set (resolution must succeed
    via OpenRouter for the `:451` ladder to run at all; every existing
    ladder test sets both keys for this reason, test_anthropic.py:1060) →
    the variant-suffix fix, not key advice; bare `thinkingmachines/` under
    the same both-keys env → the full-command fix.
  - `-P wire=anthropic` (any provider) → accepted, config records
    `"messages"`, Messages client selected.
  - explicit `wire=chat` / `responses` / `gemini-live` +
    `thinkingmachines/*` + key → `ConfigError` with the guided text.
  - unknown wire value → `ConfigError` listing the canonical set.
  - 3-strikes raise: hint present with `wire=chat` + explicit `base_url`;
    absent without `base_url`; absent on the invalid-tool-call raise.
  - existing `wire="anthropic"` string literals in tests migrate to
    `"messages"` except the ones that exist to prove the alias.
- Wire-only param errors (`speed`, `max_output_tokens`) name
  `wire=messages` in their fix strings.

## Out of scope

- Core changes of any kind; `inspect_robots.__all__` is untouched.
- Tinker's OAI-compatible endpoint, `tinker://` checkpoint ids, the 256K
  `:peft` variants (untested), audio input.
- A live canary against Tinker (paid beta endpoint; would be flaky noise).
