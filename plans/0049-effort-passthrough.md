# Effort passthrough Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** `-P effort=` becomes what-you-type-is-what-the-model-gets. Today an
unset flag injects an explicit `"low"` on the wire, while `-P effort=none` is
coerced by the CLI scalar parser to Python `None`, which the policy interprets
as "omit the field" — so the operator who asked for zero reasoning silently
gets the provider's default. After this plan: an **unset** flag omits the
effort field entirely (provider default, matching how `temperature` already
behaves), `-P effort=none` sends the true minimum (`reasoning_effort: "none"`
on chat, `reasoning: {"effort": "none"}` on responses, and
`thinking: {"type": "disabled"}` on messages — the only faithful "none" the
Anthropic API has), and the named levels are unchanged. Closes #317.

**Architecture:** one normalization point in `LLMAgentPolicy.__init__`
(`plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py:376-387`):
a received `effort=None` — reachable only via the CLI's `none`/`null`
coercion or a deliberate programmatic call — normalizes to the string
`"none"` before validation; `_UNSET` resolves to `None`, which every wire
client already treats as "omit the field". Downstream, the only wire that
needs a code change is messages: `AnthropicClient.complete`
(`_anthropic.py:117-141`) maps `reasoning_effort == "none"` to
`thinking: {"type": "disabled"}` with no `output_config`, and keeps
`thinking: {"type": "adaptive"}` for `None` and the named levels (explicit
adaptive stays: omitting `thinking` means thinking-off on Opus 4.8/4.7,
which is a level choice, not a default). The chat and responses clients
already omit on `None` and pass strings through untouched. `AgentPolicyConfig
.effort` defaults to `None` and records the *normalized* value, so the eval
log shows the truth. The per-run constancy of effort means a
thinking-disabled run never has signed thinking blocks in the replay cache,
so no replay changes are needed.

**Breaking change (behavior, not API):** scripts that omit `-P effort=`
today run at `low`; after this they run at the provider default (for
Inkling that is documented as *high*). The changelog and README call this
out with the one-line fix: add `-P effort=low` to pin the old behavior.
Programmatic callers passing `effort=None` previously meant "omit"; they now
mean the level `"none"`. Interaction: PR #315 (`feat/continuous-effort`)
touches the same validation block; whichever lands second rebases.

**Tech stack:** stdlib only, all inside `plugins/inspect-robots-agent`.
pytest with the existing fake-client patterns in
`plugins/inspect-robots-agent/tests/`.

## Global Constraints

- Gates (all blocking), run from the worktree root. The root mypy/pytest
  configs do NOT cover the plugin (root `testpaths=["tests"]`, coverage
  `source=["inspect_robots"]`), so the plugin gates must be the CI commands
  (`.github/workflows/ci.yml`, agent-plugin job; equivalent modulo
  --no-sync/-q flags):
  - `uv run --no-sync ruff check plugins/inspect-robots-agent` and
    `uv run --no-sync ruff format --check plugins/inspect-robots-agent`.
  - `uv run mypy --config-file plugins/inspect-robots-agent/pyproject.toml plugins/inspect-robots-agent/src`
  - `uv run --no-sync pytest plugins/inspect-robots-agent/tests -q` with the
    CI coverage flags **including `--cov-fail-under=0`** (plugin coverage is
    report-only per `.github/workflows/ci.yml:325-331` — there is no 100%
    plugin gate, and omitting the flag would wrongly inherit the root
    `fail_under = 100`, which is scoped to `source=["inspect_robots"]`).
- No changes outside `plugins/inspect-robots-agent/` except
  `CHANGELOG.md` and this plan file. In particular the core CLI parser
  (`src/inspect_robots/defaults.py::_parse_value`) is deliberately untouched:
  its `none → None` coercion is load-bearing for every other `-E`/`-P`
  surface (YAM config documents literal `none` as "library default").
- The capx plugin (`plugins/inspect-robots-capx/src/inspect_robots_capx/
  policy.py:120`) duplicates the old semantics. Out of scope here — it lacks
  the `_Unset` marker so unset and explicit `None` are indistinguishable —
  but the follow-up issue must be filed when this PR opens.

## Task 1: Constructor normalization + resolution

**Files:** `policy.py`, `tests/test_policy_e2e.py`,
`tests/test_gemini_live.py`

- [ ] Tests first, in `test_policy_e2e.py` (extend the existing
  `test_effort_defaults_low_and_is_tunable` cluster at ~:2473, renaming it to
  match the new contract, e.g. `test_effort_passthrough_matrix`):
  - unset (`effort` kwarg not passed) → chat body has **no**
    `reasoning_effort` key AND `logs[0].eval.policy_config["effort"] is None`.
  - `effort=None` (programmatic; what the CLI delivers for `-P effort=none`)
    → `reasoning_effort == "none"` on the wire AND
    `policy_config["effort"] == "none"`.
  - `effort="none"` (quoted escape hatch, must keep working) → same as above.
  - `effort="high"` → unchanged explicit passthrough.
  - invalid value (`effort="turbo"`) → `ConfigError` whose message names the
    accepted levels and says "omit -P effort= to use the provider default"
    (the old "or None to omit the field" wording must be gone).
  - `wire="gemini-live"` with `effort=None` → `ConfigError` (today `None`
    slips through because the guard tests `effort is not None`); with
    `effort="low"` → still `ConfigError`; with effort unset → constructs fine.
- [ ] Rewrite the two existing tests this contract change breaks, both in
  `tests/test_gemini_live.py`:
  - `test_per_wire_default_resolution_and_explicit_none` (~:766): its
    gemini-live `effort=None` construction (~:773-781) now raises — that
    branch becomes an explicit `ConfigError` assertion; its
    `config.effort == "low"` assertions for chat/responses/anthropic
    (~:808-810) become `is None`. Fold whatever it still covers into the
    passthrough matrix rather than duplicating it — with one exception: its
    per-wire `image_horizon` default assertions (~:803-810, live → `None`,
    HTTP wires → `2`) cover behavior this plan does not touch and must
    survive the rewrite in `test_gemini_live.py`, not move files.
  - `test_existing_invalid_effort_and_horizon_messages_are_preserved`
    (~:834): pins the old error string ("or None to omit the field",
    ~:843-846); update the expected wording to the new message.
- [ ] Implement in `policy.py`:
  - Normalize before validation:
    `effort_level = "none" if effort is None else effort` for non-`_Unset`
    input; validate membership on the normalized value.
  - `resolved_effort: str | None = None` when unset (all wires — the
    `wire == "gemini-live"` special case collapses into the general rule).
  - gemini-live guard becomes `if wire == "gemini-live" and effort is not
    _UNSET` (explicit `none` is still an effort request the Live wire cannot
    honor).
  - Update the stale comment at ~:604 ("default to low reasoning effort") to
    describe passthrough, and the `AgentPolicyConfig.effort` field: default
    `None`, docstring "resolved effort level; `None` means the field is
    omitted and the provider default applies".
  - The chat-wire 4xx hint at `_llm.py:292-296` ("switch to /v1/responses …
    or -P effort=none") stays **verbatim**: its trigger requires the server
    text to name both `reasoning_effort` and `/v1/responses` (the GPT-5.x
    tools rejection), and under the new semantics `-P effort=none` sends the
    literal `"none"` that endpoint asks for — the hint becomes *more*
    correct, not stale. Do not "fix" it; `test_llm.py:528-539` pins it.
- [ ] Gates pass; commit.

## Task 2: Messages wire — `"none"` → thinking disabled

**Files:** `_anthropic.py`, `tests/test_anthropic.py`

- [ ] Tests first, in `test_anthropic.py` (fake-transport `_client` pattern,
  see ~:111):
  - `reasoning_effort="none"` → body has `thinking == {"type": "disabled"}`
    and **no** `output_config` key.
  - `reasoning_effort="low"` → `thinking == {"type": "adaptive"}` and
    `output_config == {"effort": "low"}` (existing assertion at ~:125 stays).
  - `reasoning_effort=None` → adaptive, no `output_config` (provider
    default).
  - The effort-4xx guidance test (~:750) updates: the hint must name
    `minimal` as the OpenAI-only value and must no longer claim `none` is
    unsupported (it now maps to thinking-disabled client-side).
  - `speed="fast"` combined with `reasoning_effort="none"` → body carries
    both `speed: "fast"` and `thinking: {"type": "disabled"}`. Decided
    stance: **allow and pass through** — the plugin's philosophy after this
    plan is passthrough, client-side combo guards contradict it, and if the
    server rejects the pairing the existing `_rejection_guidance` speed
    branch (~:215) already explains fast-mode constraints. The test pins the
    passthrough so the stance is explicit.
- [ ] Implement in `AnthropicClient.complete`: branch on
  `reasoning_effort == "none"` when building `body` — set
  `thinking: {"type": "disabled"}`, skip `output_config`; otherwise keep the
  current adaptive + optional `output_config` behavior. Rewrite the inline
  always-adaptive rationale comment at ~:120-123 (the module docstring does
  not mention thinking and needs no change) and the `_rejection_guidance`
  effort wording at ~:226-230.
- [ ] Gates pass; commit.

## Task 3: Docs, changelog, version

**Files:** `plugins/inspect-robots-agent/README.md`,
`plugins/inspect-robots-agent/pyproject.toml`, `CHANGELOG.md`

- [ ] README: rewrite the effort paragraph (~:311-321): unset = provider
  default (parallel to `temperature`), `-P effort=none` now sends the true
  minimum with the per-wire mapping table, the `-P effort="'none'"` quoting
  escape hatch is no longer needed but still valid, and "add `-P effort=low`
  to pin the pre-0.23 behavior". Update the Inkling section (~:370-376):
  unset now inherits Inkling's own default (high per Tinker's cookbook) — the
  latency warning moves here. Update the GPT-5.x section (~:444-459): the
  `-P effort=none` advice there becomes literally correct; drop the quoting
  caveat. **Rewrite ~:420-421** ("This wire always requests adaptive
  thinking… Use `-P wire=chat`" for pre-4.6 models): after Task 2 the wire
  requests adaptive only when effort is not `"none"`, and `-P effort=none`
  makes pre-4.6 models usable on `wire=messages`. Check the xhigh/max
  sentence (~:426) still reads correctly now that `none` is legal on
  `wire=messages`.
- [ ] `pyproject.toml`: version 0.22.0 → 0.23.0.
- [ ] `CHANGELOG.md`: **Changed** entry under Unreleased, "Agent plugin
  (0.23.0)", covering: unset effort passes through to the provider default
  (breaking; pin with `-P effort=low`), `effort=none` sends true none on all
  HTTP wires including `thinking: disabled` on messages, `None` normalizes
  to `"none"`, and the gemini-live `effort=none` loophole closes
  ([plan 0049](plans/0049-effort-passthrough.md), #317).
- [ ] Gates pass (ruff format touches README code fences if any); commit.

## Out of scope

- Core CLI parser changes (`_parse_value`) — the coercion stays; the policy
  absorbs it. Its docstring example (`src/inspect_robots/defaults.py:50-56`,
  "sends the wire string none instead of omitting the parameter") goes stale
  for this flag; recorded as a core follow-up on #317, not edited here.
- Core log-view effort extractor (`src/inspect_robots/_html.py:626-636`)
  reads `reasoning_effort`/`reasoning.effort`/`output_config.effort` only, so
  an `effort=none` messages run renders as effort "n/a" in the wire view;
  same core follow-up on #317.
- capx plugin parity — follow-up issue #319, filed.
- Any run-header/console printing of resolved effort — the value already
  lands in `EvalSpec.policy_config` in the eval log; surfacing it in the CLI
  banner is core-repo work and a separate discussion.
- Fractional efforts (PR #315) — orthogonal; rebase coordination only.
