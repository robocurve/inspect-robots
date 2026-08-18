# taskgen/grader: effort channel for reasoning models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the agent policy has `-P effort=`, but the auto-task author
(`-A`) and the `vlm` grader (`-G`) have no effort channel (issue #394):
the shared chat wire sends only `model`, `messages`, and the token cap,
so a gpt-5.x author or Gemini grader always runs at provider-default
effort with no per-run control.

**Architecture:** one optional parameter threaded through three layers,
all in core. `chat_completion` gains `effort: str | None = None`; when
truthy it adds `"reasoning_effort": effort` to the request body (both the
`max_tokens` send and the #390 `max_completion_tokens` retry, since the
field lives in `_post_chat`). `generate_scene` and `vlm_grader` gain the
same optional parameter and pass it through. No CLI changes are needed:
`-A k=v` and `-G k=v` reach those functions as kwargs (cli.py:1666-1672
composes `{**defaults.taskgen_args, **parsed -A}` into `generate_scene`,
and `_build_grader` at cli.py:1023-1024 composes `{**config_kvs,
**parsed -G}` into the registry-resolved `vlm_grader`), and the
`[taskgen.args]`/`[grader.args]` config sections pick the key up through
the same composition.

**Tech stack:** stdlib-only `_chatwire` as today; mypy strict; pytest at
100% branch coverage.

**Spec:** issue #394 + this plan (the plan is the spec, per repo
convention).

**Critique:** round 1 (fresh-context subagent) verified the CLI/config
plumbing, registry forwarding, api-snapshot, docs anchors, coverage
reachability, and non-collision with #393, and found 1 major + 1 minor,
both fixed in this revision: the original `effort=none → omit` decision
inverted the agent plugin's documented contract (`-P effort=none` sends
the true minimum since agent 0.23.0; only key-absent omits), so
decisions 3/4 now adopt the same `_Unset`-sentinel + `None → "none"`
normalization for real parity; and falsy non-string parses
(`effort=0`, `effort=false`) are no longer silently swallowed — the
wire serializes non-empty values verbatim and the endpoint's 4xx
rejects them loudly. Nitpicks folded in: the CHANGELOG `### Added`
section already exists (dead fallback removed) and taskgen provenance
metadata now records the sent effort (new decision 4b).

## Global constraints

- Gates: `ruff check .`, `ruff format --check .`, `mypy` (strict,
  src + tests), `pytest --cov` at 100%.
- Unset means absent: when `effort` is not set, the request body must be
  byte-identical to today's (no `reasoning_effort` key at all), so every
  existing endpoint sees unchanged behavior and the provider default
  applies. This is the project's flags-omit-means-provider-default
  stance.
- No client-side value validation: providers disagree on the allowed set
  (`minimal`/`none`/`xhigh` exist on some, not others); the endpoint's
  own 4xx is the accurate guided error and already surfaces through the
  existing non-2xx path.
- Existing tests must pass without edits; a pre-existing test that seems
  to demand modification is a stop-and-flag conflict, not something to
  edit.
- Public API: `generate_scene` and `vlm_grader` are exported —
  adding a keyword-only optional parameter is additive. Check
  `tests/test_api_snapshot.py`: if it pins signatures (not just names),
  update it together with `inspect_robots.__all__` per repo rule; if it
  pins names only, no change is needed there.

## Binding decisions

1. **Wire field and name:** `reasoning_effort`, the OpenAI
   chat-completions parameter, which Gemini's OpenAI-compat endpoint
   also honors as its thinking-budget control. At the wire layer,
   `chat_completion(effort=...)` sends the value verbatim when it is
   neither `None` nor `""`, and omits the key otherwise. Non-string
   parses (`-A effort=0` → `0`, `effort=false` → `False`) are therefore
   serialized onto the wire and rejected loudly by the endpoint's own
   4xx through the existing guided-error path — never silently
   swallowed. (`""` omits: `-G effort=` parses to `""`, and an empty
   string must not put `"reasoning_effort": ""` on the wire.)
2. **Placement:** the field is added in `_post_chat`, so the retry send
   carries it identically to the first send. Body key order:
   `model`, `messages`, token cap, then `reasoning_effort` (appended
   last; tests assert key presence/values, not order).
3. **Signatures:** `chat_completion` gains keyword-only
   `effort: str | float | None = None` with the decision-1 semantics
   (`None`/`""` omit, everything else verbatim). `generate_scene` and
   `vlm_grader` gain keyword-only `effort` defaulting to an `_Unset`
   sentinel (module-private, mirroring the agent plugin's
   `policy.py` `_Unset` pattern): sentinel or `""` → call the wire with
   `effort=None` (omit); key-present `None` → normalize to the string
   `"none"`; any other value passes through verbatim. `_VLMGrader`
   stores the normalized value; `grade` passes it to `chat_completion`.
   `_post_chat` takes the normalized `effort` as a
   positional-after-token_param parameter (private helper, no default
   needed — both callers pass it explicitly).
4. **`effort=none` means minimum, absent means provider default —
   the agent plugin's contract, mirrored for real:** `-A effort=none` /
   `-G effort=none` parse to key-present Python `None`, which decision 3
   normalizes to the wire string `"none"` (minimum reasoning), exactly
   as `-P effort=none` sends the true minimum since agent 0.23.0
   (policy.py's `"none" if effort is None else effort` on its `_Unset`
   sentinel). Only a key that is absent altogether (no flag, no config
   key) omits the field for the provider default. The docs note states
   both halves. This keeps one CLI-wide meaning for `effort=none` and
   preserves an unquoted spelling for minimum effort.
4b. **Provenance metadata:** `generate_scene` records the taskgen
   provenance dict (`taskgen.py` ~line 220); when an effort value is
   sent on the wire, record it there as `"effort"` alongside `model`
   and `base_url`; when omitted, the key stays absent.
5. **Docs surface:** add `effort` to the two key lists in
   `docs/guide/cli.md`: the `-G` component-argument list (lines ~171-174)
   and the `-A` common-arguments list (lines ~409-411), each stating
   both halves of the contract: leaving the key out leaves the provider
   default in charge, and `effort=none` requests the minimum (it does
   not mean unset). No em dashes in the added prose.
6. **CHANGELOG:** one entry under the existing `## [Unreleased]` /
   `### Added` section (present at ~line 65), Core-scoped, linking this
   plan and issue #394, following the existing entry format.
7. **Out of scope is binding:** no `--effort` for summarize, no changes
   to `-P effort=`, no validation enums.

## Tasks

### Task 1: failing tests first (TDD)

- [ ] `tests/test_chatwire.py`: using the existing `_scripted_post`
      recorder, add tests that (a) `effort="high"` puts
      `reasoning_effort: "high"` in the request body; (b) the default
      call's body has no `reasoning_effort` key (write a new test; do
      not edit the existing happy-path test); (c) `effort=""` and
      `effort=None` both omit the key; (d) when the #390 retry fires
      with `effort="high"`, both the `max_tokens` body and the
      `max_completion_tokens` retry body carry
      `reasoning_effort: "high"`; (e) a non-string value
      (`effort=0.5`) is serialized verbatim
      (`"reasoning_effort": 0.5`), pinning the no-silent-swallow rule.
- [ ] `tests/test_taskgen.py`: tests that `generate_scene(...,
      effort="high")` produces a request body containing
      `reasoning_effort: "high"` and records `effort` in the taskgen
      provenance metadata; that `effort=None` (key present) sends the
      string `"none"`; and that omitting the kwarg sends no
      `reasoning_effort` key and records no `effort` metadata (capture
      via the injected `http_post`, following the file's existing fake
      conventions).
- [ ] `tests/test_vlm_grader.py`: tests that `vlm_grader(...,
      effort="high")` sends `reasoning_effort: "high"` when grading,
      that `effort=None` (key present) sends `"none"`, and that
      omitting the kwarg sends no key (same capture approach as that
      file's existing wire tests).
- [ ] Run the three files; every new test must fail against current code
      (each asserts a key that nothing emits yet, or passes a kwarg that
      does not exist yet — the taskgen/grader ones fail with TypeError).

### Task 2: implement

- [ ] `_chatwire.py`: extend `_post_chat` to accept `effort` and append
      `reasoning_effort` to the body dict when it is neither `None` nor
      `""`; extend `chat_completion` with keyword-only
      `effort: str | float | None = None` and pass it to both sends;
      extend the module docstring's contract sentence.
- [ ] `taskgen.py`: add the `_Unset`-sentinel keyword-only `effort` to
      `generate_scene` per binding decision 3, normalize (sentinel/`""`
      → omit, `None` → `"none"`), pass the normalized value to
      `chat_completion`, and record it in the provenance metadata when
      sent (decision 4b). Docstring states both halves of the `none`
      contract.
- [ ] `grader.py`: same sentinel + normalization on `vlm_grader`, store
      the normalized value on `_VLMGrader`, pass in `grade`'s call.
      Docstring states both halves.
- [ ] Run the three test files until green, then full gates:
      `uv run ruff check .`, `uv run ruff format --check .`,
      `uv run mypy`, `uv run pytest --cov` (must hold 100%).

### Task 3: docs + changelog

- [ ] `docs/guide/cli.md`: the two key-list additions (binding
      decision 5).
- [ ] `CHANGELOG.md`: the `### Added` entry (binding decision 6).

## Out of scope

- `--effort` for summarize (same wire, separate surface; follow-up if
  wanted).
- Any change to `-P effort=` (exists) or the agent plugin.
- Client-side effort validation or enums (binding decision 7).
- Making `reasoning_effort` configurable per-send or renaming it per
  provider: one OpenAI-compat name, passed through verbatim, is the
  contract; endpoints that reject unknown parameters surface their own
  guided 4xx through the existing error path.
