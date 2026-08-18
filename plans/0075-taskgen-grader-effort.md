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

**Critique:** pending (rounds recorded here as they complete).

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
   also honors as its thinking-budget control. Sent only when `effort`
   is truthy; `None` and `""` both mean absent. (Truthiness, not
   `is not None`: `-G effort=` parses to `""`, and an empty string must
   not put `"reasoning_effort": ""` on the wire — same reasoning as the
   agent plugin's api_key_env handling.)
2. **Placement:** the field is added in `_post_chat`, so the retry send
   carries it identically to the first send. Body key order:
   `model`, `messages`, token cap, then `reasoning_effort` (appended
   last; tests assert key presence/values, not order).
3. **Signatures:** keyword-only `effort: str | None = None` on
   `chat_completion`, `generate_scene`, and `vlm_grader`. `_VLMGrader`
   stores it and passes it in `grade`'s `chat_completion` call.
   `_post_chat` takes `effort: str | None` as a positional-after-
   token_param parameter (private helper, no default needed — both
   callers pass it explicitly).
4. **`none` parses to None:** `-A effort=none` / `-G effort=none` go
   through the shared kv parser, which yields Python `None` — the field
   is omitted (provider default), NOT the literal string `"none"`. This
   mirrors the documented `-P effort=none` behavior. The docs note names
   the parallel; no code special-cases it.
5. **Docs surface:** add `effort` to the two key lists in
   `docs/guide/cli.md`: the `-G` component-argument list (lines ~171-174)
   and the `-A` common-arguments list (lines ~409-411), each with a
   clause that unset (or `none`) leaves the provider default in charge.
   No em dashes in the added prose.
6. **CHANGELOG:** one entry under `## [Unreleased]` / `### Added`,
   Core-scoped, linking this plan and issue #394, following the existing
   entry format. (If no `### Added` exists under Unreleased, create it
   above `### Fixed`, matching Keep-a-Changelog section order used
   elsewhere in the file.)
7. **Out of scope is binding:** no `--effort` for summarize, no changes
   to `-P effort=`, no validation enums.

## Tasks

### Task 1: failing tests first (TDD)

- [ ] `tests/test_chatwire.py`: using the existing `_scripted_post`
      recorder, add tests that (a) `effort="high"` puts
      `reasoning_effort: "high"` in the request body; (b) the default
      call's body has no `reasoning_effort` key (may fold into (a) by
      asserting on the existing happy-path test's body — but do not edit
      the existing test; write a new one); (c) `effort=""` omits the
      key; (d) when the #390 retry fires with `effort="high"`, both the
      `max_tokens` body and the `max_completion_tokens` retry body carry
      `reasoning_effort: "high"`.
- [ ] `tests/test_taskgen.py`: one test that `generate_scene(...,
      effort="high")` produces a request body containing
      `reasoning_effort: "high"` (capture via the injected `http_post`,
      following the file's existing fake conventions).
- [ ] `tests/test_vlm_grader.py`: one test that `vlm_grader(...,
      effort="high")` sends `reasoning_effort: "high"` when grading
      (same capture approach as that file's existing wire tests).
- [ ] Run the three files; every new test must fail against current code
      (each asserts a key that nothing emits yet, or passes a kwarg that
      does not exist yet — the taskgen/grader ones fail with TypeError).

### Task 2: implement

- [ ] `_chatwire.py`: extend `_post_chat` to accept `effort` and append
      `reasoning_effort` to the body dict when truthy; extend
      `chat_completion` with keyword-only `effort: str | None = None`
      and pass it to both sends; extend the module docstring's contract
      sentence.
- [ ] `taskgen.py`: add keyword-only `effort: str | None = None` to
      `generate_scene`, pass to `chat_completion`. Docstring mention.
- [ ] `grader.py`: add keyword-only `effort: str | None = None` to
      `vlm_grader`, store on `_VLMGrader`, pass in `grade`'s call.
      Docstring mention.
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
