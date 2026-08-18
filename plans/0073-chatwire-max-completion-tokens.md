# chatwire: retry with max_completion_tokens when a 400 names it Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--auto-task` with an OpenAI gpt-5.x author fails before rollout
(issue #390): `chat_completion` hardcodes `"max_tokens": 8192` into every
request body, and OpenAI's reasoning models reject that parameter on
`/chat/completions` with HTTP 400 `unsupported_parameter`, whose message
says to use `max_completion_tokens` instead. Every consumer of the shared
chat wire is exposed when pointed at such a model: task generation (`-A`),
the `vlm` grader (`-G`), and summarize.

**Architecture:** one self-contained change in
`src/inspect_robots/_chatwire.py`. `chat_completion` sends the request
exactly as today; if and only if the response is a 400 whose decoded body
contains the substring `max_completion_tokens`, it re-sends the identical
request once with the token cap keyed as `max_completion_tokens` instead of
`max_tokens`, then proceeds with the normal success/error handling on the
retry response. The server itself names the substitute parameter, so there
is no model-name sniffing, and endpoints that accept `max_tokens` today
(Gemini OpenAI-compat, OpenRouter, Anthropic's compat endpoint, vLLM) see
byte-identical behavior. Callers (`taskgen.py`, `grader.py`,
`_summarize.py`) are untouched.

**Tech stack:** stdlib-only module as today (`json`, `urllib`); mypy
strict; pytest at 100% coverage.

**Spec:** issue #390 + this plan (the plan is the spec, per repo
convention).

**Critique:** round 1 (fresh-context subagent) verified every factual
claim against the code (callers, tests, `__all__`, CHANGELOG format, no
collision with the in-flight #388 taskgen-args work beyond a trivial
CHANGELOG merge) and found 3 minor issues, all fixed in this revision:
the both-400 test now scripts two different 400 bodies so it can prove
the retry response is the one surfaced (binding decision 3 was otherwise
unverifiable); the TDD must-fail claim is scoped to the three tests that
actually fail pre-change, with the other two labeled regression guards;
and the residual reasoning-token-exhaustion failure mode is named in
Out of scope so it is not rediscovered as a bug in this fix.

## Global constraints

- Gates: `ruff check .`, `ruff format --check .`, `mypy` (strict,
  src + tests), `pytest --cov` at 100%.
- The module stays dependency-free (stdlib only); the `http_post`
  injection seam keeps its `HttpPost` signature so existing callers and
  tests are unaffected.
- New helpers get contract-stating docstrings (ruff D1 exempts
  underscore-prefixed functions, but the module's own style keeps them).
- No public API change (`chat_completion` signature untouched;
  `inspect_robots.__all__` untouched).
- Error wording: existing guided-error text (`what` prefix, `fix_hint`)
  must survive unchanged; existing tests in `tests/test_chatwire.py`,
  `tests/test_taskgen.py`, `tests/test_vlm_grader.py`, and
  `tests/test_summarize.py` must pass without edits. Any pre-existing
  test that seems to demand modification is a stop-and-flag conflict, not
  something to edit.

## Binding decisions

1. **Trigger contract:** retry exactly when `status == 400` and the
   response body, decoded as UTF-8 with `errors="replace"`, contains the
   substring `"max_completion_tokens"`. Decode the full body for the
   check, not `_response_excerpt` (that helper truncates to 500 chars and
   exists for error display only). Any other status, and any 400 without
   the marker, takes today's path with zero extra requests.
2. **Retry request:** identical `url`, `headers`, `model`, `messages`,
   and cap value; the only difference is the JSON key `max_tokens` →
   `max_completion_tokens`. The cap value stays `8192` and is hoisted to
   a module constant `_MAX_COMPLETION_TOKENS = 8192` so both branches
   share it.
3. **Single retry, terminal second answer:** the retry response flows
   into the same success/error handling as a first answer, with no
   further body inspection. A 400 from the `max_completion_tokens`
   request therefore raises the normal guided `ConfigError` carrying the
   *retry* response's body (that is the actionable error). No loops.
4. **Shape of the change:** extract the request-building + posting into a
   private helper so the two sends cannot drift:

   ```python
   def _post_chat(
       post: HttpPost,
       url: str,
       headers: dict[str, str],
       model: str,
       messages: list[dict[str, Any]],
       token_param: str,
   ) -> tuple[int, bytes]:
       """Send one chat-completions request with the token cap under token_param."""
       body = json.dumps(
           {"model": model, "messages": messages, token_param: _MAX_COMPLETION_TOKENS}
       ).encode("utf-8")
       return post(url, headers, body)
   ```

   `chat_completion` then reads:

   ```python
   status, response_body = _post_chat(post, url, headers, model, messages, "max_tokens")
   if status == 400 and "max_completion_tokens" in response_body.decode(
       "utf-8", errors="replace"
   ):
       status, response_body = _post_chat(
           post, url, headers, model, messages, "max_completion_tokens"
       )
   ```

5. **Docs surface:** module docstring gains one sentence stating the
   retry contract. No user-facing docs change (the wire is internal), no
   CLI docs change.
6. **CHANGELOG:** one entry under `## [Unreleased]` / `### Fixed`,
   Core-scoped, linking this plan and issue #390, following the existing
   entry format.

## Tasks

### Task 1: failing tests first (TDD)

- [ ] In `tests/test_chatwire.py`, add a recording post factory that
      captures each `(url, headers, body_bytes)` and serves scripted
      responses in sequence (the existing `_post` helper serves one
      static response; the new tests need per-call scripting and capture).
- [ ] Test: first response is the OpenAI 400 `unsupported_parameter` body
      naming `max_completion_tokens`, second response is a 200 reply.
      Assert the wire returns the content, exactly two posts were made,
      the second body's JSON has `max_completion_tokens: 8192` and no
      `max_tokens` key, and `model`/`messages` are unchanged between the
      two posts.
- [ ] Test: both responses are 400, with two DIFFERENT bodies (first
      contains the marker, second is distinct text without it, e.g.
      `{"error": {"message": "still rejected"}}`). Assert exactly two
      posts, and the raised `ConfigError` carries the `what` prefix,
      HTTP 400, and the second body's text but not the first's — this is
      what proves binding decision 3 (the retry response is the one
      surfaced); identical bodies could not distinguish the two.
- [ ] Test (regression guard, passes pre-change): a 400 whose body lacks
      the marker raises immediately with a single post.
- [ ] Test (regression guard, passes pre-change): a non-400 failure
      (e.g. 500) whose body mentions `max_completion_tokens` raises
      immediately with a single post (the trigger requires both
      conditions).
- [ ] Test: the marker check reads past 500 bytes: a 400 body with the
      marker after 500 filler bytes still triggers the retry (guards the
      full-body-decode decision against regression to
      `_response_excerpt`).
- [ ] Run `uv run pytest tests/test_chatwire.py`. Exactly three of the
      new tests must fail against the current module: retry-then-success,
      both-400, and marker-past-500-bytes. The two regression guards
      pass against current code by design (today's behavior is already
      correct there); do not "fix" them into failing.

### Task 2: implement the retry

- [ ] Add `_MAX_COMPLETION_TOKENS = 8192` and the `_post_chat` helper
      (binding decision 4), with D1 docstrings.
- [ ] Rewire `chat_completion` per binding decisions 1–3.
- [ ] Extend the module docstring with the retry contract sentence.
- [ ] Run `uv run pytest tests/test_chatwire.py` until green, then the
      full gates: `uv run ruff check .`, `uv run ruff format --check .`,
      `uv run mypy`, `uv run pytest --cov` (must hold 100%).

### Task 3: changelog

- [ ] Add the `### Fixed` entry under `## [Unreleased]` in
      `CHANGELOG.md` (binding decision 6).

## Out of scope

- Making the token cap configurable (`-A max_tokens=...`): separate
  feature; #386/#388 is building the taskgen config surface and nothing
  here should collide with it (the only contact point is a one-hunk
  CHANGELOG Unreleased merge conflict for whichever lands second).
- Known residual, accepted: once `max_completion_tokens: 8192` is
  accepted, OpenAI reasoning models count reasoning tokens against the
  cap. A high-effort call that exhausts the cap on reasoning returns
  HTTP 200 with `content: null` and `finish_reason: "length"`, which the
  existing parse path reports as "endpoint returned a malformed reply" —
  a misleading diagnosis for this exact provider+model combination. That
  is a pre-existing property of the parse path, not introduced by the
  retry; the remedy is the cap-configurability feature above, not a
  change to the retry design.
- Teaching the agent plugin's chat wire anything: it has its own client
  and does not exhibit the bug.
- Caching the discovered parameter name across calls: each
  `chat_completion` call is one-shot today; per-call retry costs one
  extra request only on the affected provider+model combination and
  keeps the module stateless.
