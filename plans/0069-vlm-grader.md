# VLM grader: automated rubric judgement over initial and final frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the automated grader that plan 0049 deferred (issue #382): a
builtin `vlm` grader that sends a trial's initial frames, final frames, the
scene instruction, and a rubric to a vision-capable model over the
OpenAI-compatible chat wire, and records the success judgement so unattended
runs are graded without a human at the rig.

**Architecture:** A new `_VLMGrader` in core `grader.py` implementing the
existing `Grader` protocol, registered as `"vlm"`. It reuses the stdlib-urllib
chat wire that `_summarize.py` already ships (extracted to a shared private
module) and `_pngenc.png_data_url` for frames — zero new dependencies, so the
core-only-import gate is unaffected. The judgement lands on the same
`TrialRecord` fields the operator grader writes (`operator_judgement`,
`operator_note`, an `operator_event` with `source="vlm"`), so scorers stay
pure readers (R6) and the existing `operator` scorer — already the ad-hoc
fallback scorer (`_ADHOC_SCORER_FALLBACK = "operator"`, `defaults.py:47`) —
scores it with no scoring changes at all.

**Tech stack:** stdlib `urllib` + existing `_pngenc` PNG data URLs; no new
deps; mypy strict; pytest at 100% coverage.

**Spec:** issue #382 + this plan (the plan is the spec, per repo convention).

## Global constraints

- Core stays NumPy-only: no new dependencies, `core-only-import` CI job must
  stay green.
- Gates: `ruff check .`, `ruff format --check .`, `mypy` (strict, src + tests),
  `pytest --cov` at 100% (`--cov-fail-under=100`).
- Public API additions go through `inspect_robots.__all__` **and**
  `tests/test_api_snapshot.py` in the same commit.
- User-facing text (help strings, docs) follows the repo writing-style rule:
  no em dashes in prose, no slogans, no "not just X, but Y".
- D1 docstrings on every new public module/class/function: state the contract,
  do not restate the name.

## Binding decisions

1. **Core builtin, not a plugin.** Plan 0049 assumed a VLM grader needed a
   plugin, but `_summarize.py` established the precedent this plan reuses: an
   OpenAI-compatible chat call over stdlib `urllib` with an injectable
   `http_post` seam is dependency-free and already lives in core. A core
   builtin means `--grader vlm` works on a bare install, and the 100% coverage
   gate applies (the `http_post` seam makes that cheap). The wire moves to a
   shared private module `_chatwire.py`; `_summarize.py` re-imports the moved
   names so its tests and any monkeypatching keep working.
2. **Name and factory.** Registry name `vlm` (the R10 vocabulary this repo
   already uses); factory `vlm_grader(...)` exported in `__all__`:

   ```python
   def vlm_grader(
       model: str,
       rubric: str | None = None,
       rubric_file: str | None = None,
       *,
       base_url: str = "https://api.anthropic.com/v1",
       api_key_env: str = "ANTHROPIC_API_KEY",
       max_cameras: int = 4,
       http_post: HttpPost | None = None,
   ) -> _VLMGrader: ...
   ```

   Defaults mirror `inspect-robots summarize` (same wire, same default
   endpoint and key variable). `http_post` is the test seam, exactly as in
   `_summarize.py`.
3. **Fail fast at construction, degrade at grade time.** Configuration
   problems must not wait for (or waste) a rollout: the factory raises
   `ConfigError` when `rubric` and `rubric_file` are both given, when
   `rubric_file` cannot be read (read once at construction), when `model` is
   empty, or when `os.environ.get(api_key_env)` is unset/empty — each with a
   one-line `fix:` hint naming the flag or variable. Missing `model` entirely
   is a `TypeError` and both error classes already become guided messages via
   `_resolve_or_exit` (`cli.py:679`). After the rollout, `grade()` never
   raises (the `Grader` protocol contract): transport failures, non-2xx,
   malformed replies, and unparseable verdicts print one stderr note
   (`vlm grader: <reason>; trial left ungraded`) and leave the record
   unchanged.
4. **Frame selection.** Initial frames come from `record.steps[0]` (inline
   `observation.images` when present, else `image_refs` loaded via
   `FrameRef.load()`); final frames from `record.steps[-1]`
   (`result.observation.images`, else `result_image_refs`). Cameras are
   sorted by name and capped at `max_cameras` per phase (deterministic
   prompt). A trial with no steps or no images in both phases degrades per
   decision 3 (a vision judge with nothing to look at must not guess).
   Frames are encoded with `_pngenc.png_data_url` (strict uint8, already the
   repo's frame-to-data-URL path).
5. **Prompt and reply contract.** One user message whose `content` is a list
   of parts: a text part, then for each selected frame a short text label
   (`initial frame, camera 'wrist'` / `final frame, camera 'wrist'`) followed
   by an `{"type": "image_url", "image_url": {"url": <data url>}}` part. The
   text part is exactly:

   ```
   You are grading one robot trial from its camera frames.

   Task instruction: {scene.instruction}

   Rubric:
   {rubric}

   You will see the initial frames (before the robot acted) and the final
   frames (after the trial ended). Judge the trial against the rubric using
   only what is visible. Explain your judgement briefly, then end your reply
   with exactly one line:
   GRADE: success
   or
   GRADE: failure
   ```

   Default rubric (when neither `rubric` nor `rubric_file` is given): "Grade
   success if the system completed the task instruction, otherwise grade
   failure. If the outcome is ambiguous or not visible in the frames, grade
   failure." The binary 1/0 outcome the operator sees comes from the
   `operator` scorer mapping the judgement to 1.0/0.0 (decision 6).
   The reply is parsed by scanning lines for
   `^\s*GRADE:\s*(success|failure)\s*$` (case-insensitive) and taking the
   **last** match, mirroring Inspect AI's model-graded pattern; no match
   degrades per decision 3 with a bounded reply excerpt in the note.
6. **Record mutation.** If `record.operator_judgement` is already set (a
   console-typed or embodiment-definitive verdict captured during rollout),
   `grade()` returns without spending a model call — same adopt-don't-re-ask
   stance as `prompt_verdict`. Otherwise it sets
   `operator_judgement = "success" | "failure"` (`"success"` is in the
   scorer's `_OPERATOR_SUCCESS` set; `"failure"` is not, so the `operator`
   scorer maps them to 1.0/0.0 unchanged), sets `operator_note` to the reply
   text stripped and capped at 2000 characters, and appends
   `operator_event(t=len(record.steps), verdict=<judgement>, source="vlm",
   note=<same capped note>)`.
7. **CLI grader args.** New repeatable shared argument `-G` (dest
   `grader_args`, `metavar="k=v"`) on `run` and `eval-set`, declared next to
   `-P` in the shared-args helper so both commands get it (anti-drift). New
   config section `[grader.args]` with owner semantics mirroring
   `[policy.args]` (issue #44 behavior): `defaults.py` gains `grader_args`
   and `grader_args_owner` (owner is the configured `[defaults] grader`
   name), and `_build_grader` composes
   `{**_config_args("grader", name, owner, config_args),
   **_parse_kvs(args.grader_args)}` and passes the kwargs through
   `_resolve_or_exit("grader", name, **kvs)`. Passing `-G` to a grader whose
   factory rejects the key (e.g. `operator`) surfaces as the existing guided
   `TypeError` message from `_resolve_or_exit`, not a traceback. Selection
   semantics from plan 0049 are untouched: an explicit `--grader vlm` runs
   unattended, `--no-prompt` still only suppresses the operator grader, and
   the config downgrade special-cases only `operator`, so
   `[defaults] grader = vlm` grades cron runs too.
8. **The reserved `VLMScorer` stub stays** but its `NotImplementedError`
   message now points at the shipped path: "VLM judging ships as the 'vlm'
   grader (--grader vlm) with the 'operator' scorer reading its judgement".
   Deleting the class would be an API break for no gain; redirecting the
   error is the cheapest true message. (R6 is also *why* it stays a stub: a
   scorer must be a pure reader, so a VLM scorer is unimplementable — the
   grader is the correct home, as plan 0049 already resolved.)
9. **No log-schema change.** Judgement, note, and event already persist in
   `EvalLog`; recording the grader name in the log header remains the 0049
   follow-up, still out of scope.

## Tasks

- [ ] **1. `_chatwire.py` extraction.** Move `HttpPost`, `_urllib_post`,
  `_response_excerpt`, and `chat_completion` from `_summarize.py` into new
  `src/inspect_robots/_chatwire.py`; give `chat_completion` a
  `what: str = "summary"` parameter used in its error prefixes so summarize
  output stays byte-identical while the grader passes `what="grading"`, and a
  `fix_hint: str` parameter replacing the hardcoded `--base-url`/`--model`
  hint lines for the same reason. `_summarize.py` re-imports the moved names
  (`from inspect_robots._chatwire import ...`) so existing tests and
  monkeypatch targets keep working. Tests: existing summarize tests pass
  unmodified; new `tests/test_chatwire.py` covers the moved code paths that
  coverage attribution moves out of `_summarize` (2xx parse, non-2xx, malformed
  reply, URLError/HTTPError translation) via injected `http_post` and a fake
  `urllib` seam, whichever the existing summarize tests already model.
- [ ] **2. `vlm_grader` in `grader.py` + registration.** Implement decision
  2–6: `_VLMGrader` class + `vlm_grader()` factory in `grader.py`, registered
  as `"vlm"` in `_builtins.py`. Frame collection helper handles inline images
  and `FrameRef`s (use a real `FrameStore` in a tmp dir in tests). Tests
  (injected `http_post` capturing the request body): construction fail-fast
  (both rubrics, unreadable `rubric_file`, empty `model`, unset key env, each
  with `fix:` in the message); rubric file read once; skip when judgement
  already present (no HTTP call recorded); no-frames degrade with stderr
  note; camera sort + `max_cameras` cap; prompt contains instruction, rubric,
  labels, and data URLs for first-step and last-step frames (refs and inline
  both); reply parsing (success, failure, case-insensitive, last-match-wins,
  no-match degrade with excerpt); note capped at 2000 chars; event has
  `source="vlm"`; transport error degrades without raising. mypy strict.
- [ ] **3. CLI `-G` + `[grader.args]` config.** Implement decision 7:
  shared `-G` argument, `defaults.py` fields + `[grader.args]` section
  parsing (mirror the `[policy.args]` parser and owner wiring), `_build_grader`
  passes composed kwargs; add `"grader": "-G"` to the kind-to-flag map inside
  `_resolve_or_exit`'s TypeError handler so its guided message names the right
  flag. Update `VLMScorer`'s message (decision 8). Tests:
  `-G k=v` reaches the factory (register a throwaway grader capturing
  kwargs); config args apply only when the configured grader matches the
  owner and explicit `-G` overrides same-named keys; `-G` on the operator
  grader exits with the guided invalid-argument message; `--grader vlm`
  without the key env set exits with the guided `ConfigError` before any
  rollout starts; `VLMScorer` message test updated.
- [ ] **4. Public API + docs.** `__all__` += `vlm_grader` (alphabetical;
  the `grader`-decorator shadowing order from plan 0049 is unaffected because
  the submodule import already runs first); update
  `tests/test_api_snapshot.py` in the same commit. Docs: `docs/guide/scoring.md`
  gains a "Automated grading with a vision model" section (flags, config
  keys, rubric file, unattended behavior, how `operator` scorer reads the
  judgement); `docs/guide/cli.md` documents `-G` and `--grader vlm`;
  `src/inspect_robots/CLAUDE.md` module-map rows for `grader.py`,
  `_chatwire.py`, `_summarize.py`, `cli.py`, `defaults.py`. README only if it
  enumerates graders today (check; do not add a new section). Nothing
  hand-edited that `scripts/gen_api_docs.py` owns.
- [ ] **5. Gates + PR.** `ruff check .`, `ruff format --check .`,
  `uv run mypy`, `uv run pytest --cov` (100%), core-only import unaffected.
  Push, CI green, fresh-eye review loop, merge (closes #382).

## Out of scope

- Numeric/partial-credit rubric scores (the judgement path is binary; a
  `value_to_float`-able judgement string is a possible follow-up).
- Anthropic-native or Interactions wire variants (the OpenAI-compatible wire
  covers Anthropic's compat endpoint, OpenAI, OpenRouter, Ollama, vLLM).
- Mid-trial or multi-frame (video) judging; this grader sees first and last
  frames only.
- Recording the grader name in the `EvalLog` header (0049 follow-up).
- Retry/backoff on transport failures (one attempt; degrade).
