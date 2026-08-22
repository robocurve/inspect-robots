# Automatic task generation: a VLM writes the task and rubric from the initial frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship automatic task generation (issue #384): a vision-capable LLM
looks at the embodiment's initial camera frames and writes both halves of an
eval itself — a task instruction handed to the policy under test, and a
grading rubric handed to the grader (the automated VLM grader from #382 /
PR #383, or the human operator at the verdict prompt). Built for the fenced
tabletop "jungle" workflow: scatter items, run one command, the robot gets a
model-written task, the grader gets a model-written rubric.

**Architecture:** A new core module `taskgen.py` with one public factory,
`generate_scene(embodiment, *, model, ...) -> Scene`. It peek-resets the
embodiment, encodes the initial frames with `_pngenc.png_data_url`, sends
them with meta-instructions (default provided, operator-overridable) over
the OpenAI-compatible chat wire `_summarize.py` already ships, parses a
`TASK:` line plus `RUBRIC:` block, and returns a frozen `Scene` whose
`metadata` carries the rubric and generation provenance. The CLI grows
`run --auto-task` plus repeatable `-A k=v` generator args; the operator
grader displays the rubric at the verdict prompt; the per-scene log sample
persists scene metadata so the generated rubric survives into the `EvalLog`.

**Tech stack:** stdlib `urllib` chat wire + `_pngenc` data URLs; no new
deps; mypy strict; pytest at 100% coverage.

**Spec:** issue #384 + this plan (the plan is the spec, per repo convention).

**Critique:** round 1 (fresh-context subagent) found 3 major + 3 minor
issues, all fixed in this revision: peek seed now uses
`derive_seed(seed, None, 0)` to match eval's first trial; wire imports
pinned lazy (and the CLI's `taskgen` import pinned lazy for the test
seam); wire errors re-raised with taskgen wording and `-A` knobs; the
`from_dict` mechanism corrected; missing `model` made a first-class
`ConfigError`; rubric-marker-first parsing. Round 2 (fresh subagent)
found 2 major + 3 minor, all fixed: `seed` default now 0 mirroring
`eval()` with `seed=None` rejected; the wire wrap also catches
`TimeoutError`/`OSError`; the prefix rewrite covers the malformed-reply
message; the interim rubric story is the printed rubric at the verdict
prompt; task metadata keeps the `"adhoc"` marker alongside
`"auto_task"`. Round 3 (fresh subagent): 1 minor + 2 nits, fixed
(`inspect`'s saved-log step-limit hint covers task name `"auto"`; the
`derive_seed` citation and the `OSError`/`TimeoutError` subclass wording
corrected) — otherwise verified clean, loop converged.

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

1. **Generation runs before eval, never inside the rollout.** Inside
   `rollout()` the policy learns its instruction at `policy.reset(scene)`,
   which runs *before* `embodiment.reset()` produces the first observation —
   generating in-loop would reorder the core loop for every caller. Instead
   `generate_scene` performs its own peek reset before `eval()` runs.
   **The peek must see the same world as the first trial.** `eval()` resets
   each trial with `derive_seed(eval_seed, scene.init_seed, epoch)`
   (`eval.py:467`; `derive_seed` is defined at `rollout.py:55`), never the
   raw seed, and a seedable sim (CubePick
   included) re-places objects from that trial seed. So `generate_scene`
   takes `seed` documented as "the eval seed you will pass to `eval()`" and
   peeks with `embodiment.reset(peek_scene, seed=derive_seed(seed, None,
   0))` — the generated scene has `init_seed=None` and its first trial is
   epoch 0, so peek world and first-trial world match bitwise. On a real
   rig the seed is irrelevant and the table is simply untouched between
   peek and trial. With `--epochs > 1` a seedable sim still re-randomizes
   later epochs (per-epoch regeneration is out of scope; the task/rubric
   describe epoch 0's layout). Consequence for error stance: a grader runs
   after a finished rollout and must degrade, but the generator runs before
   any rollout, so it **fails fast** — every configuration, transport, and
   parse problem raises `ConfigError` with a one-line `fix:` hint. No
   rollout is ever wasted on a half-generated task.
2. **Public factory, no new registry kind.** One function in new
   `src/inspect_robots/taskgen.py`, exported via `__all__`:

   ```python
   def generate_scene(
       embodiment: Embodiment,
       *,
       model: str | None = None,
       instructions: str | None = None,
       instructions_file: str | None = None,
       base_url: str = "https://api.anthropic.com/v1",
       api_key_env: str = "ANTHROPIC_API_KEY",
       max_cameras: int = 4,
       scene_id: str = "auto-0",
       seed: int | None = 0,
       http_post: HttpPost | None = None,
   ) -> Scene: ...
   ```

   `seed` must be the same value the caller will pass to `eval()`; its
   default 0 mirrors `eval()`'s default, so the defaults agree. `eval()`
   replaces `seed=None` with a freshly drawn random seed that a peek can
   never match, so `seed=None` here raises `ConfigError`
   (`fix: pass the same integer seed you will pass to eval()`); a real
   rig ignores seeds, so such callers just keep the defaults.

   Wire defaults mirror `inspect-robots summarize` and PR #383's `vlm`
   grader (same wire, same default endpoint and key variable); `http_post`
   is the same injectable test seam. `model` defaults to `None` only so the
   missing-model error can be a first-class `ConfigError` with
   `fix: pass model=... (-A model=... from the CLI)` instead of a bare
   `TypeError` (it is the most likely first-run mistake). A registry kind
   for pluggable generators is deliberately deferred until a second
   generator exists (YAGNI). The chat call uses `chat_completion` from
   `inspect_robots._summarize`: PR #383 extracts it to `_chatwire.py` but
   keeps `_summarize` re-imports, so this import path is stable whichever
   PR lands first (switching to `_chatwire` is a one-line follow-up).
   **Both wire imports must be lazy**: `_summarize.py` imports from
   `inspect_robots.cli` at module top, so a top-level import in `taskgen.py`
   (eagerly exported from `__init__.py`) would make bare
   `import inspect_robots` execute all of `cli.py` — import `HttpPost`
   under `TYPE_CHECKING` and `chat_completion` inside `generate_scene`,
   the repo's established lazy style.
3. **Fail-fast validation (all `ConfigError` with a `fix:` hint):**
   `model` missing, empty, or whitespace; `instructions` and
   `instructions_file` both given; `instructions_file` unreadable (read
   once, at call time); `os.environ.get(api_key_env)` unset or empty;
   `max_cameras < 1`; `scene_id` empty; the peek observation carrying no
   images (a vision designer with nothing to look at must not guess); a
   reply missing the `TASK:`/`RUBRIC:` contract (message carries a bounded
   reply excerpt). Transport failures and non-2xx replies from the shared
   wire are already `ConfigError`, but their messages say "summary request
   failed" and name `--base-url`/`--model` flags that do not exist on
   `run --auto-task` — `generate_scene` wraps the `chat_completion` call,
   catches `ConfigError`, and re-raises a `ConfigError` keeping the
   original diagnosis lines (status, excerpt) but with the leading word
   "summary" replaced by "task generation" (covering both the "summary
   request failed" and "summary endpoint returned a malformed reply"
   prefixes; once #383's `_chatwire` lands, its `what=`/`fix_hint=`
   parameters replace this rewrite entirely) and the `fix:` line replaced
   by `fix: check model=, base_url=, and the ${api_key_env} key (-A k=v
   from the CLI)`. The wrap also catches `OSError` (which includes
   `TimeoutError`: a read timeout after connect is not translated by the
   wire and must not reach the operator as a traceback) and re-raises it
   as the same taskgen-worded `ConfigError`. Every `fix:` hint names a knob the
   caller actually has. Tests assert the rewritten output only on
   messages the test itself injects, so they stay correct when #383's
   neutral wording lands.
4. **Peek reset.** `embodiment.reset(Scene(id=f"{scene_id}-peek",
   instruction="Hold still and observe the scene."),
   seed=derive_seed(seed, None, 0))` (decision 1) collects the initial
   observation. Frames come from `obs.images`, cameras sorted by name and
   capped at `max_cameras` (deterministic prompt), each encoded with
   `png_data_url`. The embodiment is caller-owned and stays open; the
   caller (CLI or user code) runs `eval()` with the same object afterwards.
5. **Prompt and reply contract.** One user message whose `content` is a
   list of parts: one text part, then for each selected camera a short text
   label (`camera 'top'`) followed by
   `{"type": "image_url", "image_url": {"url": <data url>}}`. The text part
   is the meta-instructions followed by a blank line and this fixed,
   non-overridable reply contract (the parser depends on it, so overriding
   `instructions` never changes it):

   ```
   Reply with your reasoning first if you wish, then end your reply with
   exactly these two sections:
   TASK: <the task instruction, on one line>
   RUBRIC:
   <the grading rubric, one or more lines>
   ```

   Default meta-instructions (used when neither `instructions` nor
   `instructions_file` is given):

   ```
   You are designing one evaluation task for a general-purpose robot
   manipulation system. Observe the tabletop and the items on it in the
   attached camera frames. Write one concrete task a single robot arm could
   plausibly attempt in this scene, such as picking, placing, stacking,
   sorting, or grouping the visible items. Use only items you can actually
   see. Then write a grading rubric for that task: the observable conditions
   a judge should check on the final camera frames to call the trial a
   success, strict enough to be meaningful and fair enough to be achievable.
   ```

   Parsing anchors on the rubric marker first so a rubric that restates
   the task (models do this) cannot break the parse: find the **last**
   `^\s*RUBRIC:(.*)$` line, then the **last** `^\s*TASK:\s*(\S.*)$` line
   *before* it. The rubric is the marker line's inline remainder (if
   non-blank) followed by every line after the marker, joined with
   newlines and stripped. A missing task line, missing rubric marker, or
   empty rubric raises `ConfigError` per decision 3, and the parser gets
   tests pinning exactly these rules.
6. **Scene shape and the rubric contract.** The returned scene is
   `Scene(id=scene_id, instruction=<task line>, metadata={"rubric": <rubric>,
   "taskgen": {"model": model, "base_url": base_url}})`. The documented
   contract is the key: **a generated rubric lives at
   `scene.metadata["rubric"]`**. Consumers:
   - The **operator grader**: `OperatorSession.prompt_verdict` (which
     currently ignores its `scene` argument) displays the rubric before
     prompting — after the adopt-from-console and adopt-from-embodiment
     early returns (an adopted verdict spends no operator attention), print
     `rubric:` and the rubric lines indented two spaces via `write_line`.
     Non-string or blank metadata values are ignored.
   - The **`vlm` grader** (PR #383): reads `scene.metadata["rubric"]` in
     preference to its constructed rubric. That change belongs to #383's
     branch; this plan's PR posts the proposal as a comment on #383 rather
     than editing the same file in two open PRs. Until it lands, the
     working interim path is the one this plan builds: the operator reads
     the printed rubric (decision 7) and judges at the verdict prompt.
7. **CLI: `run --auto-task` + repeatable `-A k=v`.** `--auto-task`
   (store_true) joins `--task`/`--instruction` on `run` only (`eval-set`
   runs registered tasks; out of scope): exactly one of the three must be
   given. `-A` (dest `auto_task_args`, `metavar "k=v"`, repeatable) carries
   generator kwargs (`model=...`, `instructions_file=...`,
   `base_url=...`, `api_key_env=...`, `max_cameras=...`) parsed by the
   existing `_parse_kvs`; `-A` without `--auto-task` and `-T` with
   `--auto-task` are `SystemExit` usage errors. An auto task **is** an
   ad-hoc run whose instruction a model wrote: `--max-steps`/`--scorer`
   apply with the same `_ADHOC_*` fallbacks, `--epochs` re-runs the same
   generated scene. Construction order changes for this mode only: the task
   is built inside the existing `try` block after `_resolve_components`
   (generation needs the resolved embodiment), as
   `generate_scene(resolved.embodiment, seed=args.seed, **kvs)`.
   `taskgen` is imported lazily inside the run handler at call time — the
   existing `from inspect_robots import eval` pattern — both to keep CLI
   startup light and because Task 3's tests monkeypatch
   `inspect_robots.taskgen.generate_scene` (a top-level `from` import
   would bind early and defeat the patch). `ConfigError` and `TypeError`
   become guided `SystemExit` messages embedding `{exc}` and pointing at
   `-A k=v`, mirroring `_resolve_or_exit`'s handler style; the `TypeError`
   arm covers unknown keys, `-A seed=1` (duplicate of the explicit kwarg),
   and similar misuse in one message.
   Before the rollout the CLI prints the generated task and full rubric
   (the operator must see what the robot was asked and how it will be
   judged):

   ```
   auto task: <instruction>
   rubric:
     <line 1>
     ...
   ```

   The task is `Task(name="auto", scenes=[scene], scorer=<resolved>,
   max_steps=<resolved>, metadata={"adhoc": True, "auto_task": True,
   "instruction": scene.instruction})` — the `"adhoc"` marker is kept for
   symmetry with the existing ad-hoc construction (Task metadata is not
   persisted to logs today), and `"auto_task"` distinguishes the mode.
   The run-summary/step-limit notices treat it as ad-hoc (`is_adhoc`
   covers both modes so `--max-steps` hints stay correct), and
   `_cmd_inspect`'s saved-log step-limit hint extends its task-name check
   to `log.eval.task in ("adhoc", "auto")` so inspecting an auto-task log
   that hit the step limit names `--max-steps`, not a task-owned horizon
   (with a test).
8. **The rubric persists in the log.** `SceneResult` (log.py) gains
   `scene_metadata: dict[str, Any] = field(default_factory=dict)`. The
   dataclass default alone gives newer-reads-older: `from_dict` builds
   samples via `SceneResult(**sample)`, so an old log without the key just
   takes the default (the `.get` calls there exist only for list-to-tuple
   coercion, which a dict does not need). No schema version bump. `eval.py`
   populates it per scene with a **JSON-safe per-key copy**: each
   `scene.metadata` entry is kept only if `json.dumps` accepts it, dropped
   otherwise — scene metadata was never persisted before, so an existing
   user's non-serializable metadata must not start crashing `JsonLogSink`.
   Taskgen metadata is all strings and always survives. HTML/`view`
   rendering of the rubric is a follow-up, not this plan.
9. **No config section yet.** `[taskgen.args]` config-file support (the
   `[policy.args]`/`[grader.args]` pattern) is deferred until the `-G`
   machinery from #383 lands, then one follow-up can mirror it exactly
   (anti-drift beats speculative parallel implementations).

## Tasks

- [ ] **1. `taskgen.py` + tests.** Implement decisions 1–6:
  `generate_scene` in new `src/inspect_robots/taskgen.py` (module docstring
  explains the two-consumer contract: instruction to the policy, rubric to
  the grader). Tests in `tests/test_taskgen.py` with injected `http_post`
  capturing the request and the mock `CubePick` embodiment (it renders a
  `"top"` camera): every fail-fast branch from decision 3 raises
  `ConfigError` with `fix:` in the message (missing `model` included);
  wrapped wire errors say "task generation request" and name `-A` knobs,
  never `--base-url`; instructions file is read and used; default
  meta-instructions used otherwise; reply contract text is always present
  even with custom `instructions`; camera sort and `max_cameras` cap (use
  a stub embodiment with several named cameras); data URLs and camera
  labels appear in the request body; parsing (task + block rubric, inline
  `RUBRIC: text` joined as the first rubric line, last-rubric-marker
  anchoring with the last prior `TASK:` line so a rubric restating
  `TASK:` still parses, missing task / missing rubric / empty rubric each
  raise with an excerpt); returned scene has the instruction,
  `metadata["rubric"]`, and `metadata["taskgen"]` provenance; the peek
  reset receives `derive_seed(seed, None, 0)` (assert equality with
  eval's first-trial seed, and that `generate_scene` and `eval` share the
  default seed); `seed=None` raises `ConfigError`; an injected `http_post`
  raising `TimeoutError` becomes a taskgen-worded `ConfigError`; peek
  scene id is namespaced; bare
  `import inspect_robots` does not import `cli` (lazy-wire test). mypy
  strict clean.
- [ ] **2. Rubric display in `prompt_verdict` + log persistence.**
  Implement decisions 6 (operator display) and 8: session.py shows the
  rubric only when actually prompting; log.py `SceneResult.scene_metadata`
  + `from_dict` default; eval.py JSON-safe per-key population. Tests:
  verdict prompt output includes indented rubric lines when
  `scene.metadata["rubric"]` is a non-blank string and omits them otherwise
  (and on the adopt paths); a log round-trips `scene_metadata` through
  `to_dict`/`from_dict` and a pre-field dict still loads; an eval over a
  scene with mixed JSON-safe and non-JSON-safe metadata persists only the
  safe keys and completes.
- [ ] **3. CLI `--auto-task` / `-A`.** Implement decision 7. Tests
  (monkeypatching `inspect_robots.taskgen.generate_scene` to return a
  canned scene for the happy path): a full `--auto-task` mock run prints
  the task and rubric, runs, and writes a log whose sample carries the
  rubric; zero/two-of-three task-mode selections exit with the usage
  message; `-A` without `--auto-task` and `-T` with `--auto-task` exit;
  `-A` kwargs reach `generate_scene`; an unknown `-A` key and a
  `ConfigError` from generation each exit with the guided message and no
  traceback; `--max-steps`/`--scorer` fallbacks apply.
- [ ] **4. Public API + docs + coordination.** `__all__` +=
  `generate_scene` and `tests/test_api_snapshot.py` in the same commit.
  Docs: `docs/guide/writing-a-benchmark.md` gains an "Automatic task
  generation" section (the two-consumer contract, `scene.metadata["rubric"]`,
  default meta-instructions and how to override, programmatic example);
  `docs/guide/cli.md` documents `--auto-task` and `-A`;
  `src/inspect_robots/CLAUDE.md` module-map rows for `taskgen.py`,
  `session.py`, `log.py`, `eval.py`, `cli.py`. README mention only if it
  enumerates run modes today (check; do not add a new section). Post the
  #383 coordination comment (decision 6): propose
  `scene.metadata["rubric"]` precedence in the `vlm` grader.
- [ ] **5. Gates + PR.** `ruff check .`, `ruff format --check .`,
  `uv run mypy`, `uv run pytest --cov` (100%), core-only import unaffected.
  Push, CI green, fresh-eye review loop, merge (closes #384).

## Out of scope

- A `task_generator` registry kind / pluggable generators (one builtin is
  enough until a second generator exists).
- `[taskgen.args]` config section (decision 9 follow-up after #383).
- Per-trial regeneration across epochs or multi-scene generation in one
  call (the jungle workflow is one scattered table, one generated task).
- The `vlm` grader's scene-metadata rubric read (PR #383's branch, proposed
  via comment).
- Rendering the rubric in `inspect-robots view` HTML reports.
- `eval-set` support (registered tasks own their scenes).
- Anthropic-native or Interactions wire variants (the OpenAI-compatible
  wire covers Anthropic's compat endpoint, OpenAI, OpenRouter, Ollama,
  vLLM).
