# [taskgen.args] config section for automatic task generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the config-file half of `--auto-task` that plan 0070 deferred
(issue #386): a `[taskgen.args]` section so the jungle workflow becomes just
`inspect-robots run --auto-task` once `model` (and any other generator
argument) is persisted in `config.ini`.

**Architecture:** `defaults.py` parses `[taskgen.args]` into a new
`Defaults.taskgen_args` dict with the existing `_parse_args_section` helper;
the CLI's auto-task branch composes `{**defaults.taskgen_args,
**_parse_kvs(args.auto_task_args)}` so explicit `-A k=v` overrides
same-named config keys, mirroring the `-P`/`-G` precedence.

**Tech stack:** stdlib configparser via the existing defaults machinery; no
new deps; mypy strict; pytest at 100% coverage.

**Spec:** issue #386 + this plan (the plan is the spec, per repo convention).

**Critique:** round 1 (fresh-context subagent) verified every claim
against the code, confirmed the no-owner rationale and the absence of
leak paths, and found 3 minors, all fixed in this revision: the config
`seed` collision is now an explicit loud-failure decision pointing at
`--seed`; the cli.md owner-gating paragraph and the defaults.py
docstring/field comments gain the no-owner carve-out so neither claim
becomes false. The critic blessed the plan as ready with these fixes.

## Global constraints

- Core stays NumPy-only; gates: `ruff check .`, `ruff format --check .`,
  `mypy` (strict, src + tests), `pytest --cov` at 100%.
- No public API change is expected (`Defaults` is public but gains only a
  field; check whether `tests/test_api_snapshot.py` snapshots dataclass
  fields — if it does, update it in the same commit).
- User-facing text follows the repo writing-style rule: no em dashes in
  prose, no slogans, no "not just X, but Y".
- D1 docstrings state contracts, never restate names.

## Binding decisions

1. **No owner gating.** `[policy.args]`/`[grader.args]` are gated to the
   `[defaults]` component they were written for (issue #44) because `-P`/`-G`
   kwargs flow to whichever registry component the run selects; a mismatch
   injects kwargs a different constructor never asked for. Task generation
   has no registry kind and exactly one generator, so there is no
   differently-selected component to leak into: `[taskgen.args]` applies to
   every `--auto-task` run, unconditionally, and `defaults.py` gains **no**
   `taskgen_args_owner`. The section is inert for `--task`/`--instruction`
   runs (the auto branch is the only reader), so no stderr "ignoring" note
   is needed either — nothing is dropped.
2. **Parsing.** `_read_config` adds
   `taskgen_args=_parse_args_section(parser, "taskgen.args")` and the
   `Defaults` dataclass gains `taskgen_args: dict[str, Any] =
   field(default_factory=dict)`. The field sits apart from the
   owner-gated args fields with its own comment stating the exception:
   task generation is a single fixed function, not a registry-selected
   component, so this section has no owner and applies to every
   `--auto-task` run. The `defaults.py` module docstring's "an args
   section is valid only for its owner" opening gains the same one-line
   carve-out, so neither statement becomes false.
   `_parse_args_section` already handles value parsing and `~` expansion —
   the expansion is wanted here too (`instructions_file = ~/prompts/x.txt`
   is the flagship persisted path).
3. **Composition and precedence.** In `_cmd_run`'s auto-task branch the
   call becomes `generate_scene(resolved.embodiment, seed=args.seed,
   **{**defaults.taskgen_args, **_parse_kvs(args.auto_task_args)})`:
   config supplies defaults, explicit `-A` wins per key. A config key the
   factory rejects surfaces through the branch's existing
   `TypeError` handler ("invalid arguments for automatic task
   generation: ...; check -A k=v") — extend that message to also name
   `[taskgen.args]` so a config-sourced typo points at the file:
   `"invalid arguments for automatic task generation: {exc}; check
   [taskgen.args] and -A k=v"`. The `seed` kwarg stays explicit at the
   call site, so a persisted `[taskgen.args] seed` key makes **every**
   `--auto-task` run exit with that guided message ("got multiple values
   for keyword argument 'seed'") until the file is edited. That is the
   deliberate choice — loud beats a silently ignored config key — because
   the correct knob for the seed is `--seed`, which already reaches
   generation and eval together; the docs paragraph (Task 3) says so
   explicitly. Do not pop `seed` from the config dict.
4. **Config-writing surfaces stay untouched.** `inspect-robots setup` and
   `config set` do not learn to write `[taskgen.args]` in this plan; the
   wizard renders unmanaged sections through raw (plan 0011 behavior), so a
   hand-written section survives every wizard rewrite. Verify that
   carry-through with a test only if one does not already cover an
   arbitrary unmanaged section.

## Tasks

- [ ] **1. defaults.py + tests.** Decision 2. Tests in the existing
  defaults test module: a config file with `[taskgen.args]` parses values
  with type coercion and `~` expansion; an absent section yields `{}`;
  the section does not interact with any owner logic.
- [ ] **2. CLI composition + tests.** Decision 3. Tests (monkeypatching
  `inspect_robots.taskgen.generate_scene` to capture kwargs, as the
  existing auto-task CLI tests do): config-only kwargs reach the factory
  on a bare `--auto-task` run; an explicit `-A` overrides a same-named
  config key while other config keys still apply; the extended TypeError
  message names `[taskgen.args]` and `-A k=v`; a `--task` run with
  `[taskgen.args]` configured behaves identically to before (section
  unread). Update the existing "invalid arguments" test expectation.
- [ ] **3. Docs + module map.** `docs/guide/cli.md`: add `[taskgen.args]`
  to the config-file example block (line ~65 area) **and amend the
  paragraph after it** — it currently claims every `[*.args]` section is
  owner-gated with a stderr note, which `[taskgen.args]` would falsify;
  add the carve-out there (taskgen has no `[defaults]` component and no
  owner). Add a short paragraph in the auto-task section stating
  precedence (`-A` over config), that the section applies to every
  `--auto-task` run, and that the seed is set with `--seed`, never a
  `[taskgen.args] seed` key (which exits with the guided error). Update
  the `defaults.py` and `cli.py` rows in `src/inspect_robots/CLAUDE.md`.
  `tests/test_api_snapshot.py` snapshots only `__all__` names, so no
  change there.
- [ ] **4. Gates + PR.** `ruff check .`, `ruff format --check .`,
  `uv run mypy`, `uv run pytest --cov` (100%). Push, CI green, fresh-eye
  review loop, merge (closes #386).

## Out of scope

- Owner gating or a `taskgen_args_owner` field (decision 1).
- Teaching `setup`/`config set` to write the section.
- Any change to `generate_scene` itself.
