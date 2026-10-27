# Review guide

Instructions for Claude Code Review on pull requests. Project rules live in
`AGENTS.md` (loaded through `CLAUDE.md`); this file only says what a review
should prioritise.

## What to flag

Report these as important, in roughly this order:

- **Robot safety:** any change that lets an action reach an embodiment without
  passing the configured `Approver`, weakens `ClampApprover` or
  `DeltaLimitApprover` limits, or changes rollout stop and horizon handling so a
  run can exceed `max_steps`/`max_seconds`. Real arms run this code.
- **Correctness of the eval loop:** wrong observation/action alignment, off-by-one
  steps, scores computed from the wrong epoch or scene, swallowed exceptions that
  turn a failed rollout into a scored one.
- **Interface contracts:** changes to `Policy`, `Embodiment`, `Task`, `Scorer` or
  `EvalLog` that break the binding resolutions in
  `plans/0001-foundation-design.md` §9 to §11, or mutate an `EvalLog` after it is
  written. For the serialised log format, a new field with a default is fine
  when a test shows older logs still read back (see
  `plans/0027-grader-notes.md` and `plans/0081-grader-config.md`). Flag
  a rename, removal or type change that breaks reading older logs, and flag a
  `SCHEMA_VERSION` bump that has no migration path, because
  `EvalLog.from_dict` accepts only the current version.
- **Public API:** a symbol added to or removed from `inspect_robots.__all__`
  without the matching update to `tests/test_api_snapshot.py`, or the reverse.
- **NumPy-only core:** a top-level import of anything beyond NumPy and the
  standard library in `src/inspect_robots/`. Optional dependencies must be
  imported lazily inside the function that needs them.
- **Secrets:** API keys, tokens or other credentials written to logs, `EvalLog`
  files, HTML reports or test fixtures. Documented path fields such as
  `frames_dir` and `policy_checkpoint` store local paths by design; flag paths
  only when they leak somewhere the schema does not call for them.
- **Release hygiene:** a user-facing change with no `changelog.d/` fragment, a
  feature PR that edits `CHANGELOG.md` directly, or a static `version =` added to
  the root `pyproject.toml`.

## What to skip

- Formatting, import order, type annotations and docstring presence. Ruff and
  strict mypy enforce these in CI.
- Test coverage percentages. CI fails below 100%.
- Generated files: `uv.lock` and `CHANGELOG.md` sections built by towncrier.
- Wording in `plans/` design docs unless it contradicts the code in the PR.

## Tone and volume

- Lead with the most serious finding. If there are none, say so in one line.
- At most three nits per review, and only when the PR has no important findings.
- For each finding, give the failing input or sequence of calls, not just the
  suspicion.
