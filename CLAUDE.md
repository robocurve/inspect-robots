# Inspect Robots — agent guide

Inspect Robots is the **"Inspect AI for robotics"**: an open-source evaluation
framework for physical AI / VLA (vision-language-action) models. This repo is the
*framework*; concrete benchmarks and backend adapters live elsewhere (see below).

## The one big idea

LLM evals have one swappable input (the model). Robotics evals have **two**:

- **`Policy`** (the VLA "brain") — observation → `ActionChunk` (open-loop horizon).
- **`Embodiment`** (the robot/sim "body + world") — executes actions, owns spaces.

A **`Task`** (a dataset of `Scene`s + scorers) is defined independently of both.
`eval()` checks a `(policy, embodiment)` pair is compatible, runs the closed-loop
rollout, scores it, and writes an immutable `EvalLog`. Mirrors Inspect AI's
`Task = dataset + solver + scorer`, `eval()`, `EvalLog`, registry/decorator model.

## Layout

- `src/inspect_robots/` — the package (see `src/inspect_robots/CLAUDE.md` for the module map).
- `tests/` — pytest; the `CubePick` mock world exercises the whole stack with no
  hardware or sim.
- `plans/` — design docs. `plans/0001-foundation-design.md` is the authoritative
  spec (read its §9–§11 "binding resolutions" before changing core interfaces).
- `examples/` — runnable demos (`quickstart.py`).
- `plugins/*` — first-party plugin packages (concrete sims/VLAs that are out of
  scope for the numpy-only core), each its own package with its own pyproject,
  entry point, tests, and coverage scope. A uv workspace (`[tool.uv.workspace]`)
  ties them in: `uv sync --all-packages --extra dev` installs core + all plugins
  editable. They never count toward the core 100% gate (coverage is scoped to
  `inspect_robots`). E.g. `plugins/inspect-robots-isaacsim/` (Isaac Lab
  embodiment), `plugins/inspect-robots-xpolicylab/` (policy adapter speaking
  the XPolicyLab websocket protocol — 40+ served VLAs, no xpolicylab dep), and
  `plugins/inspect-robots-agent/` (LLMs as policies via the OpenAI-compatible
  wire format — httpx only, no provider SDKs; registered as `agent`).

## Working here

- Dev loop: `uv pip install -e ".[dev]"`, `uv run pre-commit install`, then
  `uv run pytest --cov`.
- Gates that must pass: `ruff check .`, `ruff format --check .`, `mypy` (strict,
  covers `src` **and** `tests`), and `pytest --cov` at **100% coverage**
  (`--cov-fail-under=100`). Pre-commit runs ruff+mypy on commit and the coverage
  gate on push (`.pre-commit-config.yaml`). CI runs all gates on Linux+macOS /
  py3.11-3.12 as **blocking, required PR checks** (coverage below 100% fails the
  build), plus a **non-blocking** `test-extra` tier for py3.10/py3.13, Windows,
  and the NumPy floor — together covering the py3.10–3.13 range pyproject claims.
- Ruff's D1 rules require a docstring on every public module, class, and function.
  State the contract or invariant the caller needs; do not restate the symbol name.
- **Core stays NumPy-only.** New deps are optional extras, lazily imported; the
  `core-only-import` CI job enforces this.
- Test-driven; commit/push in small focused steps.
- Public API is fenced by `inspect_robots.__all__` and guarded by
  `tests/test_api_snapshot.py` — update both together.

## Out of scope (separate repos / plugins)

Specific benchmarks ("Inspect Evals for robotics"), specific simulators
(ManiSkill/MuJoCo/Isaac), and specific VLA weights (OpenVLA/π0/Octo) ship as
separate plugin packages registered via entry points (`inspect_robots.embodiments`,
`inspect_robots.policies`, …) — either in their own repos or as in-repo `plugins/*`
workspace members (see Layout). Either way they stay out of the numpy-only core
and its 100% coverage gate; `plugins/inspect-robots-isaacsim/` is the reference example.

## CI, merging, and releases

- **main is PR-only** — a branch ruleset (admins included) blocks direct pushes,
  force pushes, and deletion. Merging requires the `ci-ok` check green and the
  branch up to date with main.
- **`ci-ok` is the single required status check** — an aggregate job at the end
  of `ci.yml`. When adding a CI job, add it to `ci-ok`'s `needs` list, or it
  will not gate merges.
- **Red main is stop-the-line**: if CI fails on a push to main, the
  `alert-red-main` job opens an issue. Fix forward or revert before merging
  anything else; if the failure was transient, re-run the failed jobs and close
  the issue.
- **CI installs from `uv.lock`** (`uv sync --locked`). After changing
  dependencies in `pyproject.toml`, run `uv lock` and commit the lockfile —
  otherwise CI fails with "the lockfile needs to be updated".
- A weekly **canary** (`canary.yml`) does the opposite: it installs the latest
  dependency versions the pyproject ranges allow (ignoring the lockfile), runs
  the tests, and opens an issue on failure — catching ecosystem breakage that
  locked CI can't see. A green canary means `uv lock --upgrade` is safe.
- The `test-extra` tier stays advisory: `continue-on-error: true` means it
  reports success to `ci-ok` even when its steps fail — listing it in `needs`
  does not make it blocking.
- **Releases are one-click**: Actions → Release → Run workflow → pick
  patch/minor/major. The version is derived from the git tag by hatch-vcs —
  never add a static `version =` back to pyproject (`__version__` comes from importlib.metadata. Exception: `plugins/*` packages keep static versions in their own pyprojects; bump one in a PR and it publishes alongside the next core release via its `publish-<name>` job in `release.yml` (`skip-existing` makes unchanged versions a no-op). A new plugin needs its own `publish-<name>` job and PyPI trusted-publisher environment). The same
  run publishes to PyPI via trusted publishing; nothing is pushed to main.
- **PyPI readme is transformed at build time** — `hatch-fancy-pypi-readme`
  rewrites GitHub-only alert syntax (`> [!NOTE]` etc.) in README.md into bold
  blockquotes (`> **Note:**`) that PyPI renders; keep using alert syntax in the
  README itself. Config lives at the bottom of pyproject.toml.

## Writing style (public-facing text)

READMEs, docs pages, repo/collection descriptions, and HF model cards must
avoid AI-writing tells. The full rule with the gating checklist lives in
[worldevals docs/model-cards.md, "Writing style"](https://github.com/robocurve/worldevals/blob/main/docs/model-cards.md);
short version:

- No em dashes in prose. Use periods, colons, commas, or parentheses (`—` is
  fine as an empty table cell and inside code blocks).
- Bold only for definition-list lead-ins (`**term:**`) and at most one critical
  imperative per safety bullet. Never mid-sentence for emphasis.
- No decorative emoji (functional ✅/⚠️ marks and 🤗 for Hugging Face are fine),
  no slogans or chiasmus, no "not just X, but Y".
- Headers use colons, never em dashes or italics.

Style-only edits must never touch YAML frontmatter, code blocks, numbers,
links, or safety qualifiers.
