# Contributing to RoboLens

Thanks for your interest in improving RoboLens — the evaluation framework for
physical AI. This guide covers how to get set up and what we expect from a change.

## Scope

This repository is the **framework** (the "Inspect AI for robotics"). Concrete
benchmarks live in a separate repository (the "Inspect Evals for robotics").
Backend adapters (simulators, real VLA models) are expected to ship as **separate
plugin packages** that register components through entry points — see below.

So, in scope here: core abstractions, the rollout engine, scoring, logging,
compatibility checking, the registry/plugin mechanism, and the dependency-free
mock world. Out of scope: specific sims, specific model weights, specific
benchmarks.

## Development setup

We use [uv](https://github.com/astral-sh/uv):

```bash
uv venv && uv pip install -e ".[dev]"
uv run pre-commit install      # set up the git hooks (do this once)
uv run pytest
```

Before opening a PR, the same gates CI runs must pass locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov            # must report 100% coverage
```

## Git hooks (pre-commit)

This repo ships a [pre-commit](https://pre-commit.com/) configuration
([`.pre-commit-config.yaml`](.pre-commit-config.yaml)). Install it once per clone
with `uv run pre-commit install` — this wires up **both** a `pre-commit` and a
`pre-push` hook. After that:

| Stage | When | Hooks |
|-------|------|-------|
| **pre-commit** | every `git commit` | trailing-whitespace / EOF / YAML / TOML / merge-conflict / large-file checks, **ruff** (lint + autofix), **ruff format**, **mypy** (strict) |
| **pre-push** | every `git push` | **pytest** with the **100% coverage** gate |

The `ruff`/`mypy`/`pytest` hooks are *local* — they run the exact tool versions
in your dev environment, so the hooks, your terminal, and CI all agree.

Useful commands:

```bash
uv run pre-commit run --all-files                      # run the commit-stage hooks now
uv run pre-commit run --all-files --hook-stage pre-push  # also run the coverage gate
uv run pre-commit autoupdate                           # bump the pinned hook repos
```

In a genuine emergency you can bypass hooks with `git commit --no-verify` /
`git push --no-verify`, but CI runs the same checks and **will block the PR**, so
prefer fixing the issue.

## Conventions

- **Test-driven, 100% coverage.** Write or update tests with every change; the
  mock `CubePick` world lets you exercise the whole stack without hardware or a
  simulator. **Coverage must stay at 100%** (line *and* branch) — it is enforced
  by `--cov-fail-under=100` in the pre-push hook and as a blocking PR check in CI.
  For the rare line that is genuinely unexecutable (a `Protocol` stub, a
  `__main__` guard, a defensive branch), add it to `tool.coverage.report`'s
  `exclude_also` or mark it `# pragma: no cover` with a one-line justification —
  do not lower the threshold.
- **Typed.** The package is `mypy --strict` clean and ships `py.typed`. The
  `dict[str, Any]` escape hatches (`info`/`extra`/`meta`) are the deliberate
  boundary of typing — don't widen the public API to `Any` beyond them.
- **Light core.** The core depends only on NumPy and the standard library.
  Anything else (rerun, sim/model backends) is an optional extra and must be
  lazily imported so the core-only import test stays green.
- **Small, focused units.** Prefer files and functions with one clear purpose.
- **Design first for non-trivial work.** Larger features start with a short
  design note under [`plans/`](plans/); see the existing ones for the format.

## Adding a plugin (out-of-tree)

Register your component(s) by publishing entry points in your package's
`pyproject.toml`:

```toml
[project.entry-points."robolens.embodiments"]
maniskill = "robolens_maniskill:ManiSkillEmbodiment"

[project.entry-points."robolens.policies"]
openvla = "robolens_openvla:OpenVLAPolicy"
```

Groups: `robolens.tasks`, `robolens.policies`, `robolens.embodiments`,
`robolens.scorers`, `robolens.sinks`. They will then appear in `robolens list`
and resolve by name in `eval()` / the CLI.

## Submitting changes

1. Branch from `main`.
2. Make the change with tests and docs (keep coverage at 100%).
3. Ensure all gates pass — the pre-commit/pre-push hooks run them for you.
4. Add a `CHANGELOG.md` entry under "Unreleased".
5. Open a PR describing the motivation and approach. CI re-runs lint, strict
   typing, the test matrix, and the 100% coverage gate as **required, blocking
   checks** before the PR can merge.

By contributing you agree your contributions are licensed under the project's
[MIT license](LICENSE).
