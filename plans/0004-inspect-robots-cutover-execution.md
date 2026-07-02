# 0004 — Inspect Robots cutover: outward-facing execution (Steps D/E)

**Status:** ready to execute. Steps A–C (local content rename + gates + branch
commits) are **DONE**; this doc covers only the outward-facing, irreversible
remainder (GitHub repo renames, local dir moves, push, tags). Companion spec:
`plans/0003-rename-to-inspect-robots.md` (the full, critique-hardened plan).

**Run this from a shell whose cwd is `~` (a neutral directory) — NOT from inside
`~/roboinspect`,** because Step D moves that directory and would break a session
running inside it.

## What is already done (do NOT redo)

All four repos have a branch **`rename-to-inspect-robots`** with the rename +
version bumps committed and every gate green (ruff, ruff format, mypy --strict,
pytest --cov 100%, plus per-repo functional checks). Sibling commits used
`--no-verify` because their `v0.3.0` git-tag deps aren't pushed yet.

| Repo (current dir) | Branch HEAD | Notes |
|---|---|---|
| `~/roboinspect` | `rename-to-inspect-robots` | framework → `inspect-robots`/`inspect_robots`; in-repo isaacsim plugin renamed; src dirs + plugin dir moved. Coverage 100% (109 tests) + plugin (13); `mkdocs build --strict` ok; `inspect-robots list` shows `isaacsim`. |
| `~/kitchenbench` | `rename-to-inspect-robots` | refs → inspect-robots; bumped to 0.3.0. cov 100% (74). |
| `~/worldevals` | `rename-to-inspect-robots` | refs → inspect-robots; bumped to 0.3.0. cov 100% (14). |
| `~/robolens-yam` | `rename-to-inspect-robots` | pkg `src/robolens_yam`→`src/inspect_robots_yam`; dist `robolens-yam`→`inspect-robots-yam`; bumped to 0.3.0 (incl. `__version__` + snapshot assert). cov 100% (77); preflight resolves molmoact2+yam_arms. |

Verification already passed in every repo:
`git grep -in 'roboinspect\|robolens' -- ':!plans/000[123]-*'` → 0, and the
wrong-new-forms grep (`inspect_robots-`, `import inspect-robots`,
`../inspect_robots`, `prog="inspect_robots"`, `version=f"inspect_robots `,
`Inspect Robots[A-Za-z0-9_]`) → 0.

**Merge decision (ask the user before doing it):** these branches can be merged
straight to `main`, or pushed as PRs for review first. The commands below assume
**direct merge to `main`**. If PRs are wanted, replace the merge+push lines with
`git push origin rename-to-inspect-robots` and `gh pr create`, and tag `v0.3.0`
on the merge commit after each PR lands.

## Step 0 — fetch first (collaborator safety)

A collaborator (**aris-zhu**) owns the Isaac plugin and has pushed mid-work
before. Fetch every repo and integrate (don't clobber) before pushing:

```bash
for r in roboinspect kitchenbench worldevals robolens-yam; do
  echo "== $r =="; git -C ~/$r fetch --all --tags
  git -C ~/$r log --oneline origin/main -3
done
```

If `origin/main` has commits not in these branches, rebase the
`rename-to-inspect-robots` branch onto the new `origin/main` and re-run that
repo's gates before continuing.

## Step D — rename GitHub repos, move local dirs, fix remotes

Only two repos rename (kitchenbench and worldevals keep their names):

```bash
# GitHub repo renames
gh repo rename inspect-robots     -R robocurve/roboinspect
gh repo rename inspect-robots-yam -R robocurve/robolens-yam

# local dir moves (cwd must be ~ or elsewhere neutral)
mv ~/roboinspect   ~/inspect-robots
mv ~/robolens-yam  ~/inspect-robots-yam

# point remotes at the renamed GitHub repos
git -C ~/inspect-robots     remote set-url origin git@github.com:robocurve/inspect-robots.git
git -C ~/inspect-robots-yam remote set-url origin git@github.com:robocurve/inspect-robots-yam.git
```

### Recreate venvs from the NEW hyphen paths

Never use `../inspect_robots` (underscore) — the moved dir is `../inspect-robots`
(hyphen). The framework repo is a uv workspace; the siblings resolve the
framework from a git tag that isn't pushed yet, so gate them with the **local
editable** framework instead. Force uv to target `.venv` (conda base is active in
these shells and a bare `uv pip install` lands there):

```bash
# framework workspace (installs core + isaacsim plugin editable)
cd ~/inspect-robots && rm -rf .venv && uv sync --all-packages --extra dev

# each sibling: fresh py3.11 venv, editable framework (+ kitchenbench for yam), dev tools, numpy<2.5
recreate() {  # $1 = repo dir ; $2.. = extra editable deps
  cd ~/"$1" && rm -rf .venv && uv venv --python 3.11
  P=.venv/bin/python
  uv pip install --python $P -e ../inspect-robots
  shift; for d in "$@"; do uv pip install --python $P --no-deps -e "$d"; done
  uv pip install --python $P --no-deps -e .
  uv pip install --python $P pytest pytest-cov ruff mypy "numpy<2.5"
}
recreate kitchenbench
recreate worldevals
recreate inspect-robots-yam ../kitchenbench
```

### Re-run all gates from the new paths

```bash
# framework
cd ~/inspect-robots && uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov
cd ~/inspect-robots && uv run pytest plugins/inspect-robots-isaacsim/tests -q
cd ~/inspect-robots/plugins/inspect-robots-isaacsim && uv run --project ~/inspect-robots mypy --config-file pyproject.toml src
cd ~/inspect-robots && uv run inspect-robots list   # must show cubepick + isaacsim

# siblings (use the .venv binaries directly to avoid conda)
for r in kitchenbench worldevals inspect-robots-yam; do
  echo "== $r =="; cd ~/$r
  .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest --cov
done
cd ~/inspect-robots-yam && .venv/bin/inspect-robots-yam-preflight   # -> "OK: ... compatible."
```

Final zero-check in every repo (from `~/inspect-robots`, `~/kitchenbench`,
`~/worldevals`, `~/inspect-robots-yam`):

```bash
git grep -in 'roboinspect\|robolens' -- ':!plans/000[123]-*'   # -> empty
git grep -in 'inspect_robots-\|import inspect-robots\|\.\./inspect_robots\b\|prog="inspect_robots"\|version=f"inspect_robots \|Inspect Robots[A-Za-z0-9_]' -- ':!plans/*'  # -> empty
```

## Step E — merge, push, and tag in dependency order

Order matters: a sibling's CI does `uv pip install -e ".[dev]"` which resolves the
framework from the **git tag**, so the upstream tag must exist first. Order:
**inspect-robots → kitchenbench → (worldevals, inspect-robots-yam)**.

```bash
release() {  # $1 = repo dir
  cd ~/"$1"
  git checkout main
  git merge --no-ff rename-to-inspect-robots
  git push origin main
  git tag v0.3.0
  git push origin v0.3.0
}
release inspect-robots
# wait for inspect-robots CI + the v0.3.0 tag to be visible, then:
release kitchenbench
# then the two that depend on the above:
release worldevals
release inspect-robots-yam
```

After each push, confirm CI is green before releasing the next
(`gh run watch` or `gh run list -R robocurve/<repo>`).

## Step F — finish-up

```bash
# GitHub descriptions
gh repo edit robocurve/inspect-robots     --description "Inspect Robots — the Inspect AI for robotics (VLA/physical-AI eval framework)"
gh repo edit robocurve/inspect-robots-yam --description "Inspect Robots adapters for I2RT YAM bimanual arms driven by MolmoAct2"

# Pages redeploys from the docs workflow; confirm it lands at:
#   https://robocurve.github.io/inspect-robots/
gh run list -R robocurve/inspect-robots --workflow docs.yml
```

## Done = all true

- Both GitHub repos renamed; local dirs are `~/inspect-robots` +
  `~/inspect-robots-yam`; remotes updated.
- All four repos: gates green from the new paths; both greps above return zero.
- `main` pushed on all four; `v0.3.0` tags pushed in dependency order; all CIs green.
- `inspect-robots list` (isaacsim), `worldevals tasks`, and
  `inspect-robots-yam-preflight` all work; docs live at
  `robocurve.github.io/inspect-robots`.

## Gotchas (carried from 0002/0003 + execution notes)

- **Never `../inspect_robots` (underscore)** — the dir is `../inspect-robots`.
- **Conda base is active** in the sibling shells; a bare `uv pip install` lands
  there, not in `.venv`. Always pass `--python .venv/bin/python` (or run the
  `.venv/bin/` tools directly).
- **`numpy<2.5`** — numpy 2.5's stubs use 3.12-only syntax that mypy
  (python_version 3.10) rejects; keep the pin when recreating venvs.
- **Reinstall the plugin itself** after entry-point edits so
  `entry_points.txt` regenerates (a fresh `uv sync`/`uv pip install -e` does this).
- **`site/` is gitignored** — never hand-edit; `mkdocs build` regenerates it.
- **Old tags** (`v0.2.0` etc.) stay; we cut fresh `v0.3.0`.
- `git fetch` before pushing (aris-zhu). Integrate, don't clobber.
