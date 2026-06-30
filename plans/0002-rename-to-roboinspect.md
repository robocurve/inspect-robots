# 0002 — Rename `robolens` → `roboinspect` (repo + package, all repos)

**Status:** design + execution plan (pre-execution)
**Goal:** rename the framework from **robolens** to **roboinspect** end-to-end:
the GitHub repo, the local checkout, the Python package/distribution/CLI, the
entry-point groups, and every reference in the three sibling repos
(kitchenbench, worldevals, robolens-yam).

## Naming (locked)

| Thing | Old | New |
|---|---|---|
| Display name | RoboLens | **RoboInspect** |
| GitHub repo | `robocurve/robolens` | `robocurve/roboinspect` |
| Local dir | `~/robolens` | `~/roboinspect` |
| Distribution / import / CLI | `robolens` | `roboinspect` |
| Source dir | `src/robolens/` | `src/roboinspect/` |
| Entry-point groups | `robolens.{tasks,policies,embodiments,scorers,sinks}` | `roboinspect.{…}` |

**Sibling repo names stay** (`kitchenbench`, `worldevals`, `robolens-yam`) — the
task is to rename *references to robolens*, not to rename the siblings themselves.
`robolens-yam` keeps its name/package for now; renaming it to `roboinspect-yam` is
called out as an **optional follow-up** (§7), not done here.

## Versioning & tags (the riskiest part — order matters)

Siblings pin the framework by git tag, so a rename without a new tag breaks them.
Plan: bump everything to **0.2.0** and cut fresh tags **in dependency order**.

1. **roboinspect**: version `0.1.0 → 0.2.0`; after the rename commit is pushed,
   tag `v0.2.0` (contains the `roboinspect` package). The old `v0.1.0` tag stays
   for history (it still builds the old `robolens` package) but nothing points at it.
2. **kitchenbench**: deps `roboinspect>=0.2`; `tool.uv.sources roboinspect =
   {git=.../roboinspect, tag="v0.2.0"}`; version → `0.2.0`; tag `v0.2.0` (this also
   releases the already-merged task-instances work). Depends on (1) being tagged.
3. **robolens-yam**: deps `roboinspect>=0.2` + `kitchenbench` source `tag="v0.2.0"`;
   version → `0.2.0`. Depends on (1) and (2) being tagged.
4. **worldevals**: deps `roboinspect>=0.2`; source `roboinspect@v0.2.0`; version → `0.2.0`.

Local dev uses **editable sibling installs**, so local test runs don't need the
tags — only CI (clean-env, installs from tags) does. Push tags before relying on
CI of the dependent repo.

## Execution order (per-repo, verifying gates at each step)

### Phase A — roboinspect (the framework), still in `~/robolens`
1. `git mv src/robolens src/roboinspect`.
2. In-tree code rename (sed, then verify): `robolens` → `roboinspect` across
   `src/`, `tests/`, `examples/`, `docs/`, `mkdocs.yml`, `pyproject.toml`,
   `.pre-commit-config.yaml`, `.github/`, `CHANGELOG.md`, `CONTRIBUTING.md`,
   `SECURITY.md`, `README.md`, `CLAUDE.md`, `plans/`. **Skip** `site/` (gitignored,
   regenerated) and `reference/` (none here).
   - Identifier/string `robolens` → `roboinspect` (covers imports, `name=`,
     `[project.scripts] roboinspect = "roboinspect.cli:main"`, `_GROUPS` values,
     `known-first-party`, `files=`, coverage `source=`, mkdocstrings autorefs
     `[`X`][robolens.…]`).
   - Display `RoboLens` → `RoboInspect` (prose, `site_name`, badges).
   - URLs `github.com/robocurve/robolens` → `…/roboinspect`,
     `robocurve.github.io/robolens` → `…/roboinspect`.
   - Entry-point GROUP strings in `registry.py` `_GROUPS` → `roboinspect.*`
     (and the module docstring listing them).
   - The `robolens` CLI command name in docs/prose → `roboinspect`.
3. Recreate the venv (the dir name is about to change anyway; do it after the dir
   move in Phase D) — for now reinstall editable: `uv pip install -e ".[dev]"`.
4. Gates: `ruff check . && ruff format --check . && mypy && pytest --cov` (100%).
   `test_api_snapshot` / `test_package` may assert the dist name — update them.
5. `mkdocs build --strict` to confirm docs build under the new name.

### Phase B — kitchenbench (sibling)
1. Rename references: `from robolens` → `from roboinspect`, `import robolens` →
   `import roboinspect`, `robolens.rollout`/`robolens.registry`/`robolens.logging`
   submodule imports, `robolens>=0.1` → `roboinspect>=0.2`, `tool.uv.sources`
   key+url+tag, entry-point group keys `"robolens.tasks"` → `"roboinspect.tasks"`
   (+ embodiments/policies), URLs, badges, prose (`RoboLens`→`RoboInspect`,
   `robolens run`→`roboinspect run`, `../robolens`→`../roboinspect`).
2. Reinstall editable framework: `uv pip install -e ../robolens` (path still old
   until Phase D) → re-point to `../roboinspect` after the dir move.
3. Gates: ruff/format/mypy/pytest --cov (100%). Bump version 0.2.0.

### Phase C — worldevals & robolens-yam (siblings, same treatment)
- worldevals: `cli.py` imports `robolens.registry` → `roboinspect.registry`;
  `catalog.py` install strings / prose; dep + source; `gen_catalog.py`; docs;
  `mkdocs.yml`; README; `test_catalog_cli.py` (it monkeypatches
  `robolens.registry.registered` → now `roboinspect.registry`). Bump 0.2.0.
- robolens-yam: all `from robolens` imports, `robolens.compat`/`.rollout`/`.policy`/
  `.embodiment`/`.spaces`/`.logging.sink` submodule imports, entry-point group keys
  `"robolens.embodiments"`/`"robolens.policies"` → `roboinspect.*`, deps + sources
  (roboinspect@v0.2.0 + kitchenbench@v0.2.0), `keywords` list, URLs, prose, the
  `robolens-yam-preflight`/`robolens run` command mentions. **Keep** the package
  name `robolens_yam`, the dir `src/robolens_yam`, and the repo name. Bump 0.2.0.

### Phase D — GitHub + local dir + venvs
1. `gh repo rename roboinspect -R robocurve/robolens` (leaves a redirect).
2. Update local remote: `git -C ~/robolens remote set-url origin
   git@github.com:robocurve/roboinspect.git`.
3. Rename local dir: `mv ~/robolens ~/roboinspect`. **After this, always use the new
   absolute path** (the shell's cwd-reset target changes).
4. Recreate/repair venvs (venvs hardcode the old path): simplest is
   `rm -rf .venv && uv venv && uv pip install -e ".[dev]"` in roboinspect, and in
   each sibling re-point the editable framework install to `../roboinspect`
   (`uv pip install -e ../roboinspect`, `--no-deps -e ../kitchenbench` where needed).
5. Re-run every repo's full gate suite from the new paths.

### Phase E — push, tag, branch protection, CI
1. Commit each repo (focused messages); push.
2. Tag in order: roboinspect `v0.2.0` → kitchenbench `v0.2.0` (push tags), then the
   dependent CIs go green; bump/commit worldevals + robolens-yam.
3. Branch protection: robolens-yam's required-check contexts are unchanged (job
   names didn't change). The roboinspect repo keeps its existing protection (rename
   carries it over). Verify required checks still match job names.
4. Verify: every repo CI green; `gh repo view robocurve/roboinspect`;
   docs site redeploys at `robocurve.github.io/roboinspect`.
5. Update GitHub repo **description** if it names RoboLens.

## Gotchas to respect

- **`site/` is gitignored** — never hand-edit; `mkdocs build` regenerates it.
- **Entry-point groups are magic strings on BOTH sides** — registry `_GROUPS` and
  every plugin pyproject must change together, or discovery silently finds nothing.
- **Editable installs hardcode the old path** — reinstall after the dir move.
- **cwd-reset target disappears** when `~/robolens` is moved — switch to
  `~/roboinspect` for all later commands.
- **Old `v0.1.0` tags** still build the old package — leave them; just stop pinning
  them.
- **GitHub keeps a redirect** from the old repo URL, so an un-updated pin won't hard
  404 immediately — but we update all of them for cleanliness.
- **Don't rename inside `reference/`** (kitchenbench's gitignored methodology copy).
- Word-boundary care in sed: `robolens_yam`/`robolens-yam` must NOT become
  `roboinspect_yam` in Phase C (we keep that name) — rename only the framework
  references, not the sibling's own identifier. Use targeted replaces, then grep to
  confirm the only remaining `robolens` hits are the intended `robolens_yam`/repo-name ones.

## Verification (done = all true)

- `grep -rin robolens` in each repo (excl. `.git`/`.venv`/`reference`/`site`) returns
  **only** intended residue: in robolens-yam, the kept `robolens_yam` package + repo
  name; elsewhere, zero.
- All 4 repos: ruff + format + mypy + pytest --cov 100% green locally.
- `roboinspect list` / `roboinspect run` work; `roboinspect.tasks` discovery finds
  kitchenbench tasks; preflight finds `molmoact2`/`yam_arms`.
- All 4 CIs green; tags pushed in order; docs redeployed.

## Critique resolutions (rev 2 — adopted before execution)

1. **[BLOCKER] Hardcoded version assert.** `robolens-yam/tests/test_api_snapshot.py`
   asserts `__version__ == "0.1.0"`. Update to `"0.2.0"` with the version bump.
   (robolens itself is dynamic via hatch-vcs — safe.)
2. **[BLOCKER] `robolens-yam` is protected in worldevals too.** worldevals README
   links the `robolens-yam` repo (`github.com/robocurve/robolens-yam`) in its
   Backends table — that must survive. The protection rule applies to **all four
   repos**, and verification allows `robolens-yam` residue in worldevals (but
   `robolens run` → `roboinspect run` there).
3. **[Mechanical] Use `perl` negative-lookahead, not sed** (BSD sed lacks lookahead).
   The single robust rule, applied per text file in all repos:
   `perl -pi -e 's/robolens(?![_-]yam)/roboinspect/g'` then `s/RoboLens/RoboInspect/g`.
   The lookahead protects `robolens_yam`, `robolens-yam`, `robolens-yam-preflight`,
   `src/robolens_yam`, `known-first-party=["robolens_yam"]`, `source=["robolens_yam"]`,
   `files=["src/robolens_yam"]`, `import robolens_yam`, `find_spec("robolens_yam")`,
   and (crucially) keeps `robocurve/robolens-yam` intact even though
   `robocurve/robolens` is a prefix of it. It still rewrites `from robolens.*`,
   `import robolens`, `"robolens"`, `robolens>=`, `robolens.tasks`, `robolens run`,
   `robocurve/robolens"`, etc. Exclude `.git/.venv/reference/site/*.lock` and binaries.
4. **[Mechanical] Reinstall the *plugin itself* after editing its entry-point groups.**
   Editable installs bake group names into `.dist-info/entry_points.txt` at install
   time. After editing each plugin's `[project.entry-points]`, run
   `uv pip install -e .` (or `--no-deps -e .`) in that plugin's venv *before* its
   gates, or `entry_points(group="roboinspect.tasks")` is empty and discovery +
   `test_tasks`/`test_api_snapshot` fail.
5. **[Mechanical] Phase A recreates the venv** (`rm -rf .venv && uv venv && uv pip
   install -e ".[dev]"`) so the old `robolens-*.dist-info` (stale import + stale
   `robolens.*` groups) doesn't linger.
6. **[Mechanical] Explicitly covered** (the perl rule handles them, but verify):
   `[project.scripts] roboinspect = "roboinspect.cli:main"`, the `robolens[rerun]`
   self-extra → `roboinspect[rerun]`, sdist `include=`, `packages=`,
   `known-first-party`, `files=`, coverage `source=`, `registry.py` `_GROUPS` +
   docstring, and `tests/test_registry_cli.py` group literal `"robolens.policies"`.
7. **[Nice] `uv.lock`** (untracked, gitignored in robolens) still says
   `name = "robolens"`; regenerate via `uv lock` after the rename (or ignore — it's
   not committed). Excluded from the verification grep.
8. **[Confirmed] Tag order** roboinspect → kitchenbench → (robolens-yam, worldevals)
   is correct; `[tool.uv.sources]` is not transitively inherited, so each repo keeps
   its own `roboinspect`/`kitchenbench` source pins. GitHub required-check contexts
   and Pages env survive `gh repo rename`.

## §7 Optional follow-up (not in this change)

Rename `robolens-yam` → `roboinspect-yam` (repo + `roboinspect_yam` package +
`robolens-yam-preflight` → `roboinspect-yam-preflight`). Deferred because the user
asked to rename *robolens* and *references*, not the sibling's own name; called out
so it's a deliberate choice, not an oversight.
