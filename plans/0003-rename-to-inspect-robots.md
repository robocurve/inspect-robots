# 0003 — Rename `roboinspect` → `inspect-robots` (Inspect Robots), incl. plugins

**Status:** design + execution plan (pre-execution)
**Goal:** rebrand the whole ecosystem to **Inspect Robots**, including the two
plugins (which still carry the original `robolens` brand). Everything: GitHub
repos, local checkouts, Python packages/imports/CLIs, entry-point groups, and every
cross-repo reference.

## Naming map (locked)

| Context | Old | New |
|---|---|---|
| Display | RoboInspect | **Inspect Robots** |
| Framework repo / dist / CLI | `roboinspect` | **`inspect-robots`** (hyphen) |
| Framework import package | `roboinspect` | **`inspect_robots`** (underscore) |
| Framework src dir | `src/roboinspect/` | `src/inspect_robots/` |
| Entry-point groups | `roboinspect.{tasks,…}` | **`inspect_robots.{tasks,…}`** (underscore) |
| YAM plugin repo/dist | `robolens-yam` | **`inspect-robots-yam`** |
| YAM plugin package | `robolens_yam` | **`inspect_robots_yam`** |
| YAM preflight CLI | `robolens-yam-preflight` | **`inspect-robots-yam-preflight`** |
| Isaac plugin dir/dist | `robolens-isaacsim` | **`inspect-robots-isaacsim`** |
| Isaac plugin package | `robolens_isaacsim` | **`inspect_robots_isaacsim`** |
| Benchmark / collection | `kitchenbench`, `worldevals` | **unchanged** (only their refs change) |

**Key rule:** `roboinspect` maps to `inspect_robots` in Python-identifier
positions and `inspect-robots` in packaging-name / URL / CLI positions. The two
plugin names always carry a separator (`robolens-…` / `robolens_…`), which makes
their hyphen/underscore split unambiguous.

## The replacement strategy (ordered; the load-bearing part)

Applied per text file (via a Python script, not blind sed), excluding
`.git/.venv/site/reference/*.lock` and binaries, and **excluding ALL prior design
docs** (`plans/0001-*`, `plans/0002-*`, `plans/0003-*`). These are immutable
historical records: `plans/0002` documents the *previous* `robolens`→`roboinspect`
rename and legitimately contains ~68 bare-`robolens` and many `roboinspect`
mentions; rewriting them corrupts the history AND (a) trips Stage 0's bare-`robolens`
assertion below and (b) makes the final verification grep impossible to bring to
zero. BSD sed lacks lookahead, so use Python `re` (use `\1` for backrefs, not `$1`).

**Stage 0 — plugins' `robolens` brand (separator-preserving, unambiguous):**
- `robolens-` → `inspect-robots-`
- `robolens_` → `inspect_robots_`
- Then assert no bare `robolens` remains (there shouldn't be, given the `plans/*`
  exclusion above — the previous rename left `robolens` only inside
  `robolens-yam`/`robolens_yam`/`robolens-isaacsim`/`robolens_isaacsim`).

**Stage 1 — framework, blanket to the Python form:**
- `RoboInspect` → `Inspect Robots`
- `roboinspect` → `inspect_robots`   (makes every import / group / path correct)

**Stage 2 — fix the packaging / URL / CLI spots that need the hyphen form**
(each is a targeted, enumerated regex; order after Stage 1):
- `robocurve/inspect_robots\b` → `robocurve/inspect-robots`   (repo URLs; `\b` so
  `inspect-robots-yam` — already hyphen from Stage 0 — is untouched)
- `github\.io/inspect_robots` → `github.io/inspect-robots`     (Pages URL)
- `name = "inspect_robots"` → `name = "inspect-robots"`         (framework dist name)
- `"inspect_robots>=` → `"inspect-robots>=`                     (dependency specs)
- `inspect_robots\[` → `inspect-robots[`                        (extras, e.g. `[rerun]`)
- `inspect_robots = {` → `inspect-robots = {`                   (`[tool.uv.sources]` key
  + `{ workspace = true }`)
- `inspect_robots = "inspect_robots.cli:main"` →
  `inspect-robots = "inspect_robots.cli:main"`                  (`[project.scripts]` LHS
  is the CLI name → hyphen; the `.cli:main` target stays underscore)
- `inspect_robots (run|list|inspect|--version)\b` → `inspect-robots $1`  (CLI usage in
  docs — covers *all four* subcommands/flags, not just run/list)
- `\.\./inspect_robots\b` → `../inspect-robots`                 (**editable-install paths**;
  Stage 1 turned `-e ../roboinspect` into `-e ../inspect_robots`, but the moved dir is
  hyphen `inspect-robots`. `\b` leaves `../inspect-robots-yam` — already hyphen — alone.
  Sites: `kitchenbench/{pyproject.toml,CLAUDE.md}`, `robolens-yam/{pyproject.toml,CLAUDE.md}`)
- `inspect_robots-` → `inspect-robots-`                         (**mixed-form dist/plugin
  names**; Stage 1 mangles hyphen dist names like `roboinspect-openvla` →
  `inspect_robots-openvla`, invalid. This fixup is safe: it does not touch
  `inspect_robots_`, `inspect_robots.`, `inspect_robots[`, or `inspect_robots>`.
  Sites incl. `src/inspect_robots/registry.py` docstring, prose in
  `README.md`/`docs/index.md`/`docs/guide/plugins.md`)
- **`cli.py` runtime identity** (explicit, not regex-covered above): `prog="inspect_robots"`
  → `prog="inspect-robots"` and `version=f"inspect_robots {__version__}"` →
  `version=f"inspect-robots {__version__}"` (the argparse usage + `--version` output are
  the user-facing CLI name → hyphen). No test asserts these today, so they will pass gates
  while mislabelled unless fixed by hand — see `src/inspect_robots/cli.py`.
- **Leave underscore** (do NOT fix): `["inspect_robots"]` in ruff
  `known-first-party` and coverage `source` (import-package name), the
  `inspect_robots.*` entry-point groups, and `src/inspect_robots` paths.

**Stage 3 — code+prose sweep (NOT just `*.md`):** after Stages 0–2, grep across
`*.md`, `*.py`, `*.toml`, `*.yml`/`*.yaml` for lowercase `inspect_robots` that reads
as prose/brand or a hyphen-form site (not an import / group / path) and change to
`Inspect Robots` or `inspect-robots` as fits; grep for any stray
`roboinspect`/`RoboInspect`/bare `robolens`, **and** for the wrong *new* forms that
silently pass an old-brand grep: `inspect_robots-`, `import inspect-robots`,
`\.\./inspect_robots\b`, `prog="inspect_robots"`. Verification target: zero
`roboinspect`/`robolens` anywhere **and** zero of those wrong-new-forms (these brands
are fully retired — unlike 0002 there are **no** protected tokens).

## Directory / repo renames

**git mv (contents already fixed by the script):**
- `src/roboinspect` → `src/inspect_robots` (framework)
- `src/robolens_yam` → `src/inspect_robots_yam` (yam)
- `plugins/robolens-isaacsim` → `plugins/inspect-robots-isaacsim`, and inside it
  `src/robolens_isaacsim` → `src/inspect_robots_isaacsim`
  (root `[tool.uv.workspace] members=["plugins/*"]` still matches — no change)

**GitHub + local dirs:**
- `gh repo rename inspect-robots -R robocurve/roboinspect`; `~/roboinspect` →
  `~/inspect-robots`; update remote URL.
- `gh repo rename inspect-robots-yam -R robocurve/robolens-yam`; `~/robolens-yam` →
  `~/inspect-robots-yam`; update remote URL.
- (Isaac plugin is in-repo — no separate GitHub repo.)

## Versioning & tags (bump to 0.3.0, dependency order)

The package identity changes (`roboinspect` → `inspect-robots`), so cut fresh
tags. Order: **inspect-robots → kitchenbench → (inspect-robots-yam, worldevals)**.
- inspect-robots: dynamic version via hatch-vcs; push, then tag `v0.3.0`.
- kitchenbench: dep `inspect-robots>=0.3`, source `inspect-robots @ v0.3.0`,
  `pyproject` version 0.3.0 **and** the hardcoded `src/kitchenbench/__init__.py`
  `__version__ = "0.2.0"` → `"0.3.0"`; tag `v0.3.0`.
- inspect-robots-yam: deps `inspect-robots>=0.3` + `kitchenbench @ v0.3.0`,
  `pyproject` version 0.3.0. **Three coupled version sites** — miss any and the
  snapshot test fails: (1) `pyproject` `version`, (2) the hardcoded constant
  `src/inspect_robots_yam/__init__.py` `__version__ = "0.2.0"` → `"0.3.0"`,
  (3) the assert `tests/test_api_snapshot.py` `__version__ == "0.2.0"` → `"0.3.0"`.
- worldevals: dep `inspect-robots>=0.3`, source `@ v0.3.0`, `pyproject` version
  0.3.0 **and** `src/worldevals/__init__.py` `__version__ = "0.1.0"` → `"0.3.0"`.
- isaacsim plugin (in-repo): dep `inspect-robots>=0.3`, workspace source — version
  bump with the repo.

## Execution order (ONE linear order — content first, dirs last)

The dir-move-vs-venv chicken/egg is resolved by a fixed rule: **edit contents and
gate with dirs at their OLD names (old editable paths still valid), rename all
dirs/repos in one atomic step, then recreate every venv from the NEW hyphen
paths.** Never `-e ../inspect_robots` (underscore) — the moved dir is
`../inspect-robots` (hyphen); Stage 2's `\.\./inspect_robots\b` fixup makes the
committed files say so.

A) **inspect-robots repo** (still in `~/roboinspect`): Stage 0–3 script over `git
ls-files`; git mv the src dirs (framework `src/roboinspect`→`src/inspect_robots` +
isaacsim `src/robolens_isaacsim`→`src/inspect_robots_isaacsim` + the plugin dir
`plugins/robolens-isaacsim`→`plugins/inspect-robots-isaacsim`; yam is a separate
repo); recreate venv via `uv sync --all-packages --extra dev`; run **core**
(ruff/format/mypy/pytest --cov 100%) **and plugin** (ruff/mypy/pytest) gates;
`mkdocs build --strict`; `inspect-robots list` shows tasks/embodiments incl.
`isaacsim`. Commit.
B) **kitchenbench** (still in `~/kitchenbench`): script; recreate venv with the
**still-valid old path** `-e ../roboinspect` (dirs not moved yet). Gates. Commit
`--no-verify` (git-tag dep not pushed yet).
C) **worldevals** & **robolens-yam** (still in `~/robolens-yam`): script (yam repo
also git mv `src/robolens_yam`→`src/inspect_robots_yam`); recreate venvs with the
still-valid old paths (`-e ../roboinspect -e ../kitchenbench`); gates; bump versions
(all coupled `__version__` sites per Versioning §). Commit `--no-verify`.
D) **GitHub + dirs (atomic)**: rename both repos on GitHub, move
`~/roboinspect`→`~/inspect-robots` and `~/robolens-yam`→`~/inspect-robots-yam`,
update remotes, then **recreate every venv from the new hyphen paths**
(`-e ../inspect-robots`, `-e ../inspect-robots-yam` — never underscore), re-run all
gates from the new paths.
E) **Push + tag** in dependency order; verify all CIs green; update both GitHub
descriptions; confirm Pages redeploys at `robocurve.github.io/inspect-robots`.

## Gotchas (carried from 0002 + new)

- **Hyphen vs underscore** is the #1 risk — see Stage 2. A blanket
  `s/roboinspect/inspect-robots/g` would produce invalid Python (`import
  inspect-robots`); a blanket underscore would produce wrong dist names/URLs/CLI.
  The script does underscore-blanket then targeted hyphen fixes.
- **Reinstall the *plugin itself*** after editing its entry-point groups so
  `.dist-info/entry_points.txt` regenerates (else discovery finds nothing).
- **Editable installs + venvs hardcode the old path** — recreate after the dir move.
- **cwd-reset target disappears** when a dir is moved — use the new absolute paths.
- **`site/` gitignored** — never hand-edit; `mkdocs build` regenerates.
- **A collaborator (aris-zhu) owns the Isaac plugin** and pushed mid-work last
  time — `git fetch` every repo before pushing; integrate, don't clobber.
- **Move the `v0.2.0`/old tags? No** — leave them; cut fresh `v0.3.0`.
- **`numpy<2.5` dev pin** already present in all repos — keep (mypy determinism).
- **CLI script name** is `inspect-robots` (hyphen). `inspect-robots run --task …`.
- **CamelCase compound identifiers** (found in execution): the blanket
  `RoboInspect`→`Inspect Robots` (Stage 1) inserts a *space* into any CamelCase
  identifier built on the brand — the framework has `class RoboInspectError` (base
  exception, internal, not in `__all__`) used across `errors.py`/`rollout.py`. That
  becomes the invalid `Inspect RobotsError` and breaks parse/mypy/ruff. Fixup:
  `Inspect Robots([A-Za-z0-9_])` → `InspectRobots\1` (CamelCase, no space) after
  Stage 1. Verify with `git grep -nE 'Inspect Robots[A-Za-z0-9_]'` → zero.
- **Line-length regressions**: `roboinspect`→`inspect_robots` is +3 chars, so a few
  docstring/comment lines cross ruff's 100-col limit — reflow them. Run
  `ruff check --fix` (sorts the now-unsorted `InspectRobotsError` import) then fix
  residual `E501`s by hand; `ruff format` may reformat a file whose string contents
  shifted.

## Verification (done = all true)

- **Old brand gone:** `git grep -in 'roboinspect\|robolens' -- ':!plans/000[123]-*'`
  in every repo → **zero** (use `git grep`, not `grep -r`: it skips gitignored
  `uv.lock`/`site/`/`.venv`/`.mypy_cache`/`.ruff_cache`/`.pytest_cache` automatically;
  plain `grep -r` recurses into `uv.lock` and the caches and reports false hits).
  Exclude the historical design docs (`plans/0001-*`, `plans/0002-*`, `plans/0003-*`) —
  they legitimately record both retired brands.
- **No wrong *new* forms** (these pass an old-brand grep but are broken):
  `git grep -in 'inspect_robots-\|import inspect-robots\|\.\./inspect_robots\b\|prog="inspect_robots"\|version=f"inspect_robots ' -- ':!plans/000[123]-*'`
  → **zero** (the `version=f"inspect_robots ` alt catches a forgotten `cli.py`
  `--version` f-string fix).
- All repos: ruff + format + mypy + pytest --cov 100% green from the new paths;
  plugin suites green; `mkdocs --strict` builds.
- `inspect-robots list` works; `inspect_robots.tasks` discovery finds kitchenbench;
  `inspect-robots-yam-preflight` runs and finds `molmoact2`/`yam_arms`; `isaacsim`
  embodiment discovered.
- All CIs green; tags pushed in order; both repos renamed; docs redeployed.
