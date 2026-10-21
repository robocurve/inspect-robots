# 0086 — Retire the Docusaurus docs site

Issue: [#580](https://github.com/robocurve/inspect-robots/issues/580)

## Outcome

The Docusaurus site published at docs.inspectrobots.org is no longer used.
This change removes the site, its build and deploy automation, and the
generated API reference. The guides in `docs/` stay and become the canonical
documentation, read as plain Markdown on GitHub. Every link that pointed into
the site is replaced with one that works on GitHub (and on PyPI for README
links).

Out of scope (repo settings / DNS, done by a maintainer after merge):
unpublishing GitHub Pages and removing the `docs.inspectrobots.org` CNAME.
Historical design notes in `plans/` and released `CHANGELOG.md` entries are
left untouched.

## Changes

### Delete

- `website/` (whole directory: Docusaurus config, sidebars, landing page,
  `scripts/check-api-docs.mjs`, `scripts/gen-llms-txt.mjs`, `static/` incl.
  `CNAME`, lockfile).
- `scripts/gen_api_docs.py` (Griffe → `docs/api/index.md`; only consumer was
  the site).
- `.github/workflows/docs.yml` (build + GitHub Pages deploy).

### CI and tooling

- `.github/workflows/ci.yml`: remove the `docs-build` job and drop it from
  the `needs` of both `ci-ok` and `alert-red-main`. `ci-ok` is the only
  required status check, so no ruleset change is needed.
- `.github/dependabot.yml`: npm `directories` becomes `[/tools/issue-bot]`.
- `.pre-commit-config.yaml`: drop the `^website/package-lock\.json$`
  large-file exclusion.
- `.gitignore`: drop the Docusaurus block (`website/build`,
  `website/node_modules`, `website/.docusaurus`) and the `docs/api/index.md`
  ignore if present.
- `pyproject.toml`: remove the `docs = ["griffe>=1.0"]` extra (only
  `gen_api_docs.py` used it) and the `"inspect-robots[docs]"` entry in the
  `dev` extra; remove `scripts/gen_api_docs.py` from mypy
  `files`; `[project.urls] Documentation` →
  `https://github.com/robocurve/inspect-robots/tree/main/docs`.
  Re-run `uv lock`; nothing else depends on griffe, so `griffe`,
  `griffecli` and `griffelib` drop out.
- `tools/issue-bot/Dockerfile`: the `griffe==2.1.0` pin is kept deliberately.
  It becomes dead weight but is harmless, and editing the Dockerfile forces an
  issue-bot sandbox image rebuild/deploy that is out of scope here.
- `CITATION.cff`: `url` → `https://github.com/robocurve/inspect-robots`.

### Docs content (`docs/`)

- **API links.** 67 links of the form
  `` [`Name`](/api/#inspect_robots.mod.Name) `` across 8 guide pages point at
  the generated API page. Rewrite each to a relative link to the defining
  source file, e.g. `` [`Observation`](../../src/inspect_robots/types.py) ``.
  The module path is resolved by a one-off script: take the longest prefix of
  the dotted anchor that exists as `src/<path>.py` or
  `src/<path>/__init__.py`. Any anchor that does not resolve fails the script
  and is fixed by hand. The script is not committed.
- **Admonitions.** The three `:::note` / `:::warning` blocks (quickstart,
  examples, gr00t cookbook) become GitHub alerts (`> [!NOTE]`,
  `> [!WARNING]`).
- **Front matter / other Docusaurus-only syntax.** Check for front matter,
  `{#custom-id}` heading IDs, MDX/JSX imports and site-absolute links
  (`](/...)`); convert any found to plain Markdown equivalents.

- **Index.** Add `docs/README.md` (GitHub shows it on the `docs/` tree page)
  listing the guides in the old sidebar order (quickstart, examples,
  rig-reference, concepts, writing-a-benchmark, policies-and-embodiments,
  scoring, live-view, logging-and-rerun, plugins, voice-mode, adapters, cli)
  then the gr00t-on-yam cookbook, each with a one-line description.

### Permanent link check

The strict Docusaurus build (`onBrokenLinks`/`onBrokenAnchors: 'throw'`,
plus the API-page anchors) and the `docs-build` "reject stale mkdocstrings
autorefs" step go away. `tests/test_docs_links.py` replaces them. It reads
`docs/**/*.md` (and `README.md` for the GitHub URL check) with
`encoding="utf-8"` and blanks fenced (```` ``` ````/`~~~`) blocks before
scanning, including when collecting heading slugs. Links are matched on the
raw line (so `` [`Name`](...) `` link text survives) and skipped only when the
link itself starts inside an inline code span; the forbidden-syntax check runs
on the line with code spans blanked. A sanity test asserts the symbol check
examined at least 60 links. It asserts:

1. Every inline link with a relative target (not `http(s):`, `mailto:`)
   resolves to an existing file or directory.
2. Every `#fragment` (same-file or cross-file `.md`) matches a GitHub-style
   heading slug in the target: lowercase, punctuation other than `-`/`_`
   removed, spaces → `-`, `-1`, `-2` suffixes for duplicates.
3. Every link whose text is a single backticked identifier and whose target
   is a `src/**/*.py` file points at a file that defines it
   (`class Name`, `def Name`, `async def Name`, or a module-level
   `Name =`/`Name:`).
4. Every `https://github.com/robocurve/inspect-robots/(blob|tree)/main/<path>`
   URL in `README.md` and `docs/` names an existing repo path.
5. No `](/api/`, `][inspect_robots` autorefs, `:::` admonition fences, or
   reference-style link definitions (`[x]: target`) remain in `docs/`.

Runs in the normal pytest job (the 100% coverage gate measures `src/` only).

### README, contributor and agent docs

- `README.md`:
  - Remove the Docs workflow badge (the workflow no longer exists).
  - Header links and in-body links to `docs.inspectrobots.org/guide/<page>/`
    → absolute GitHub URLs
    `https://github.com/robocurve/inspect-robots/blob/main/docs/guide/<page>.md`
    (absolute because the README is also the PyPI long description). The
    "Documentation" header link → the `docs/` tree URL (renders
    `docs/README.md`). Drop the
    "For LLMs" link.
  - "Documentation" section: point to `docs/guide/` on GitHub; drop the
    llms.txt sentence and the API-reference claim (docstrings are the API
    reference).
  - "Development": delete the "build the Docusaurus documentation site"
    paragraph and code block.
- `CONTRIBUTING.md` "Documentation": rewrite — docs are Markdown in `docs/`
  rendered by GitHub; keep public docstrings accurate; link API symbols to
  their source file with relative links; use GitHub alert syntax for
  callouts. No build step.
- `AGENTS.md`: replace the Docusaurus bullet with one line: guides live in
  `docs/` as plain GitHub Markdown; there is no docs site or build.
- `.oss-scanner/threat_model.md`: drop `website/` from the static-site line
  (and reword: `docs/` is Markdown only, no site).
- `changelog.d/580.removed.md`: the docs site is retired (guides now live in
  `docs/` on GitHub) and the `inspect-robots[docs]` extra is removed.

## Verification

- `git grep -n -i -e docusaurus -e docs.inspectrobots -e "website/" -e
  gen_api_docs -e llms.txt -e llms-full -e griffe -- ':!plans'
  ':!CHANGELOG.md' ':!changelog.d' ':!uv.lock' ':!docs/assets'
  ':!tools/issue-bot/Dockerfile'` returns nothing.
- `tests/test_docs_links.py` passes (covers `/api/` links, admonitions and
  link resolution).
- `uv lock --check`, `uv run ruff check .`, `uv run mypy`,
  `uv run pytest --cov` all pass; `uv run pre-commit run --all-files` passes.
- CI green on the PR (`ci-ok` no longer waits on `docs build`).

## Follow-ups (not in this PR)

- Maintainer: disable GitHub Pages and remove the `docs` CNAME DNS record for
  inspectrobots.org (or point it at the GitHub docs tree via a redirect).
- Close PR #579 (website npm overrides) as superseded; dismiss website-only
  Dependabot alerts once `website/package-lock.json` is gone (they auto-close
  when the manifest disappears).
