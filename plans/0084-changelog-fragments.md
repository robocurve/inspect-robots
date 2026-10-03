# Changelog fragments with towncrier

Issue: [#528](https://github.com/robocurve/inspect-robots/issues/528)

## Outcome

Contributor PRs never edit `CHANGELOG.md`, so they cannot conflict with each
other there. Each PR adds one small fragment file under `changelog.d/`; a
maintainer compiles the fragments into `CHANGELOG.md` in the release-prep PR
with towncrier. This is the pattern used by Twisted (towncrier's origin),
pytest, pip and attrs; CPython does the same with blurb.

## Why now

Every PR inserts its entry at the top of the same `## [Unreleased]` subsection.
`main` requires branches to be up to date (ruleset 18603265), so each merge
leaves every other open PR conflicting there, and GitHub's "Update branch"
cannot resolve it. On 2026-10-02/03, five of nine serial merges needed a
maintainer push to the contributor's fork just to union two bullets, and each
push triggered a paid automated re-review.

## Current state (verified)

- `CHANGELOG.md` follows Keep a Changelog, but no version section has been cut
  since `## [0.6.0]`. `## [Unreleased]` is an ~800-line running log covering
  releases 0.7.0 through 0.60.0; plugin entries carry their own versions inline
  ("**Agent plugin (0.27.0):** …").
- GitHub Releases use auto-generated notes ("What's Changed"); the release
  workflow (`.github/workflows/release.yml`) never touches `CHANGELOG.md`, and it
  cannot push to `main` (PR-only rulesets, no bot bypass).
- Core releases are one-click (Actions → Release → Run workflow; root
  `CLAUDE.md`, "CI, merging, and releases"). Plugin releases go through a
  version-bump PR first (e.g. #513), which already edits `CHANGELOG.md`.
- `CONTRIBUTING.md` step 4 and the PR template both tell contributors to edit
  `CHANGELOG.md` under "Unreleased". There is no written release procedure.

## Design

### Fragments

- Directory: `changelog.d/`.
- Name: `<issue>.<type>.md`, where `<issue>` is the GitHub issue number. A PR
  with no issue uses an orphan fragment, `+<slug>.<type>.md` (contributors do not
  know their PR number before opening the PR; `towncrier create +.added.md`
  generates `+<hex>.added.md`). A second fragment for the same issue and type is
  `<issue>.<type>.1.md`, towncrier's own form.
- `<type>` is one of the Keep a Changelog sections: `added`, `changed`,
  `deprecated`, `removed`, `fixed`, `security`.
- Content: the entry prose in the existing style, e.g.
  `**Core:** a run with no clean scene now warns instead of passing silently.`
  Plugin entries keep their existing prefix form, e.g.
  `**Agent plugin (0.28.0):** …` (CHANGELOG.md:12).
  Do **not** add the issue link; towncrier appends
  `([#N](https://github.com/robocurve/inspect-robots/issues/N))` from the file
  name (verified with towncrier 26.9.0: a link in the content renders twice).
- `changelog.d/README.md` explains the convention; towncrier ignores it
  (verified).
- Output formatting is accepted as towncrier renders it: entries within a
  subsection form a tight list (no blank line between bullets, unlike the legacy
  entries), and there are two blank lines before the next `##` heading. Neither
  affects rendering; no custom template.

### Configuration (`pyproject.toml`)

```toml
[tool.towncrier]
directory = "changelog.d"
filename = "CHANGELOG.md"
start_string = "<!-- towncrier release notes start -->\n"
title_format = "## [{version}] - {project_date}"
issue_format = "[#{issue}](https://github.com/robocurve/inspect-robots/issues/{issue})"
underlines = ["", "", ""]
# one [[tool.towncrier.type]] per Keep a Changelog section, showcontent = true,
# in Keep a Changelog order: Added, Changed, Deprecated, Removed, Fixed, Security
```

`towncrier` is added to the `dev` extra only. Core stays NumPy-only.

### Guarding against silently dropped entries

towncrier silently ignores a fragment whose type is misspelled
(`9.bogus.md` is dropped from the build; verified). Add
`tests/test_changelog_fragments.py`, run by the normal pytest gate. It must pass
strict mypy at `python_version = "3.10"`, so it does **not** parse TOML
(`tomllib` is 3.11+, and a `tomli` fallback fails mypy on the 3.11 quality job).
Instead it hardcodes the six types as a tuple and cross-checks them against the
whole `pyproject.toml` read as plain text with a regex anchored on the type
tables, `re.findall(r'^\[\[tool\.towncrier\.type\]\]\s*\ndirectory = "(\w+)"',
text, re.M)`, asserting the result is non-empty and equal to the tuple in
order, so the two cannot drift. It fails
on any file in `changelog.d/` other than `README.md` / `.gitkeep` that does not
match `^(\d+|\+[a-z0-9][a-z0-9-]*)\.(<types>)(\.\d+)?\.md$` or that is empty
(an empty fragment renders as a bare link; verified). The sdist ships `tests`
but not `changelog.d/`, so the test skips when `changelog.d/` is absent. The
optional `.N` counter matches towncrier's own form for two fragments with the
same issue and type. It does not require a fragment per PR (see Non-goals).

### `CHANGELOG.md` migration (done entirely in this PR)

The migration happens in this PR, not at the next release, so there is never a
window where new entries land in the legacy block.

1. **Move unreleased legacy entries into fragments.** Every entry added to
   `CHANGELOG.md` since the `v0.60.0` tag (`git diff v0.60.0..HEAD --
   CHANGELOG.md`; today #450, #498, #502, #504, #479, #509, plus anything merged
   before this PR lands, e.g. #517-#519 and #438) has not shipped yet. Each
   becomes a fragment: type from the subsection it sat in, name from its single
   linked issue (`<issue>.<type>.md`) with the link stripped from the prose, or
   an orphan `+<slug>.<type>.md` when it links zero or several issues (then the
   links stay in the prose). The entries are removed from the legacy block.
   This is redone against the latest `main` immediately before merge.
2. **Retitle the legacy block.** `## [Unreleased]` becomes
   `## [0.7.0 to 0.60.0]: consolidated log`, unchanged otherwise: after step 1
   it holds exactly what shipped through v0.60.0. Delete the stale
   `[Unreleased]: …/compare/v0.3.0...HEAD` link definition (CHANGELOG.md:1049).
3. **Note and marker.** Directly above the retitled heading: a one-line note
   that unreleased changes live in `changelog.d/` and are compiled at release,
   then the towncrier start marker on its own line (note above marker, so builds
   insert new release sections below the note and above the legacy block).

After this PR there is no `[Unreleased]` section.

**Open PRs that still edit `CHANGELOG.md`.** 22 open PRs touch it today (525,
520, 514, 510, 487, 483, 482, 481, 472, 456, 455, 438, 437, 435, 434, 429, 427,
411, 376, 375, 349, 307). Rule, documented in the maintainer section of
`CONTRIBUTING.md` and in root `CLAUDE.md`: before merging **any** PR that
touches `CHANGELOG.md`, conflict or not, the maintainer moves its entry into a
`changelog.d/` fragment on the PR branch and drops the `CHANGELOG.md` hunk. The
retitle in step 2 makes most of those PRs conflict, which is harmless because
their `CHANGELOG.md` hunk is dropped anyway.

### Release procedure (new, documented in `CONTRIBUTING.md` and `CLAUDE.md`)

This adds one step before the one-click release: a small "changelog" PR, or
the plugin version-bump PR when there is one. `release.yml` computes the tag
from the bump chosen at dispatch, but the changelog PR needs the version first,
so the maintainer computes it the same way (latest `v*` tag plus the intended
bump) and must pick that same bump at dispatch. Fragments merged between the
changelog PR and the dispatch, or a skipped changelog PR, are not lost: those
entries are listed under a later version. Add a short
"Releasing (maintainers)" section to `CONTRIBUTING.md`, and amend the root
`CLAUDE.md` releases bullet to match: before dispatching the release, run
`uv run towncrier build --draft --version X.Y.Z` to preview, then
`uv run towncrier build --version X.Y.Z --yes`, commit the updated
`CHANGELOG.md` and deleted fragments, merge, then dispatch the release
workflow as today. Plugin-only releases include their fragments in the same way;
plugin entries keep their inline prefix, e.g. `**Agent plugin (0.28.0):**`.

### Contributor docs

- `CONTRIBUTING.md` step 4: "Add a changelog fragment
  `changelog.d/<issue>.<type>.md`, or `changelog.d/+<slug>.<type>.md` if there
  is no issue (slug: lowercase letters, digits, hyphens); see
  `changelog.d/README.md`. Do not edit `CHANGELOG.md`."
- PR template checkbox: "Changelog fragment added in `changelog.d/`".
- Root `CLAUDE.md` "Working here": one bullet with the same rule, since agents
  read it first; plus the releases bullet above.
- `tests/test_api_snapshot.py` module docstring: "note it in the changelog"
  becomes "add a changelog fragment".
- All new prose (README, CONTRIBUTING, CLAUDE.md) follows the repo's writing
  style: no em dashes.

### Dogfooding

This PR adds its own fragment, `changelog.d/528.changed.md`, and no
`CHANGELOG.md` entry.

## Non-goals

- No CI check that every PR has a fragment (`towncrier check`). Docs-only and
  CI-only PRs legitimately have none, and a blocking check would bounce
  external contributors. Revisit after a release cycle.
- No automated build in `release.yml`: it cannot push to `main`.
- No rewrite of the legacy section beyond the migration above.
- GitHub Releases keep their auto-generated notes.

## Files

```
pyproject.toml                       # [tool.towncrier] + towncrier in dev extra
uv.lock                              # lock update for the new dev dependency
CHANGELOG.md                         # entries since v0.60.0 moved out; retitle; note + marker
changelog.d/README.md                # fragment convention
changelog.d/528.changed.md           # this PR's own entry
changelog.d/*.md                     # entries migrated from the legacy block
tests/test_changelog_fragments.py    # fragment-name validation
CONTRIBUTING.md                      # step 4 + "Releasing (maintainers)"
.github/PULL_REQUEST_TEMPLATE.md     # checkbox
CLAUDE.md                            # agent guide bullets
tests/test_api_snapshot.py           # docstring wording
plans/0084-changelog-fragments.md    # this plan
```

## Verification

- `uv run towncrier build --draft --version 0.61.0` renders this PR's fragment
  under `## [0.61.0] - <date>` / `### Changed`, with the issue link appended
  once, above `## [Unreleased]`.
- The new test passes with the real `changelog.d/`, and fails for a misspelled
  type, an empty fragment, and a bad name (parametrized cases using `tmp_path`);
  it skips when `changelog.d/` is absent; its type tuple matches the pyproject
  block. It passes `mypy` at `python_version = "3.10"` with no TOML import.
- After the migration, `git diff v0.60.0..HEAD -- CHANGELOG.md` adds no
  entries inside the retitled block, every removed entry has a fragment with the
  same prose, and `towncrier build --draft` lists them all.
- Expected towncrier output quirks, not to be hand-edited: a blank line between
  the note and the marker, wrapped fragment lines indented 4 spaces, tight lists,
  two blank lines before the next `##` heading.
- Full gates: ruff, ruff format, strict mypy (covers tests), pytest at 100%
  coverage.
