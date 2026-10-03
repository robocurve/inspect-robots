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
- Name: `<issue>.<type>.md`, where `<issue>` is the GitHub issue (or PR) number,
  or `+<slug>.<type>.md` for an orphan entry with no issue.
- `<type>` is one of the Keep a Changelog sections: `added`, `changed`,
  `deprecated`, `removed`, `fixed`, `security`.
- Content: the entry prose in the existing style, e.g.
  `**Core:** a run with no clean scene now warns instead of passing silently.`
  Do **not** add the issue link; towncrier appends
  `([#N](https://github.com/robocurve/inspect-robots/issues/N))` from the file
  name (verified with towncrier 26.9.0: a link in the content renders twice).
- `changelog.d/README.md` explains the convention; towncrier ignores it
  (verified).

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
`tests/test_changelog_fragments.py`, run by the normal pytest gate, which reads
the type list from `pyproject.toml` and fails on any file in `changelog.d/`
other than `README.md` / `.gitkeep` that does not match
`^(\d+|\+[a-z0-9][a-z0-9-]*)\.(<types>)(\.\d+)?\.md$` or that is empty. The
optional `.N` counter matches towncrier's own form for two fragments with the
same issue and type. It does not require a fragment per PR (see Non-goals).

### `CHANGELOG.md` and migration of the legacy section

This PR inserts the towncrier start marker on its own line directly above
`## [Unreleased]`, with a one-line note that new entries go in `changelog.d/`.
The legacy `## [Unreleased]` block is left as is, so the ~40 open PRs that
already add an old-style entry still merge (they only conflict with each other,
as today). Nothing else in the file changes now.

At the **first release-prep after this lands**, the maintainer:

1. Runs `uv run towncrier build --version X.Y.Z --yes`, which writes
   `## [X.Y.Z] - <date>` with the fragment entries above the legacy block and
   deletes the fragments.
2. Moves the legacy `## [Unreleased]` entries into that same `## [X.Y.Z]`
   section (merging subsection by subsection) and deletes the empty
   `## [Unreleased]` heading. From then on there is no `[Unreleased]` section:
   unreleased changes live in `changelog.d/`.

Old-style entries that land after that release are moved into fragments by the
maintainer when merging (or into the next release section at release-prep).

### Release procedure (new, documented in `CONTRIBUTING.md` and `CLAUDE.md`)

This adds one step before the one-click release: a small "changelog" PR, or
the plugin version-bump PR when there is one. Skipping it is harmless:
fragments simply accumulate and are compiled at the next release. Add a short
"Releasing (maintainers)" section to `CONTRIBUTING.md`, and amend the root
`CLAUDE.md` releases bullet to match: before dispatching the release, run
`uv run towncrier build --draft --version X.Y.Z` to preview, then
`uv run towncrier build --version X.Y.Z --yes`, commit the updated
`CHANGELOG.md` and deleted fragments, merge, then dispatch the release
workflow as today. Plugin-only releases include their fragments in the same way;
plugin entries keep their inline "(plugin X.Y.Z)" prefix.

### Contributor docs

- `CONTRIBUTING.md` step 4: "Add a changelog fragment
  `changelog.d/<issue>.<type>.md` (see `changelog.d/README.md`); do not edit
  `CHANGELOG.md`."
- PR template checkbox: "Changelog fragment added in `changelog.d/`".
- Root `CLAUDE.md` "Working here": one bullet with the same rule, since agents
  read it first; plus the releases bullet above.

### Dogfooding

This PR adds its own fragment, `changelog.d/528.changed.md`, and no
`CHANGELOG.md` entry.

## Non-goals

- No CI check that every PR has a fragment (`towncrier check`). Docs-only and
  CI-only PRs legitimately have none, and a blocking check would bounce
  external contributors. Revisit after a release cycle.
- No automated build in `release.yml`: it cannot push to `main`.
- No rewrite of the legacy 800-line section beyond the one-time move above.
- GitHub Releases keep their auto-generated notes.

## Files

```
pyproject.toml                       # [tool.towncrier] + towncrier in dev extra
uv.lock                              # lock update for the new dev dependency
CHANGELOG.md                         # start marker + one-line note
changelog.d/README.md                # fragment convention
changelog.d/528.changed.md           # this PR's own entry
tests/test_changelog_fragments.py    # fragment-name validation
CONTRIBUTING.md                      # step 4 + "Releasing (maintainers)"
.github/PULL_REQUEST_TEMPLATE.md     # checkbox
CLAUDE.md                            # agent guide bullet
plans/0084-changelog-fragments.md    # this plan
```

## Verification

- `uv run towncrier build --draft --version 0.61.0` renders this PR's fragment
  under `## [0.61.0] - <date>` / `### Changed`, with the issue link appended
  once, above `## [Unreleased]`.
- The new test passes with the real `changelog.d/`, and fails for a misspelled
  type, an empty fragment, and a bad name (parametrized cases using `tmp_path`).
- Full gates: ruff, ruff format, strict mypy (covers tests), pytest at 100%
  coverage.
