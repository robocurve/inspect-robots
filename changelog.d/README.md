# Changelog fragments

Each PR describes its change in one small file here instead of editing
`CHANGELOG.md`. Every PR adds a different file, so PRs never conflict over the
changelog. A maintainer compiles the fragments into `CHANGELOG.md` with
[towncrier](https://towncrier.readthedocs.io/) when cutting a release.

## Naming

- `<issue>.<type>.md`: the GitHub issue the change addresses, e.g.
  `440.fixed.md`. towncrier appends the issue link, so do not add it yourself.
- `+<slug>.<type>.md`: for a change with no issue, e.g.
  `+doctor-device-slots.added.md`. The slug uses lowercase letters, digits and
  hyphens. Keep any links you need in the text.
- A second fragment for the same issue and type: `<issue>.<type>.1.md`.

`<type>` is one of the Keep a Changelog sections: `added`, `changed`,
`deprecated`, `removed`, `fixed`, `security`. A misspelled type would be
silently dropped from the release notes, so `tests/test_changelog_fragments.py`
rejects it.

`uv run towncrier create 440.fixed.md` (or `+.added.md` for a random slug)
creates the file for you.

## Content

One entry, written the way existing `CHANGELOG.md` entries read, without the
leading `- `:

```markdown
**Core:** a run in which no scene completed cleanly now warns instead of
passing silently.
```

Plugin entries keep their prefix, e.g. `**Agent plugin (0.28.0):** ...`.

## Releasing (maintainers)

See "Releasing (maintainers)" in `CONTRIBUTING.md`.
