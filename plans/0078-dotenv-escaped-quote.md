# 0078: read_dotenv must not close a quoted value at an escaped quote

Closes #291.

## Problem

`read_dotenv` (`src/inspect_robots/_dotenv.py`) finds a quoted value's
closing quote with `value.find(value[0], 1)`, which has no escape awareness.
A backslash-escaped quote inside the value is taken as the closer and
everything after it is dropped: `KEY="a\"b"` parses as `a\`, and
`CFG="{\"k\": 1}"` parses as `{\`. `init_dotenv` then exports the truncated
value to `os.environ` with no warning. This runs on every CLI invocation
with a `.env` in the working directory, and via the public
`inspect_robots.defaults.init_dotenv` re-export for plugins. Reproduced on
`main` (`f4b6f03f`); the existing `tests/test_dotenv.py` passes, so nothing
pins the correct behaviour.

## Design

Escape policy first, because the reporter asked for it and two contributors
have paused on it. This module deliberately does **no** escape processing:
the existing `LITERAL_ESCAPE="left\\nright"` case in
`test_read_dotenv_parses_supported_syntax` pins that `\n` stays two literal
characters, which already diverges from python-dotenv. Keep that policy.
The fix is therefore:

- A quote is escaped, and does not terminate the value, when it is preceded
  by an **odd-length** run of backslashes; an even-length run (zero
  included) leaves the quote as the closer. Backslashes are kept literally
  either way. So `KEY="a\"b"` parses as `a\"b` (four characters) and
  `KEY="a\\"` still parses as `a\\` exactly as on main today. This is how
  python-dotenv tokenises `\\"` too, and it is the rule that keeps every
  value that works today byte-identical (Windows paths and regexes ending in
  `\\` are the realistic cases). No unescaping is introduced.
- Implementation: replace the `find` with a helper
  `_closing_quote(value: str) -> int` that scans from index 1, counts the
  run of backslashes immediately before each candidate quote, and returns
  the first index where that run has even length; `-1` when unmatched. Keep
  the rest of the branch (`value[1:closing]`, comment after the closer
  ignored) unchanged. About ten lines.
- Consequence to state in code comment and CHANGELOG, not to special-case:
  a quoted value whose content ends in a lone backslash (`KEY="a\"`,
  `W="C:\data\"`) can no longer be closed under any escape-aware rule, so
  it falls through the existing unmatched-quote path and is kept literally,
  surrounding quotes included (`"a\"`). python-dotenv rejects such lines
  outright, so there is no parity to preserve; the behaviour change versus
  main is confined to this case.
- Update the two `# python-dotenv semantics` comments to say the parity is
  for quoting and comment rules only, and that backslashes are literal.

## Tests

`tests/test_dotenv.py`, one new test
`test_read_dotenv_escaped_quote_does_not_close_value(tmp_path)` writing a
single file with these lines and asserting the parsed dict exactly. Write
the fixture as a raw string (`r"""..."""`) or with doubled backslashes: the
existing test uses a non-raw literal where `"left\\nright"` becomes one
backslash in the file, and a `\"` typed in a non-raw literal silently
becomes a bare `"`, which would pin the wrong bytes.

- `A="a\"b"` → `a\"b`
- `B='a\'b'` → `a\'b`
- `C="{\"k\": 1}"` → `{\"k\": 1}`
- `D="a\"" # comment` → `a\"` (escaped quote, then a real closer, then a
  comment that is dropped)
- `F="a\\"` → `a\\` (even run: the quote closes; unchanged from main)
- `E="a\"` → `"a\"` (odd run at the end: unmatched, kept literally with
  its quotes)

Also assert one fixture line's bytes (e.g. `r'A="a\"b"' in
path.read_text()`) so a future non-raw edit of the fixture fails loudly.
Keep `LITERAL_ESCAPE` in the existing test untouched so the no-unescape
policy stays pinned. Coverage runs with branch measurement; the helper's
found and not-found arms and the even/odd run branches are all exercised.

## Docs

- `CHANGELOG.md` under `## [Unreleased]` / `### Fixed`, `**Core:**` prefix,
  in the neighbouring entries' link style
  `([plan 0078](plans/0078-dotenv-escaped-quote.md), [#291](https://github.com/robocurve/inspect-robots/issues/291))`.
  State the rule in one sentence (an escaped quote no longer terminates a
  quoted `.env` value; backslashes stay literal) and the one visible change
  (a quoted value ending in a lone backslash is now kept literally with its
  quotes).
- No docs page describes the quoting rules (`docs/guide/cli.md:90-93`,
  `docs/guide/quickstart.md`, `README.md:106-107`, `.env.example` only
  mention `.env`), so no docs page changes. Extend the `_dotenv.py` row in
  `src/inspect_robots/CLAUDE.md` with "backslashes are literal; an escaped
  quote does not close a value". Do not claim python-dotenv parity anywhere.

## Out of scope

Unescaping (`\"` → `"`, `\n` → newline), multi-line values, variable
expansion. If those are ever wanted they are a separate, deliberate change
to the module's contract.
