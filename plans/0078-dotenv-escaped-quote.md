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

- A quote preceded by a backslash does not terminate the value, but the
  backslash is kept literally. `KEY="a\"b"` parses as `a\"b` (four
  characters). No unescaping is introduced.
- Implementation: replace the `find` with a small forward scan from index 1
  that returns the first index `i` where `value[i] == quote` and
  `value[i - 1] != "\\"`. Keep the rest of the branch (`value[1:closing]`,
  comment after the closer ignored) unchanged. Roughly six lines; a helper
  `_closing_quote(value: str) -> int` returning `-1` when unmatched keeps
  the walrus expression readable.
- Consequence to document, not to special-case: a quoted value whose last
  character before the closing quote is a backslash (`KEY="a\"`) can no
  longer be closed, so it falls through the existing unmatched-quote path
  and is kept literally as `"a\"`. python-dotenv rejects that line outright,
  so there is no parity to preserve. Same for `KEY="a\\"`.
- Update the two `# python-dotenv semantics` comments to say the parity is
  for quoting and comment rules only, and that backslashes are literal.

## Tests

`tests/test_dotenv.py`, one new test
`test_read_dotenv_escaped_quote_does_not_close_value(tmp_path)` writing a
single file with these lines and asserting the parsed dict exactly:

- `A="a\"b"` → `a\"b`
- `B='a\'b'` → `a\'b`
- `C="{\"k\": 1}"` → `{\"k\": 1}`
- `D="a\"" # comment` → `a\"` (escaped quote, then a real closer, then a
  comment that is dropped)
- `E="a\"` → `"a\"` (trailing backslash: unmatched, kept literally)

Keep `LITERAL_ESCAPE` in the existing test untouched so the no-unescape
policy stays pinned. Coverage stays at 100%: the helper's found and
not-found paths are both exercised.

## Docs

- `CHANGELOG.md` under `## [Unreleased]` / `### Fixed`, `**Core:**` prefix,
  referencing #291; state the rule in one sentence ("a backslash-escaped
  quote no longer terminates a quoted `.env` value; backslashes stay
  literal").
- Grep `docs/` and `README.md` for `.env` / `dotenv`; if any page describes
  the quoting rules, add the same one sentence there. Do not claim
  python-dotenv parity anywhere.

## Out of scope

Unescaping (`\"` → `"`, `\n` → newline), multi-line values, variable
expansion. If those are ever wanted they are a separate, deliberate change
to the module's contract.
