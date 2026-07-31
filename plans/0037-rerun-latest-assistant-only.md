# 0037 — RerunSink: limit the llm/latest reading pane to the assistant message

Issue: #243.

## Problem

`RerunSink` emits each step's transcript delta twice: as `TextLog` rows at
`{prefix}/llm` (one row per message, leveled INFO/DEBUG/TRACE by role) and as
one markdown `TextDocument` at `{prefix}/llm/latest`. The document currently
concatenates **every** rendered row, so a typical step reads

```
**[INFO]** user: Current observation ...
---
**[INFO]** assistant: tool_call move_by({"dx": 0.1})
---
**[DEBUG]** tool: result
```

The pane opens on observation boilerplate and the policy's actual decision is
buried behind a separator. The `llm/latest` pane should answer "what did the
policy just decide": one entry, the assistant message.

## Design

All changes in `src/inspect_robots/logging/rerun_sink.py`. The `{prefix}/llm`
`TextLog` stream is untouched — it stays the complete leveled conversation.

### 1. Carry the role through the payload

`_render_message` returns `(role, level, text)` instead of `(level, text)`:

- dict messages: the existing `str(message.get("role", "unknown"))`.
- non-dict messages: role `""` (they render as bare INFO rows with no
  `role:` prefix today; that stays).

`_TranscriptPayload.entries` becomes `tuple[tuple[str, str, str], ...]`
(role, level, text). Both are private, so this is not an API change; the
public-API snapshot is unaffected.

Every consumer of `entries` unpacks the new triple: the `TextLog` row loop in
`_emit_transcript` becomes `for _role, level, text in payload.entries:` (rows
themselves are emitted exactly as today), and the document filter below uses
the role.

Accepted trade-off: non-dict messages carry role `""`, so they can never
appear in `llm/latest` (previously they did, as bare INFO rows). The live
sink path does not normalize message shapes (`sink.py` contract), but the
pane is defined as "the policy's decision", and a policy that streams plain
strings still gets them in the `{prefix}/llm` TextLog stream. Documented in
Non-goals.

### 2. Filter the document (`_emit_transcript_document`)

Select `[(level, text) for role, level, text in entries if role == "assistant"]`:

- Non-empty selection: log the document exactly as today (same
  `**[{level}]** ` prefix, same `\n\n---\n\n` separator, same hard-break
  newline handling, same `media_type="text/markdown"`). A delta normally
  holds one assistant message; if it holds several, all of them appear in
  order — the filter is by role, not "last one wins".
- Empty selection (tool-result-only delta, plain-string rows, system/user
  only): **skip `rr.log` entirely.** Rerun's latest-at-time semantics then
  keep the previous assistant document visible at the cursor instead of
  blanking the pane. This also subsumes the existing
  `if ... not payload.entries: return` guard, which stays as written.

The `**[INFO]**` prefix is retained even though the filtered pane is always
INFO: it keeps the format identical to today's entries and to the TextLog
levels, and costs nothing.

### 3. Docs

- Module docstring (lines 44–46): "...paired with a markdown `TextDocument`
  at `{prefix}/llm/latest` holding the step's assistant message(s) for a
  wrapped, timeline-synced reading pane."
- `docs/guide/logging-and-rerun.md` "Live transcript in the viewer": the
  `llm/latest` sentence becomes "The most recent assistant message is also
  available as markdown at `trial/<scene>/e<epoch>/llm/latest`; add a Text
  Document view for a wrapped reading pane that stays synchronized with the
  timeline cursor and always shows the policy's latest decision."

## Non-goals

No change to `{prefix}/llm` row emission, levels, or `_render_message` text
formatting; no change to backpressure/eviction accounting (transcript payloads
still count as one unit whether or not a document is emitted); no config knob
for choosing which roles the pane shows. Non-dict (role-less) rows are
deliberately excluded from the pane (see Design §1); they remain in the
TextLog stream.

## Tests (tests/test_rerun_sink.py)

Updated expectations:

- `test_policy_messages_emit_ordered_levels_on_the_step_timeline` (the
  5-role `log_policy_messages` test): document body becomes
  `"**[INFO]** assistant: answer"` only; the five TextLog rows are unchanged.
- `test_transcript_emission_logs_rows_and_markdown_document`: document body
  `"**[INFO]** assistant: answer"`; the DEBUG tool row still appears in the
  TextLog stream.
- `test_transcript_document_composes_multiline_entries_with_hard_breaks`:
  rebuild around an assistant multiline message (content parts) so the
  hard-break (`"  \n"`) behavior is still exercised on the surviving path.
- `test_transcript_payload_emits_exactly_one_document_for_multiple_entries`:
  payload entries gain roles; use two assistant entries plus one tool entry —
  still exactly one document.
- `test_policy_message_rendering_table`: expected tuples gain the role
  element (`("assistant", "INFO", ...)`, `("", "INFO", "plain row")`, ...).
- Payload literals elsewhere in the suite
  (`("INFO", "first")` → `("", "INFO", "first")` or similar) updated
  mechanically.

New:

- Tool-only delta: `_emit_transcript` on entries
  `(("tool", "DEBUG", "tool: result"),)` logs the TextLog row and **no**
  document (assert no `llm/latest` path in the fake-rerun log).
- Mixed delta ordering: user + assistant + tool in one payload yields a
  document containing only the assistant line (regression for the issue's
  exact complaint).

Coverage stays at the 100% gate; the skip branch is exercised by the
tool-only test.

## Rollout

Changelog `### Changed` under `[Unreleased]`: the `llm/latest` Rerun pane now
shows only the step's assistant message(s); the full conversation remains in
the `llm` TextLog stream (plan 0037, #243).
