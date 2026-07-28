# 0030 — rerun sink: wrapped LLM transcript reading pane (TextDocument)

Issue: #203. PR: #204. Status: gate-clear after critique round 2 (round 1 findings applied (named the
exact-equality test the new emission changes; intra-entry newlines now become
markdown hard breaks so the sink's own multi-line structure survives
rendering); round 2: no substantive issues, minors folded in).

## Problem

The rerun sink emits each rendered transcript entry as a `rr.TextLog` row at
`{prefix}/llm` (`_emit_transcript`, `src/inspect_robots/logging/rerun_sink.py`).
The viewer renders `TextLog` as a table with single-line rows, and Rerun's
`TextLogView` has no word-wrap option (verified against rerun-sdk 0.34.1:
`TextLogFormat` exposes only `monospace_body`). LLM messages are long, so the
one view we give operators for the transcript truncates almost everything;
reading a message requires clicking its row (Selection panel) or hovering for
a tooltip, and neither follows the timeline cursor. During failure triage —
scrubbing a rollout while watching the cameras — the transcript is effectively
unreadable.

Replacing `TextLog` with `TextDocument` was considered and rejected: it would
lose the scannable history table, per-entry level coloring/filtering, and the
ability for multiple same-step entries to coexist (`TextDocument` has
latest-at semantics — same entity + same time index = overwrite).

## Design

Keep the `TextLog` emission exactly as is, and **additionally** log each
transcript payload's entries joined into one markdown `TextDocument` at
`{prefix}/llm/latest`:

```python
def _emit_transcript(self, rr: Any, payload: _TranscriptPayload) -> None:
    self._set_step(rr, payload.t)
    for level, text in payload.entries:
        rr.log(f"{payload.prefix}/llm", rr.TextLog(text, level=level))
    self._emit_transcript_document(rr, payload)
```

- **Latest-at is the feature.** A `TextDocumentView` on `{prefix}/llm/latest`
  shows the document current at the timeline cursor, so transcript reading
  stays in lockstep with scrubbing. Word wrap is on by default in that view.
- **One document per payload, not per entry.** Entries logged separately at
  the same step would overwrite each other; joining sidesteps that and gives
  the operator the whole step's exchange in one pane. Cross-payload overwrite
  at the same `t` is out of scope by contract: the sink protocol documents
  that the rollout calls `log_policy_messages` at most once per control step
  (`src/inspect_robots/logging/sink.py:8-9`).
- **Markdown.** `media_type="text/markdown"` (string literal, so we don't
  depend on the `rr.MediaType` constant existing). Each entry is prefixed
  with its level so the information `TextLog` carries in row color isn't lost
  in the document:

  ```
  **[INFO]** assistant: I can see the bowl…

  ---

  **[DEBUG]** tool: move_arm(…)
  ```

  Join separator: `\n\n---\n\n`. Raw LLM text may itself contain markdown;
  rendering it is desired (code blocks, lists), and a message that renders
  oddly is still strictly more readable than a truncated table cell.

- **Intra-entry newlines become markdown hard breaks.** `_render_message`
  joins content parts and tool-call lines with bare `"\n"`
  (`rerun_sink.py:92,103`), which markdown renders as a soft break — the
  entry would flatten into one run-on paragraph, losing structure the sink
  itself created (e.g. `"user: camera 'top':\n[image_url part]\nafter"`).
  The compose step therefore applies `text.replace("\n", "  \n")` (trailing
  two-space hard break) to each entry before prefixing the level. Known
  cosmetic caveat: if an entry contains a fenced code block, the added
  trailing spaces land inside the fence; they are invisible in rendering and
  acceptable. Same posture for two sibling ambiguities: raw content whose own
  `---` line follows a blank line renders like the entry separator (the
  `**[LEVEL]**` prefix disambiguates), and an unclosed code fence in one
  entry swallows the rendered separator and subsequent entries — renders
  oddly still beats a truncated table cell.

### Compatibility and edge cases

- **Old SDK surface without `TextDocument`:** resolve via
  `getattr(rr, "TextDocument", None)`; when absent, skip the document
  emission silently — the `TextLog` table still works, and warning once per
  eval about a purely additive nicety is noise. (This matches the sink's
  existing tiered-degradation philosophy but is even weaker than the
  compress fallback, deliberately: nothing is lost that the sink emitted
  before this plan.)
- **Empty `payload.entries`:** skip the document (don't log an empty body
  that would blank the pane at that step). The `TextLog` loop already
  no-ops naturally.
- **Where the join happens:** in the worker-thread emit path
  (`_emit_transcript_document`), not at enqueue time. Entries are already
  snapshotted tuples, the join is cheap, and this keeps the queue payloads
  and drop-accounting untouched — a dropped `_TranscriptPayload` drops both
  representations together, as it should.
- **Failure isolation:** `_worker_loop` already wraps `_emit` in the
  once-warned `try/except`; the document emission inherits that. No new
  failure surface on the control path.

### Non-changes

- No blueprint shipping. The sink has never sent a blueprint; choosing one is
  a separate discussion (would also cover camera/plot layout). Operators add
  a Text Document view manually or via their own blueprint.
- No changes to `_render_message`, payload dataclasses, queueing,
  backpressure, or drop accounting.
- No new config knob. The document costs ~the same bytes as the rows we
  already send; a toggle would be a knob nobody turns.

## Implementation steps

1. `_emit_transcript_document(rr, payload)` in `RerunSink`, called from
   `_emit_transcript`. Guard: `TextDocument` attr present, entries non-empty.
   Compose
   `"\n\n---\n\n".join(f"**[{level}]** " + text.replace("\n", "  \n") for
   level, text in entries)`, log at `f"{payload.prefix}/llm/latest"` with
   `media_type="text/markdown"`.
2. Module docstring: **add** a sentence stating that transcripts are emitted
   as `TextLog` rows at `{prefix}/llm` paired with a markdown `TextDocument`
   at `{prefix}/llm/latest` (wrapped, timeline-synced reading pane). The
   current docstring never introduces transcript entities at all — its only
   transcript mention is the backpressure drop-ordering clause, which is the
   wrong place to extend.
3. Tests (`tests/test_rerun_sink.py`), following the existing
   `_install_fake_rerun` pattern:
   - fake gains a `TextDocument` class capturing `(text, media_type)` **by
     default** — the fake should model the modern SDK surface; the legacy
     surface is the opt-out (`del`/flag), mirroring how the modern fake
     already carries `set_time`/`Scalars`.
   - **existing test updated:**
     `test_policy_messages_emit_ordered_levels_on_the_step_timeline`
     (`tests/test_rerun_sink.py:387-394`) asserts the logged list by exact
     equality; it gains the new `("trial/scene/e2/llm/latest", …)` entry.
     Audit the rest of the suite for other exact-equality assertions over
     `logged` that a transcript emission feeds.
   - transcript emission logs both the `TextLog` rows at `{prefix}/llm` and
     one document at `{prefix}/llm/latest`; document body contains every
     entry's text, level prefixes, and the separator; `media_type` is
     `"text/markdown"`.
   - a multi-line entry (content parts / tool_call lines) renders with hard
     breaks: assert the **full composed body** by exact equality for a known
     multi-line fixture (e.g. the existing
     `user: camera 'top':\n[image_url part]\nafter` rendering-table case),
     pinning level prefix, `"  \n"` transform, and separator in one
     assertion, matching the suite's exact-equality style.
   - multiple entries in one payload → exactly one document log call.
   - empty-entries payload → no document log call.
   - fake *without* a `TextDocument` attribute → no error, `TextLog` rows
     still logged, nothing logged at `…/llm/latest`.
4. Docs: `docs/guide/logging-and-rerun.md` § "Live transcript in the viewer"
   (currently "add a TextLog view") gains the `…/llm/latest` entity and the
   suggestion to add a Text Document view for wrapped, timeline-synced
   reading. No README change (its viewer-streams line is generic and does
   not enumerate transcript entities).

## Acceptance

- All existing tests pass; new tests cover every new branch (repo gate:
  100% coverage).
- `uv run inspect-robots run --rerun …` on a live viewer shows the existing
  `llm` table plus a `llm/latest` entity that renders wrapped markdown in a
  Text Document view and tracks the timeline cursor.

## Out of scope

- Shipping a default blueprint (table + document side by side).
- Any change to yam/agent packages; rig enablement is release + venv bump.
