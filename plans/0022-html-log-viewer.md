# 0022 — `inspect-robots view`: render an eval log as a self-contained HTML page

Issue: #132. Status: draft.

## Problem

`inspect-robots inspect LOG.json --transcript` prints a faithful but flat text
dump. For an agent trial with dozens of turns, the operator wants what Inspect
AI's log viewer gives LLM evals: a scannable page with the run header, scores,
and a conversation view where roles, tool calls, and the model's mandatory
per-call notes (issue #130) are visually distinct. Nothing in the ecosystem
renders robotics eval logs today.

## Design

A new subcommand and one new private module. No new dependencies (core stays
NumPy-only; rendering is stdlib: `html.escape` + string templates).

```
inspect-robots view LOG.json [-o OUT.html] [--open]
```

- Writes a **single self-contained HTML file** (inline CSS, no JS, no
  external assets, no network) rendering the full `EvalLog`.
- Default output path: the log path with its suffix replaced by `.html`
  (`with_suffix(".html")`; a suffixless path gains `.html` — covered by a
  test). `-o` overrides. `-o -` writes the document to stdout for piping.
- File mode prints `wrote OUT.html` to stdout; **stdout mode suppresses the
  wrote line** (it would corrupt the piped document).
- Guard rails (guidance-over-traceback, like `video`'s output handling —
  note `video --out` names a directory so its guard condition is the
  inverse; do not copy its message verbatim): an output path that exists
  **and is a directory** is a `SystemExit` with guidance; a missing parent
  directory is created (`parent.mkdir(parents=True, exist_ok=True)`, the
  same courtesy `video` extends); `-o -` together with `--open` is a
  `SystemExit` ("no file to open").
- `--open` opens the written file with stdlib `webbrowser.open` on the
  `resolve()`d path's `as_uri()` (relative paths make `as_uri()` raise).
  Best-effort: a `False` return **or** an exception degrades to a stderr
  warning; exit code unchanged.
- Exit code: 0 when the page was produced, nonzero only on production
  failure — the `video` artifact-producer convention, not `inspect`'s
  status mirror. The command's job is rendering; `view ... && xdg-open`
  must not read a failed *run* as a failed *render*. (The page itself shows
  the status badge front and center.)

### Module: `src/inspect_robots/_html.py`

Private, like `_video.py`; `__all__` and the API snapshot are untouched.
Entry point:

```python
def render_html(log: EvalLog, *, title: str) -> str
```

pure function from log to document text — trivially testable, no IO. The CLI
command owns path derivation, writing, and `--open`, and passes
`title=f"{log.eval.task} - {Path(path).name}"` (task identity plus which
file, ASCII separator).

**Import direction:** `_html.py` never imports from `cli.py`. The shared
transcript predicates `_is_chat_transcript` and `_chat_content`, and the
status display mapping (`_STATUS_DISPLAY` / `_display_status`), migrate into
`_html.py`; `cli.py` imports them from there (top-level is fine — `_html`
is stdlib-only, so it cannot violate the lazy-import rule that exists for
heavy optional deps). The text renderer's behavior is unchanged.

**Encoding:** the file handle opens with `encoding="utf-8",
errors="replace"` — lone surrogates survive the log's JSON round-trip but
not strict UTF-8. Stdout mode funnels through the same degrade
(`doc.encode("utf-8", errors="replace").decode("utf-8")` before writing),
because `sys.stdout` is strict on most terminals. Encode-time `replace`
substitutes ASCII `?` (U+FFFD is the *decode*-time replacement); the tests
assert `?` appears in place of the surrogate rather than an exception.

### Page structure (Inspect AI-viewer-inspired, dependency-free)

1. **Header bar** — task name, status badge, created timestamp,
   inspect-robots version, git commit (or an "unknown" chip when `None`).
   Badge classes: completed=green, error=red, cancelled=grey,
   started/anything-else=neutral. Labels come from the shared
   `_display_status` mapping so the page and the CLI never diverge
   (`success` displays as `completed`); unknown statuses render their raw
   text escaped in the neutral badge.
2. **Spec strip** — policy, embodiment, `policy_config` entries as
   definition pairs, seed / max_steps / mean inference latency (each with a
   None-omitted branch), duration, total steps.
3. **Metrics row** — one stat tile per `results.metrics` entry, plus
   scenes/trials/errored counts.
4. **Scene cards** — one `<section>` per `SceneResult`: instruction, status,
   reduced scores, per-epoch score chips, termination reasons (None entries
   render as an em-dash-free "n/a" chip), operator judgements, error text
   when present.
5. **Transcript panels** — inside each scene card, one `<details>` per trial
   with a recorded transcript. Default-open rule: open exactly when the
   whole log carries **exactly one non-None transcript**; otherwise all
   collapsed (both states tested). Chat-shaped transcripts (shared
   `_is_chat_transcript`) render as a conversation:
   - `system` → nested collapsed `<details>` (long, static);
   - `user` → observation bubble; content lists collapse media parts to an
     `[image]` chip exactly like `_chat_content`; content that is neither
     str nor list renders as a role-only bubble (the `_chat_content` → None
     branch);
   - `assistant` → response bubble; each entry of `tool_calls` renders as a
     monospace call line `move_by({...})`, with the same defensive guards
     as `_render_chat_transcript` (non-dict message/call/function,
     non-list `tool_calls` — each guard tested);
   - **notes** — for every tool call whose arguments carry a non-empty
     string under `"note"` (and for `done`/`give_up`'s `summary`/`reason`),
     the text renders as a highlighted callout labeled "agent note" above
     the call line. Arguments may arrive as a JSON string **or** as a dict
     (the wire format allows both; the text renderer already tolerates
     non-str); both are consulted. A string that fails `json.loads` renders
     escaped-verbatim with no callout. This is the issue-#130 visibility
     channel and the reason this page exists;
   - `tool` → result line attached under the calling bubble.
   Non-chat transcripts fall back to pretty-printed JSON in a `<pre>`,
   with long string values elided at 2048 chars and an explicit ASCII
   `[... N chars truncated]` marker (ASCII to match every other
   CLI-emitted string and to survive a C-locale stdout in `-o -` mode) — a
   transcript that fails the chat predicate can carry full base64 image
   parts, and an unbounded dump produces a file browsers cannot open.
   Elision runs on the **raw** value, then the result is escaped, so a cut
   can never land mid-HTML-entity.

Collapsing uses native `<details>/<summary>` only (no JavaScript). Colors
respect `prefers-color-scheme` (light and dark rules in one inline
stylesheet).

### Foreign-data rules (the security section)

Transcript text, instructions, scene ids, error strings, metric names —
everything from the log — is **foreign data** and is passed through
`html.escape(..., quote=True)` exactly once at interpolation time. No code
path emits log text unescaped; a transcript containing `<script>` must
render as literal text.

## CLI touchpoints (enumerated so none is forgotten)

- New `sub.add_parser("view", ...)` with help text, `-o`, `--open`.
- `_SUBCOMMANDS` tuple gains `"view"` (instruction-sugar guard; `view` has
  no interior whitespace so behavior is identical, but the tuple must not
  drift).
- `main()` dispatch chain gains the branch.
- Module docstring's subcommand list gains the entry.
- The two existing hints reading `hint: view it with: inspect-robots
  inspect …` (`_print_run_summary` and the KeyboardInterrupt path) are
  reworded to `hint: inspect it with: …` in the same PR — "view" now names
  a different command. Five assertions in `tests/test_registry_cli.py`
  hard-code the old wording and are updated with it.
  `_print_run_summary`, `inspect`'s "policy transcripts: recorded" line,
  and the eval-set surfaces that print log hints
  (`_print_eval_set_summary` and eval-set's KeyboardInterrupt path) each
  gain one hint advertising `inspect-robots view <log>`.
- README's "Read the recorded agent conversation with `inspect-robots
  inspect ... --transcript`" line gains the view alternative.

## Compatibility

- Works on any schema-v1 log: every field the renderer touches has a
  reading-old-logs default. Logs with no transcripts render
  header/metrics/scenes and a "no policy transcripts recorded" line.
- Notes are duck-typed from tool-call arguments; pre-#130 agent logs show
  no callouts. Nothing imports the agent plugin.
- Core version: minor bump on release (new user-facing command).

## Tests

Renderer tests in a new `tests/test_html_view.py` (mirroring `_video.py`'s
dedicated test file); CLI wiring in `tests/test_registry_cli.py` (the
existing `main([...])` + capsys/tmp_path/monkeypatch convention).

`render_html` (pure, no IO):
- Header/status/metrics/scene content present and escaped; badge class per
  status: success ("completed"), error, cancelled, started, and an unknown
  string (neutral fallback).
- None-vs-present branches: `git_commit`, `seed`, `max_steps`,
  `mean_inference_latency_s`, `scene.error`, `scene.instruction`, empty
  `epochs`/`operator_judgements`, None entries in `termination_reasons`.
- XSS: `<script>alert(1)</script>` in instruction, message content, tool
  arguments, scene id, error string, metric name, and status → appears only
  escaped.
- Chat transcript: roles, tool-call lines, tool results; system message in
  a nested collapsed details; note in string-JSON arguments → exactly one
  callout; note in dict arguments → callout; `done` summary and `give_up`
  reason → callouts; malformed argument string → escaped verbatim, no
  callout; non-string/empty/whitespace note → no callout.
- Defensive guards: non-dict message, non-list `tool_calls`, non-dict call,
  non-dict function, content neither str nor list.
- Media parts collapse to `[image]`.
- Non-chat transcript → `<pre>` fallback; a long string value is elided
  with the truncation marker; `None` transcripts skipped; transcript-free
  log renders the "none recorded" line.
- Default-open: exactly-one-transcript log renders an open panel;
  two-transcript log renders all collapsed.

CLI:
- Default path derivation (including a suffixless log path), `-o` override,
  `-o -` stdout mode (document on stdout, **no** wrote line).
- `-o` pointing at an existing directory → SystemExit guidance; `-o -`
  plus `--open` → SystemExit; `-o` with a missing parent directory →
  parents created and the file written.
- Surrogate content: file mode writes replacement chars; stdout mode
  degrades identically instead of crashing.
- `--open`: `webbrowser.open` receives the resolved `file://` URI
  (monkeypatched); a `False` return warns on stderr; a raising open warns
  on stderr; exit code unchanged in both.
- Exit code 0 for success logs, 1 otherwise.
- New hint lines from `run` summary and `inspect`; reworded existing
  "inspect it with" hints asserted.

Gates: 100% core coverage (`--cov-fail-under=100`), `mypy --strict`
(src+tests), ruff incl. D1.

## Docs

- README subcommand list gains `view` (drafted to the repo's public-text
  style rules: no em dashes, no mid-sentence bold).
- `docs/guide/cli.md` gains a `## inspect-robots view` section beside
  `inspect` and `video`.
- `docs/guide/logging-and-rerun.md`'s transcript-reading passage mentions
  the HTML viewer.
- `CHANGELOG.md` entry under core.

## Rollout

Single PR (`Closes #132`), core only, after the mandatory-notes PR
(issue #130) merges — the note-callout fixtures mirror the real agent's
argument shape. Release the next core minor immediately after merge.
