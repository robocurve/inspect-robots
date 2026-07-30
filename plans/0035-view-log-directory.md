# 0035 — `view` on a logs directory: browsable index of all runs

Issue: #234.

## Problem

`inspect-robots view` renders exactly one log to one self-contained HTML
report. A working `logs/` directory accumulates hundreds of runs, and the
package offers no way to browse them: which runs happened, with which policy,
which succeeded, which errored, and where the report for each one is. Users
hand-roll index builders that shell out to `view` per log (the rig carries a
~170-line `build_log_index.py` doing exactly this).

The post-run hint block advertises `inspect`, `view`, and `video` for the log
just written, but nothing points at the directory-level story, because none
exists.

## Design

Directory support folds into the existing subcommand rather than adding a new
one: `inspect-robots view logs/`. One command, one mental model — "view
renders logs to HTML" — with the argument deciding scope.

### 1. CLI surface (cli.py)

- `view` positional `log` accepts either an EvalLog JSON file (unchanged) or a
  directory. Help text: `path to an EvalLog JSON file, or a logs directory to
  render a browsable index`.
- Directory mode semantics:
  - Renders every top-level `*.json` in the directory (no recursion; frames/
    transcript/wire/learnings subdirectories are not logs; the sink's atomic
    `.json.tmp` + `os.replace` write means the glob never sees partials).
  - Per-log pages land in `<dir>/html/<log-stem>.html`; the index at
    `<dir>/html/index.html`. A log whose stem is exactly `index` renders to
    `html/index_log.html` so it cannot clobber the index (documented in the
    subcommand help). If `<dir>/html` exists and is not a directory:
    `SystemExit` with a clear message, before any rendering.
  - `-o PATH` (metavar changes from FILE to PATH): in single-file mode an
    output HTML file, exactly as today; in directory mode an output
    *directory*, created if missing, replacing the `<dir>/html` default. If
    it exists and is a regular file in directory mode: `SystemExit`. The
    existing "is a directory" guard applies only in single-file mode. `-o -`
    is rejected in directory mode: there is no single document to stream.
  - `--open` opens `index.html`.
  - `--no-frames` / `--frames-budget` pass through to every per-log render.
    The budget is **per page**; help text says so explicitly, since a
    first render of a large directory multiplies it (the totals line reports
    cumulative bytes written, so the cost is visible, and `--no-frames`
    or a lower budget is the stated remedy).
  - New flag `--force`: re-render pages that already exist. Without it,
    rendering is incremental — a page is re-rendered only when missing or
    older than its log (mtime comparison), so re-running after a few new runs
    is quick. Changing `--frames-budget`/`--no-frames` between runs does not
    invalidate existing pages; `--force` is the refresh path and its help
    text says so. `--force` is accepted and ignored in single-file mode.
    Incremental applies to page rendering only: index metadata still parses
    every log each run (`read_eval_log` is O(log bytes)), which is fine for
    v1 and stated here so nobody expects a no-op re-run to be free.
- Progress goes to stderr (`[i/N] rendering <name>`), one line per log
  actually rendered, so a large first run is visibly alive while stdout stays
  clean. The final totals line goes to stdout (mirroring plan 0022's `wrote`
  line): `index: <path> (N logs, M pages, X MB)`.
- Exit codes: 0 when the index is written, even if individual logs were
  unreadable — those are surfaced as error rows in the index and warnings on
  stderr. A directory with no top-level `*.json` at all is a runtime usage
  error: `SystemExit` with a message (exit 1, matching every other runtime
  validation error in cli.py), no empty index.

### 2. Index rendering (`_html_index.py`)

New private module (naming per `_video.py`/`_summarize.py`), exporting one
function:

```python
def render_index(entries: Sequence[IndexEntry], *, title: str = "Inspect Robots runs") -> str
```

with a small frozen dataclass `IndexEntry` carrying: `name` (log filename),
`page` (relative href or None), `created`, `instruction`, `policy`, `model`,
`status` (display string), `status_class` (badge class), `metrics` (mapping,
run-level), `termination`, `error`, `size_mb`. The CLI builds entries; the
module renders. This keeps the module pure (string in, string out) and
testable without touching the filesystem.

Index page properties:

- Single self-contained document: inline CSS + a few lines of inline JS, no
  external assets, `color-scheme: light dark` with a `prefers-color-scheme`
  dark palette — same constraints `_html.py` already honors.
- Table sorted newest-first by `created` (falling back to file mtime when a
  log is unreadable), columns: When, Instruction, Policy (policy / model
  tail), Status, Metrics, Termination, Error, Log. Instruction and Log cells
  link to the page when one exists.
- **Status, not verdict guessing.** The badge shows the run status through
  the existing display mapping (`_display_status`): completed / error /
  cancelled, with error also when any trial errored. The Metrics column
  shows the **run-level** `log.results.metrics` (`name=value`, 4 sig figs) —
  the same aggregate `run`/`inspect` print. No success/failure inference
  from a 1.0 threshold: reduced values are epoch means (1-of-2 trials is
  0.5) and metrics like `min_distance_to_goal` invert the scale, so any
  threshold badge would contradict the repo's existing outcome vocabulary
  (`_outcome_line`, `_display_status`). Users judge from the metrics they
  chose; the report page has the full per-scene story.
- **Multi-scene aware.** Instruction column: the shared instruction when all
  scenes share one (the `shared` logic `_cmd_inspect` already applies);
  otherwise `"<n> scenes"`. Termination column: the union of termination
  reasons across all scenes, deduplicated, order-preserved. `log.samples`
  may be empty (cancelled before the first trial): every per-sample access
  is guarded and such logs still get a row (status badge carries the story).
- A filter input that hides non-matching rows on substring match across the
  row text, persisted to `localStorage` so a reload keeps the filter.
- All user-controlled strings HTML-escaped; errors truncated in-cell with the
  full text in `title=`.

Non-goals, deliberately: no serve/watch mode, no auto-refresh meta tag (the
index is a static artifact; serving it is the user's business), no pagination
(a table of a few hundred rows is fine), no camera-grid changes to the per-log
report (separate concern, separate issue if wanted).

### 3. Per-log rendering in directory mode (cli.py)

Reuses the exact single-file path: `read_eval_log` → `resolve_frames_dir`
(unless `--no-frames`) → `render_html` → write UTF-8. Factored so both modes
call one helper rather than duplicating the frames-dir/budget handling. A log
that `read_eval_log` rejects (truncated write, foreign JSON) contributes an
index row with `error="unreadable: …"`, no page, and a stderr warning — one
bad file must not sink the index.

### 4. Hints (cli.py)

- Post-run block (`hint: render videos with…` neighborhood) gains, last (it
  is the widest-scope hint):
  `hint: browse all logs: inspect-robots view <log_dir>`
- Eval-set completion block: the existing
  `hint: HTML viewer: inspect-robots view {log_dir}/<task>_<id>.json`
  placeholder line (which the user must hand-edit) is **replaced** by
  `hint: browse all logs: inspect-robots view {log_dir}` — the directory form
  strictly dominates it, and two adjacent view hints would be noise.

## Tests (tests/test_html_index.py + tests/test_registry_cli.py)

`test_html_index.py` (pure renderer):

- Escaping (instruction containing `<script>`), link vs no-link rows, badge
  class per status, newest-first ordering, metrics formatting, filter JS and
  localStorage key present in the document.

`test_registry_cli.py` (CLI-level, alongside the existing `test_view_*`):

- Directory mode end-to-end (tmp_path): two small synthetic logs → pages +
  index exist, index references both; unreadable third file → error row, no
  page, exit 0; multi-scene log shows shared instruction and run-level
  metrics; empty-samples log gets a row without crashing.
- Incremental behavior: second invocation renders nothing; a log made newer
  than its page via explicit `os.utime` (not sleep — same-second mtimes on
  coarse filesystems) re-renders exactly that page; `--force` re-renders all.
- Collisions and flags: log named `index.json` → page at `index_log.html`,
  index intact; `<dir>/html` as regular file → clean `SystemExit`; `-o -`
  with a directory → clean `SystemExit`; `-o` pointing at an existing file in
  directory mode → clean `SystemExit`; empty directory → clean `SystemExit`;
  `--open` targets `index.html` (monkeypatched webbrowser).
- Hint lines: post-run output includes the browse-all hint with the right
  directory; eval-set output includes the directory hint and no longer the
  placeholder per-file hint.

## Rollout

Pure addition to the CLI + one new private module; no schema, wire, or plugin
surface changes. Changelog: `### Added` entry under `[Unreleased]` in the
root `CHANGELOG.md` with the `(plan 0035, #234)` reference recent entries
use. No version gate needed by downstream embodiment packages.
