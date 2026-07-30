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
    transcript/wire subdirectories live below and are not logs).
  - Per-log pages land in `<dir>/html/<log-stem>.html`; the index at
    `<dir>/html/index.html`.
  - `-o DIR` overrides the output directory (created if missing). `-o -` is
    rejected in directory mode: there is no single document to stream.
  - `--open` opens `index.html`.
  - `--no-frames` / `--frames-budget` pass through to every per-log render.
  - New flag `--force`: re-render pages that already exist. Without it,
    rendering is incremental — a page is re-rendered only when missing or
    older than its log (mtime comparison), so re-running after a few new runs
    is quick. `--force` is accepted and ignored in single-file mode (a single
    render is always "forced"); documented as directory-mode only.
- Progress goes to stderr (`[i/N] rendering <name>`), one line per log
  actually rendered, so a large first run is visibly alive while stdout stays
  clean. A final line reports totals: `index: <path> (N logs, M pages)`.
- Exit code 0 when the index is written, even if individual logs were
  unreadable — those are surfaced as error rows in the index and warnings on
  stderr. (A directory with no `*.json` at all is a usage error: exit 2 with
  a message, no empty index.)

### 2. Index rendering (`_html_index.py`)

New private module, sibling to `_html.py`, exporting one function:

```python
def render_index(entries: Sequence[IndexEntry], *, title: str = "Inspect Robots runs") -> str
```

with a small frozen dataclass `IndexEntry` carrying: `name` (log filename),
`page` (relative href or None), `created`, `instruction`, `policy`, `model`,
`status`, `errored` (bool), `reduced` (mapping), `termination`, `error`,
`size_mb`. The CLI builds entries; the module renders. This keeps the module
pure (string in, string out) and testable without touching the filesystem.

Index page properties:

- Single self-contained document: inline CSS + a few lines of inline JS, no
  external assets, `color-scheme: light dark` with a `prefers-color-scheme`
  dark palette — same constraints `_html.py` already honors.
- Table sorted newest-first by `created` (falling back to file mtime when a
  log is unreadable), columns: When, Instruction, Policy (policy / model
  tail), Outcome, Score, Termination, Error, Log. Instruction and Log cells
  link to the page when one exists.
- Outcome badge derived in the CLI, not the template: `error` when the log
  status is error or any trial errored; `cancelled`; else `success` /
  `failure` from whether any reduced metric reaches 1.0; `?` when no reduced
  metrics exist. (The reduced scorer metric is the task verdict; run status
  only says the run completed. `operator_judgements` is not it either —
  stock rollout never populates it.)
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

Entry metadata for readable logs comes from the parsed `EvalLog` (first
sample's instruction, `eval.policy`, `policy_config["model"]`, reduced
metrics of the first sample, termination reasons), mirroring what the per-log
report shows.

### 4. Hints (cli.py)

The two hint blocks gain one directory-level line, emitted only when the log's
parent directory is plausible (always is — logs are written into a directory):

- Post-run block (`hint: render videos with…` neighborhood):
  `hint: browse all logs: inspect-robots view <log_dir>`
- Eval-set completion block: same line with the set's `log_dir`.

Placed last in each block: it is the widest-scope hint.

## Tests (tests/test_html_index.py + additions to tests/test_cli_view.py area)

- `render_index`: escaping (instruction with `<script>`), link vs no-link
  rows, badge classes for each outcome, newest-first ordering, empty-entries
  rejected upstream (never called with zero entries).
- Directory mode end-to-end (tmp_path): two small synthetic logs → pages +
  index exist, index references both; unreadable third file → warning row, no
  page, exit 0.
- Incremental behavior: second invocation renders nothing (mtimes honored);
  touching a log re-renders its page; `--force` re-renders all.
- Flag validation: `-o -` with a directory exits with a usage error; `--open`
  targets index.html (monkeypatched webbrowser).
- Hint lines: post-run output includes the browse-all hint with the right
  directory.

## Rollout

Pure addition to the CLI + one new private module; no schema, wire, or plugin
surface changes. Changelog entry under core. No version gate needed by
downstream embodiment packages.
