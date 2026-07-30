"""Render a directory of evaluation logs as a self-contained HTML index.

The CLI owns filesystem discovery and log parsing; this module accepts plain
row data and returns one dependency-free document. Keeping that boundary
pure makes escaping, ordering, and presentation independently testable.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexEntry:
    """One evaluation-log row in the directory index."""

    name: str
    page: str | None
    created: str
    instruction: str
    policy: str
    model: str | None
    status: str
    status_class: str
    metrics: Mapping[str, float]
    errored_trials: int
    termination: str
    error: str | None


_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #20242b;
  --muted: #68707d;
  --line: #dfe3e8;
  --link: #245ca6;
  --green: #19723b;
  --green-bg: #e9f6ed;
  --red: #a12a2a;
  --red-bg: #fbecec;
  --grey: #626a75;
  --grey-bg: #eef0f2;
  --neutral: #45546a;
  --neutral-bg: #edf1f6;
}
@media (prefers-color-scheme: light) {
  :root { color-scheme: light; }
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111419;
    --panel: #191d24;
    --text: #e7eaf0;
    --muted: #a4acb8;
    --line: #343a45;
    --link: #8ebcff;
    --green: #7ed99a;
    --green-bg: #193b27;
    --red: #ff9b9b;
    --red-bg: #492323;
    --grey: #c0c5cd;
    --grey-bg: #343943;
    --neutral: #b9c9df;
    --neutral-bg: #293342;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header { border-bottom: 1px solid var(--line); background: var(--panel); }
.header-inner, main { width: min(1500px, calc(100% - 32px)); margin: auto; }
.header-inner { padding: 26px 0 20px; }
h1 { margin: 0; font-size: 24px; font-weight: 650; }
.meta { color: var(--muted); margin-top: 7px; }
main { margin-bottom: 64px; }
.toolbar { margin: 22px 0 14px; }
label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
input {
  width: min(460px, 100%);
  padding: 9px 11px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  font: inherit;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; vertical-align: top; }
th {
  color: var(--muted);
  border-bottom: 1px solid var(--line);
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .04em;
  text-transform: uppercase;
  white-space: nowrap;
}
td { border-top: 1px solid var(--line); }
tbody tr:first-child td { border-top: 0; }
tbody tr:hover { background: var(--bg); }
a { color: var(--link); }
.when, .policy, .metrics, .termination, .log { white-space: nowrap; }
.instruction { min-width: 230px; max-width: 440px; }
.error-cell { color: var(--red); min-width: 150px; max-width: 280px; overflow-wrap: anywhere; }
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 9px;
  font-size: 12px;
  white-space: nowrap;
}
.status-completed { color: var(--green); background: var(--green-bg); }
.status-error { color: var(--red); background: var(--red-bg); }
.status-cancelled { color: var(--grey); background: var(--grey-bg); }
.status-neutral { color: var(--neutral); background: var(--neutral-bg); }
.errored { color: var(--red); font-weight: 650; white-space: nowrap; }
.muted, .empty { color: var(--muted); }
.empty { padding: 28px 12px; text-align: center; }
""".strip()

_ERROR_LIMIT = 160
_FILTER_KEY = "inspect-robots-index-filter"


def _escape(value: object) -> str:
    """Escape one foreign value at its HTML interpolation boundary."""
    return html.escape(str(value), quote=True)


def _number(value: int | float) -> str:
    """Format numeric log values compactly before their interpolation boundary."""
    return f"{value:.4g}"


def _link(page: str | None, content: str) -> str:
    """Wrap escaped cell content in a relative report link when one exists."""
    if page is None:
        return content
    return f'<a href="{_escape(page)}">{content}</a>'


def _error_cell(error: str | None) -> str:
    """Render a compact error with its complete value available as a tooltip."""
    if not error:
        return ""
    short = error if len(error) <= _ERROR_LIMIT else f"{error[: _ERROR_LIMIT - 1]}…"
    return f'<span title="{_escape(error)}">{_escape(short)}</span>'


def _row(entry: IndexEntry) -> str:
    """Render one fully escaped table row."""
    model_tail = "" if entry.model is None else entry.model.rsplit("/", 1)[-1]
    policy = _escape(entry.policy)
    if model_tail:
        policy += f' <span class="muted">/ {_escape(model_tail)}</span>'
    metrics = ", ".join(
        f"{_escape(name)}={_escape(_number(value))}"
        for name, value in sorted(entry.metrics.items())
    )
    if entry.errored_trials:
        marker = f'<span class="errored">({_escape(entry.errored_trials)} errored)</span>'
        metrics = f"{metrics} {marker}" if metrics else marker
    instruction = _link(entry.page, _escape(entry.instruction))
    log_name = _link(entry.page, _escape(entry.name))
    return (
        "<tr>"
        f'<td class="when"><time>{_escape(entry.created)}</time></td>'
        f'<td class="instruction">{instruction}</td>'
        f'<td class="policy">{policy}</td>'
        f'<td><span class="badge {_escape(entry.status_class)}">'
        f"{_escape(entry.status)}</span></td>"
        f'<td class="metrics">{metrics}</td>'
        f'<td class="termination">{_escape(entry.termination)}</td>'
        f'<td class="error-cell">{_error_cell(entry.error)}</td>'
        f'<td class="log">{log_name}</td>'
        "</tr>"
    )


def render_index(entries: Sequence[IndexEntry], *, title: str = "Inspect Robots runs") -> str:
    """Return one self-contained HTML document indexing evaluation logs."""
    ordered = sorted(entries, key=lambda entry: entry.created, reverse=True)
    rows = "".join(_row(entry) for entry in ordered)
    empty = (
        ""
        if rows
        else '<tr class="empty"><td class="empty" colspan="8">no evaluation logs found</td></tr>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_STYLES}</style>
</head>
<body>
<header><div class="header-inner">
  <h1>{_escape(title)}</h1>
  <div class="meta">{len(entries)} logs</div>
</div></header>
<main>
  <div class="toolbar">
    <label for="filter">Filter runs</label>
    <input id="filter" type="search" placeholder="instruction, policy, status, metric…">
  </div>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>When</th><th>Instruction</th><th>Policy</th><th>Status</th>
      <th>Metrics</th><th>Termination</th><th>Error</th><th>Log</th>
    </tr></thead>
    <tbody>{rows}{empty}</tbody>
  </table></div>
</main>
<script>
const key = "{_FILTER_KEY}", input = document.querySelector("#filter");
const rows = document.querySelectorAll("tbody tr:not(.empty)");
function applyFilter() {{
  const query = input.value.toLocaleLowerCase();
  rows.forEach(row => row.hidden = !row.textContent.toLocaleLowerCase().includes(query));
  try {{ localStorage.setItem(key, input.value); }} catch (_) {{}}
}}
try {{ input.value = localStorage.getItem(key) || ""; }} catch (_) {{}}
input.addEventListener("input", applyFilter);
applyFilter();
</script>
</body>
</html>
"""
