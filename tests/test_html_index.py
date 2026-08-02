"""Pure rendering tests for the self-contained eval-log directory index."""

from __future__ import annotations

from dataclasses import replace

from inspect_robots._html_index import IndexEntry, render_index

_EXPECTED_INDEX_SCRIPT = """const key = "inspect-robots-index-filter",
  sortKey = "inspect-robots-index-sort-key",
  sortDirectionKey = "inspect-robots-index-sort-direction",
  input = document.querySelector("#filter");
const rows = document.querySelectorAll("tbody tr:not(.empty)");
function applyFilter() {
  const query = input.value.toLocaleLowerCase();
  rows.forEach(row => {
    const searchable = `${row.textContent} ${row.dataset.id} ${row.dataset.stamp}`;
    row.hidden = !searchable.toLocaleLowerCase().includes(query);
  });
  try { localStorage.setItem(key, input.value); } catch (_) {}
}
try { input.value = localStorage.getItem(key) || ""; } catch (_) {}
input.addEventListener("input", applyFilter);
applyFilter();
const head = document.querySelector("thead"), body = document.querySelector("tbody");
function sortRows(column, direction, persist = true) {
  const index = column.cellIndex, dataKey = column.dataset.key;
  const ordered = Array.from(body.querySelectorAll("tr:not(.empty)"));
  ordered.sort((left, right) => {
    const a = left.cells[index].dataset.sort ?? left.cells[index].textContent.trim();
    const b = right.cells[index].dataset.sort ?? right.cells[index].textContent.trim();
    if (dataKey === "number" && (!a || !b)) return !a && !b ? 0 : (!a ? 1 : -1);
    const compared = a.localeCompare(b, undefined, { numeric: dataKey === "number" });
    return direction === "asc" ? compared : -compared;
  });
  ordered.forEach(row => body.append(row));
  head.querySelectorAll("th.sorted").forEach(th => th.classList.remove("sorted", "asc", "desc"));
  column.classList.add("sorted", direction);
  if (persist) {
    try {
      localStorage.setItem(sortKey, dataKey);
      localStorage.setItem(sortDirectionKey, direction);
    } catch (_) {}
  }
}
head.addEventListener("click", event => {
  const column = event.target.closest('th[data-key]:not([data-key=""])');
  if (!column) return;
  const direction = column.classList.contains("sorted") && column.classList.contains("asc")
    ? "desc" : "asc";
  sortRows(column, direction);
});
try {
  const savedKey = localStorage.getItem(sortKey);
  const savedDirection = localStorage.getItem(sortDirectionKey);
  const column = savedKey && head.querySelector(`th[data-key="${savedKey}"]`);
  if (column && (savedDirection === "asc" || savedDirection === "desc")) {
    sortRows(column, savedDirection, false);
  }
} catch (_) {}
document.querySelector("tbody").addEventListener("click", event => {
  const row = event.target.closest("tr[data-href]");
  if (!row || event.target.closest("a") || getSelection().toString()) return;
  if (event.shiftKey || event.altKey) return;
  if (event.metaKey || event.ctrlKey) {
    const opened = window.open(row.dataset.href, "_blank");
    if (opened) opened.opener = null;
  } else {
    location.href = row.dataset.href;
  }
});"""


def _entry(
    name: str,
    *,
    page: str | None = "run.html",
    created: str = "2026-07-30T12:00:00Z",
    instruction: str = "pick up the cube",
    status: str = "completed",
    status_class: str = "status-completed",
    metrics: dict[str, float] | None = None,
    errored_trials: int = 0,
    number: int | None = 1,
    stamp: str | None = "20260730_120000_deadbeef",
) -> IndexEntry:
    return IndexEntry(
        name=name,
        page=page,
        created=created,
        instruction=instruction,
        policy="agent",
        model="provider/models/claude-test",
        status=status,
        status_class=status_class,
        metrics={"success_at_end": 0.75} if metrics is None else metrics,
        errored_trials=errored_trials,
        termination="succeeded",
        error=None,
        number=number,
        stamp=stamp,
    )


def test_escaping_and_link_vs_no_link_rows() -> None:
    document = render_index(
        [
            _entry("linked.json", instruction="<script>alert(1)</script>"),
            _entry("broken.json", page=None),
        ]
    )

    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert '<a href="run.html">linked.json</a>' in document
    assert '<a href="run.html">&lt;script&gt;' in document
    assert ">broken.json</a>" not in document


def test_badge_classes_are_used_verbatim_for_display_status() -> None:
    document = render_index(
        [
            _entry("complete.json"),
            _entry("error.json", status="error", status_class="status-error"),
            _entry(
                "cancelled.json",
                status="cancelled",
                status_class="status-cancelled",
            ),
        ]
    )

    assert '<span class="badge status-completed">completed</span>' in document
    assert '<span class="badge status-error">error</span>' in document
    assert '<span class="badge status-cancelled">cancelled</span>' in document


def test_rows_are_newest_first_and_metrics_use_four_significant_figures() -> None:
    document = render_index(
        [
            _entry(
                "old.json",
                created="2026-07-29T12:00:00Z",
                metrics={"distance": 0.0123456},
            ),
            _entry(
                "new.json",
                created="2026-07-30T12:00:00Z",
                metrics={"success": 0.666666},
                errored_trials=1,
            ),
        ]
    )

    assert document.index("new.json") < document.index("old.json")
    assert "distance=0.01235" in document
    assert "success=0.6667" in document
    assert '<span class="errored">(1 errored)</span>' in document
    assert 'agent <span class="muted">/ claude-test</span>' in document


def test_filter_script_and_persisted_key_are_present() -> None:
    document = render_index([_entry("run.json")])

    assert 'id="filter"' in document
    assert "row.dataset.id" in document
    assert "row.dataset.stamp" in document
    assert "localStorage.setItem" in document
    assert "localStorage.getItem" in document
    assert "inspect-robots-index-filter" in document


def test_row_with_page_has_escaped_data_href() -> None:
    document = render_index([_entry("run.json", page='run"&report.html')])

    assert 'data-href="run&quot;&amp;report.html"' in document
    assert 'data-href="run"&amp;report.html"' not in document


def test_row_without_page_has_no_data_href_attribute() -> None:
    document = render_index([_entry("run.json", page=None)])

    assert '<tr data-href="' not in document
    assert ' data-href="' not in document


def test_delegated_row_click_listener_is_present_once() -> None:
    document = render_index([_entry("run.json")])

    assert document.count('document.querySelector("tbody").addEventListener("click"') == 1
    assert 'event.target.closest("tr[data-href]")' in document
    assert 'event.target.closest("a")' in document
    assert "getSelection().toString()" in document
    assert "event.shiftKey || event.altKey" in document
    assert "event.metaKey || event.ctrlKey" in document
    assert "row.dataset.href" in document
    assert "opened.opener = null" in document


def test_clickable_row_cursor_style_is_present() -> None:
    document = render_index([_entry("run.json")])

    assert "tbody tr[data-href] { cursor: pointer; }" in document


def test_empty_index_has_no_data_href_attribute() -> None:
    document = render_index([])

    assert "<!doctype html>" in document
    assert '<tr data-href="' not in document


def test_number_column_sort_values_and_missing_number_are_exact() -> None:
    """Run numbers are zero-padded, sortable, and missing numbers use an em dash."""
    document = render_index(
        [
            _entry("numbered.json", number=12),
            _entry("unreadable.json", page=None, number=None),
        ]
    )

    assert '<th data-key="number">Run Id</th>' in document
    assert '<td class="number" data-sort="0012">0012</td>' in document
    assert '<td class="number" data-sort="">—</td>' in document
    assert 'class="when" data-sort="2026-07-30T12:00:00Z"' in document
    assert '<span class="date">2026-07-30</span>' in document
    assert '<span class="time">T12:00:00Z</span>' in document


def test_rows_expose_escaped_log_id_and_stamp_for_filtering() -> None:
    """Each row carries its filename stem and run stamp as escaped search data."""
    document = render_index([_entry("run<&>.json", stamp="stamp<&>")])

    assert 'data-id="run&lt;&amp;&gt;"' in document
    assert 'data-stamp="stamp&lt;&amp;&gt;"' in document


def test_sort_headers_persistence_and_empty_row_skip_ship_in_one_script() -> None:
    """The sortable-index script persists both fields and excludes placeholder rows."""
    document = render_index([_entry("run.json")])

    assert document.count("<script>") == 1
    assert "inspect-robots-index-sort-key" in document
    assert "inspect-robots-index-sort-direction" in document
    assert 'body.querySelectorAll("tr:not(.empty)")' in document
    assert 'head.addEventListener("click"' in document
    assert 'th[data-key]:not([data-key=""])' in document
    assert "localStorage.setItem(sortKey, dataKey)" in document
    assert f"<script>\n{_EXPECTED_INDEX_SCRIPT}\n</script>" in document


def test_static_index_has_no_meta_refresh() -> None:
    document = render_index([_entry("run.json")], refresh_seconds=None)

    assert '<meta http-equiv="refresh"' not in document


def test_served_index_has_exact_meta_refresh() -> None:
    document = render_index([_entry("run.json")], refresh_seconds=60)

    assert '<meta http-equiv="refresh" content="60">' in document


def test_long_error_is_truncated_with_full_escaped_tooltip() -> None:
    error = 'failed <badly> "' + "x" * 200
    entry = replace(
        _entry("broken.json"),
        status="error",
        status_class="status-error",
        error=error,
    )

    document = render_index([entry])

    assert f'title="failed &lt;badly&gt; &quot;{"x" * 200}"' in document
    assert "…" in document
