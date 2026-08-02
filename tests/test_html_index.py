"""Pure rendering tests for the self-contained eval-log directory index."""

from __future__ import annotations

from dataclasses import replace

from inspect_robots._html_index import IndexEntry, render_index


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


def test_non_finite_metric_written_as_null_renders_as_not_available() -> None:
    """Regression for #253: sanitize() writes inf/nan metrics as JSON null;
    the index row must render that ``None`` instead of crashing on ``.4g``."""
    document = render_index(
        [_entry("run.json", metrics={"min_distance_to_goal": None})]  # type: ignore[dict-item]
    )

    assert "min_distance_to_goal=n/a" in document


def test_filter_script_and_persisted_key_are_present() -> None:
    document = render_index([_entry("run.json")])

    assert 'id="filter"' in document
    assert "row.textContent.toLocaleLowerCase().includes(query)" in document
    assert "localStorage.setItem" in document
    assert "localStorage.getItem" in document
    assert "inspect-robots-index-filter" in document


def test_row_with_page_has_escaped_data_href() -> None:
    document = render_index([_entry("run.json", page='run"&report.html')])

    assert '<tr data-href="run&quot;&amp;report.html">' in document
    assert '<tr data-href="run"&amp;report.html">' not in document


def test_row_without_page_has_no_data_href_attribute() -> None:
    document = render_index([_entry("run.json", page=None)])

    assert '<tr data-href="' not in document


def test_delegated_row_click_listener_is_present_once() -> None:
    document = render_index([_entry("run.json")])

    assert document.count('addEventListener("click"') == 1
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
