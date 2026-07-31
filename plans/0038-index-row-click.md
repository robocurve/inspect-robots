# 0038 — Index row-click navigation

Issue: #245. Make each row of the directory index (`inspect-robots view
logs/`) clickable: clicking anywhere on a row whose run has a rendered
report opens that report, instead of requiring a hit on the instruction or
log-name link.

## Scope

`src/inspect_robots/_html_index.py` only, plus its tests. No CLI changes,
no changes to per-run report rendering (`_html.py`).

## Behavior contract

1. A row whose entry has `page is not None` carries
   `data-href="<escaped page>"` on its `<tr>` and shows `cursor: pointer`.
2. Clicking such a row navigates to `data-href` (same tab). Cmd/ctrl-click
   opens it via `window.open(..., "_blank")`, matching platform link
   conventions.
3. Clicks that land on an `<a>` (or inside one) are left entirely to the
   browser — no double navigation, native modifier handling preserved.
4. A click that concludes a text selection (`getSelection().toString()`
   non-empty) does not navigate; selecting an error message must stay
   possible.
5. A row whose entry has `page is None` gets no `data-href`, no pointer
   cursor, and no click behavior. The `.empty` placeholder row likewise.
6. The two existing in-cell links are kept as-is: they remain the
   accessible/keyboard path and the honest URL under hover, and row click
   is a convenience layered on top. (A row is not focusable; keyboard users
   tab to the links exactly as today.)

## Implementation

- `_row(entry)`: emit `<tr data-href="...">` when `entry.page` is not
  None, `<tr>` otherwise. The value goes through `_escape` like every
  other interpolation (it already does for the `<a href>`).
- CSS: `tbody tr[data-href] { cursor: pointer; }` appended to `_STYLES`.
- Script: extend the single existing `<script>` block with one delegated
  listener on `tbody` (not per-row listeners — rows are static but
  delegation is smaller and matches the report's `_FRAME_CLICK_SCRIPT`
  style):

  ```js
  document.querySelector("tbody").addEventListener("click", event => {
    const row = event.target.closest("tr[data-href]");
    if (!row || event.target.closest("a") || getSelection().toString()) return;
    if (event.metaKey || event.ctrlKey) window.open(row.dataset.href, "_blank");
    else location.href = row.dataset.href;
  });
  ```

  Inside `render_index`'s f-string this needs `{{ }}` brace doubling,
  same as the filter code above it.

## Tests (tests/test_html_index.py)

- Row with a page: `data-href="<page>"` present on the `<tr>`, value
  escaped (page name containing `"` and `&` round-trips escaped, no raw
  quote in the attribute).
- Row without a page: rendered `<tr>` has no `data-href` anywhere.
- Document contains the delegated listener exactly once, asserted against
  an independent hardcoded literal (not a constant imported from the
  module — the injection guard must not be self-referential; same lesson
  as plan 0036).
- CSS rule `tr[data-href]` present.
- Empty index: document renders, no `data-href` present.

## Non-goals

- No keyboard focusability for rows (links already serve keyboard users).
- No middle-click/auxclick handling: middle-click paste/scroll semantics
  vary by platform and the in-cell links already provide open-in-new-tab.
- No visual affordance beyond the cursor (the existing row hover
  highlight already signals interactivity).

## Release

Ships in the next minor alongside whatever else lands; no dedicated
release required — the rig can pick it up at the next venv refresh.
