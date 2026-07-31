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
   opens it in a new tab via `window.open(..., "_blank")` with
   `window.opener` neutralized on the opened page. Shift-click and
   alt-click do nothing (early return): their native meanings (new
   window, download) belong to real links, and the in-cell links remain
   available for them.
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
    if (event.shiftKey || event.altKey) return;
    if (event.metaKey || event.ctrlKey) {
      const opened = window.open(row.dataset.href, "_blank");
      if (opened) opened.opener = null;
    } else {
      location.href = row.dataset.href;
    }
  });
  ```

  (`opener = null` rather than a `"noopener"` features string: the
  features form demotes the tab to a window in some browsers.)

  Inside `render_index`'s f-string this needs `{{ }}` brace doubling,
  same as the filter code above it.

## Tests (tests/test_html_index.py)

The strings `data-href` and `tr[data-href]` appear in EVERY document via
the listener's selector and the CSS rule, so document-level substring
checks on those are vacuous or unsatisfiable. All assertions below target
the attribute form `<tr data-href="` specifically:

- Row with a page: `<tr data-href="<escaped page>">` present, value
  escaped (a page name containing `"` and `&` round-trips escaped — no
  raw quote inside the attribute value).
- Row without a page: the substring `<tr data-href="` does not appear in
  a document whose only entry lacks a page.
- Document contains the delegated listener exactly once, asserted against
  an independent hardcoded literal (not a constant imported from the
  module — the injection guard must not be self-referential; same lesson
  as plan 0036).
- The full CSS rule text `tbody tr[data-href] { cursor: pointer; }` is
  present (substring `tr[data-href]` alone would pass via the JS
  selector even with the style missing).
- Empty index: document renders, substring `<tr data-href="` absent.

## Non-goals

- No keyboard focusability for rows (links already serve keyboard users).
- No middle-click/auxclick, shift-click, or alt-click handling: their
  semantics vary by platform and the in-cell links already provide every
  native link behavior.
- No visual affordance beyond the cursor (the existing row hover
  highlight already signals interactivity).

## Release

Ships in the next minor alongside whatever else lands; no dedicated
release required — the rig can pick it up at the next venv refresh.
