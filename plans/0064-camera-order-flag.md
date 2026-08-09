# 0064 — `--camera-order`: viewer-chosen camera ordering for rows and run video

- **Status:** draft (R0)
- **Issue:** #350

## Problem

Turn frame rows render cameras in wire order — the order the policy received
them — and plan 0063's composite run video deliberately mirrors that order.
But wire order is a policy-facing artifact, not a presentation choice: a rig
that sends `top_cam` first produces reports reading `top · left · right`,
while a human reading the page expects the physical left-to-right arrangement
(`left · top · right`). There is no way to change it today.

## Design

One new `view` option:

```
--camera-order left_cam,top_cam,right_cam
```

Comma-separated **display** camera names. Cameras named in the flag render
first, in the given order; cameras not named keep their existing relative
order after them. The flag reorders **both** presentation surfaces at once —
per-turn frame rows and (through them) the composite run video — preserving
0063's pinned invariant that the video and the frame rows read identically.
The raw transcript (wire fidelity) is untouched, as are stored frames, logs,
and the `video` subcommand.

### Mechanism: sort the rows; the composite follows

- `_FrameContext` gains `camera_order: tuple[str, ...] = ()`. The context is
  already threaded to every render site, so no signature churn.
- `_render_turn_frames` sorts its references **stably** by
  `_camera_sort_key(camera) = (index of camera in camera_order if present
  else len(camera_order), original position)` before rendering cells.
- `frame_ctx.rendered` is appended in `_frame_image` call order, which after
  the sort *is* the displayed row order; `_render_trial_media`'s composite
  ordering (first-appearance of display cameras in `rendered`, plan 0063)
  therefore follows automatically — **no composite-side change**. The
  survivor-driven caption keeps working unchanged.
- Sorting is per-turn and stable, so cameras outside the flag preserve wire
  order among themselves, and multi-step turns keep their within-camera step
  order.
- `_place_feedback` reads `turn.references[0].step` — the anchor may become a
  different camera's reference at the same turn; the plan pins that feedback
  placement must anchor on the turn's **minimum** step, not "first reference
  after sorting", if those can differ. (They differ only when a turn mixes
  steps; the fix is `min(reference.step for ...)` at the two anchor sites if
  the existing behavior is first-reference — verify against code and keep
  observable placement identical for flag-less renders byte-for-byte.)

### CLI

- `view` gains `--camera-order NAMES` (comma-separated, names trimmed of
  surrounding whitespace; empty items ignored; empty/absent flag = today's
  wire order; unknown names are inert; a duplicated name takes its first
  position). Plumbed `cli.py` → `render_html(camera_order=...)` →
  `_FrameContext`.
- Applies to every `view` mode that renders frames (plain, `--serve`, live
  pages) — it is presentation-only, so live flipbook panels and their tab
  order inherit it exactly as the fallback path inherits row order today
  (fallback camera order derives from `streams`+`flipbook`; the flipbook
  dict is built from `rendered_frames`, so tabs follow too; stream-only
  cameras keep sorted-key order after).
- Directory-mode staleness stamps do not encode flag values (same as
  `--frames-budget`); changing the flag needs `--force` on already-rendered
  pages, matching existing flags' documented behavior.

## Tests

`tests/test_html_view.py`:
- Flagged order reorders row figcaptions within every turn and the composite
  caption matches it (fixture wire order differs from flag order; a
  space-bearing display name in the flag to pin display-name matching, not
  `_safe` keys).
- Unnamed camera sorts after all named ones, keeping wire order among
  unnamed; unknown flag name is inert; duplicate name takes first position.
- Empty `camera_order` renders byte-identically to today (regression pin on
  an existing fixture).
- Stable within-camera step order in a multi-step turn.
- Feedback placement unchanged by reordering (anchor test at a mixed-step
  turn if the code proves anchor-sensitive; otherwise assert current
  placement holds under the flag).

`tests/test_registry_cli.py` (or wherever `view` flags are covered today —
mirror `--no-video`'s tests):
- `--camera-order` parses, trims, ignores empties, and reaches
  `render_html`; default is `()`.

## Docs & changelog

- CHANGELOG `[Unreleased]` → Added: `--camera-order` (plan 0064, issue #350).
- `docs/guide/cli.md`: option row/paragraph in the `view` section next to
  `--frames-budget`/`--no-video`, noting rows + run video reorder together
  and `--force` for already-rendered directory pages.
- `src/inspect_robots/CLAUDE.md`: `cli.py` row mentions the flag; `_html.py`
  row mentions presentation-order override.
