# 0091: `view --video-url` plays a hosted run video

Issue #610.

## Problem

`inspect-robots view` renders each trial's "Run video" block in one of two
ways: a composite MP4 embedded as base64 (plans 0055 and later), or a
per-camera flipbook over the frame images already on the page. Publishers who
keep pages small render with `--no-video --frames-budget 12` and upload each
run's composite MP4 to a public URL on their own. The page cannot play that
hosted file, so readers see a short flipbook while a full video sits one click
away.

## Scope

- A new `view` option, `--video-url TEMPLATE`, makes every trial's "Run video"
  block a `<video>` that streams the URL instead of an embedded MP4 or a
  flipbook.
- The template may use three placeholders, each percent-encoded on
  substitution (`urllib.parse.quote(value, safe="")`):
  - `{log}`: the log file's stem (`run.json` gives `run`);
  - `{scene}`: the scene id;
  - `{trial}`: the trial index within the scene (`0`, `1`, ...).
  `{{` and `}}` are literal braces. Any other field, a positional `{}`, a
  conversion (`!r`) or a format spec (`:>3`) is rejected.
- Validation at the CLI, before any render, and again in `render_html` for
  library callers: the template must parse and use only the fields above; it
  must not contain whitespace or `\` (browsers read `\` as `/`, Python does
  not); and when the template has a field, the literal text before the first
  field must be a full `http://host/` or `https://host/` prefix: a non-empty
  hostname followed by a literal `/` (`urllib.parse.urlsplit` path starting
  with `/`). So `https://cdn{log}.x/` and `https://h:{trial}/` are rejected,
  and a substituted value can only land in the path, query or fragment. Field
  names are compared whole, so `{log.x}` and `{log[0]}` are unknown fields. CLI
  failures exit through `SystemExit` with a message naming `--video-url`.
- Single-file mode: a literal URL without placeholders is allowed. A template
  without `{scene}` or `{trial}` gives every trial of a log the same video;
  that is the intended publisher layout (one composite per single-trial log),
  so it is documented rather than warned about.
- Directory mode: the template must contain `{log}`, since a literal URL would
  give every page the same video. The option applies to every completed page
  of the pass. Changing `--video-url` needs `--force` to re-render pages that
  are already current, the same rule as `--no-video` today; the help text for
  `--force` lists it.
- Started (running) logs ignore `--video-url`: the hosted file does not exist
  until the run finishes, and the live render loop should not reload a
  missing video every two seconds. They keep today's flipbook. A single-file
  render of a started log prints one stderr note saying the option was
  ignored.
- Trials with no policy transcript get no media block today (media lives
  inside the transcript `<details>`), and they get no linked video either.
- `--video-url` replaces the embedded composite for that page: no ffmpeg
  encode runs and the video budget is not charged. `--no-video` and
  `--video-url` combine (the publishing case); `--no-frames` and
  `--video-url` combine too, leaving the linked video with no flipbook to
  fall back to.

## Out of scope

- Rail seeking, turn highlight and Follow for linked videos. These need the
  video's step timeline (`data-steps`, `data-fps`), which is not known for an
  external file. Linked videos render without `data-steps` and without the
  Follow button, so the existing rail code never attaches. A later change
  could accept a sidecar timeline.
- A per-log URL mapping file. The `{log}` template covers the known publisher
  layout (`.../videos/<log stem>.mp4`). A mapping can be added later without
  changing this option.
- Camera labels under the video. The external composite's layout is unknown.
- Fetching, probing or validating that the URL exists at render time. The
  page stays network-free to render; the browser decides at view time.

## Design

### Python

- `_html.py` gains `validate_video_url_template(template: str) -> str`, which
  raises `ValueError` with a human message, and `_expand_video_url(template,
  *, log, scene, trial)`. Both live in `_html.py` because the page is the only
  consumer; the CLI imports the validator.
- `render_html(..., video_url: str | None = None)`. When set and the log is
  not `started`, each trial's media block is built by a new
  `_render_linked_media(trial_prefix, url, flipbook, cameras)` instead of
  `_render_trial_media`. `{log}` comes from `log_path.stem`; a template using
  `{log}` with `log_path=None` raises `ValueError` (the CLI always passes a
  path). `render_html` validates the template itself too, so library callers
  get the same guarantees.
- The flipbook panels are still built from the rendered frames
  (`rendered_frames`), exactly as the flipbook-only path builds them, so the
  fallback needs no extra frame payload.

### Markup

```html
<div class="run-media" data-trial="scene-0-e0" data-linked-video>
  <div class="run-media-head">Run video
    <a class="linked-video-link" href="https://...">Open video</a></div>
  <div class="linked-video video-panel" data-state="loading">
    <video muted loop playsinline controls preload="auto"
           data-src="https://..."></video>
    <div class="linked-video-status" role="status">Loading video…</div>
  </div>
  <div class="camera-tabs">...</div> ...flipbook panels...
</div>
```

The URL sits in `data-src` and the script assigns `video.src`. Assigning it
in the script, after the `error` listener is attached, removes the race where
a fast failure fires before `DOMContentLoaded` on a large page. It also lets
the script defer loading for collapsed transcripts. Every interpolated value
goes through `_escape`.

### CSS

- `.linked-video` is a dark box with `aspect-ratio: 16 / 9` and a
  `max-height`, so the placeholder has a size before metadata arrives. On
  `loadedmetadata` the script sets the box's `aspect-ratio` to
  `videoWidth / videoHeight`.
- The `<video>` stays visible with its controls, so a browser that waits for
  a user gesture can still be started by hand. The status text sits over the
  dark box with `pointer-events: none` and is hidden once ready, so the
  reader sees "Loading video…" instead of a bare black box.
- The "Open video" link gives readers without JavaScript, or with a failed
  load, a direct way to the file.
- A stalled network never fires `error`, so the status can stay on
  "Loading video…"; the docs note this, and the controls and link stay usable.
  An `http:` URL on an `https:` page may be blocked as mixed content and then
  falls back like any other error.
- `.run-media[data-linked-video]:not([data-linked-failed]) .camera-tabs` and
  `.camera-panel` are hidden, so the flipbook stays out of sight unless the
  video fails.

### Script

Inside the existing per-block `DOMContentLoaded` handler (`_FLIPBOOK_SCRIPT`):

- `linked = block.querySelector('.linked-video')`. While `linked` is present
  and not failed, `activate()` does not start the flipbook timer.
- `load()` assigns `video.src = video.dataset.src` once. It runs immediately
  when the trial's `<details class="transcript">` is open (or absent), and on
  the first `toggle` to open otherwise, so a directory of collapsed trials
  does not download every video at page load.
- After assigning `src`, `load()` calls `play()` (errors swallowed), since
  some browsers (iOS Safari, data saver) fetch only metadata until asked to
  play and would never fire `canplay`.
- `loadedmetadata`: set the aspect ratio. `loadeddata` (once): set
  `data-state="ready"`.
- `error` (once): if the block has flipbook panels, remove the linked box,
  set `data-linked-failed` on the block, and call `activate(active)` so the
  flipbook starts. Otherwise set `data-state="error"` and change the status
  text to "Video unavailable".
- `syncVideo` (play on open, pause on collapse) already queries
  `.video-panel video`; the linked box carries `video-panel`, so it is
  covered. On open it first calls `load()`.

## Tests (test-first)

`tests/test_html_view.py`:

- template validation: accepts literal http/https URLs and each placeholder,
  `{{`/`}}`; rejects `ftp:`, `javascript:`, relative URLs, a missing host,
  unknown fields, positional fields, conversions, format specs, and unbalanced
  braces.
- linked markup: `data-src` holds the expanded, escaped URL with
  percent-encoded scene ids; no `data-steps`, no Follow button, no
  `data:video/mp4`; ffmpeg encode is never called even when ffmpeg is found;
  flipbook panels are still present for fallback; a page with no flipbook
  source still renders the linked box.
- started logs ignore the option; `{log}` without `log_path` raises.
- a multi-trial log with a `{log}`-only template gives every trial the same
  URL without a warning.
- the script contains the `loadeddata`, `error` and `data-src` hooks (string
  checks, as the existing script tests do).

`tests/test_registry_cli.py`:

- `--video-url` is forwarded to `render_html` in single-file and directory
  mode; an invalid template exits with a message before rendering; directory
  mode without `{log}` exits; a started single-file log prints the
  ignored-option note.

Browser behaviour is checked by hand in headless Chromium (Playwright): a
render pointing at a served MP4 shows "Loading video…" then plays; a bad URL
falls back to the flipbook. The repo has no browser test harness, and this
change does not add one.

## Docs

- `docs/guide/cli.md`: a "Linking a hosted run video" paragraph in the
  `view` section, including the rail-seek limitation and the `--force` note.
- argparse help for `--video-url` and `--force`.
- Module map (`src/inspect_robots/CLAUDE.md`): `cli.py` and `_html.py` rows.
- `changelog.d/610.added.md`.
