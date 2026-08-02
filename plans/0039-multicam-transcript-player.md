# 0039 — Multicam player with synced transcript, run numbering, sortable index

Issue: #247. The `view` report gains, per trial, a synchronized multi-camera
video player (one labeled pane per camera, one scrubber) with the trial's
transcript acting as a live rail: the message under the playhead highlights,
and clicking a message seeks the videos. The directory index gains sequential
run numbers, click-to-sort column headers, and id-aware filtering. Run
identity creation becomes collision-proof under concurrent writers.

## Problem

A run's camera footage is only viewable as per-camera MP4s opened one at a
time (written by `inspect-robots video` into the frames dir), or as static
frame grids inside the transcript. There is no way to watch the cameras in
parallel, no way to scrub, and no connection between playback time and the
model's narration. Separately: runs are identified only by random hex, the
index cannot be sorted or searched by id, and two concurrent eval processes
writing one log dir can in principle collide on a filename or run stamp.

## Design

### 1. Trial video discovery (`_video.py`)

New helper:

```python
def trial_videos(frames_dir: str | None, log_path: Path,
                 trial_prefixes: Sequence[str]) -> dict[str, list[tuple[str, Path]]]:
    """Locate per-camera MP4s per trial, keyed by trial prefix."""
```

Candidate roots, first root that yields any file wins per trial:

1. `resolve_frames_dir(frames_dir, log_path)` when `frames_dir` is not None
   (where `inspect-robots video` writes by default).
2. `log_path.parent / "videos" / stamp`, where `stamp` is the basename of
   `frames_dir` derived with the same `PureWindowsPath` backslash handling as
   `resolve_frames_dir`. This is a new convention introduced and documented by
   this plan (`docs/guide/cli.md`): a place to keep rendered videos when the
   raw frames have been deleted or archived.

File-to-trial assignment scans `*.mp4` in a root and assigns each file to the
longest `trial_prefix` in `trial_prefixes` that prefixes its stem followed by
`_`; files matching no known prefix are ignored. This resolves the underscore
ambiguity documented in `_video.py` as far as it can be resolved; the pane
label is the stem segment after `{prefix}_` (which for cameras with unsafe
characters shows the `_safe()`-mangled form; accepted). Returns `{}` when
nothing is found; trials without entries render no player.

### 2. One discovery pass, explicit plumbing (`cli.py` -> `_html.py`)

Discovery runs once, in the CLI. `render_html` gains one keyword:

```python
def render_html(log, *, title, log_path=None, frames_dir=None,
                frames_budget_bytes=50_000_000,
                trial_media: Mapping[str, Sequence[tuple[str, str]]] | None = None) -> str:
```

`trial_media` maps trial prefix to `(label, href)` pairs, hrefs already
page-relative (`media/<stamp>/<name>.mp4`). `_html.py` touches no filesystem
for videos. The CLI builds it in `_render_log_page` for both modes:

- Directory mode (`_render_view_directory`): media links land in
  `out_dir / "media" / <stamp> / <name>.mp4`. Discovery runs for every log on
  every pass (it is a directory scan); the incremental skip predicate becomes
  "page mtime >= max(log mtime, newest discovered video mtime)", so videos
  rendered after a page was built trigger a re-render without `--force`.
- Single-file mode with a file target: media links land next to the report,
  `out_path.parent / "media" / <stamp> / ...`.
- `-o -` (stdout): `trial_media` stays `None`; no player is emitted. The
  stdout document remains fully self-contained.

Linking is a relative `os.symlink`; on `OSError` (platforms or filesystems
without symlink support) fall back to `shutil.copy2`. Idempotency contract,
per target path: if a symlink exists and `os.readlink` already matches, leave
it; if it mismatches, replace it; if a regular file exists (copy fallback)
with equal size and a copy mtime not older than the source (tolerant of
coarse filesystem timestamp granularity), leave it, else re-copy.

Docs updated accordingly: `docs/guide/cli.md` loses both stale claims (the
report "has no network or JavaScript dependency" and, for pages with players,
the implication that a single HTML file carries everything; the page itself
stays single-file, the videos ride alongside under `media/`).

### 3. Run page player (`_html.py`)

`_scene_section` wraps each trial in a container so player and transcript
share an anchor:

```html
<div class="trial" id="trial-scene-0-e0"> <!-- id via _safe() + escaping -->
  <section class="player" data-fps="30">
    <div class="cams">
      <figure class="cam"><figcaption>top_cam</figcaption>
        <video src="media/<stamp>/scene-0-e0_top_cam.mp4" preload="metadata" muted playsinline></video>
      </figure>...
    </div>
    <div class="controls">
      <button class="playpause" type="button">Play</button>
      <input class="scrub" type="range" min="0" max="1000" value="0" step="1">
      <span class="clock">step 0 · 0.0s</span>
      <button class="follow on" type="button">Follow</button>
    </div>
  </section>
  <details class="transcript" ...>...</details>
</div>
```

The wrapper is emitted for every trial (with or without player) so trial
markup is uniform; the player section only when `trial_media` has entries for
the prefix. A trial with videos but no transcript gets a player and no
`<details>`; association is by shared wrapper, never adjacency.

CSS: `.player { position: sticky; top: 0; }` scoped inside `.trial`; when a
player is present the wrapper gets class `has-player` and its transcript body
gets `max-height: 60vh; overflow-y: auto;` so the rail scrolls inside the
page while the player stays put. `.message { scroll-margin-top: … }` keeps
highlighted messages clear of the sticky player.

`data-fps` carries `default_fps(log.eval.embodiment_info)[0]` formatted `:g`
(10 Hz fallback branch included).

JS: the run-page script constant is renamed `_VIEW_SCRIPT` and extended (the
existing frame-cell toggle moves in unchanged). Behavior, per `.trial`:

- Master election waits for `loadedmetadata` on all panes (panes firing
  `error`, e.g. a dangling symlink, are excluded); the longest-duration pane
  is the clock master. Until election, controls are disabled.
- Play/pause drives all panes; the scrubber maps 0..1000 over the master
  duration and sets `currentTime` on every pane on `input`; while playing, a
  `requestAnimationFrame` loop updates scrubber, clock (`step N · T.Ts`,
  `N = round(T * fps)`), and the live message; shorter panes clamp at their
  final frame (native post-`ended` behavior).
- Live rail: the last `.message[data-step]` with `data-step <= N` inside this
  trial's wrapper gets `.live`; when Follow is on the rail scrolls to it via
  `container.scrollTop` arithmetic (never `scrollIntoView`, which would also
  scroll ancestors and hijack the page). Any manual scroll/wheel/touch on the
  rail turns Follow off; the button toggles it back.
- Click-to-seek: clicks inside the transcript seek all panes to
  `data-step / fps`, except clicks on `.frame-cell`, on links, or while
  `getSelection()` is non-empty (plan 0038 guards).

Scrub latency under `--serve` is accepted: the stdlib handler serves whole
files without Range support. To keep startup and seeking workable,
`_ffmpeg_argv` in `_video.py` adds `-movflags +faststart` (also speeds
`file://` playback; re-rendered videos benefit, existing files still play).

### 4. Message step anchors (`_html.py`)

Step extraction becomes independent of frame rendering: a helper scans a user
message's content parts for `_FRAME_LABEL_RE` matches and returns the minimum
step named, whether or not `frame_ctx` is set (so anchors survive
`--no-frames` and missing frames dirs — exactly the archived-footage case).
`_render_chat_transcript` threads a running anchor: messages without labels
(assistant/tool) inherit the most recent prior anchor, initial value 0
(system messages render as `<details class="system-message">`, not
`.message`, and carry no anchor). Every `.message` gets `data-step="N"`. `_render_message` gains the anchor
as a defaulted keyword so existing callers and tests stay valid.

The tool-result phrase `executing ... over N steps` is deliberately not
parsed; camera labels are the single source of step truth. Known accepted
skew: `encode_stream` drops empty warm-up frames, so a camera's frame 0 can
sit a few steps after step 0 (plan 0016); a per-camera offset map is a
non-goal.

### 5. Index numbering (`cli.py`, `_html_index.py`)

`_render_view_directory` sorts readable entries by `(created, stem)` and
assigns `number = position + 1`; unreadable entries get no number (rendered
as `—`) so a concurrent writer's transient file cannot renumber history
between serve cycles beyond its own row. `IndexEntry` gains
`number: int | None = None` and `stamp: str | None = None`. The table gains a
leading `Run Id` column rendering `f"{number:04d}"`. Numbers live only on the
index: run pages are unchanged by numbering, so the incremental mtime skip in
`_render_view_directory` can never leave a stale number baked into a page.
Files are never renamed; numbers are a presentation of the collection.

### 6. Sortable headers and id-aware filter (`_html_index.py`)

- Sortable columns and their keys: `#` (numeric, `data-sort` zero-padded),
  When (`data-sort` = ISO `created`), Instruction / Policy / Status /
  Termination (lexical on `textContent`). No new data columns are added.
  Every `<th>` gets `data-key`; a delegated `thead` click handler sorts
  ascending, then descending on repeat click (CSS arrow indicator),
  reordering `tbody` rows in place, skipping the `.empty` row, sorting
  unnumbered (`—`) rows to the bottom under a `#` sort, and leaving
  filter-hidden rows hidden. Sort key and direction persist in
  `localStorage` beside `_FILTER_KEY` (so the 60 s serve refresh keeps both
  filter and sort).
- Rows gain `data-id` (log stem) and `data-stamp`; the filter matches on
  `textContent` plus both attributes, so the 8-hex id, the run stamp, and the
  zero-padded number all match.
- The index JS lives in an f-string: JS braces are `{{ }}`-doubled (plan 0038).

### 7. Collision-proof run identity (`logging/json_log.py`, `eval.py`)

- `json_log`: keep the write-then-rename pattern (no visible half-written
  logs). The tmp file gets a per-attempt unique name
  (`.{final_stem}.{uuid4().hex}.tmp`). Claiming the final name uses
  `os.link(tmp, final)` — atomic, fails with `FileExistsError` if the name is
  taken — then the tmp is unlinked. On collision a fresh `uuid4().hex[:8]`
  stem is drawn, up to 16 attempts, then the full `uuid4().hex` is used. If
  `os.link` raises `OSError` other than `FileExistsError` (filesystem without
  hard links), fall back to `os.replace` for that attempt (the pre-plan
  behavior: last-writer-wins, still no partial visibility).
- `eval.py`: the run stamp is claimed by creating `log_dir/frames/<stamp>`
  with `mkdir(parents=True, exist_ok=False)` in a redraw loop,
  unconditionally (frames stored or not), before the stamp is used anywhere.
  The redraw loop is bounded at 16 attempts like the sink's. When
  `store_frames` is false the directory is removed in a `finally` at eval end
  if still empty (`rmdir` guarded by `OSError` pass, covering the became-
  non-empty case), so non-frames runs do not litter even when cancelled.
  `FrameStore` keeps its current behavior against the pre-created dir.

## Non-goals

- Per-camera warm-up offset correction (plan 0016 territory).
- Recording or detecting a custom `--fps` used at encode time; sync assumes
  the default (`control_hz`, 10 Hz fallback), matching `default_fps`.
- HTTP Range support in the stdlib server.
- Embedding video bytes into the HTML document (data URLs stay images-only).
- Reading the transcript JSONL sidecar; the inline transcript is the rail.
- Audio playback (panes stay `muted`; robot cams have no audio).
- Mapping `_safe()`-mangled camera filenames back to original names.

## Tests

- `tests/test_video.py`: `trial_videos` — frames-dir hit, `videos/<stamp>`
  hit, frames dir preferred, `{}` on nothing, `frames_dir=None` guard,
  longest-prefix assignment (scene ids containing `_`), unknown-prefix files
  ignored, Windows-style `frames_dir` stamp derivation; `+faststart` present
  in `_ffmpeg_argv`.
- `tests/test_html_view.py`: trial wrapper always present; player present
  exactly once per trial with media and absent otherwise; absent under
  `trial_media=None`; labels/hrefs escaped and exact; `data-fps` value and
  10 Hz fallback; `data-step` anchors (min label wins, inheritance across
  assistant/tool messages, leading 0, anchors present with `frames_dir=None`);
  `has-player` class and follow button only with media; script tag count
  stays 1 with the `_VIEW_SCRIPT` literal duplicated in the guard test per
  the plan-0038 convention.
- `tests/test_html_index.py`: `#` column zero-padded and `—` for
  `number=None`; `data-sort` on `#` and When cells; `data-id`/`data-stamp`
  on rows; sort-persistence key present; sort script literal duplicated
  (injection guard); `.empty` row skip covered via the script literal.
- `tests/test_registry_cli.py`: directory mode numbers follow `created` order
  regardless of glob order; unreadable logs unnumbered; media symlinks land
  under `out_dir/media` and resolve; symlink-mismatch replacement; copy
  fallback via monkeypatched `os.symlink` raising `OSError`; copy-fallback
  idempotency (equal size+mtime skips, mismatch re-copies); single-file mode
  places `media/` next to `-o` target; `-o -` emits no player and no media
  dir; `--serve` serves a linked video URL (second `urlopen` in the
  `_serve_sleep` monkeypatch pattern).
- `tests/test_eval_orchestration.py` / `tests/test_strict_json.py` (the
  sink's existing homes): filename redraw on link collision (monkeypatched
  `uuid4` colliding once), full-uuid fallback after exhaustion, hard-link
  `OSError` fallback to `os.replace`, per-attempt tmp names; stamp mkdir
  redraw loop with 16-attempt bound; empty stamp dir removed for non-frames
  runs (including via the `finally` on a cancelled run), kept when frames
  were stored, and the rmdir `OSError` pass branch via a stamp dir that
  gained content in a non-frames run.
- Coverage stays at 100% line and branch; mypy strict (tests included) and
  ruff (incl. D1 docstrings) clean.

## Rollout

- CHANGELOG under Unreleased / Added: multicam player with synced transcript
  rail, index run numbers, sortable and id-searchable index (plan 0039, #PR);
  under Changed: collision-proof log filenames and run stamps, faststart
  MP4s.
- `docs/guide/cli.md`: document the player, the `media/` layout, the
  `videos/<stamp>/` convention; remove both stale self-containment claims.
- `src/inspect_robots/CLAUDE.md` module table: update the `_html.py`,
  `_html_index.py`, `_video.py` row descriptions (no new modules).
- No release; version stays hatch-vcs dev until the next tag.
