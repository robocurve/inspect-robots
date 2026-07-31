# 0037 — `view --serve`: serve a rendered logs directory and print the URL

Issue: #241.

## Problem

`view logs/` (plan 0035) renders a browsable index, but viewing it from
another machine — the normal case for a headless robot host — still demands
side knowledge: run a separate static file server, keep re-running the
render as new logs land, and derive the right URL yourself. The README
currently *teaches* the workaround (`python3 -m http.server -d logs/html`).
On the rig this grew into a hand-rolled tmux-then-systemd arrangement;
that is infrastructure a user has to know to build, for what should be one
command.

## Design

### 1. CLI surface (cli.py)

- New `view` flags:
  - `--serve`: after rendering, serve the output directory over HTTP until
    interrupted. Directory mode only — with a single-file log argument,
    `SystemExit` (`--serve requires a logs directory`).
  - `--port N` (default 8300): listen port. Only meaningful with `--serve`;
    silently ignored otherwise is unacceptable — `SystemExit` if given
    without `--serve`.
  - `--host HOST` (default `0.0.0.0`): bind address. Same guard as
    `--port`. The default is deliberately non-loopback: the whole point of
    the flag is browsing from another machine, the served content is the
    user's own local files, and the printed URL warns exactly what is
    exposed. `--host 127.0.0.1` opts into loopback-only.
- Composition with existing flags: `--open` opens the served URL (not the
  file path). `--force`, `--no-frames`, `--frames-budget`, `-o DIR` apply
  to the render pass as today. `-o -` remains rejected in directory mode.
- Flow of `--serve`:
  1. Render pass, exactly today's incremental directory render.
  2. Bind an `http.server.ThreadingHTTPServer` with
     `SimpleHTTPRequestHandler` rooted at the output directory (via the
     handler's `directory` parameter — no `os.chdir`). Bind failures
     (port taken) surface as `SystemExit` with the OS error text.
  3. Print, to stdout, the URL block after the render totals line:
     `serving logs at: http://<display-host>:<port>/` where
     `<display-host>` is the machine's hostname when bound to `0.0.0.0`
     (falling back to the literal bind address if hostname resolution
     fails), or the bind address otherwise. A second dim line notes
     `serving to your network; use --host 127.0.0.1 for this machine only`
     when bound non-loopback, and `press Ctrl-C to stop` in either case.
  4. Serve forever; a re-render loop re-runs the incremental render every
     `_SERVE_RERENDER_SECONDS = 60` in the main thread while the server
     runs in a daemon thread (`serve_forever` + threading — the render
     loop stays in the main thread so `KeyboardInterrupt` lands there,
     shuts the server down cleanly, and exits 0).
  5. Request logging is suppressed (`log_message` no-op override): render
     progress belongs on stderr, per-request noise helps nobody.

### 2. Browser auto-refresh while served (`_html_index.py`)

`render_index` gains `refresh_seconds: int | None = None`; when set, the
document includes `<meta http-equiv="refresh" content="{refresh_seconds}">`.
The CLI passes `_SERVE_RERENDER_SECONDS` in serve mode and `None`
otherwise — a statically rendered index stays static (plan 0035's non-goal
unchanged), but a served one tracks new runs in an open tab with no user
action. The existing localStorage filter persistence (plan 0035) already
makes the refresh non-destructive to a typed filter.

The index page also gains row-click navigation, proven downstream: rows
with a report carry `data-href`; a click anywhere on the row (delegated
handler in the existing inline script) navigates unless the click hit the
in-row anchor or text is selected; ctrl/cmd-click opens a new tab. Rows
without a page are unaffected. This is an index-usability change that
serves both static and served modes; it rides along because the serve
workflow makes the index the primary browsing surface.

### 3. Re-render loop details (cli.py)

- The loop calls the same internal render function as the initial pass
  (incremental: unchanged logs cost one parse each, no page writes), then
  sleeps. Render errors in a loop iteration must not kill the server: the
  iteration logs `warning: re-render failed: <exc>` to stderr and the loop
  continues (a transient half-written foreign file must not take the
  viewer down; per-log unreadability is already an error row).
- Each re-render pass rewrites `index.html`; per-log pages are only
  touched for new/changed logs, so a page being viewed is never rewritten
  mid-request in the steady state. (`index.html` rewrite during a request
  is the same benign race any static server has; accepted.)

## Non-goals

WebSockets/live push, TLS, auth (the printed warning line plus `--host`
is the boundary — same posture as `python3 -m http.server`, which this
replaces), daemonization (foreground process is the feature: Ctrl-C is
the lifecycle), and any change to single-file mode.

## Tests

`tests/test_html_index.py`:
- `refresh_seconds=None` → no meta refresh tag; `refresh_seconds=60` →
  exact tag present.
- Row-click: `data-href` present exactly on rows with pages; delegation
  script present; rows without pages carry no `data-href`.

`tests/test_registry_cli.py`:
- `--serve` with a single-file argument, `--port`/`--host` without
  `--serve`: clean `SystemExit`s.
- End-to-end serve test: run `main(["view", dir, "--serve", "--port", "0",
  "--host", "127.0.0.1"])` in a thread? No — inverted: monkeypatch the
  serve loop's sleep to raise `KeyboardInterrupt` after the first
  iteration, bind port 0 (ephemeral), and in that first iteration fetch
  `http://127.0.0.1:<bound-port>/index.html` with `urllib` asserting 200
  and the refresh tag — then the injected interrupt exercises clean
  shutdown and exit 0. (Port 0 requires printing the *bound* port, which
  the URL construction takes from the socket, not the flag — test pins
  that.)
- Re-render loop resilience: monkeypatched render function that raises
  once then succeeds; loop continues (warning on stderr, server still
  answering).
- URL display: bound `127.0.0.1` prints that literal; bound `0.0.0.0`
  prints a hostname-based URL (monkeypatch hostname lookup for
  determinism) plus the network-exposure note.

## README

The **Browse your runs** section replaces the `python3 -m http.server` tip
with `--serve`: one command, shows the URL, new runs appear while it runs.
Static usage (no `--serve`) stays documented first.

## Rollout

CLI + index-renderer addition; no schema or plugin surface changes.
Changelog `### Added` under `[Unreleased]` with `(plan 0037, #241)`.
