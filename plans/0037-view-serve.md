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
    stopped. Directory mode only — with a single-file log argument,
    `SystemExit` (`--serve requires a logs directory`).
  - `--port N` (default 8300): listen port. `SystemExit` if given without
    `--serve` (a silently ignored flag is worse than an error).
  - `--host HOST` (default `127.0.0.1`): bind address. Same guard.
    **Loopback by default**: rendered pages embed stored camera frames —
    images of the user's physical space — with no auth and no TLS, and the
    tools this repo's audience already knows (`inspect view`, Jupyter)
    bind loopback. Remote browsing is one documented flag away
    (`--host 0.0.0.0`), and the printed hint teaches it:
    when bound to loopback, a dim line reads
    `serving to this machine only; pass --host 0.0.0.0 to serve to your network`.
    When bound non-loopback, the exposure note is the **loud** line:
    `serving to your network: anyone who can reach this machine can view these logs`.
- Composition with existing flags: `--open` opens the served URL (not a
  file path). `--no-frames`, `--frames-budget`, `-o DIR` apply to every
  render pass. `--force` applies to the **initial pass only** — loop
  passes are always incremental, otherwise `--serve --force` would rewrite
  every page every minute forever. `-o -` remains rejected in directory
  mode (transitively covers `--serve`).
- Flow of `--serve`:
  1. Initial render pass, exactly today's incremental directory render,
     including the totals line.
  2. Bind `http.server.ThreadingHTTPServer` with a
     `SimpleHTTPRequestHandler` subclass rooted at the output directory
     (the handler's `directory` parameter — no `os.chdir`) and
     `log_message` overridden to a no-op (per-request noise helps nobody).
     Bind failure (port taken) is `SystemExit` with the OS error text.
  3. Print the URL block to stdout after the totals line:
     `serving logs at: http://<display-host>:<actual-port>/`.
     `<actual-port>` always comes from the bound socket
     (`server.server_address[1]`), never the flag — `--port 0` works and
     the test pins it. `<display-host>` is the bind address, except for
     `0.0.0.0`, where it is `socket.gethostname()` and a second line also
     prints the `http://localhost:<port>/` form (hostnames like Debian's
     `127.0.1.1` pattern may not resolve from other machines; giving both
     never strands the user). Then the audience line from above, then
     `press Ctrl-C to stop`.
  4. Start the server thread (`serve_forever`, daemon), then — order
     matters and gets a comment: the socket is already bound and listening
     from step 2 (`__init__` binds; the backlog holds early requests), so
     printing/`--open` before `serve_forever` cannot race — invoke
     `--open` on the localhost-form URL if requested.
  5. Main thread runs the re-render loop: sleep `_SERVE_RERENDER_SECONDS
     = 60` (module-level seam, monkeypatchable), then an incremental
     render pass.
  6. Shutdown, spelled out: `KeyboardInterrupt` (which may land in the
     sleep *or* mid-render) is caught in the main thread →
     `server.shutdown()` (blocks until the serve loop exits) →
     `server.server_close()` in a `finally` (releases the listening
     socket even on unexpected errors) → exit 0. `SIGTERM` is mapped to
     raise `KeyboardInterrupt` via `signal.signal` before serving starts,
     so `kill`/process managers get the same clean path as Ctrl-C.

### 2. Re-render loop (cli.py)

- The initial pass and loop passes share one factored render core. Loop
  passes are **quiet**: no stdout totals; stderr `[i/N] rendering` lines
  only when a page is actually rendered. A quiet no-change pass costs one
  `read_eval_log` per log (acknowledged O(parse); an mtime-keyed entry
  cache is a named follow-up, not v1).
- Loop exception policy, explicit because the render path raises
  `SystemExit` for conditions that can arise mid-serve (all logs deleted,
  output dir replaced): the loop catches `(Exception, SystemExit)`,
  prints `warning: re-render failed: <exc>` to stderr, and continues —
  the viewer must not die because a directory was mid-mutation.
  `KeyboardInterrupt` propagates (it is neither).
- Each pass rewrites `index.html`; per-log pages are touched only for
  new/changed logs, so a page being viewed is not rewritten in the steady
  state (`index.html` mid-request rewrite is the same benign race any
  static server has; accepted).

### 3. Browser auto-refresh while served (`_html_index.py`)

`render_index` gains `refresh_seconds: int | None = None`; when set, the
document includes `<meta http-equiv="refresh" content="{refresh_seconds}">`.
The CLI passes `_SERVE_RERENDER_SECONDS` in serve mode and `None`
otherwise — a statically rendered index stays static (plan 0035's
non-goal unchanged), but a served one tracks new runs in an open tab. The
existing localStorage filter persistence makes the refresh non-destructive
to the typed filter *value*; focus/caret do reset once a minute — accepted
for v1 over a fetch-and-swap (code comment says so).

## Non-goals

WebSockets/live push, TLS, auth (loopback default + the explicit
`--host 0.0.0.0` opt-in and its loud exposure line are the boundary),
daemonization (foreground is the feature: Ctrl-C/SIGTERM is the
lifecycle), entry caching across loop passes (named follow-up), any
change to single-file mode, and index row-click navigation (separable
index-usability change; its own small PR).

## Tests

`tests/test_html_index.py`:
- `refresh_seconds=None` → no meta refresh tag; `refresh_seconds=60` →
  exact tag present.

`tests/test_registry_cli.py`:
- Guards: `--serve` with a single-file argument; `--port`/`--host`
  without `--serve` — clean `SystemExit`s.
- End-to-end serve: monkeypatch the sleep seam to raise
  `KeyboardInterrupt` on its first call after performing one
  `urllib.request.urlopen` (with `timeout=`, so a dead server fails
  instead of hanging) against `http://127.0.0.1:<bound-port>/index.html`
  — asserts 200, the refresh tag, the socket-derived port in the printed
  URL (invoked with `--port 0`), and exit 0 through the clean-shutdown
  path.
- Loop resilience: monkeypatched render core raising once then
  succeeding, sleep seam allowing two iterations before interrupting —
  warning on stderr, server still answering, exit 0.
- URL display: bound `127.0.0.1` prints that literal and the
  loopback-audience hint; bound `0.0.0.0` (monkeypatched
  `socket.gethostname` for determinism) prints the hostname URL, the
  localhost URL, and the loud exposure line.
- SIGTERM: handler registered (assert via `signal.getsignal`) — the full
  kill-delivery path is covered by construction through the
  KeyboardInterrupt tests.

## README

The **Browse your runs** section makes the serve workflow the headline
and spells out the remote case explicitly, since that is the whole point:

- Local: `inspect-robots view logs/ --serve --open` — renders, serves,
  opens the browser; leave it running and new runs appear on their own.
- From another machine (headless robot host): run
  `inspect-robots view logs/ --serve --host 0.0.0.0` on the host, open
  the URL it prints from your laptop. State plainly what `--host 0.0.0.0`
  exposes (anyone who can reach the machine can view the logs, camera
  frames included) and that the index auto-refreshes while served.
- The `python3 -m http.server` tip is deleted — `--serve` replaces it.
- Static rendering (no `--serve`) stays documented first.

## Rollout

CLI + index-renderer addition; no schema or plugin surface changes.
Changelog `### Added` under `[Unreleased]` with `(plan 0037, #241)`.
