"""The ``inspect-robots view`` command: render saved eval logs as HTML reports,
optionally serving a logs directory live.."""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, cast

from inspect_robots._html import _display_status, _status_class
from inspect_robots._html_index import IndexEntry, render_index

from ._shared import _DIM, _OUTCOME_PHRASES, _YELLOW, _styled

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog

_SERVE_RERENDER_SECONDS = 60


_SERVE_LIVE_RERENDER_SECONDS = 2


# Two-level directory stamp (plan 0060): a suppressed-tier page (serve pass or
# --no-video) is stamped this far below the source log's mtime so a later
# full-tier pass can recognize and upgrade it. The delta spans many filesystem
# timestamp ticks (NTFS rounds to 100 ns) and the gate compares with slack so
# tick rounding can never turn a skip into a per-tick re-render.
_SUPPRESSED_STAMP_DELTA_NS = 2_000


_STAMP_TICK_SLACK_NS = 1_000


_serve_sleep = time.sleep


def _write_html(document: str, out_path: Path | None) -> int:
    """Write one document as UTF-8, returning the encoded byte count."""
    encoded = document.encode("utf-8", errors="replace")
    if out_path is None:
        sys.stdout.write(encoded.decode("utf-8"))
        return len(encoded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    return len(encoded)


def _write_html_atomic(document: str, out_path: Path) -> int:
    """Replace one UTF-8 document atomically and return its encoded byte count."""
    encoded = document.encode("utf-8", errors="replace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f"{out_path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    os.replace(tmp, out_path)
    return len(encoded)


def _render_log_page(
    log: EvalLog,
    log_path: Path,
    out_path: Path | None,
    *,
    no_frames: bool,
    frames_budget: float,
    live_frames_budget: float | None = None,
    refresh_seconds: int | None = None,
    atomic: bool = False,
    no_video: bool = False,
    serve_pass: bool = False,
    index_href: str | None = None,
) -> int:
    from inspect_robots import cli

    """Render one already-parsed log through the shared single-page pipeline."""
    frames_dir = None
    if not no_frames and log.stats.frames_dir is not None:
        from inspect_robots._video import resolve_frames_dir

        frames_dir = resolve_frames_dir(log.stats.frames_dir, log_path)
    document = cli.render_html(
        log,
        title=f"{log.eval.task} - {log_path.name}",
        log_path=log_path,
        frames_dir=frames_dir,
        frames_budget_bytes=int(frames_budget * 1_000_000),
        live_frames_budget_bytes=(
            None if live_frames_budget is None else int(live_frames_budget * 1_000_000)
        ),
        wire_media_elided=log.status == "started",
        refresh_seconds=refresh_seconds,
        no_video=no_video,
        serve_pass=serve_pass,
        index_href=index_href,
    )
    if atomic and out_path is not None:
        return _write_html_atomic(document, out_path)
    return _write_html(document, out_path)


def _open_browser(uri: str, display: str | Path) -> None:
    """Open one rendered artifact, degrading browser failures to warnings."""
    import webbrowser

    try:
        opened = webbrowser.open(uri)
    except Exception as exc:
        print(f"warning: could not open browser for {display}: {exc}", file=sys.stderr)
    else:
        if not opened:
            print(f"warning: could not open browser for {display}", file=sys.stderr)


def _index_instruction(log: EvalLog) -> str:
    """Return the shared instruction or the multi-scene fallback."""
    instructions = {scene.instruction for scene in log.samples}
    shared = next(iter(instructions)) if len(instructions) == 1 else None
    if shared:
        return shared
    count = len(log.samples)
    return f"{count} scene" if count == 1 else f"{count} scenes"


def _index_termination(log: EvalLog) -> str:
    """Return the order-preserved union of human-facing termination reasons."""
    seen: set[str] = set()
    phrases: list[str] = []
    for scene in log.samples:
        for reason in scene.termination_reasons:
            if reason is None:
                continue
            text = str(reason)
            if text in seen:
                continue
            seen.add(text)
            phrases.append(_OUTCOME_PHRASES.get(text, text))
    return ", ".join(phrases)


def _index_entry(log: EvalLog, log_path: Path, page: str) -> IndexEntry:
    """Build one directory-index row from a successfully parsed log."""
    model = log.eval.policy_config.get("model")
    return IndexEntry(
        name=log_path.name,
        page=page,
        created=log.eval.created,
        instruction=_index_instruction(log),
        policy=log.eval.policy,
        model=None if model is None else str(model),
        status=_display_status(log.status),
        status_class=_status_class(log.status),
        metrics=log.results.metrics,
        errored_trials=log.results.errored_trials,
        termination=_index_termination(log),
        error=log.error,
    )


def _unreadable_index_entry(log_path: Path, error: Exception) -> IndexEntry:
    """Build an error row for a file that is not a readable EvalLog."""
    try:
        created = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        created = ""
    return IndexEntry(
        name=log_path.name,
        page=None,
        created=created,
        instruction="",
        policy="",
        model=None,
        status="error",
        status_class="status-error",
        metrics={},
        errored_trials=0,
        termination="",
        error=f"unreadable: {error}",
    )


def _directory_page_names(log_paths: Sequence[Path]) -> dict[Path, str]:
    """Assign collision-free report names without consulting existing files."""
    names = {path: f"{path.stem}.html" for path in log_paths if path.stem != "index"}
    reserved = set(names.values())
    for path in log_paths:
        if path.stem != "index":
            continue
        candidate = "index_log.html"
        suffix = 2
        while candidate in reserved:
            candidate = f"index_log_{suffix}.html"
            suffix += 1
        names[path] = candidate
        reserved.add(candidate)
    return names


class _DirectoryRenderResult(NamedTuple):
    """Paths produced by one logs-directory render pass."""

    out_dir: Path
    index_path: Path


_LIVE_REDIRECT_SENTINEL = "<!-- inspect-robots live redirect stub -->"


def _write_live_redirect_stub(path: Path) -> int:
    """Atomically replace an orphaned live page with an idempotent index redirect."""
    document = f"""{_LIVE_REDIRECT_SENTINEL}
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=index.html">
<title>Run completed</title></head>
<body><a href="index.html">Continue to the run index</a></body></html>
"""
    return _write_html_atomic(document, path)


def _render_view_directory(
    args: argparse.Namespace,
    log_dir: Path,
    *,
    force: bool,
    quiet: bool,
    refresh_seconds: int | None,
) -> _DirectoryRenderResult:
    from inspect_robots import cli

    """Render one incremental logs-directory pass and return its output paths."""
    from inspect_robots import read_eval_log

    if args.out == "-":
        raise SystemExit("-o - cannot be used with a logs directory; pass an output directory")

    log_paths = sorted(path for path in log_dir.glob("*.json") if path.is_file())
    if not log_paths and not args.serve:
        raise SystemExit(f"no top-level *.json logs found in {log_dir}")

    out_dir = log_dir / "html" if args.out is None else Path(args.out)
    if (out_dir.exists() or out_dir.is_symlink()) and not out_dir.is_dir():
        if args.out is not None:
            raise SystemExit(f"--out {out_dir} is not a directory; pass an output directory")
        raise SystemExit(
            f"default output path {out_dir} exists and is not a directory; move it or pass -o DIR"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    page_names = _directory_page_names(log_paths)
    # Derived from this pass's own glob, not the caller's earlier one: a live
    # log appearing between the two globs must not have its freshly rendered
    # page immediately stubbed as an orphan.
    backed_live_pages = {page_names[path] for path in log_paths if path.name.endswith(".live.json")}
    entries: list[IndexEntry] = []
    pages_written = 0
    bytes_written = 0
    total = len(log_paths)
    # One pass, one parse per log, nothing retained beyond its row: a log that
    # fails to read or render (e.g. replaced mid-run by a concurrent writer)
    # becomes an error row instead of sinking the index.
    for index, log_path in enumerate(log_paths, start=1):
        page_name = page_names[log_path]
        page_path = out_dir / page_name
        is_live_path = log_path.name.endswith(".live.json")
        try:
            source_stat = log_path.stat()
            log = read_eval_log(str(log_path))
            suppressed_tier = args.serve or args.no_video
            # The suppressed-tier stamp sits a full 2 microseconds below the
            # source mtime, and the gate allows 1 microsecond of slack:
            # filesystems round timestamps to their tick (100 ns on NTFS), so
            # a 1 ns delta would floor below its own target and re-render
            # suppressed pages on every serve tick on Windows.
            stamp_ns = max(
                0,
                source_stat.st_mtime_ns - (_SUPPRESSED_STAMP_DELTA_NS if suppressed_tier else 0),
            )
            render_page = (
                force
                or log.status == "started"
                or not page_path.exists()
                or page_path.stat().st_mtime_ns < stamp_ns - _STAMP_TICK_SLACK_NS
            )
            if render_page:
                if not quiet:
                    print(f"[{index}/{total}] rendering {log_path.name}", file=sys.stderr)
                bytes_written += cli._render_log_page(
                    log,
                    log_path,
                    page_path,
                    no_frames=args.no_frames,
                    frames_budget=args.frames_budget,
                    live_frames_budget=(
                        args.live_frames_budget if args.serve and log.status == "started" else None
                    ),
                    refresh_seconds=(
                        _SERVE_LIVE_RERENDER_SECONDS
                        if args.serve and log.status == "started"
                        else None
                    ),
                    atomic=log.status == "started",
                    no_video=args.no_video,
                    serve_pass=args.serve,
                    # Pages sit beside index.html in out_dir.
                    index_href="index.html",
                )
                if log.status != "started":
                    # Started pages re-render every pass by design (plan 0058);
                    # the two-level stamp is a completed-page contract only.
                    os.utime(
                        page_path,
                        ns=(source_stat.st_atime_ns, stamp_ns),
                    )
                pages_written += 1
            entries.append(_index_entry(log, log_path, page_name))
        except FileNotFoundError:
            if is_live_path:
                continue
            exc = FileNotFoundError(log_path)
            print(f"warning: could not read or render {log_path.name}: {exc}", file=sys.stderr)
            entries.append(_unreadable_index_entry(log_path, exc))
        except Exception as exc:
            print(f"warning: could not read or render {log_path.name}: {exc}", file=sys.stderr)
            entries.append(_unreadable_index_entry(log_path, exc))

    for live_page in out_dir.glob("*.live.html"):
        if live_page.name in backed_live_pages:
            continue
        try:
            with live_page.open(encoding="utf-8", errors="replace") as handle:
                if handle.readline().rstrip("\r\n") == _LIVE_REDIRECT_SENTINEL:
                    continue
            bytes_written += cli._write_live_redirect_stub(live_page)
        except OSError as exc:
            print(f"warning: could not redirect {live_page.name}: {exc}", file=sys.stderr)

    index_path = out_dir / "index.html"
    bytes_written += _write_html_atomic(
        render_index(entries, refresh_seconds=refresh_seconds),
        index_path,
    )
    if not quiet:
        print(
            f"index: {index_path} "
            f"({total} logs, {pages_written} pages, {bytes_written / 1_000_000:.1f} MB)"
        )
    return _DirectoryRenderResult(out_dir=out_dir, index_path=index_path)


class _QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static reports without per-request stderr logging."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress the standard request log."""
        return None


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _serve_display_urls(bind_host: str, port: int) -> tuple[str, ...]:
    """Return the URLs shown for one bound address and socket-derived port."""
    display_host = socket.gethostname() if bind_host == "0.0.0.0" else bind_host
    primary = f"http://{display_host}:{port}/"
    if bind_host == "0.0.0.0":
        return primary, f"http://localhost:{port}/"
    return (primary,)


def _serve_open_url(bind_host: str, port: int) -> str:
    """Return the reachable URL that ``--open`` should target."""
    open_host = "127.0.0.1" if bind_host in _LOOPBACK_HOSTS | {"0.0.0.0"} else bind_host
    return f"http://{open_host}:{port}/"


def _raise_keyboard_interrupt(_signum: int, _frame: FrameType | None) -> NoReturn:
    """Map a termination signal onto the normal Ctrl-C shutdown path."""
    raise KeyboardInterrupt


def _serve_view_directory(
    args: argparse.Namespace,
    log_dir: Path,
    render_result: _DirectoryRenderResult,
) -> int:
    from inspect_robots import cli

    """Serve rendered logs and refresh the artifacts incrementally until stopped."""
    bind_host = cast(str, args.host)
    port = cast(int, args.port)
    handler = partial(_QuietHTTPRequestHandler, directory=str(render_result.out_dir))
    try:
        server = cli.ThreadingHTTPServer((bind_host, port), handler)
    except OSError as exc:
        raise SystemExit(str(exc)) from exc

    actual_port = int(server.server_address[1])
    urls = cli._serve_display_urls(bind_host, actual_port)
    # ThreadingHTTPServer.__init__ has already bound and started listening, so
    # printing the URLs and opening one before serve_forever cannot race a request.
    print(f"serving logs at: {urls[0]}")
    for url in urls[1:]:
        print(f"                 {url}")
    if bind_host in _LOOPBACK_HOSTS:
        print(
            _styled(
                "serving to this machine only; pass --host 0.0.0.0 to serve to your network",
                _DIM,
            )
        )
    else:
        print(
            _styled(
                "serving to your network: anyone who can reach this machine can view these logs",
                _YELLOW,
            )
        )
    print("press Ctrl-C to stop")

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    sigterm_installed = False
    server_thread_started = False
    previous_live_set: set[str] = set()
    accumulated_seconds = 0
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        sigterm_installed = True
        server_thread.start()
        server_thread_started = True
        if args.open:
            open_url = _serve_open_url(bind_host, actual_port)
            cli._open_browser(open_url, open_url)
        while True:
            cli._serve_sleep(_SERVE_LIVE_RERENDER_SECONDS)
            accumulated_seconds += _SERVE_LIVE_RERENDER_SECONDS
            live_set = {path.name for path in log_dir.glob("*.live.json")}
            if not (
                live_set != previous_live_set
                or live_set
                or accumulated_seconds >= _SERVE_RERENDER_SECONDS
            ):
                continue
            try:
                cli._render_view_directory(
                    args,
                    log_dir,
                    force=False,
                    quiet=True,
                    refresh_seconds=(
                        _SERVE_LIVE_RERENDER_SECONDS if live_set else _SERVE_RERENDER_SECONDS
                    ),
                )
            except (Exception, SystemExit) as exc:
                print(f"warning: re-render failed: {exc}", file=sys.stderr)
                # Reset the accumulator on failure too: an idle directory with
                # a persistent render error must warn once per baseline period,
                # not once per 2s tick.
                accumulated_seconds = 0
            else:
                previous_live_set = live_set
                accumulated_seconds = 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            # shutdown() waits on an event only serve_forever sets; calling it
            # when the serve thread never started would hang forever.
            if server_thread_started:
                server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                if sigterm_installed:
                    # A non-Python-installed prior handler reads back as None,
                    # which signal.signal rejects; SIG_DFL is the sane restore.
                    restore = signal.SIG_DFL if previous_sigterm is None else previous_sigterm
                    signal.signal(signal.SIGTERM, restore)


def _cmd_view_directory(args: argparse.Namespace, log_dir: Path) -> int:
    """Render a logs directory and optionally serve it until stopped."""

    from inspect_robots import cli

    live_set = {path.name for path in log_dir.glob("*.live.json")}
    render_result = cli._render_view_directory(
        args,
        log_dir,
        force=args.force,
        quiet=False,
        refresh_seconds=(
            (_SERVE_LIVE_RERENDER_SECONDS if live_set else _SERVE_RERENDER_SECONDS)
            if args.serve
            else None
        ),
    )
    if args.serve:
        return cli._serve_view_directory(args, log_dir, render_result)
    if args.open:
        index_uri = render_result.index_path.resolve().as_uri()
        cli._open_browser(index_uri, render_result.index_path)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Render one saved log or a logs directory to UTF-8 HTML artifacts."""

    from inspect_robots import cli

    if args.port is not None and not args.serve:
        raise SystemExit("--port requires --serve")
    if args.host is not None and not args.serve:
        raise SystemExit("--host requires --serve")
    if args.port is not None and not (0 <= args.port <= 65535):
        raise SystemExit(f"--port must be between 0 and 65535, got {args.port}")
    args.port = 8300 if args.port is None else args.port
    args.host = "127.0.0.1" if args.host is None else args.host

    if not (math.isfinite(args.frames_budget) and args.frames_budget >= 0):
        raise SystemExit("--frames-budget must be a non-negative finite number")
    if not (math.isfinite(args.live_frames_budget) and args.live_frames_budget >= 0):
        raise SystemExit("--live-frames-budget must be a non-negative finite number")

    log_path = Path(args.log)
    if args.serve and not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    if log_path.is_dir():
        return _cmd_view_directory(args, log_path)
    if args.serve:
        raise SystemExit("--serve requires a logs directory")

    stdout_mode = args.out == "-"
    if stdout_mode and args.open:
        raise SystemExit("--open cannot be used with -o -: no file to open")
    out_path = (
        None
        if stdout_mode
        else (log_path.with_suffix(".html") if args.out is None else Path(args.out))
    )
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out {out_path} is a directory; pass an HTML file path")
    if out_path is not None and out_path.resolve() == log_path.resolve():
        # The one data-loss path in this command: rendering over the log it reads.
        raise SystemExit(f"--out {out_path} would overwrite the input log; pass a different path")

    from inspect_robots import read_eval_log

    document_size = cli._render_log_page(
        read_eval_log(str(log_path)),
        log_path,
        out_path,
        no_frames=args.no_frames,
        frames_budget=args.frames_budget,
        no_video=args.no_video,
    )
    if stdout_mode:
        return 0

    file_path = cast(Path, out_path)
    size_suffix = f" ({document_size / 1_000_000:.1f} MB)" if document_size > 1_000_000 else ""
    print(f"wrote {file_path}{size_suffix}")
    if args.open:
        cli._open_browser(file_path.resolve().as_uri(), file_path)
    return 0
