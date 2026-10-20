"""The ``inspect_robots`` command-line interface.

Subcommands:

- ``inspect-robots list [tasks|policies|embodiments|scorers|sinks|operator_inputs]`` — show
  registered components (builtins + installed plugins).
- ``inspect-robots run --task T --policy P --embodiment E`` — run an eval, resolving
  components from the registry. ``--instruction`` runs an operator-written
  ad-hoc task, while ``--auto-task`` and repeatable ``-A k=v`` generate one
  from the initial camera frames. Pass constructor args with ``-T/-P/-E k=v``;
  ``--epochs``, ``--fail-on-error``, and ``--store-frames`` tune the run. The
  written log's path is printed at the end.
- ``inspect-robots eval-set TASK [TASK ...] --policy P --embodiment E`` — run several
  registered tasks (exact names or ``fnmatch`` globs, e.g. ``'kitchenbench/*'``) against
  one resolved policy/embodiment pair via
  [`eval_set`][inspect_robots.eval.eval_set]. Prints one status line and a compact
  per-task row instead of a full summary per task.
- ``inspect-robots inspect LOG.json [--transcript] [--wire [CALL]]`` — print a
  saved eval log and optionally append policy conversations or captured wire calls.
- ``inspect-robots summarize LOG.json [--model M]`` — distill a saved eval log
  into a deterministic digest or model-written learnings file.
- ``inspect-robots view LOG.json|LOG_DIR [-o PATH] [--open] [--serve]`` — render
  one saved eval log or a browsable directory index as self-contained HTML,
  optionally serving a directory until stopped.
- ``inspect-robots video LOG.json`` — render a ``--store-frames`` run's stored
  camera frames to one MP4 per (trial, camera) stream via the ffmpeg binary.
- ``inspect-robots setup`` — interactively configure defaults and camera devices.

Zero-config form (plan 0005): ``inspect-robots "place the spoon on the plate"``
is sugar for ``run --instruction "..."`` — a single ad-hoc scene on the user's
default policy/embodiment (flags > ``INSPECT_ROBOTS_POLICY``/``_EMBODIMENT``
env vars > ``~/.config/inspect-robots/config.ini``). The sugar only fires for
a first argument with interior whitespace, so a mistyped subcommand
(``inspect-robots isnpect``) errors instead of starting a robot rollout;
single-word instructions use the explicit ``run --instruction`` form.

Layout: this module owns the parser (``build_parser``), the dispatch
(``main``), and the instruction sugar; each subcommand's handler and its
helpers live in the private ``inspect_robots._cli_commands`` package. Every
name this module defined before is re-exported from here unchanged except the
private ``_format_metric`` helper, which now lives only in the new package,
and moved code reaches the collaborators tests patch (``claim_devices``,
``_resolve_components``, ``render_html``, ``_print_degraded``, and the
``view --serve`` cluster) through this module's namespace at call time, so
monkeypatching ``inspect_robots.cli.<name>`` keeps intercepting them.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from http.server import ThreadingHTTPServer as ThreadingHTTPServer
from pathlib import Path

from inspect_robots._claims import claim_devices as claim_devices
from inspect_robots._cli_commands._parser import (
    _add_config_arg as _add_config_arg,
)
from inspect_robots._cli_commands._parser import (
    _add_shared_eval_args as _add_shared_eval_args,
)
from inspect_robots._cli_commands._parser import (
    _port_number as _port_number,
)
from inspect_robots._cli_commands._parser import build_parser as build_parser
from inspect_robots._cli_commands._shared import (
    _BOLD as _BOLD,
)
from inspect_robots._cli_commands._shared import (
    _BOLD_BRIGHT_MAGENTA as _BOLD_BRIGHT_MAGENTA,
)
from inspect_robots._cli_commands._shared import _CYAN as _CYAN
from inspect_robots._cli_commands._shared import _DIM as _DIM
from inspect_robots._cli_commands._shared import _ENV_BY_KIND as _ENV_BY_KIND
from inspect_robots._cli_commands._shared import _GREEN as _GREEN
from inspect_robots._cli_commands._shared import (
    _KIND_BY_PLURAL as _KIND_BY_PLURAL,
)
from inspect_robots._cli_commands._shared import (
    _LIVE_FRAMES_BUDGET_MB as _LIVE_FRAMES_BUDGET_MB,
)
from inspect_robots._cli_commands._shared import _OUTCOME_PHRASES as _OUTCOME_PHRASES
from inspect_robots._cli_commands._shared import _PLURAL_BY_KIND as _PLURAL_BY_KIND
from inspect_robots._cli_commands._shared import _RED as _RED
from inspect_robots._cli_commands._shared import _SUBCOMMANDS as _SUBCOMMANDS
from inspect_robots._cli_commands._shared import _YELLOW as _YELLOW
from inspect_robots._cli_commands._shared import (
    DEFAULT_RERUN_CONNECT_URL as DEFAULT_RERUN_CONNECT_URL,
)
from inspect_robots._cli_commands._shared import _parse_kvs as _parse_kvs
from inspect_robots._cli_commands._shared import _styled as _styled
from inspect_robots._cli_commands.components import (
    _announce_components as _announce_components,
)
from inspect_robots._cli_commands.components import (
    _apply_epochs_or_exit as _apply_epochs_or_exit,
)
from inspect_robots._cli_commands.components import (
    _check_shared_run_conflicts as _check_shared_run_conflicts,
)
from inspect_robots._cli_commands.components import _config_args as _config_args
from inspect_robots._cli_commands.components import _match_tasks as _match_tasks
from inspect_robots._cli_commands.components import _pick_component as _pick_component
from inspect_robots._cli_commands.components import (
    _pick_sim_embodiment as _pick_sim_embodiment,
)
from inspect_robots._cli_commands.components import (
    _resolve_components as _resolve_components,
)
from inspect_robots._cli_commands.components import (
    _resolve_or_exit as _resolve_or_exit,
)
from inspect_robots._cli_commands.components import (
    _ResolvedComponents as _ResolvedComponents,
)
from inspect_robots._cli_commands.config import _cmd_config as _cmd_config
from inspect_robots._cli_commands.doctor import _cmd_doctor as _cmd_doctor
from inspect_robots._cli_commands.eval_set import (
    _EVAL_SET_OUTPUT_LOCK as _EVAL_SET_OUTPUT_LOCK,
)
from inspect_robots._cli_commands.eval_set import (
    _cmd_eval_set as _cmd_eval_set,
)
from inspect_robots._cli_commands.eval_set import (
    _run_cli_eval_set as _run_cli_eval_set,
)
from inspect_robots._cli_commands.inspect import _WIRE_BLOB_RE as _WIRE_BLOB_RE
from inspect_robots._cli_commands.inspect import (
    _available_wire_trials as _available_wire_trials,
)
from inspect_robots._cli_commands.inspect import _cmd_inspect as _cmd_inspect
from inspect_robots._cli_commands.inspect import (
    _load_wire_trials as _load_wire_trials,
)
from inspect_robots._cli_commands.inspect import (
    _print_wire_call as _print_wire_call,
)
from inspect_robots._cli_commands.inspect import (
    _print_wire_capture as _print_wire_capture,
)
from inspect_robots._cli_commands.inspect import (
    _print_wire_table as _print_wire_table,
)
from inspect_robots._cli_commands.inspect import (
    _wire_blob_shas as _wire_blob_shas,
)
from inspect_robots._cli_commands.inspect import _wire_request as _wire_request
from inspect_robots._cli_commands.inspect import _WireTrial as _WireTrial
from inspect_robots._cli_commands.list import _cmd_list as _cmd_list
from inspect_robots._cli_commands.run import _cmd_run as _cmd_run
from inspect_robots._cli_commands.sessions import (
    _announce_live_view as _announce_live_view,
)
from inspect_robots._cli_commands.sessions import _attended as _attended
from inspect_robots._cli_commands.sessions import (
    _build_and_announce_guardrails as _build_and_announce_guardrails,
)
from inspect_robots._cli_commands.sessions import _build_grader as _build_grader
from inspect_robots._cli_commands.sessions import (
    _build_guardrails as _build_guardrails,
)
from inspect_robots._cli_commands.sessions import (
    _build_operator_session as _build_operator_session,
)
from inspect_robots._cli_commands.sessions import (
    _build_speaker_sink as _build_speaker_sink,
)
from inspect_robots._cli_commands.sessions import (
    _build_voice_input as _build_voice_input,
)
from inspect_robots._cli_commands.sessions import (
    _close_speaker_sink as _close_speaker_sink,
)
from inspect_robots._cli_commands.sessions import (
    _close_voice_input as _close_voice_input,
)
from inspect_robots._cli_commands.sessions import (
    _connect_grader_session as _connect_grader_session,
)
from inspect_robots._cli_commands.sessions import (
    _headless_session as _headless_session,
)
from inspect_robots._cli_commands.sessions import (
    _select_grader_name as _select_grader_name,
)
from inspect_robots._cli_commands.sessions import (
    _start_speaker_sink as _start_speaker_sink,
)
from inspect_robots._cli_commands.sessions import (
    _start_voice_input as _start_voice_input,
)
from inspect_robots._cli_commands.setup import _cmd_setup as _cmd_setup
from inspect_robots._cli_commands.summarize import _cmd_summarize as _cmd_summarize
from inspect_robots._cli_commands.summary import (
    _has_policy_transcripts as _has_policy_transcripts,
)
from inspect_robots._cli_commands.summary import (
    _outcome_line as _outcome_line,
)
from inspect_robots._cli_commands.summary import (
    _print_degraded as _print_degraded,
)
from inspect_robots._cli_commands.summary import (
    _print_eval_set_summary as _print_eval_set_summary,
)
from inspect_robots._cli_commands.summary import (
    _print_policy_transcripts as _print_policy_transcripts,
)
from inspect_robots._cli_commands.summary import (
    _print_run_summary as _print_run_summary,
)
from inspect_robots._cli_commands.summary import (
    _print_step_limit_notice as _print_step_limit_notice,
)
from inspect_robots._cli_commands.summary import (
    _print_survivor_warning as _print_survivor_warning,
)
from inspect_robots._cli_commands.summary import (
    _render_chat_transcript as _render_chat_transcript,
)
from inspect_robots._cli_commands.summary import (
    _seconds_horizon as _seconds_horizon,
)
from inspect_robots._cli_commands.summary import (
    _seconds_horizon_text as _seconds_horizon_text,
)
from inspect_robots._cli_commands.summary import _step_limit_count as _step_limit_count
from inspect_robots._cli_commands.video import _cmd_video as _cmd_video
from inspect_robots._cli_commands.view import (
    _LIVE_REDIRECT_SENTINEL as _LIVE_REDIRECT_SENTINEL,
)
from inspect_robots._cli_commands.view import (
    _LOOPBACK_HOSTS as _LOOPBACK_HOSTS,
)
from inspect_robots._cli_commands.view import (
    _SERVE_LIVE_RERENDER_SECONDS as _SERVE_LIVE_RERENDER_SECONDS,
)
from inspect_robots._cli_commands.view import (
    _SERVE_RERENDER_SECONDS as _SERVE_RERENDER_SECONDS,
)
from inspect_robots._cli_commands.view import (
    _STAMP_TICK_SLACK_NS as _STAMP_TICK_SLACK_NS,
)
from inspect_robots._cli_commands.view import (
    _SUPPRESSED_STAMP_DELTA_NS as _SUPPRESSED_STAMP_DELTA_NS,
)
from inspect_robots._cli_commands.view import (
    _cmd_view as _cmd_view,
)
from inspect_robots._cli_commands.view import (
    _cmd_view_directory as _cmd_view_directory,
)
from inspect_robots._cli_commands.view import (
    _directory_page_names as _directory_page_names,
)
from inspect_robots._cli_commands.view import (
    _DirectoryRenderResult as _DirectoryRenderResult,
)
from inspect_robots._cli_commands.view import (
    _index_entry as _index_entry,
)
from inspect_robots._cli_commands.view import (
    _index_instruction as _index_instruction,
)
from inspect_robots._cli_commands.view import (
    _index_termination as _index_termination,
)
from inspect_robots._cli_commands.view import _open_browser as _open_browser
from inspect_robots._cli_commands.view import (
    _QuietHTTPRequestHandler as _QuietHTTPRequestHandler,
)
from inspect_robots._cli_commands.view import (
    _raise_keyboard_interrupt as _raise_keyboard_interrupt,
)
from inspect_robots._cli_commands.view import (
    _render_log_page as _render_log_page,
)
from inspect_robots._cli_commands.view import (
    _render_view_directory as _render_view_directory,
)
from inspect_robots._cli_commands.view import (
    _serve_display_urls as _serve_display_urls,
)
from inspect_robots._cli_commands.view import (
    _serve_open_url as _serve_open_url,
)
from inspect_robots._cli_commands.view import (
    _serve_sleep as _serve_sleep,
)
from inspect_robots._cli_commands.view import (
    _serve_view_directory as _serve_view_directory,
)
from inspect_robots._cli_commands.view import (
    _unreadable_index_entry as _unreadable_index_entry,
)
from inspect_robots._cli_commands.view import (
    _write_html as _write_html,
)
from inspect_robots._cli_commands.view import (
    _write_html_atomic as _write_html_atomic,
)
from inspect_robots._cli_commands.view import (
    _write_live_redirect_stub as _write_live_redirect_stub,
)
from inspect_robots._dotenv import init_dotenv as init_dotenv
from inspect_robots._html import render_html as render_html


def _apply_instruction_sugar(argv: list[str]) -> list[str]:
    """``inspect-robots "wipe the table"`` → ``run --instruction "wipe the table"``.

    Fires only for a first argument that is not a subcommand or flag AND has
    interior whitespace after stripping: a mistyped subcommand
    (``inspect-robots isnpect``) or a whitespace-padded one
    (``inspect-robots " list "``) must never silently start a rollout.
    """
    if not argv:
        return argv
    tok = argv[0].strip()
    if tok in _SUBCOMMANDS or tok.startswith("-") or not any(ch.isspace() for ch in tok):
        return argv
    return ["run", "--instruction", argv[0], *argv[1:]]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch one subcommand, and return its process exit code."""
    init_dotenv(os.environ)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(_apply_instruction_sugar(argv_list))
    if config_override := getattr(args, "config", None):
        # os.environ, not a local mapping: external plugins (inspect-robots-yam's
        # default loader) re-read the config from os.environ while constructing
        # components, and subprocesses inherit it.
        override_path = Path(config_override).expanduser()
        if not override_path.is_absolute():
            override_path = Path.cwd() / override_path
        os.environ["INSPECT_ROBOTS_CONFIG"] = str(override_path)
    if args.command == "list":
        return _cmd_list(args.what)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "eval-set":
        return _cmd_eval_set(args)
    if args.command == "inspect":
        return _cmd_inspect(
            args.log,
            transcript=args.transcript,
            wire=args.wire,
            trial=args.trial,
        )
    if args.command == "summarize":
        return _cmd_summarize(args)
    if args.command == "view":
        return _cmd_view(args)
    if args.command == "video":
        return _cmd_video(args)
    if args.command == "config":
        return _cmd_config(args)
    if args.command == "setup":
        return _cmd_setup()
    if args.command == "doctor":
        return _cmd_doctor(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
