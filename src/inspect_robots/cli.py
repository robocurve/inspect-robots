"""The ``inspect_robots`` command-line interface.

Subcommands:

- ``inspect-robots list [tasks|policies|embodiments|scorers|sinks]`` — show registered
  components (builtins + installed plugins).
- ``inspect-robots run --task T --policy P --embodiment E`` — run an eval, resolving
  components from the registry. Pass constructor args with ``-T/-P/-E k=v``;
  ``--epochs``, ``--fail-on-error``, and ``--store-frames`` tune the run. The
  written log's path is printed at the end.
- ``inspect-robots inspect LOG.json`` — print a saved eval log.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from inspect_robots import __version__

_KIND_BY_PLURAL = {
    "tasks": "task",
    "policies": "policy",
    "embodiments": "embodiment",
    "scorers": "scorer",
    "sinks": "sink",
}


def _parse_value(text: str) -> Any:
    """Best-effort scalar parse for ``k=v`` CLI args (bool/int/float/None/str)."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def _parse_kvs(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key] = _parse_value(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect-robots",
        description="Inspect Robots — the Inspect AI for robotics.",
    )
    parser.add_argument("--version", action="version", version=f"inspect-robots {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="list registered components")
    p_list.add_argument(
        "what",
        nargs="?",
        choices=sorted(_KIND_BY_PLURAL),
        help="component kind to list (default: all)",
    )

    p_run = sub.add_parser("run", help="run an evaluation")
    p_run.add_argument("--task", required=True, help="registered task name")
    p_run.add_argument("--policy", required=True, help="registered policy name")
    p_run.add_argument("--embodiment", required=True, help="registered embodiment name")
    p_run.add_argument("-T", dest="task_args", action="append", metavar="k=v")
    p_run.add_argument("-P", dest="policy_args", action="append", metavar="k=v")
    p_run.add_argument("-E", dest="embodiment_args", action="append", metavar="k=v")
    p_run.add_argument("--log-dir", default="logs")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--epochs", type=int, default=None, help="override the task's epoch count")
    p_run.add_argument(
        "--fail-on-error",
        type=float,
        default=None,
        metavar="X",
        help="halt on PolicyErrors: 1 = first error, 0<X<1 = proportion, X>1 = count",
    )
    p_run.add_argument(
        "--store-frames",
        action="store_true",
        help="stream camera frames to <log-dir>/frames instead of keeping them in memory",
    )

    p_inspect = sub.add_parser("inspect", help="print a saved eval log")
    p_inspect.add_argument("log", help="path to an EvalLog JSON file")
    return parser


def _cmd_list(what: str | None) -> int:
    from inspect_robots.registry import registered

    plurals = [what] if what else sorted(_KIND_BY_PLURAL)
    for plural in plurals:
        kind = _KIND_BY_PLURAL[plural]
        names = sorted(registered(kind))
        print(f"{plural}:")
        for name in names:
            print(f"  - {name}")
        if not names:
            print("  (none)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from inspect_robots import eval
    from inspect_robots.logging import JsonLogSink
    from inspect_robots.registry import resolve

    task = resolve("task", args.task, **_parse_kvs(args.task_args))
    policy = resolve("policy", args.policy, **_parse_kvs(args.policy_args))
    embodiment = resolve("embodiment", args.embodiment, **_parse_kvs(args.embodiment_args))
    if args.epochs is not None:
        task = replace(task, epochs=args.epochs)

    # Construct the sink explicitly so we can tell the user where the log went.
    sink = JsonLogSink(args.log_dir)
    logs = eval(
        task,
        policy,
        embodiment,
        log_dir=args.log_dir,
        seed=args.seed,
        sinks=[sink],
        fail_on_error=args.fail_on_error if args.fail_on_error is not None else False,
        store_frames=args.store_frames,
    )
    log = logs[0]
    print(f"status: {log.status}")
    print(f"scenes: {log.results.total_scenes}  trials: {log.results.total_trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}")
    print(f"log: {sink.path}")
    return 0 if log.status == "success" else 1


def _cmd_inspect(path: str) -> int:
    from inspect_robots import read_eval_log

    log = read_eval_log(path)
    print(f"task:        {log.eval.task}")
    print(f"policy:      {log.eval.policy}")
    print(f"embodiment:  {log.eval.embodiment}")
    print(f"status:      {log.status}")
    print(f"created:     {log.eval.created}")
    print(f"git:         {log.eval.git_commit}")
    print(f"scenes:      {log.results.total_scenes}   trials: {log.results.total_trials}")
    print("metrics:")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}")
    print("scenes:")
    for scene in log.samples:
        reduced = "  ".join(f"{k}={v:.4g}" for k, v in sorted(scene.reduced.items()))
        print(f"  [{scene.status}] {scene.scene_id}: {reduced}")
    if log.error:
        print(f"error: {log.error}")
    return 0 if log.status == "success" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args.what)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "inspect":
        return _cmd_inspect(args.log)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
