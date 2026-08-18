"""Run the attended Actuate conference demo in a continuous sequence.

Run with:  python examples/actuate/run.py [-- extra inspect-robots run args]
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _roster import EFFORT, ROSTER

from inspect_robots.log import read_eval_log

HERE = Path(__file__).parent
# The setup wizard writes the rig folder's config.ini; the XDG-global config
# can go stale (it pointed the top cam at the D435's mono IR node). Always
# run against the rig folder's file. Booth-editable; a later --config in the
# passthrough args after "--" overrides it.
RIG_CONFIG = Path.home() / "robocurve" / "rig-1" / "config.ini"
STATE_DIR = HERE / "state"
LOGS_DIR = HERE / "logs"
STATUS_PATH = STATE_DIR / "status.json"
RESULTS_PATH = STATE_DIR / "results.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extra_args(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def _write_status(status: dict[str, Any]) -> None:
    temporary_path = STATUS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(status) + "\n", encoding="utf-8")
    os.replace(temporary_path, STATUS_PATH)


def _role_args(flag: str, model: dict[str, str]) -> list[str]:
    return [
        flag,
        f"model={model['model']}",
        flag,
        f"base_url={model['base_url']}",
        flag,
        f"api_key_env={model['api_key_env']}",
        flag,
        f"effort={EFFORT}",
    ]


def _final_log_paths() -> set[Path]:
    return {path for path in LOGS_DIR.glob("*.json") if not path.name.endswith(".live.json")}


def _newest_final_log(existing_paths: set[Path]) -> Path | None:
    try:
        return max(
            _final_log_paths() - existing_paths,
            key=lambda path: path.stat().st_mtime,
        )
    except (OSError, ValueError):
        return None


def _log_outcome(path: Path | None) -> tuple[float | None, str | None]:
    if path is None:
        return None, None
    try:
        log = read_eval_log(path)
        scores = [
            epoch["operator"]
            for sample in log.samples
            for epoch in sample.epochs
            if "operator" in epoch
        ]
        task = log.samples[0].instruction if log.samples else None
    except Exception:
        return None, None
    score = sum(scores) / len(scores) if scores else None
    return score, task if isinstance(task, str) else None


def _append_result(result: dict[str, Any]) -> None:
    with RESULTS_PATH.open("a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(result) + "\n")


def _positive_index(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def _next_eval_index() -> int:
    try:
        result_lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        result_lines = []

    for line in reversed(result_lines):
        try:
            result = json.loads(line)
            index = _positive_index(result.get("eval_index"))
        except (AttributeError, json.JSONDecodeError):
            continue
        if index is not None:
            return index + 1

    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        index = _positive_index(status.get("eval_index"))
    except (AttributeError, OSError, json.JSONDecodeError):
        index = None
    return index + 1 if index is not None else 1


def _command(
    tasker: dict[str, str],
    test_taker: dict[str, str],
    grader: dict[str, str],
    extra_args: list[str],
) -> list[str]:
    return [
        "inspect-robots",
        "run",
        "--config",
        str(RIG_CONFIG),
        "--auto-task",
        *_role_args("-A", tasker),
        "--grader",
        "vlm",
        *_role_args("-G", grader),
        "--policy",
        "agent",
        *_role_args("-P", test_taker),
        *(["-P", f"wire={test_taker['policy_wire']}"] if "policy_wire" in test_taker else []),
        "--log-dir",
        str(LOGS_DIR.resolve()),
        *extra_args,
    ]


def main() -> None:
    if not RIG_CONFIG.is_file():
        raise SystemExit(f"rig config not found: {RIG_CONFIG}\nfix: edit RIG_CONFIG in run.py")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    extra_args = _extra_args(sys.argv[1:])
    roster = list(ROSTER.items())
    eval_index = _next_eval_index()

    try:
        while True:
            tasker_name, tasker = random.choice(roster)
            test_taker_name, test_taker = random.choice(roster)
            grader_name, grader = random.choice(roster)
            roles = {
                "tasker": tasker_name,
                "test_taker": test_taker_name,
                "grader": grader_name,
            }
            models = {
                "tasker": tasker["model"],
                "test_taker": test_taker["model"],
                "grader": grader["model"],
            }
            _write_status(
                {
                    "eval_index": eval_index,
                    "started_at": _utc_now(),
                    "roles": roles,
                    "models": models,
                }
            )
            print(
                f"Eval {eval_index}: tasker={tasker_name}, test-taker={test_taker_name}, "
                f"grader={grader_name}",
                flush=True,
            )

            existing_logs = _final_log_paths()
            subprocess.run(
                _command(tasker, test_taker, grader, extra_args),
                check=False,
            )

            log_path = _newest_final_log(existing_logs)
            score, task = _log_outcome(log_path)
            _append_result(
                {
                    "ts": _utc_now(),
                    "eval_index": eval_index,
                    "roles": roles,
                    "models": models,
                    "task": task,
                    "score": score,
                    "log": log_path.name if log_path else None,
                }
            )
            score_text = f"{score:.2f}" if score is not None else "unscored"
            log_text = log_path.name if log_path else "no log"
            print(f"Eval {eval_index} complete: score={score_text}, log={log_text}")

            answer = input("Enter to run the next eval, q then Enter to quit: ")
            if answer.strip().lower() == "q":
                break
            eval_index += 1
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
