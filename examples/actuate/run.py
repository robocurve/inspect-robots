"""Run the attended Actuate conference demo in a continuous sequence.

Run with:  python examples/actuate/run.py [-- extra inspect-robots run args]
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _roster import EFFORT, ROSTER
from _thermal import (
    COOL_POLL_S,
    COOL_PROBE_ERRORS_GIVE_UP,
    TEMP_RESUME_C,
    TEMP_WAIT_C,
    ThermalProbeError,
    ThermalProbeUnavailable,
    config_channels,
    hottest,
    read_motor_temps,
)

from inspect_robots.log import read_eval_log

HERE = Path(__file__).parent
# The setup wizard writes the rig folder's config.ini; the XDG-global config
# can go stale (it pointed the top cam at the D435's mono IR node). Always
# run against the rig folder's file. Booth-editable; a later --config in the
# passthrough args after "--" overrides it.
RIG_CONFIG = Path.home() / "robocurve" / "rig-1" / "config.ini"
# Rig cheat-sheet notes passed to the embodiment docs channel (-E docs_extra),
# concatenated from whichever of these exist; booth-editable like the roster.
DOCS_EXTRA_PATHS = [
    Path.home() / "robocurve" / "test-dir" / f"rig-{name}.md"
    for name in ("facts", "formulas", "advice")
]
# Seconds between evals; the longer pause applies after an eval that produced
# no log (hardware or config failure) so a broken booth does not hammer retries.
PAUSE_S = 10
FAILURE_PAUSE_S = 30
# Consecutive log-less evals (hardware or config faults fail before any model
# call) before the loop halts loudly instead of spinning junk all night; a
# 2026-08-18 arm joint-limit fault produced ~385 junk evals in 5 hours.
FAILURE_STREAK_STOP = 10
STATE_DIR = HERE / "state"
LOGS_DIR = HERE / "logs"
STATUS_PATH = STATE_DIR / "status.json"
RESULTS_PATH = STATE_DIR / "results.jsonl"
MEDIA_DIR = STATE_DIR / "media"


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


def _role_args(flag: str, model: dict[str, Any]) -> list[str]:
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


def _policy_args(model: dict[str, Any]) -> list[str]:
    """The complete -P set for the drawn test-taker, plus the shared effort."""
    kvs = (
        dict(model["policy"])
        if isinstance(model.get("policy"), dict)
        else {key: model[key] for key in ("model", "base_url", "api_key_env")}
    )
    kvs["effort"] = EFFORT
    args: list[str] = []
    for key, value in kvs.items():
        args += ["-P", f"{key}={value}"]
    return args


def _docs_extra_args() -> list[str]:
    """-E docs_extra from whichever rig cheat-sheet files exist, else nothing."""
    texts = []
    for path in DOCS_EXTRA_PATHS:
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    if not texts:
        return []
    return ["-E", "docs_extra=" + "\n\n".join(texts)]


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


def _render_clips(log_path: Path | None, eval_index: int) -> dict[str, str]:
    """Render per-camera MP4 clips for a completed eval; empty on any failure.

    The video command names outputs by scene/epoch only, identical across
    evals, so render into a scratch dir and move to per-eval names.
    """
    if log_path is None:
        return {}
    scratch = MEDIA_DIR / f"render_{eval_index}"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["inspect-robots", "video", str(log_path), "--out", str(scratch), "--fps", "30"],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        return {}
    clips: dict[str, str] = {}
    for path in sorted(scratch.glob("*.mp4")):
        for cam in ("left", "top", "right"):
            if cam in path.name:
                destination = MEDIA_DIR / f"eval{eval_index:04d}_{cam}.mp4"
                os.replace(path, destination)
                clips[cam] = destination.name
    shutil.rmtree(scratch, ignore_errors=True)
    return clips


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


def _thermal_gate(eval_index: int, gate_state: dict[str, Any]) -> None:
    if gate_state["disabled"]:
        return

    try:
        hottest_motor = hottest(read_motor_temps(gate_state["channels"]))
    except ThermalProbeUnavailable as error:
        print(f"warning: thermal probe unavailable; disabling gate: {error}", flush=True)
        gate_state["disabled"] = True
        return
    except ThermalProbeError as error:
        print(f"warning: thermal probe failed; proceeding with eval: {error}", flush=True)
        return

    if hottest_motor.max_c < TEMP_WAIT_C:
        print(
            f"motor temps ok: max {hottest_motor.max_c:.1f}C ({hottest_motor.label})",
            flush=True,
        )
        return

    cooling_since = _utc_now()
    probe_error_streak = 0
    while True:
        cooling = {
            "max_c": hottest_motor.max_c,
            "hottest": hottest_motor.label,
            "wait_c": TEMP_WAIT_C,
            "resume_c": TEMP_RESUME_C,
            "since": cooling_since,
        }
        _write_status({"eval_index": eval_index, "cooling": cooling})
        print(
            f"Cooling motors: {hottest_motor.max_c:.1f}C at {hottest_motor.label}. "
            f"Evals resume below {TEMP_RESUME_C:g}C.",
            flush=True,
        )
        time.sleep(COOL_POLL_S)
        try:
            hottest_motor = hottest(read_motor_temps(gate_state["channels"]))
        except ThermalProbeError as error:
            probe_error_streak += 1
            print(
                f"warning: thermal probe failed while cooling "
                f"({probe_error_streak}/{COOL_PROBE_ERRORS_GIVE_UP}): {error}",
                flush=True,
            )
            if probe_error_streak >= COOL_PROBE_ERRORS_GIVE_UP:
                print(
                    "THERMAL GATE: TOO MANY PROBE ERRORS; PROCEEDING WITH THE EVAL.",
                    flush=True,
                )
                return
            continue

        probe_error_streak = 0
        if hottest_motor.max_c < TEMP_RESUME_C:
            return


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
        # The leaderboard reads the vlm grader's verdict through the operator
        # scorer; pin it so a rig config's scorer choice cannot unscore evals.
        "--scorer",
        "operator",
        "--auto-task",
        *_role_args("-A", tasker),
        "--grader",
        "vlm",
        *_role_args("-G", grader),
        "--policy",
        "agent",
        *_policy_args(test_taker),
        *_docs_extra_args(),
        "--log-dir",
        str(LOGS_DIR.resolve()),
        *extra_args,
    ]


def main() -> None:
    if not RIG_CONFIG.is_file():
        raise SystemExit(f"rig config not found: {RIG_CONFIG}\nfix: edit RIG_CONFIG in run.py")
    # Resolved once from RIG_CONFIG only: a --config override after "--"
    # re-routes the eval but not the gate's probe channels, so on a rig with
    # different channel names the gate degrades to per-round warnings.
    channels = config_channels(RIG_CONFIG)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    extra_args = _extra_args(sys.argv[1:])
    roster = list(ROSTER.items())
    eval_index = _next_eval_index()
    failure_streak = 0
    gate_state: dict[str, Any] = {"channels": channels, "disabled": False}

    try:
        while True:
            _thermal_gate(eval_index, gate_state)
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
            clips = _render_clips(log_path, eval_index)
            _append_result(
                {
                    "ts": _utc_now(),
                    "eval_index": eval_index,
                    "roles": roles,
                    "models": models,
                    "task": task,
                    "score": score,
                    "log": log_path.name if log_path else None,
                    "clips": clips,
                }
            )
            score_text = f"{score:.2f}" if score is not None else "unscored"
            log_text = log_path.name if log_path else "no log"
            print(f"Eval {eval_index} complete: score={score_text}, log={log_text}")

            failure_streak = 0 if log_path is not None else failure_streak + 1
            if failure_streak >= FAILURE_STREAK_STOP:
                raise SystemExit(
                    f"{failure_streak} consecutive evals produced no log; the rig "
                    "is likely faulted (check arms, CAN, cameras). Fix it and "
                    "relaunch."
                )
            pause = PAUSE_S if log_path is not None else FAILURE_PAUSE_S
            print(f"next eval in {pause}s (Ctrl-C to stop)", flush=True)
            time.sleep(pause)
            eval_index += 1
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
